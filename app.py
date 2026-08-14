import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import re
import ast

load_dotenv()

st.set_page_config(
    page_title="RepairLink Support", 
    page_icon="🛠️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Manrope:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: #221D14;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #EDE7D8;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid rgba(34,29,20,0.14);
        padding-top: 1rem;
    }
    
    div.stButton > button:first-child {
        background: #FFB800;
        color: #2A1E00;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 1rem;
        box-shadow: 0 4px 14px rgba(34,29,20,0.15);
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        height: auto;
    }
    
    div.stButton > button:hover {
        background: #ffc935;
        color: #2A1E00;
        box-shadow: 0 6px 15px rgba(34,29,20,0.1);
        transform: translateY(-2px);
    }

    [data-testid="stChatInput"] {
        border-radius: 99px !important;
        border: 1px solid rgba(34,29,20,0.14) !important;
        box-shadow: 0 10px 28px rgba(34,29,20,0.14) !important;
        padding: 0.5rem !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #221D14 !important;
    }

    [data-testid="stChatMessage"] {
        padding: 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 1.2rem;
        font-size: 14px;
        line-height: 1.65;
        animation: slideUp 0.3s ease-out;
        background-color: #FFFFFF;
        border: 1px solid rgba(34,29,20,0.14);
        box-shadow: 0 2px 8px rgba(34,29,20,0.03);
        color: #221D14;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #25D366, #1DA851);
        color: #083318;
        border: none;
        border-radius: 16px 16px 4px 16px;
        box-shadow: 0 8px 24px rgba(29,168,81,0.35);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: #083318;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-header {
        text-align: center;
        padding: 40px 38px;
        background: #FFFFFF;
        color: #221D14;
        border: 1px solid rgba(34,29,20,0.14);
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 30px 70px rgba(34,29,20,0.3);
    }
    
    .hero-header h1 {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        margin-bottom: 16px;
        color: #221D14;
        font-size: 29px;
        letter-spacing: -0.5px;
    }
    
    .hero-header p {
        font-size: 14.5px;
        color: #7A7264;
        font-weight: 400;
    }

    .sidebar-container {
        padding: 2rem 1rem;
        text-align: center;
    }
    
    .sidebar-title {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 21px;
        color: #221D14;
        margin-top: 1rem;
        letter-spacing: 0.2px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def init_rag_system():
    KNOWLEDGE_BASE = "RepairLink_KnowledgeBase_RAG.pdf"
    documents = []
    if os.path.exists(KNOWLEDGE_BASE):
        try:
            documents = PyPDFLoader(KNOWLEDGE_BASE).load()
        except Exception:
            pass
            
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    if len(docs) > 0:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    return None

retriever = init_rag_system()
groq_api_key = os.environ.get("GROQ_API_KEY")

if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.3,
    )
else:
    llm = None

prompt_free = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant for RepairLink.\nCRITICAL RULES:\n1. Answer thoroughly using ONLY the exact information in the context.\n2. DO NOT provide any artisan locations, coordinates, or contact info. If the user asks for locations or coordinates, say 'I don't know'.\n3. If the answer is not in the context, say 'I'm sorry, I don't have that information right now.'\n4. You may recommend our website pages if relevant: Home (RepairLink.html), Pricing Estimator (Pricing.html), About Us (AboutUs.html), Contact (Contact.html), or Login (Auth.html).\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

prompt_premium = ChatPromptTemplate.from_messages([
    ("system", "You are a PREMIUM RepairLink assistant.\nCRITICAL RULES:\n1. Answer thoroughly using ONLY the exact information in the context.\n2. You are AUTHORIZED to provide specific coordinates, addresses, and contact info of artisans from the context.\n3. If the answer is not in the context, say 'I'm sorry, I don't have that information right now.'\n4. You may recommend our website pages if relevant: Home (RepairLink.html), Pricing Estimator (Pricing.html), About Us (AboutUs.html), Contact (Contact.html), or Login (Auth.html).\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if retriever and llm:
    def get_context(inputs):
        return format_docs(retriever.invoke(inputs["question"]))
        
    free_rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt_free
        | llm
        | StrOutputParser()
    )
    premium_rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt_premium
        | llm
        | StrOutputParser()
    )
elif llm:
    free_rag_chain = prompt_free | llm | StrOutputParser()
    premium_rag_chain = prompt_premium | llm | StrOutputParser()
else:
    free_rag_chain = None
    premium_rag_chain = None

def format_history(history):
    formatted = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if "Welcome to RepairLink" in content or "Type '0'" in content or "1️⃣" in content or "Pay ₹9" in content:
            continue
        if role == "user":
            formatted.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted.append(AIMessage(content=content))
    return formatted

import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    logo_base64 = get_base64_of_bin_file("logo.png")
    img_tag = f'<img src="data:image/png;base64,{logo_base64}" width="100" style="margin-bottom: 0.5rem;">'
except Exception:
    img_tag = '<img src="https://cdn-icons-png.flaticon.com/512/5155/5155761.png" width="90" style="margin-bottom: 0.5rem;">'

st.markdown(f"""
<div class="hero-header">
    {img_tag}
    <h1>RepairLink Support</h1>
    <p>Seamless repairs, transparent pricing, and trusted local artisans.</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "MAIN_MENU"

MAIN_MENU_TEXT = """**Welcome to RepairLink!** 👋\n\nPlease choose an option below or ask me a question directly:\n\n**1️⃣** View our business hours  \n**2️⃣** Chat with our Support AI (Free)  \n**3️⃣** Find a Local Artisan (Premium)  \n**4️⃣** View pricing estimates  \n**5️⃣** Login / Access Dashboard  \n**6️⃣** About Us & Contact"""

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": MAIN_MENU_TEXT})

for message in st.session_state.messages:
    avatar = "logo.png" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt_text = st.chat_input("Type your message here...")

if st.session_state.chat_state == "MAIN_MENU":
    col1, col2 = st.columns(2)
    if col1.button("🕒 View Business Hours", use_container_width=True):
        prompt_text = "1"
    if col2.button("🤖 Chat with Support AI (Free)", use_container_width=True):
        prompt_text = "2"
    
    col3, col4 = st.columns(2)
    if col3.button("💎 Find a Local Artisan (Premium)", use_container_width=True):
        prompt_text = "3"
    if col4.button("💰 View Pricing Estimates", use_container_width=True):
        prompt_text = "4"
        
    col5, col6 = st.columns(2)
    if col5.button("🔐 Login / Access Dashboard", use_container_width=True):
        prompt_text = "5"
    if col6.button("ℹ️ About Us & Contact", use_container_width=True):
        prompt_text = "6"

if prompt_text:
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="logo.png"):
        message_placeholder = st.empty()
        response = ""
        
        if st.session_state.chat_state == "MAIN_MENU":
            if prompt_text.strip() == "1":
                response = "🕒 **Our Business Hours:**\n\n*   **Monday - Saturday:** 9:00 AM - 7:00 PM\n*   **Sunday:** Closed\n\n---\n*Type **0** to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "2":
                st.session_state.chat_state = "CUSTOM_QUESTION"
                response = "🤖 **Support AI Connected!** \n\nI can help answer questions about our services and policies. *(Note: Artisan coordinates are only available in Premium)*\n\nWhat would you like to know?\n\n---\n*Type **0** anytime to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "3":
                st.session_state.chat_state = "PREMIUM_PAYMENT"
                response = "💎 **Unlock Artisan Coordinates**\n\nTo view the exact coordinates, locations, and contact details of our trusted local artisans, you must upgrade to Premium.\n\n**Price:** ₹9 (One-time fee)\n\n*(Please go to your [User Dashboard](UserDashboard.html) to scan the QR code and make the payment of ₹9. Once paid, type **Pay ₹9** here to confirm, or type **0** to cancel and return to the menu)*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "4":
                response = "💰 **RepairLink Pricing Estimates:**\n\nOur transparent pricing scales based on the issue and device:\n*   **Laptops / PCs:** Screen (₹3k-4.5k), Battery (₹1.8k-2.8k)\n*   **Mobiles:** Screen (₹1.6k-3.4k), Battery (₹1.1k-1.8k)\n*   **Shoes, Bags & Clothes:** ₹100 - ₹300\n\n*Note: We apply a flat ₹50 platform convenience fee on top of the artisan's quote.*\n\nWant to calculate your exact cost? [Try our Pricing Estimator here!](Pricing.html)\n\n---\n*Type **0** to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "5":
                response = "🔐 **Login & Dashboards:**\n\n*   **Customers:** Manage your active repairs and track artisan status in your [User Dashboard](UserDashboard.html).\n*   **Artisans / Kaarigars:** Accept jobs and manage your earnings in the [Kaarigar Dashboard](KaarigarDashboard.html).\n*   **New?** Create an account or sign in on our [Auth page](Auth.html).\n\n---\n*Type **0** to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "6":
                response = "ℹ️ **About Us & Contact:**\n\n**Mend, Don't Replace.** We are mapping local kaarigars—cobblers, mechanics, and tailors—to make repair the first choice.\n\n*   Learn more about our mission on the [About Us](AboutUs.html) page.\n*   Have questions? Reach out via our [Contact Page](Contact.html).\n\n---\n*Type **0** to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.session_state.chat_state = "CUSTOM_QUESTION"
                formatted_history = format_history(st.session_state.messages[:-1])
                inputs = {"question": prompt_text, "history": formatted_history}
                if not retriever:
                    inputs["context"] = ""

                if free_rag_chain:
                    try:
                        with st.spinner("Finding the best answer..."):
                            rag_response = free_rag_chain.invoke(inputs)
                        
                        if isinstance(rag_response, str) and rag_response.strip().startswith("[{'text':"):
                            try:
                                parsed = ast.literal_eval(rag_response.strip())
                                if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                                    rag_response = parsed[0]['text']
                            except Exception:
                                pass
                        
                        rag_response += "\n\n---\n*Type **0** to return to the main menu.*"
                        message_placeholder.markdown(rag_response)
                        st.session_state.messages.append({"role": "assistant", "content": rag_response})
                        
                    except Exception:
                        message_placeholder.error("I'm currently experiencing high traffic. Please try asking your question again in a moment.")
                else:
                    message_placeholder.error("Our support system is currently offline. Please contact us via phone or email.")
            
        elif st.session_state.chat_state == "PREMIUM_PAYMENT":
            if prompt_text.strip() == "0":
                st.session_state.chat_state = "MAIN_MENU"
                response = MAIN_MENU_TEXT
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            elif prompt_text.strip().lower() == "pay ₹9":
                st.session_state.chat_state = "PREMIUM_CHAT"
                response = "✅ **Payment Successful!**\n\nYou are now connected to the **Premium Artisan Finder**. I can provide direct coordinates and contact details for our certified artisans. What location or service are you looking for?\n\n---\n*Type **0** to return to the main menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "⚠️ **Invalid Input**\n\nTo unlock the artisan coordinates, please pay ₹9 via your [User Dashboard](UserDashboard.html) and then type **Pay ₹9** to verify, or type **0** to cancel."
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        elif st.session_state.chat_state in ["CUSTOM_QUESTION", "PREMIUM_CHAT"]:
            if prompt_text.strip() == "0":
                st.session_state.chat_state = "MAIN_MENU"
                response = MAIN_MENU_TEXT
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                formatted_history = format_history(st.session_state.messages[:-1])
                inputs = {"question": prompt_text, "history": formatted_history}
                if not retriever:
                    inputs["context"] = ""

                chain_to_use = premium_rag_chain if st.session_state.chat_state == "PREMIUM_CHAT" else free_rag_chain
                
                if chain_to_use:
                    try:
                        with st.spinner("Finding the best answer..."):
                            rag_response = chain_to_use.invoke(inputs)
                        
                        if isinstance(rag_response, str) and rag_response.strip().startswith("[{'text':"):
                            try:
                                parsed = ast.literal_eval(rag_response.strip())
                                if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                                    rag_response = parsed[0]['text']
                            except Exception:
                                pass
                                
                        rag_response += "\n\n---\n*Type **0** to return to the main menu.*"
                        message_placeholder.markdown(rag_response)
                        st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    except Exception:
                        message_placeholder.error("I'm currently experiencing high traffic. Please try asking your question again in a moment.")
                else:
                    message_placeholder.error("Our support system is currently offline. Please contact us via phone or email.")
