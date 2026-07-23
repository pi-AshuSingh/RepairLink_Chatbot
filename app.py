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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #f4f7f6;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eef2f5;
        padding-top: 1rem;
    }
    
    div.stButton > button:first-child {
        background: #ffffff;
        color: #0f172a;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        height: auto;
    }
    
    div.stButton > button:hover {
        border-color: #059669;
        color: #059669;
        box-shadow: 0 8px 15px -3px rgba(5, 150, 105, 0.15);
        transform: translateY(-2px);
    }

    [data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06) !important;
        padding: 0.5rem !important;
        background-color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        padding: 1.25rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.6;
        animation: slideUp 0.3s ease-out;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        border: none;
        box-shadow: 0 8px 15px -3px rgba(16, 185, 129, 0.25);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: #ffffff;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-header {
        text-align: center;
        padding: 3rem 1.5rem;
        background: linear-gradient(135deg, #064e3b 0%, #059669 50%, #10b981 100%);
        color: white;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 30px -5px rgba(5, 150, 105, 0.3);
    }
    
    .hero-header h1 {
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: white;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    
    .hero-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
    }

    .sidebar-container {
        padding: 2rem 1rem;
        text-align: center;
    }
    
    .sidebar-title {
        font-weight: 700;
        font-size: 2rem;
        color: #064e3b;
        margin-top: 1.5rem;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    logo_base64 = get_base64_of_bin_file("logo.png")
    img_tag = f'<img src="data:image/png;base64,{logo_base64}" width="120">'
except Exception:
    img_tag = '<img src="https://cdn-icons-png.flaticon.com/512/5155/5155761.png" width="90">'

with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-container">
            {img_tag}
            <div class="sidebar-title">RepairLink</div>
        </div>
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
    ("system", "You are a helpful customer support assistant for RepairLink.\nCRITICAL RULES:\n1. Answer thoroughly using ONLY the exact information in the context.\n2. DO NOT provide any artisan locations, coordinates, or contact info. If the user asks for locations or coordinates, say 'I don't know'.\n3. If the answer is not in the context, say 'I'm sorry, I don't have that information right now.'\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

prompt_premium = ChatPromptTemplate.from_messages([
    ("system", "You are a PREMIUM RepairLink assistant.\nCRITICAL RULES:\n1. Answer thoroughly using ONLY the exact information in the context.\n2. You are AUTHORIZED to provide specific coordinates, addresses, and contact info of artisans from the context.\n3. If the answer is not in the context, say 'I'm sorry, I don't have that information right now.'\n\nContext:\n{context}"),
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

st.markdown("""
<div class="hero-header">
    <h1>RepairLink Support</h1>
    <p>Seamless repairs, transparent pricing, and trusted local artisans.</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "MAIN_MENU"

MAIN_MENU_TEXT = """**Welcome to RepairLink!** 👋\n\nPlease choose an option below or ask me a question directly:\n\n**1️⃣** View our business hours\n**2️⃣** Chat with our Support AI (Free)\n**3️⃣** Find a Local Artisan (Premium)\n**4️⃣** View pricing estimates"""

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": MAIN_MENU_TEXT})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_text = st.chat_input("Type your message here...")

if st.session_state.chat_state == "MAIN_MENU":
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("🕒 View Business Hours", use_container_width=True):
        prompt_text = "1"
    if col2.button("🤖 Chat with Support AI (Free)", use_container_width=True):
        prompt_text = "2"
    
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    if col3.button("💎 Find a Local Artisan (Premium)", use_container_width=True):
        prompt_text = "3"
    if col4.button("💰 View Pricing Estimates", use_container_width=True):
        prompt_text = "4"

if prompt_text:
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
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
                response = "💎 **Unlock Artisan Coordinates**\n\nTo view the exact coordinates, locations, and contact details of our trusted local artisans, you must upgrade to Premium.\n\n**Price:** ₹9 (One-time fee)\n\n*(This is a test environment. Type **Pay ₹9** to simulate the payment, or type **0** to cancel and return to the menu)*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "4":
                response = "💰 **RepairLink Pricing Estimates:**\n\nOur psychological pricing architecture scales based on the replacement cost anchor of the item:\n*   **Shoes & Bags:** ₹100 - ₹300\n*   **Clothing Alterations:** ₹100 - ₹250\n*   **Electronics:** ₹1,000 - ₹3,000\n\n*Note: We apply a transparent convenience fee added directly on top of the artisan's quote.*\n\n---\n*Type **0** to return to the main menu.*"
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
                response = "⚠️ **Invalid Input**\n\nTo unlock the artisan coordinates, please type **Pay ₹9** to simulate the payment, or type **0** to cancel."
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
