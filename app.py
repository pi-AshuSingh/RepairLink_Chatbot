import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. Configuration & Document Loading
# ==========================================
KNOWLEDGE_BASE = "RepairLink_KnowledgeBase_RAG.pdf"

print("Loading knowledge base...")
documents = []
if os.path.exists(KNOWLEDGE_BASE):
    try:
        documents = PyPDFLoader(KNOWLEDGE_BASE).load()
        print("✅ Knowledge base loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load {KNOWLEDGE_BASE}: {e}")
else:
    print(f"⚠️ {KNOWLEDGE_BASE} not found in the current directory.")

print(f"📄 Total document pages/elements loaded: {len(documents)}")

# ==========================================
# 2. Text Splitting & Embeddings
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
docs = text_splitter.split_documents(documents)

if len(docs) > 0:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print("⚠️ No documents loaded! RAG will not have context.")
    retriever = None

# ==========================================
# 3. LLM Setup (Google Gemini API)
# ==========================================
# Make sure GOOGLE_API_KEY is set in your environment or .env file
google_api_key = os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    st.warning("⚠️ GOOGLE_API_KEY is not set. The chatbot will not be able to generate responses.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=google_api_key,
    temperature=0.3,
)

# ==========================================
# 4. RAG Prompt & Chain
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and detailed customer support assistant for RepairLink.\nCRITICAL RULES:\n1. Answer the question thoroughly using ONLY the exact information provided in the context below.\n2. If the user asks for a list or details (like the 5 Cs), provide all of them clearly.\n3. If the answer is not in the context, say 'I don't know.' Do not elaborate.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if retriever:
    def get_context(inputs):
        return format_docs(retriever.invoke(inputs["question"]))
        
    rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    # Fallback if no documents are uploaded
    rag_chain = prompt | llm | StrOutputParser()

# ==========================================
# 5. Streamlit Interface
# ==========================================
def format_history(history):
    formatted = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        # Skip injecting our own menu-system prompts into the LLM history so it doesn't get confused
        if "Welcome to RepairLink" in content or "Type '0'" in content or "1️⃣" in content:
            continue
        if role == "user":
            formatted.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted.append(AIMessage(content=content))
    return formatted

# Streamlit Page Config
st.set_page_config(page_title="RepairLink Support", page_icon="🛠️")
st.title("🛠️ RepairLink Support")
st.write("Welcome to our WhatsApp-style automated support system!")

# Initialize chat history and state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "MAIN_MENU"

MAIN_MENU_TEXT = """Please select an option by typing the corresponding number:
1️⃣ Track my repair
2️⃣ View business hours
3️⃣ Ask a custom question to our AI Assistant"""

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": MAIN_MENU_TEXT})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Custom WhatsApp-style CSS
st.markdown("""
<style>
    /* WhatsApp Green Theme for Buttons */
    div.stButton > button:first-child {
        background-color: #25D366;
        color: white;
        border-radius: 20px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #128C7E;
        color: white;
    }
    
    /* WhatsApp Chat Bubble Styling */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background-color: #DCF8C6 !important; /* User bubble (light green) */
    }
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(odd) {
        background-color: #FFFFFF !important; /* Assistant bubble (white) */
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Accept user input
prompt_text = st.chat_input("Type your message here...")

# Inject clickable option buttons if the user is in the MAIN_MENU state
if st.session_state.chat_state == "MAIN_MENU":
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if col1.button("1️⃣ Track my repair", use_container_width=True):
        prompt_text = "1"
    if col2.button("2️⃣ View business hours", use_container_width=True):
        prompt_text = "2"
    if col3.button("3️⃣ Ask AI Assistant", use_container_width=True):
        prompt_text = "3"

if prompt_text:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = ""
        
        # State Machine Logic
        if st.session_state.chat_state == "MAIN_MENU":
            if prompt_text.strip() == "1":
                st.session_state.chat_state = "TRACK_REPAIR"
                response = "Please enter your Order ID (e.g., RL-12345):"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "2":
                response = "🕒 **Our business hours are:**\n- Monday - Friday: 9 AM to 7 PM\n- Saturday: 10 AM to 5 PM\n- Sunday: Closed\n\n*Type '0' to return to the Main Menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt_text.strip() == "3":
                st.session_state.chat_state = "CUSTOM_QUESTION"
                response = "🤖 You are now chatting with our AI Assistant! Ask me anything about our services.\n\n*Type '0' anytime to return to the Main Menu.*"
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # If they didn't type 1, 2, or 3, assume they are asking a custom question directly!
                st.session_state.chat_state = "CUSTOM_QUESTION"
                
                # Format inputs for RAG chain
                formatted_history = format_history(st.session_state.messages[:-1])
                inputs = {"question": prompt_text, "history": formatted_history}
                if not retriever:
                    inputs["context"] = "No documents available."

                try:
                    with st.spinner("Thinking..."):
                        rag_response = rag_chain.invoke(inputs)
                    
                    if isinstance(rag_response, str) and rag_response.strip().startswith("[{'text':"):
                        try:
                            import ast
                            parsed = ast.literal_eval(rag_response.strip())
                            if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                                rag_response = parsed[0]['text']
                        except:
                            pass
                            
                    rag_response += "\n\n*Type '0' to return to the Main Menu.*"
                    message_placeholder.markdown(rag_response)
                    st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    message_placeholder.error(error_msg)
                    if "Authorization" in str(e) or "token" in str(e).lower():
                        st.error("Please make sure you have a valid GOOGLE_API_KEY set in your environment.")
            
        elif st.session_state.chat_state == "TRACK_REPAIR":
            if prompt_text.strip() == "0":
                st.session_state.chat_state = "MAIN_MENU"
                response = MAIN_MENU_TEXT
            else:
                import re
                order_id = prompt_text.strip().upper()
                if not re.match(r'^RL-\d+$', order_id):
                    response = "❌ **Invalid Order ID format.**\nPlease use the format **RL-12345** (e.g., RL-567).\n\n*Type '0' to return to the Main Menu.*"
                else:
                    # Mock tracking response
                    response = f"📦 Order **{order_id}** is currently: **In Progress**.\nIt should be ready in 1-2 business days.\n\n*Type '0' to return to the Main Menu.*"
            
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        elif st.session_state.chat_state == "CUSTOM_QUESTION":
            if prompt_text.strip() == "0":
                st.session_state.chat_state = "MAIN_MENU"
                response = MAIN_MENU_TEXT
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Format inputs for RAG chain
                formatted_history = format_history(st.session_state.messages[:-1]) # exclude current message
                inputs = {"question": prompt_text, "history": formatted_history}
                if not retriever:
                    inputs["context"] = "No documents available."

                try:
                    with st.spinner("Thinking..."):
                        rag_response = rag_chain.invoke(inputs)
                    
                    # Clean up response if needed
                    if isinstance(rag_response, str) and rag_response.strip().startswith("[{'text':"):
                        try:
                            import ast
                            parsed = ast.literal_eval(rag_response.strip())
                            if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                                rag_response = parsed[0]['text']
                        except:
                            pass
                            
                    message_placeholder.markdown(rag_response)
                    st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    message_placeholder.error(error_msg)
                    if "Authorization" in str(e) or "token" in str(e).lower():
                        st.error("Please make sure you have a valid GOOGLE_API_KEY set in your environment.")
