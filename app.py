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
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0.3,
)

# ==========================================
# 4. RAG Prompt & Chain
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant for RepairLink.\nCRITICAL RULES:\n1. Keep your answers EXTREMELY short and concise (1-2 sentences maximum).\n2. Answer the question using ONLY the exact information provided in the context below.\n3. If the answer is not in the context, say 'I don't know.' Do not elaborate.\n\nContext:\n{context}"),
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
        if role == "user":
            formatted.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted.append(AIMessage(content=content))
    return formatted

# Streamlit Page Config
st.set_page_config(
    page_title="RepairLink Support",
    page_icon="🛠️",
    layout="centered"
)

# Header
st.title("🛠️ RepairLink Support")
st.caption("Ask questions about RepairLink's services and strategies.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_text := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Format inputs for RAG chain
        formatted_history = format_history(st.session_state.messages[:-1]) # exclude current message
        inputs = {"question": prompt_text, "history": formatted_history}
        if not retriever:
            inputs["context"] = "No documents available."

        try:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(inputs)
            
            # Clean up response if needed
            if isinstance(response, str) and response.strip().startswith("[{'text':"):
                try:
                    import ast
                    parsed = ast.literal_eval(response.strip())
                    if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                        response = parsed[0]['text']
                except:
                    pass
                    
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            message_placeholder.error(error_msg)
            if "Authorization" in str(e) or "token" in str(e).lower():
                st.error("Please make sure you have a valid HUGGINGFACEHUB_API_TOKEN set in your environment.")
