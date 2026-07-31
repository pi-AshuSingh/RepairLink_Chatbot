---
title: RepairLink Support Bot
emoji: 🛠️
colorFrom: green
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

# 🛠️ RepairLink Support Assistant

**Live Demo:** [https://repairlink.streamlit.app/](https://repairlink.streamlit.app/)

This is a modern, WhatsApp-style menu-driven customer support bot built for RepairLink. It uses **Streamlit** for the frontend, **LangChain** and **FAISS** for knowledge retrieval, and **Groq (Llama 3.3 70B)** for lightning-fast AI responses.


## Key Features
- **WhatsApp-Style Interface:** Custom CSS styling for chat bubbles and intuitive UI.
- **State Machine Menu:** Users can type "1", "2", or "3" (or click the interactive buttons) to navigate quickly.
- **Instant Hardcoded Actions:** Options like "Order Tracking" and "Business Hours" bypass the AI for zero latency and guaranteed accuracy.
- **RAG-Powered AI Fallback:** Natural language questions are automatically routed to the Groq LLM, which answers using context directly extracted from the `RepairLink_KnowledgeBase_RAG.pdf` document.

## Running Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup Environment Variables:**
Create a `.env` file in the root directory and add your Groq API Token:
```env
GROQ_API_KEY=your_groq_api_key_here
```

3. **Run the application:**
```bash
streamlit run app.py
```

## Deployment
This app can be deployed for free on **Streamlit Community Cloud** or **Hugging Face Spaces** (using the Streamlit SDK). Ensure you add your `GROQ_API_KEY` to your host's secrets manager, rather than uploading your `.env` file.
