---
title: RepairLink Support Bot
emoji: 🛠️
colorFrom: green
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

<div align="center">
  <h1>🛠️ RepairLink AI Support Assistant</h1>
  <p><strong>"Mat Feko, Fix Karo" — Bridging the gap between time-strapped urban consumers and skilled local repair artisans.</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
  [![LangChain](https://img.shields.io/badge/LangChain-Integration-gray.svg?logo=langchain)](https://langchain.com)
  [![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue.svg)](https://github.com/facebookresearch/faiss)
  [![Groq](https://img.shields.io/badge/Groq-LLaMA%203-f55036.svg)](https://groq.com)
</div>

<br>

**Live Demo:** [https://repairlink.streamlit.app/](https://repairlink.streamlit.app/)

Welcome to the **RepairLink AI Support Assistant**! This is a modern, menu-driven intelligent chatbot built to handle customer inquiries, guide users to local Kaarigars (artisans), explain Eco-Points, and serve as the frontline support for the circular economy.

---

## 🚀 Key Features

*   💬 **WhatsApp-Style Interface:** Custom CSS styling featuring an elegant glassmorphism chat UI, complete with a professional user avatar silhouette (`👤`).
*   🧭 **Interactive State Machine:** Users can seamlessly navigate using one-click interactive buttons or type numeric commands (e.g., "1", "2") for instantaneous menu routing.
*   🧠 **Advanced RAG (Retrieval-Augmented Generation):**
    *   **Unified Master Knowledge Base:** Fully integrated with `RepairLink_KnowledgeBase_RAG.pdf`, an optimized 22-page document combining all operational and strategic data.
    *   **Robust Text Processing:** Custom parsers securely filter out stray spaces and artifact characters from PDFs, guaranteeing highly accurate semantic chunking.
    *   **Vector Search:** Employs `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in a blazing-fast **FAISS** vector store to instantly retrieve relevant paragraphs.
*   ⚡ **Groq Llama 3.3 70B:** Harnesses the power of Groq's LPU inference engine to generate natural, accurate, and highly contextual responses with near-zero latency.
*   🔒 **Strict Agent Guardrails:** The AI is strictly programmed to answer *only* from the unified knowledge base and is completely restricted from revealing sensitive Kaarigar GPS coordinates unless authorized via premium flows.

---

## 🆕 Recent Updates & Changelog

- **[UI/UX]** Upgraded the generic user chat icon to a professional silhouette (`👤`).
- **[Data Pipeline]** Deployed the newly compiled `RepairLink_KnowledgeBase_RAG.pdf` which cleanly unifies four separate legacy knowledge documents.
- **[Bug Fix]** Fixed a critical `PyPDFLoader` extraction bug where extreme kerning added spaces between every letter (`R e p a i r`), causing zero-match RAG failures. A custom regex sanitizer now ensures 100% vector match accuracy.
- **[Prompt Engineering]** Updated system prompts to actively cross-sell RepairLink website subpages (Pricing, Dashboards) and remind users about Waste Points.

---

## 🛠️ Running Locally

Follow these steps to run the chatbot on your local machine:

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Setup Environment Variables:**
Create a `.env` file in the root directory and add your Groq API Token:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**3. Run the application:**
```bash
streamlit run app.py
```

---

## 🌍 Deployment

This application is fully compatible with **Streamlit Community Cloud** and **Hugging Face Spaces**. 
> [!IMPORTANT]
> When deploying, do not upload your `.env` file. Instead, inject your `GROQ_API_KEY` directly into your hosting provider's Secrets Manager.

---
<div align="center">
  <i>Built for the circular economy by Team Delta</i>
</div>
