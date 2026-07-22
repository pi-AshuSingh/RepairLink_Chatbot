---
title: RepairLink Assistant
emoji: 🛠️
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# RepairLink Support Assistant

This is a RAG-based customer support assistant built with Streamlit, Langchain, and Hugging Face API.

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup Environment Variables:
Create a `.env` file in the root directory and add your Hugging Face API Token:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

3. Run the application:
```bash
streamlit run app.py
```
