# 📄 Chat with your PDF (BasicRAG)

A simple Streamlit app to chat with your own PDF documents using Retrieval-Augmented Generation (RAG) powered by LangChain, HuggingFace Embeddings, Chroma vector store, and Groq LLM.

---

## 🚀 Features
- Upload any PDF and ask questions about its content
- Uses state-of-the-art open-source embeddings (all-MiniLM-L6-v2)
- Fast, context-aware answers via Groq LLM (Llama-3.1-8b-instant)
- Maintains chat history for your session

---

## 🛠️ Requirements
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [Chroma Vector Store](https://www.trychroma.com/)
- [Groq LLM](https://console.groq.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

Install dependencies (example):
```bash
pip install streamlit langchain langchain-community langchain-huggingface langchain-groq chromadb python-dotenv
```

---

## 🔑 Environment Variables
You need a Groq API key. Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💻 Usage
1. Run the Streamlit app:
   ```bash
   streamlit run BasicRAG.py
   ```
2. Upload a PDF document using the web interface.
3. Ask questions about the document in natural language.
4. View answers and chat history in the app.

---

## 📝 Troubleshooting
- **Missing API Key:** Make sure your `.env` file is present and contains a valid `GROQ_API_KEY`.
- **Dependency Errors:** Double-check that all required packages are installed.
- **PDF Not Loading:** Ensure the PDF is not corrupted and is under any file size limits imposed by Streamlit.
- **Groq LLM Errors:** If you see errors related to the LLM, verify your API key and model parameters.

---

## 📚 Credits
- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace](https://huggingface.co/)
- [Chroma](https://www.trychroma.com/)
- [Groq](https://groq.com/)

---

## 🏷️ License
MIT License 
