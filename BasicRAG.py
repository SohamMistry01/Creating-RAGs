import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
from dotenv import load_dotenv

# ---- Set your GROQ API key ----
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---- Streamlit UI ----
st.title("📄 Chat with your PDF")
st.divider()

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents(pages)

    # Create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding_model)

    # Setup GROQ LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        streaming=True  # Set to True if you want streaming response
    )

    # Alternate RAG chain using RetrievalQA
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    # User input
    st.divider()
    user_query = st.text_input("Ask a question about the document:")
    st.divider()

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        answer = result["result"]
        st.write("🤖", answer)
        st.divider()

        st.session_state.chat_history.append(("▶You", user_query))
        st.session_state.chat_history.append(("🤖AI", answer))


    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        st.divider()
        for speaker, message in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {message}")
            st.divider()
