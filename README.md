# RAG ChatBot (Streamlit + Groq + FAISS)

This project is a simple Retrieval-Augmented Generation (RAG) chatbot built using
Streamlit, LangChain, Groq LLMs, and FAISS.

The app allows users to upload documents and ask questions that are answered
strictly based on the uploaded content.

--------------------------------------------------

FEATURES

- Upload multiple PDF, TXT, and DOCX files
- Automatic document chunking
- Vector embeddings using HuggingFace sentence-transformers
- FAISS vector store for similarity search
- LLM-powered answers using Groq (LLaMA 3.1)
- Chat-style UI with session-based chat history

--------------------------------------------------

TECH STACK

- Python
- Streamlit
- LangChain
- Groq (ChatGroq)
- HuggingFace Embeddings
- FAISS (CPU)
