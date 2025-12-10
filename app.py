import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load ENV variables
groq_api = os.getenv("Groq_API")  # make sure your .env has Groq_API=...

if not groq_api:
    raise ValueError("Groq_API not found in environment variables.")

# 2. Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api,
    model="llama-3.1-8b-instant"
)


# 3. Data loading from uploaded files
def load_documents_from_uploaded_files(files):
    """
    Convert uploaded files (Streamlit) into LangChain Document objects.
    """
    docs = []

    if not files:
        return docs

    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        file_name = uploaded_file.name
        file_path = os.path.join(temp_dir, file_name)

        # Save uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Choose loader based on extension
        if file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_name.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file type: {file_name}")
            continue  # important

        file_docs = loader.load()
        docs.extend(file_docs)
        st.write(f"Loaded {len(file_docs)} documents from {file_name}")

    return docs


def split_documents(documents, chunk_size=800, chunk_overlap=200):
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks


def build_vector_store(chunks):
    if not chunks:
        return None

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vector_store = FAISS.from_documents(chunks, embeddings)
    st.success("Vector store created with FAISS ✅")
    return vector_store


def get_context(vector_store, query, k: int = 4):
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    return context


def build_prompt(context: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant. Use ONLY the information in the CONTEXT to answer.
If the answer is not in the context, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (be concise and clear):
"""
    )
    return prompt.format(context=context, question=question)


def answer_question(vector_store, question):
    if vector_store is None:
        return "The knowledge base is not ready yet. Please upload files and build it first."

    # a) retrieve relevant context
    context = get_context(vector_store, question)

    # b) build final prompt
    final_prompt = build_prompt(context, question)

    # c) ask Groq
    response = llm.invoke(final_prompt)
    return response.content


def init_session_state():
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def main():
    st.set_page_config(page_title="RAG Chatbot")
    st.title("RAG ChatBot")
    st.write("Upload **PDF / TXT / DOCX** files and ask questions based on them.")

    init_session_state()

    # File upload section
    st.subheader("Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing documents..."):
                documents = load_documents_from_uploaded_files(uploaded_files)
                if not documents:
                    st.error("No documents loaded. Please check your files.")
                else:
                    chunks = split_documents(documents)
                    vector_store = build_vector_store(chunks)
                    st.session_state["vector_store"] = vector_store
                    st.success("Knowledge base is ready! ✅")

    st.markdown("----")

    # Chat section
    st.subheader("Chat with your documents")

    if st.session_state["vector_store"] is None:
        st.info("Upload documents and click **Build Knowledge Base** to start chatting.")
        return

    # Display previous chat
    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    # New user input
    user_question = st.chat_input("Ask a question about the uploaded documents...")

    if user_question:
        st.session_state["chat_history"].append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(st.session_state["vector_store"], user_question)
                st.markdown(answer)

        st.session_state["chat_history"].append(("assistant", answer))


if __name__ == "__main__":
    main()
