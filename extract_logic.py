import streamlit as st
import pdfplumber
import os
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Groq client with API key from environment
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
model = "llama-3.3-70b-versatile"

# 1. Extract raw text from PDF
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return "\n".join(extracted_text)

# 2. Create FAISS vector store from PDF text using HuggingFace
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No valid text chunks were created from the PDF.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore

# === 3. Send a raw prompt ===
def summarize_pdf(text: str) -> str:
    """Summarize the given PDF text."""
    prompt = f"Summarize the following document:\n\n{text}"
    return run_groq_prompt(prompt)

# === 4. Query vector store ===
def ask_pdf_question(text: str, question: str) -> str:
    """Answer a question based on the content of a PDF."""
    vectorstore = create_vector_store(text)
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the context below.

Context:
{context}

Question: {question}
Answer:
"""
    return run_groq_prompt(prompt)

# === 5. Compare two PDF texts ===
def compare_pdfs(text1: str, text2: str) -> str:
    """Compare the content of two PDFs and summarize key differences."""
    prompt = f"""
Compare the following two documents and summarize the key differences.

Document 1:
{text1}

Document 2:
{text2}

Differences:
"""
    return run_groq_prompt(prompt)

# === 6. Groq utility ===
def run_groq_prompt(prompt: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error from Groq: {str(e)}"