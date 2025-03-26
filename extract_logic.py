import streamlit as st
import pdfplumber
import os
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Groq client with API key from environment
client = Groq(api_key=st.secrets("GROQ_API_KEY"))
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

# 2. Send a raw prompt to Groq (reusable utility)
def run_groq_prompt(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error from Groq: {str(e)}"

# 3. Create FAISS vector store from PDF text using HuggingFace
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No valid text chunks were created from the PDF.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore

# 4. Query the vector store and send result to Groq
def ask_with_groq(vectorstore, query):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful AI assistant. Answer the following question based ONLY on the context provided.

Context:
{context}

Question: {query}
Answer:
"""
        return run_groq_prompt(prompt)

    except Exception as e:
        return f"❌ Error during vector search or prompt: {str(e)}"
