import streamlit as st
import tempfile
import os
from extract_logic import extract_text_from_pdf, run_groq_prompt

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Chat with Your PDF", layout="centered")

# 🔒 Hide GitHub icon, menu, and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 Chat with Your PDF")
st.write("Upload a PDF and ask questions about its content using a Groq LLM.")

pdf_context = ""
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        tmp_path = tmp_file.name

    pdf_context = extract_text_from_pdf(tmp_path)
    st.success("✅ PDF loaded. You can now ask questions!")
    os.remove(tmp_path)

if pdf_context:
    query = st.text_input("Ask a question about the PDF:")

    if query:
        prompt = f"""
You are a helpful assistant. Use the following PDF content to answer the user's question.

PDF Content:
{pdf_context}

Question: {query}
Answer:
"""
        response = run_groq_prompt(prompt)
        st.markdown(f"**Answer:** {response}")