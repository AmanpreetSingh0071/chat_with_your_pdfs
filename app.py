import streamlit as st
import tempfile
import os
from extract_logic import extract_text_from_pdf, run_groq_prompt, ask_with_groq, create_vector_store

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
st.write("Upload a PDF and ask questions about its content.")

pdf_context = ""
pdf = st.file_uploader("Upload PDF", type="pdf")
vectorstore = None

if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        tmp_path = tmp_file.name

    pdf_context = extract_text_from_pdf(tmp_path)
    try:
        vectorstore = create_vector_store(pdf_context)
        st.success("✅ PDF loaded. You can now ask questions!")
    except ValueError:
        st.warning("⚠️ No text could be extracted from this file. Please upload a valid PDF with selectable text.")
    os.remove(tmp_path)

if vectorstore:
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = ask_with_groq(vectorstore, query)
        st.markdown(f"**Answer:** {response}")