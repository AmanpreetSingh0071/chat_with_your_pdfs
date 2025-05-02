import streamlit as st
import tempfile
import os

from extract_logic import (
    extract_text_from_pdf,
    summarize_pdf,
    ask_pdf_question,
    compare_pdfs,
    run_groq_prompt,
)

# Streamlit UI setup
st.set_page_config(page_title="Groq Assistant: Chat with Your PDFs", layout="centered")
st.title("🤖 AI Assistant: Summarize, Compare, Ask from PDFs")

# Upload PDFs
col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("Upload PDF 1", type="pdf")
with col2:
    pdf2 = st.file_uploader("Upload PDF 2 (optional)", type="pdf")

# Extract text
pdf1_text = pdf2_text = ""
if pdf1:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf1.read())
        pdf1_text = extract_text_from_pdf(tmp.name)
        os.remove(tmp.name)

if pdf2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf2.read())
        pdf2_text = extract_text_from_pdf(tmp.name)
        os.remove(tmp.name)

# User interaction
if pdf1_text:
    user_input = st.text_input("What would you like to do? (e.g., summarize, compare, ask a question)")

    if user_input:
        with st.spinner("Groq Assistant is thinking..."):

            # Use Groq to classify intent
            intent_prompt = f"""
You are an AI assistant. Decide what the user wants based on the message:
- summarize
- compare
- question

User input: "{user_input}"

Respond with one word: summarize, compare, or question.
"""
            try:
                intent = run_groq_prompt(intent_prompt).strip().lower()

                if intent == "summarize":
                    response = summarize_pdf(text=pdf1_text)

                elif intent == "compare":
                    if not pdf2_text:
                        response = "❌ Please upload PDF 2 to compare."
                    else:
                        response = compare_pdfs(text1=pdf1_text, text2=pdf2_text)

                elif intent == "question":
                    response = ask_pdf_question(text=pdf1_text, question=user_input)

                else:
                    response = "❌ Sorry, I couldn't understand your request."

                st.success("✅ Response:")
                st.write(response)

            except Exception as e:
                st.error(f"Groq assistant error: {str(e)}")

else:
    st.info("Please upload at least one PDF to begin.")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size: 12px; color: gray;">
        Created by <strong>Amanpreet Singh</strong>
    </div>
    """,
    unsafe_allow_html=True
)