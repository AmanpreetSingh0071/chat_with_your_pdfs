import pdfplumber
import os
from groq import Groq
from dotenv import load_dotenv

print("🔍 API Key from secrets:", os.getenv("GROQ_API_KEY"))

# Initialize Groq client with API key from environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "deepseek-r1-distill-llama-70b"

def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return "\n".join(extracted_text)

def run_groq_prompt(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"