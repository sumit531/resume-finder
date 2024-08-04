import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re

# Function to extract text from PDF using PdfReader

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    return text

# Load sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Streamlit app
st.title("Resume and Job Description Matcher")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    profile_text = extract_text_from_pdf(uploaded_file)
    profile_text = preprocess_text(profile_text)
    st.write("Extracted Profile Text:")
    st.write(profile_text)

    job_description = st.text_area("Enter Job Description")
    if job_description:
        job_description = preprocess_text(job_description)

        # Compute embeddings for profile and job description
        embeddings = model.encode([profile_text, job_description])

        # Compute similarity
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item() * 100

        st.write(f"Similarity: {similarity:.2f}%")
