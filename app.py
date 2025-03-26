import os
import streamlit as st
from PyPDF2 import PdfFileReader, PdfReader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract text from PDF file
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description and resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between job description and resumes
    job_description_vector = vectors[0].reshape(1, -1)
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(
        job_description_vector, resume_vectors
    ).flatten()

    return cosine_similarities


# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job Description input
st.header("Job Description")
job_description = st.text_area("Enter the job description:")

# File Upload
st.header("Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload resumes:", type=["pdf"], accept_multiple_files=True
)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resume = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resume.append(text)

    # Rank resumes based on job description
    cosine_similarities = rank_resumes(job_description, resume)

    # Display Scores
    results = pd.DataFrame(
        {"Resume": [file.name for file in uploaded_files], "Score": cosine_similarities}
    )
    results = results.sort_values("Score", ascending=False)

    st.table(results)
import streamlit as st

st.title("Streamlit Debugging Test")
st.write("If you see this, Streamlit is working!")
