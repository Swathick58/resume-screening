import streamlit as st
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx

# Sample job description
job_desc = """
Looking for a Python developer with knowledge of machine learning,
data analysis, and web development.
"""

# Simple preprocess (NO NLTK)
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    return " ".join(tokens)

# Extract text from different file types
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text

    else:
        return None

# UI
st.title("AI Resume Screening System")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["txt", "pdf", "docx"]
)

if uploaded_file:
    resume = extract_text(uploaded_file)

    if resume is None or resume.strip() == "":
        st.error("Could not read file. Please upload a valid resume.")
    else:
        resume = preprocess(resume)
        job = preprocess(job_desc)

        vectorizer = CountVectorizer().fit_transform([resume, job])
        vectors = vectorizer.toarray()

        similarity = cosine_similarity([vectors[0]], [vectors[1]])
        score = round(similarity[0][0] * 100, 2)

        st.subheader(f"Match Score: {score}%")

        if score > 70:
            st.success("Good match for the job!")
        else:
            st.warning("Improve your resume for better match.")