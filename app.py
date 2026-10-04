
import streamlit as st
import string
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

st.title("🤖 AI Resume Screening System")
st.write("Analyze resumes against your job description.")

# Common skills to identify
SKILLS = [
    "python", "java", "javascript", "typescript",
    "react", "node.js", "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy",
    "tensorflow", "scikit-learn", "docker",
    "aws", "azure", "html", "css",
    "rest api", "git", "flask", "django"
]

# Extract text from uploaded files
def extract_text(file):
    try:
        if file.name.lower().endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")

        elif file.name.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            return "\n".join(
                page.extract_text() or ""
                for page in pdf.pages
            )

        elif file.name.lower().endswith(".docx"):
            document = docx.Document(file)
            return "\n".join(
                para.text for para in document.paragraphs
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    return None

# Basic text preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Identify skills in text
def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found

# Job description input
st.subheader("1. Enter Job Description")

job_desc = st.text_area(
    "Paste the job description here:",
    height=180,
    placeholder="Example: Looking for a Python developer with SQL, machine learning and data analysis skills..."
)

# Resume upload
st.subheader("2. Upload Resume")

uploaded_file = st.file_uploader(
    "Choose a resume",
    type=["pdf", "docx", "txt"]
)

# Analyze button
if st.button("Analyze Resume", type="primary"):

    if not job_desc.strip():
        st.warning("Please enter a job description.")

    elif not uploaded_file:
        st.warning("Please upload a resume.")

    else:
        resume_text = extract_text(uploaded_file)

        if not resume_text or not resume_text.strip():
            st.error("Could not extract text from this resume.")
        else:
            # Process text
            resume_clean = preprocess(resume_text)
            job_clean = preprocess(job_desc)

            # Calculate text similarity
            try:
                vectorizer = CountVectorizer()
                vectors = vectorizer.fit_transform(
                    [resume_clean, job_clean]
                )

                score = cosine_similarity(
                    vectors[0], vectors[1]
                )[0][0] * 100

                score = round(score, 2)

                # Extract skills
                resume_skills = extract_skills(resume_text)
                job_skills = extract_skills(job_desc)

                matching = [
                    skill for skill in job_skills
                    if skill in resume_skills
                ]

                missing = [
                    skill for skill in job_skills
                    if skill not in resume_skills
                ]

                # Display results
                st.divider()
                st.subheader("3. Screening Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Text Similarity", f"{score}%")

                with col2:
                    st.metric(
                        "Skills Matched",
                        f"{len(matching)}/{len(job_skills)}"
                    )

                st.progress(int(score))

                if score >= 70:
                    st.success("High text similarity")
                elif score >= 40:
                    st.info("Moderate text similarity")
                else:
                    st.warning("Low text similarity")

                st.caption(
                    "Similarity is based on word frequency. "
                    "It is not a measure of candidate suitability."
                )

                st.divider()
                st.subheader("4. Skills Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Matching Skills**")
                    if matching:
                        for skill in matching:
                            st.write(f"🟢 {skill}")
                    else:
                        st.write("No matching skills detected.")

                with col2:
                    st.markdown("**❌ Missing Skills**")
                    if missing:
                        for skill in missing:
                            st.write(f"🔴 {skill}")
                    else:
                        st.write("No missing skills detected.")

                # Download report
                report = pd.DataFrame({
                    "Category": (
                        ["Matching Skills"] * len(matching)
                        + ["Missing Skills"] * len(missing)
                    ),
                    "Skill": matching + missing
                })

                st.divider()
                st.subheader("5. Download Report")

                csv = report.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Skills Report",
                    data=csv,
                    file_name="resume_analysis.csv",
                    mime="text/csv"
                )

            except ValueError as e:
                st.error(
                    f"Unable to compare the texts: {e}"
                )