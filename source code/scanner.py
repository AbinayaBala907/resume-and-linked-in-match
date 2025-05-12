pip install pdfplumber scikit-learn sentence-transformers
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Load a pre-trained NLP model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    """Extracts all text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_embedding(text):
    """Converts text to a vector embedding."""
    return model.encode([text])[0]

def match_score(resume_text, linkedin_text, job_description):
    """Computes similarity scores for resume and LinkedIn profile vs job description."""
    job_vec = get_text_embedding(job_description)
    resume_vec = get_text_embedding(resume_text)
    linkedin_vec = get_text_embedding(linkedin_text)

    resume_score = cosine_similarity([resume_vec], [job_vec])[0][0]
    linkedin_score = cosine_similarity([linkedin_vec], [job_vec])[0][0]

    return resume_score, linkedin_score

# === Example Usage ===

resume_file_path = 'resume.pdf'  # path to PDF file
linkedin_text = """
John Doe is a software engineer with 5 years of experience in Python, machine learning,
and web development. Previously worked at Google and Microsoft. Passionate about AI applications.
"""  # scraped or input LinkedIn summary

job_description = """
We are hiring a Python Developer with experience in machine learning, data science,
and cloud platforms like AWS. Candidates should have strong analytical and communication skills.
"""

if os.path.exists(resume_file_path):
    resume_text = extract_text_from_pdf(resume_file_path)
    resume_score, linkedin_score = match_score(resume_text, linkedin_text, job_description)
    print(f"Resume Match Score: {resume_score:.2f}")
    print(f"LinkedIn Match Score: {linkedin_score:.2f}")
else:
    print("Resume file not found.")
