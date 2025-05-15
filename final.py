import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file
import os
import docx2txt 
import PyPDF2 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re
import pickle
import google.generativeai as genai
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load models with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_classifier_categorization = pickle.load(open(os.path.join(BASE_DIR, 'models/rf_classifier_categorization.pkl'), 'rb'))
tfidf_vectorizer_categorization = pickle.load(open(os.path.join(BASE_DIR, 'models/tfidf_vectorizer_categorization.pkl'), 'rb'))
rf_classifier_job_recommendation = pickle.load(open(os.path.join(BASE_DIR, 'models/rf_classifier_job_recommendation.pkl'), 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open(os.path.join(BASE_DIR, 'models/tfidf_vectorizer_job_recommendation.pkl'), 'rb'))

# Text extraction functions
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return ''.join([page.extract_text() for page in reader.pages])

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""

def pdf_to_text(file):
    reader = PdfReader(file)
    return ''.join([page.extract_text() for page in reader.pages])

# Resume processing functions
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(resume_tfidf)[0]

# Resume parsing functions
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = ['Python', 'Data Analysis', 'Machine Learning', ...]  # Your full skills list
    return [skill for skill in skills_list if re.search(r"\b{}\b".format(re.escape(skill)), text, re.IGNORECASE)]

def extract_education_from_resume(text):
    education_keywords = ['Computer Science', 'Information Technology', ...]  # Your full education list
    return [kw for kw in education_keywords if re.search(r"(?i)\b{}\b".format(re.escape(kw)), text)]

def extract_name_from_resume(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else None

# Routes
@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/resume_matcher")
def resume_matcher():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')
        resumes = []
        
        for resume_file in resume_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                resume_file.save(tmp.name)
                resumes.append(extract_text(tmp.name))
                os.unlink(tmp.name)

        if not resumes or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity([vectors[0]], vectors[1:])[0]
        top_indices = similarities.argsort()[-5:][::-1]
        
        return render_template('matchresume.html', 
                             message="Top matching resumes:",
                             top_resumes=[resume_files[i].filename for i in top_indices],
                             similarity_scores=[round(similarities[i], 2) for i in top_indices])
    
    return render_template('matchresume.html')

@app.route("/resume_recommendation")
def resume_recommendation_route():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template("resume.html", message="No resume file uploaded.")
    
    file = request.files['resume']
    if file.filename.endswith('.pdf'):
        text = pdf_to_text(file)
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    else:
        return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

    return render_template('resume.html',
                         predicted_category=predict_category(text),
                         recommended_job=job_recommendation(text),
                         phone=extract_contact_number_from_resume(text),
                         name=extract_name_from_resume(text),
                         email=extract_email_from_resume(text),
                         extracted_skills=extract_skills_from_resume(text),
                         extracted_education=extract_education_from_resume(text))

@app.route("/resume_screening")
def resume_screening():
    return render_template('index.html')

class ATSAnalyzer:
    @staticmethod
    def get_gemini_response(input_prompt, pdf_text, job_description):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([input_prompt, pdf_text, job_description])
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @staticmethod
    def extract_text_from_pdf(uploaded_file):
        try:
            return pdf_to_text(uploaded_file)
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Please upload your resume and provide the job description."}), 400

    pdf_text = ATSAnalyzer.extract_text_from_pdf(request.files['resume'])
    if not pdf_text or pdf_text.startswith("Error"):
        return jsonify({"error": pdf_text}), 400

    prompt = """
    As an experienced Technical Human Resource Manager, provide a detailed professional evaluation 
    of the candidate's resume against the job description.
    """ if request.form.get('analysis_type') == "Detailed Resume Review" else """
    As an ATS (Applicant Tracking System) expert, provide analysis.
    """

    response = ATSAnalyzer.get_gemini_response(prompt, pdf_text, request.form['job_description'])
    return jsonify({"analysis": response}) if not response.startswith("Error") else jsonify({"error": response}), 400

@app.route('/export', methods=['POST'])
def export():
    analysis = request.json.get('analysis')
    if not analysis:
        return jsonify({"error": "No analysis data to export."}), 400

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(analysis)
        tmp_path = tmp.name
    
    try:
        return send_file(tmp_path, as_attachment=True, download_name='resume_analysis.txt')
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
