from flask import Flask, render_template, request, jsonify, send_file
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Please upload your resume and provide the job description."}), 400

    uploaded_file = request.files['resume']
    job_description = request.form['job_description']
    analysis_type = request.form.get('analysis_type', 'Detailed Resume Review')

    pdf_text = ATSAnalyzer.extract_text_from_pdf(uploaded_file)
    if not pdf_text or pdf_text.startswith("Error"):
        return jsonify({"error": pdf_text}), 400

    if analysis_type == "Detailed Resume Review":
        prompt = """
        As an experienced Technical Human Resource Manager, provide a detailed professional evaluation 
        of the candidate's resume against the job description. Please analyze:
        1. Overall alignment with the role
        2. Key strengths and qualifications that match
        3. Notable gaps or areas for improvement
        4. Specific recommendations for enhancing the resume
        5. Final verdict on suitability for the role
        
        Format the response with clear headings and professional language.
        """
    else:
        prompt = """
        As an ATS (Applicant Tracking System) expert, provide:
        1. Overall match percentage (%)
        2. Key matching keywords found
        3. Important missing keywords
        4. Skills gap analysis
        5. Specific recommendations for improvement
        
        Start with the percentage match prominently displayed.
        """

    response = ATSAnalyzer.get_gemini_response(prompt, pdf_text, job_description)
    if response.startswith("Error"):
        return jsonify({"error": response}), 400

    return jsonify({"analysis": response})

@app.route('/export', methods=['POST'])
def export():
    analysis = request.json.get('analysis')
    if not analysis:
        return jsonify({"error": "No analysis data to export."}), 400

    with open("resume_analysis.txt", "w") as file:
        file.write(analysis)

    return send_file("resume_analysis.txt", as_attachment=True)

if __name__ == '__main__':
    app.run(port=5003, debug=True)  # Running on port 5003
