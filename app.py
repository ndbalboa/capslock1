import os
import re
import pytesseract
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)

# Configure CORS
CORS(app, supports_credentials=True)

# Load pre-trained Naive Bayes model and vectorizer
model = joblib.load('document_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Regular expression patterns for field extraction
patterns = {
    'document_no': r'Document\s*No\.?\s*[:\s]+(\d+)',
    'date_issued': r'(Date\s*Issued|Dated)\.?\s*[:\s]+([\d-]+)',
    'from': r'(From|Period)\.?\s*[:\s]+([A-Za-z\s\d,]+)',
    'to': r'To\.?\s*[:\s]+([A-Za-z\s\d,]+)',
    'subject': r'Subject\.?\s*[:\s]+([^\n]+)',
    'employee_names': r'To\.?\s*[:\s]+([A-Za-z\s,]+)',
}

# Extract specific field from the text using regex
def extract_field(pattern, text, default=""):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default

# Extract text from a PDF or image file
def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image)
        elif ext in ['jpeg', 'jpg', 'png']:
            text = pytesseract.image_to_string(file_path)
    except Exception as e:
        return "", str(e)  # Return empty text and error message if extraction fails
    return text, None  # Return extracted text and no error

@app.route('/api/admin/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    # Save the file securely
    file_path = os.path.join(os.getcwd(), 'uploads', file.filename)  # Ensure the uploads directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Extract text from the file
    text, extraction_error = extract_text_from_file(file_path)

    if extraction_error:
        return jsonify({'error': f'Failed to extract text: {extraction_error}'}), 400

    if not text.strip():
        return jsonify({'error': 'No text could be extracted from the file.'}), 400

    # Predict document type
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    # Extract other fields using regex
    extracted_data = {
        'document_no': extract_field(patterns['document_no'], text),
        'date_issued': extract_field(patterns['date_issued'], text),
        'from': extract_field(patterns['from'], text),
        'to': extract_field(patterns['to'], text),
        'subject': extract_field(patterns['subject'], text),
        'description': text,  # Full text as description
        'employee_names': extract_field(patterns['employee_names'], text).split(',') if extract_field(patterns['employee_names'], text) else []
    }

    return jsonify({
        'extracted_data': extracted_data,
        'document_type': prediction
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Specify port if necessary
