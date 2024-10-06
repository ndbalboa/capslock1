import os
import re
import pytesseract
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from docx import Document
import logging
import spacy
from spacy.matcher import Matcher
from PIL import Image

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Setup logging for error/debug tracking
logging.basicConfig(level=logging.DEBUG)

# Load pre-trained Naive Bayes model and vectorizer
model = joblib.load('document_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load SpaCy language model
nlp = spacy.load("en_core_web_sm")

# Initialize SpaCy's Matcher for rule-based matching
matcher = Matcher(nlp.vocab)

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]), None
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {str(e)}")
        return "", str(e)

# Function to extract text from image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

# Function to extract text from PDF or image files
def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image)
        elif ext in ['jpeg', 'jpg', 'png']:
            text = extract_text_from_image(file_path)
        elif ext == 'docx':
            text, error = extract_text_from_docx(file_path)
            if error:
                return "", error
        else:
            return "", f"Unsupported file format: {ext}"
    except Exception as e:
        logging.error(f"Error extracting text from file {file_path}: {str(e)}")
        return "", str(e)
    return text, None

# Function to clean the extracted text
def clean_extracted_text(text, replace_newline_with=' '):
    return text.replace('\n', replace_newline_with)

def extract_subject_and_description(text):
    # Try different patterns for subject extraction
    subject_patterns = [
        r'SUBJECT[:\s]+([A-Z\s]+)',  # Standard 'SUBJECT:' format
        r'(?<=attend the )(.*?)(?=\son)',  # "attend the ... on" pattern
        r'Subject[:\s]+([A-Za-z0-9\s,]+)',  # 'Subject:' with more flexible character sets
        r'Title[:\s]+([A-Za-z0-9\s,]+)',  # Alternative key like 'Title:'
        r'benchmark on the (.*?)\son',  # New pattern for "benchmark on the" structure
        r'(to benchmark on the (.*?))\sfor',  # Additional benchmark-related pattern
    ]
    
    # Try different patterns for description extraction
    description_patterns = [
        r'(You are hereby informed.*?FOR YOUR GUIDANCE AND COMPLIANCE\.)',
        r'(In the exigency.*?FOR YOUR GUIDANCE AND COMPLIANCE\.)',
        r'(It is hereby requested.*?FOR YOUR GUIDANCE AND COMPLIANCE\.)',
        r'(Please be informed that.*?FOR YOUR INFORMATION AND GUIDANCE\.)',
        r'(directed to travel on Official Business.*?FOR YOUR GUIDANCE AND COMPLIANCE\.)',  # New pattern based on document
        r'(This travel being official.*?FOR YOUR GUIDANCE AND COMPLIANCE\.)'  # New pattern based on official travel
    ]
    
    # Search for subject using multiple patterns
    subject = None
    for pattern in subject_patterns:
        subject_match = re.search(pattern, text, re.DOTALL)
        if subject_match:
            subject = subject_match.group(1).strip()
            break

    if not subject:
        subject = "Subject not found"
    
    # Search for description using multiple patterns
    description = None
    for pattern in description_patterns:
        description_match = re.search(pattern, text, re.DOTALL)
        if description_match:
            description = description_match.group(1).strip()
            break

    if not description:
        description = "Description not found"
    
    return subject, description

    return subject, description

# Regex patterns for extracting fields
def extract_field_patterns(text):
    fields = {
        'document_no': None,
        'series_no': None,
        'date_issued': None,
        'from': None,
        'to': None,
        'subject': None,
        'description': None,
        'employee_names': []
    }

    # Updated pattern for Document No.
    document_no_pattern = re.compile(
        r'(TRAVEL ORDER|SPECIAL ORDER|OFFICE ORDER|No\.?|NO\.?|Doc(?:ument)? No\.?)[:\s]*\.*(\d+)', 
        re.IGNORECASE
    )
    document_no_match = document_no_pattern.search(text)
    if document_no_match:
        fields['document_no'] = document_no_match.group(2).strip()

    # Pattern for Series No.
    series_no_pattern = re.compile(r'(Series No|Series)[^\d]*(\d+)', re.IGNORECASE)
    series_no_match = series_no_pattern.search(text)
    if series_no_match:
        fields['series_no'] = series_no_match.group(2)

    # Pattern for Date Issued (e.g., DATE: January 5, 2024)
    date_issued_pattern = re.compile(r'DATE:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', re.IGNORECASE)
    date_issued_match = date_issued_pattern.search(text)
    if date_issued_match:
        fields['date_issued'] = date_issued_match.group(1)

    # Pattern for From Date (Start Date)
    from_date_pattern = re.compile(r'Official.*on\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', re.IGNORECASE)
    from_date_match = from_date_pattern.search(text)
    if from_date_match:
        fields['from'] = from_date_match.group(1)

    # No explicit 'To Date' was found, so handling this as None
    fields['to'] = None

    return fields

# Function to extract specific fields using SpaCy and regex patterns
def extract_fields_with_spacy(text):
    doc = nlp(text)
    fields = extract_field_patterns(text)

    # SpaCy extraction for employee names
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            fields['employee_names'].append(ent.text)

    # Extract subject and description using regex
    subject, description = extract_subject_and_description(text)
    fields['subject'] = subject
    fields['description'] = description

    return fields

# Function to clean up the file after processing
def cleanup_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File {file_path} removed successfully.")
    except Exception as e:
        logging.error(f"Error removing file {file_path}: {str(e)}")

# New method: Process files from a given directory
def process_directory_files(directory_path):
    processed_results = []
    if not os.path.exists(directory_path):
        logging.error(f"Directory {directory_path} does not exist.")
        return {"error": f"Directory {directory_path} does not exist."}, 400

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        logging.info(f"Processing file: {file_name}")
        text, extraction_error = extract_text_from_file(file_path)

        if extraction_error:
            logging.error(f"Failed to extract text from file {file_path}: {extraction_error}")
            continue

        if not text.strip():
            logging.warning(f"No text could be extracted from file {file_path}")
            continue

        # Clean the extracted text
        cleaned_text = clean_extracted_text(text)

        # Transform the cleaned text for document type prediction
        try:
            transformed_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_text)[0]
            logging.debug(f"Document type predicted: {prediction}")
        except Exception as e:
            logging.error(f"Error predicting document type for {file_name}: {str(e)}")
            continue

        # Extract fields using SpaCy and regex patterns
        extracted_fields = extract_fields_with_spacy(cleaned_text)

        # Add document type to the extracted fields
        extracted_fields['document_type'] = prediction

        # Append results for this file
        processed_results.append({
            'file_name': file_name,
            'raw_text': cleaned_text,
            'document_type': prediction,
            'extracted_fields': extracted_fields
        })

        # Clean up the file after extraction
        cleanup_file(file_path)

    return processed_results, 200

@app.route('/api/admin/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error("No file provided in the request.")
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    file_name = file.filename
    file_path = os.path.join(os.getcwd(), 'uploads', file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Extract text from the uploaded file
    text, extraction_error = extract_text_from_file(file_path)

    if extraction_error:
        logging.error(f"Failed to extract text from file {file_path}: {extraction_error}")
        cleanup_file(file_path)
        return jsonify({'error': f'Failed to extract text: {extraction_error}'}), 400

    if not text.strip():
        logging.warning(f"No text could be extracted from file {file_path}")
        cleanup_file(file_path)
        return jsonify({'error': 'No text could be extracted from the file.'}), 400

    # Clean the extracted text
    cleaned_text = clean_extracted_text(text)

    # Transform the cleaned text for document type prediction
    try:
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]
        logging.debug(f"Document type predicted: {prediction}")
    except Exception as e:
        logging.error(f"Error predicting document type: {str(e)}")
        cleanup_file(file_path)
        return jsonify({'error': f'Error predicting document type: {str(e)}'}), 500

    # Extract fields using SpaCy and regex patterns
    extracted_fields = extract_fields_with_spacy(cleaned_text)

    # Add document type to the extracted fields
    extracted_fields['document_type'] = prediction

    # Clean up the file after extraction
    cleanup_file(file_path)

    return jsonify({
        'file_name': file_name,
        'document_type': prediction,
        'extracted_fields': extracted_fields
    }), 200

@app.route('/api/admin/upload_directory', methods=['POST'])
def upload_directory():
    data = request.get_json()
    directory_path = data.get('directory_path')

    processed_results, status_code = process_directory_files(directory_path)

    return jsonify({'results': processed_results}), status_code

if __name__ == '__main__':
    app.run(debug=True)
