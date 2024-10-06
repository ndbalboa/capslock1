import spacy
import re

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

def extract_fields_with_spacy(text):
    # Initialize a dictionary to store the extracted fields
    fields = {
        "document_no": None,
        "series_no": None,
        "date_issued": None,
        "receivers": [],
        "from": None,
        "subject": None,
        "description": None,
        "start_date": None,
        "finish_date": None,
    }

    # Use regex to extract the document_no and series_no
    document_no_match = re.search(r'No\. (\d+),?\s*s\. (\d{4})', text)
    if document_no_match:
        fields['document_no'] = document_no_match.group(1)
        fields['series_no'] = document_no_match.group(2)

    # Process the text using spaCy
    doc = nlp(text)
    
    # Extract entities from the text using NER
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if fields["date_issued"] is None:
                fields["date_issued"] = ent.text
            elif fields["start_date"] is None:
                fields["start_date"] = ent.text
            else:
                fields["finish_date"] = ent.text
        elif ent.label_ == "PERSON":
            fields["receivers"].append(ent.text)
        elif ent.label_ == "ORG" and fields["from"] is None:
            fields["from"] = ent.text
    
    # Use regex to extract the subject
    subject_match = re.search(r'Activity entitled\s*"(.+?)"', text)
    if subject_match:
        fields['subject'] = subject_match.group(1)

    # Use regex to extract the description (if available)
    description_match = re.search(r'You are hereby directed to (.+)', text)
    if description_match:
        fields['description'] = description_match.group(1)

    return fields

# Sample text extracted from the image
sample_text = """
Republic of the Philippines
LEYTE NORMAL UNIVERSITY
Tacloban City
COMMUNITY EXTENSION SERVICES OFFICE

February 5, 2024

TRAVEL ORDER
No. 41, s. 2024

To:
IT UNIT FACULTY EXTENSIONISTS

DR. ROMMEL VERECIO
DR. LAS JOHANSEN CALUZA
DR. MICHELINE GOTARDO
DR. LOWELL QUISUMBING
DR. MARK LESTER LAURENTE
PROF. RAPHY DALAN
PROF. DEVINE GRACE FUNCION
PROF. JEFFREY CINCO
MS. MARIE JANE FERNANDEZ
PROF. DENNIS TIBE
MS. LOUVESSA IDDA GALBAN
PROF. DEBBY TURCO

FROM:
DR. EVELYN B. AGUIRRE

You are hereby directed to travel on OFFICIAL TIME only to BRGY. SAN MIGUELAY, STA. FE, LEYTE to conduct the Community Extension Activity entitled "Community Extension and Gift Giving" on February 7, 2024 at 8:00 AM - 5:00 PM.

Participation in this activity is considered part of your delivery of extension services. As such, you are required to submit the following to the Community Extension Services Office:
"""

# Call the function to extract fields
extracted_fields = extract_fields_with_spacy(sample_text)

# Print the extracted fields
print("Extracted Fields:")
for key, value in extracted_fields.items():
    print(f"{key}: {value}")
