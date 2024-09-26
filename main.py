import os
import pytesseract
from pdf2image import convert_from_path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Directory of files
data_dir = r'C:\Users\Acer\Videos\docx'

def extract_text_from_image(file_path):
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1]
    if ext in ['pdf']:
        return extract_text_from_image(file_path)
    elif ext in ['jpeg', 'jpg', 'png']:
        return pytesseract.image_to_string(file_path)

# Initialize empty lists for documents and labels
documents = []
labels = []  # Labels such as Travel Order, Office Order, Special Order

# Go through all files and extract text
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    print(f"Processing file: {filename}")  # Indicate the file being processed
    
    text = extract_text_from_file(file_path)
    
    if text.strip():  # Ensure the file contains text
        documents.append(text)
        print(f"Extracted text from {filename[:50]}...")  # Display a sample of the extracted text
    else:
        print(f"No text found in {filename}")

# Assuming you have 58 documents based on the previous error message
labels = ['Travel Order'] * 21 + ['Office Order'] * 10 + ['Special Order'] * 21

# Check if the number of documents matches the number of labels
assert len(documents) == len(labels), f"Number of documents ({len(documents)}) does not match number of labels ({len(labels)})."

# Use CountVectorizer to convert the text into feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Ensure the feature matrix and labels have the same length
assert X.shape[0] == len(labels), "Mismatch between the number of documents and labels."

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")

# Print precision, recall, F1 score, and confusion matrix
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer for future use
joblib.dump(model, 'document_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")
