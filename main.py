import os
import pytesseract
from pdf2image import convert_from_path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

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


documents = []
labels = [] 

for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    print(f"Processing file: {filename}") 
    
    text = extract_text_from_file(file_path)
    
    if text.strip():
        documents.append(text)
        print(f"Extracted text from {filename[:50]}...")  
    else:
        print(f"No text found in {filename}")

labels = ['Travel Order'] * 21 + ['Office Order'] * 10 + ['Special Order'] * 21


assert len(documents) == len(labels), f"Number of documents ({len(documents)}) does not match number of labels ({len(labels)})."

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
 
assert X.shape[0] == len(labels), "Mismatch between the number of documents and labels."


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'document_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")
