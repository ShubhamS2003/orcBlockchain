import streamlit as st
import cv2
import pytesseract
import tempfile
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import re
import pandas as pd

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Sample data for training the SVM model (you need to replace this with your labeled data)
data = [
    {"text": "1234567890", "label": "identification_number"},
    {"text": "DL No. : ABC1234567 1234", "label": "identification_number"},
    {"text": "Aadhar Card No. 123456789012", "label": "identification_number"},
    {"text": "PAN Card No. ABCDE1234F", "label": "identification_number"},
    {"text": "Voter ID No. ABCDE1234F", "label": "identification_number"},
    {"text": "Passport No. P12345678", "label": "identification_number"},
    {"text": "GSTIN No. 12ABCDE3456F7G8", "label": "identification_number"},
    {"text": "EPIC No. ABCDE1234F", "label": "identification_number"},
    {"text": "IFSC Code: ABCD1234567", "label": "identification_number"},
    {"text": "MICR Code: 123456789", "label": "identification_number"},
    {"text": "UPC Code: 0A1B2C3D", "label": "identification_number"},
    {"text": "ESIC No. 1234567890", "label": "identification_number"},
    {"text": "TIN No. AB123456789012", "label": "identification_number"},
    {"text": "UAN No. 123456789012", "label": "identification_number"},
    {"text": "IMEI No. 123456789012345", "label": "identification_number"},
    {"text": "SIM No. 12345678901234567890", "label": "identification_number"},
    {"text": "Patent No. 315456", "label": "patent_number"},
    {"text": "WHI7595242", "label": "election_number"},
    {"text": "Certificate No. IN-at78944260000018", "label": "certificate_number"},
    {"text": "Certcate No IN-at78944260000018", "label": "certificate_number"},  # Example certificate number
    # Add more labeled examples here
]

# Convert the labeled data into a DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Create TF-IDF vectors from the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train_tfidf, y_train)

# Define regular expression patterns for identification numbers using RE2
identification_patterns = [
    # Patterns for unique identification numbers
    r'\b\d{10}\b',                     # Matches 10-digit numbers
    r'DL No\. : [A-Z0-9]+\s\d{4,}',    # DL No. : 
    r'Civil Case No\. \d{4}/\d{4}',    # Civil Case No. 1234/5678
    r'VA-\d+',                          # VA-123
    r'Aff-\d+',                         # Aff-456
    r'APL-\d+',                         # APL-789
    r'Evidence ID-\d+',                 # Evidence ID-987
    r'Witness ID-[A-Z]\d+',             # Witness ID-A123
    r'Order No\. \d{3}/\d{4}',          # Order No. 123/5678
    r'BB-\d{4}-\d{3}',                  # BB-1234-567
    r'CS-\d{4}-\d{3}',                  # CS-1234-567
    r'CD-\d+',                          # CD-123
    r'MP-\d{4}-\d{3}',                  # MP-1234-567
    r'CA-\d{4}-\d{3}',                  # CA-1234-567
    r'ER-\d{4}-\d{3}',                  # ER-1234-567
    r'PIL No\. \d{3}/\d{4}',            # PIL No. 123/5678
    r'Motion No\. [A-Z]{2}-\d{3}',      # Motion No. AB-123
    r'Adjournment Application - AA-\d{4}-\d{3}',  # Adjournment Application - AA-1234-567
    r'EA-\d{4}-\d{3}',                  # EA-1234-567
    r'LN-\d{4}-\d{3}',                  # LN-1234-567
    r'SA-\d{4}-\d{3}',                  # SA-1234-567
    r'RA-\d{4}-\d{3}',                  # RA-1234-567
    r'CC-\d{4}-\d{3}',                  # CC-1234-567
    r'MA-\d{4}-\d{3}',                  # MA-1234-567
    r'Patent No\. \d+', # Matches "Patent No. 315456" format
    r'Certificate No\. [A-Z0-9-]+', # Matches "Certificate No." followed by alphanumeric characters and hyphens
    r'Certcate No\.  IN-at78944260000018', # Matches the specific certificate number
    # Patterns for other court-related identifiers and information
    r'Case No\. [A-Z0-9]+\s\d{4,}',      # Case No. XYZ 1234
    r'Suit No\. \d{4}/\d{4}',           # Suit No. 1234/5678
    r'Writ Petition No\. \d{4}/\d{4}',  # Writ Petition No. 1234/5678
    r'Criminal Case No\. [A-Z]+\s\d{4}',# Criminal Case No. ABC 1234
    r'Appeal No\. \d+',                 # Appeal No. 123
    r'Revision Petition No\. [A-Z]+\s\d{4}', # Revision Petition No. DEF 1234
    r'ID [A-Z0-9]+\s\d{10}',            # ID ABC1234567 1234567890
    r'Adhar Card No\. \d{12}',          # Adhar Card No. 123456789012
    r'Permanent Account Number Card\. [A-Z]{5}\d{4}[A-Z]',# PAN Card No.ABCDE1234F
    r'FIR No\. [A-Z]{3}/\d{3}/\d{4}',   # FIR No. XYZ/123/2023
    r'Complaint No\. \d{4}/\d{4}',      # Complaint No. 1234/5678
    r'Suit for Declaration',            # Suit for Declaration
    r'Petition for Divorce',            # Petition for Divorce
    r'Ruling in Case No\. [A-Z0-9]+\s\d{4,}',  # Ruling in Case No. XYZ 1234
    r'Order of [A-Z]+\sCourt',         # Order of High Court
    # r'Judgment dated \d{2}/\d{2}/\d{4}', # Judgment dated 01/12/2023
    # r'\d{2}/\d{2}/\d{4}',               # Date in DD/MM/YYYY format
    # r'\d{1,2}-\d{1,2}-\d{4}',           # Date in D-M-YYYY format
    r'Registered Office: [A-Za-z\s]+', # Registered Office: ABC Law Firm
    r'Advocate [A-Za-z\s]+',            # Advocate Mr. John Doe
    # Patterns for Indian government document identification numbers
    r'Aadhar Card No\. \d{12}',          # Aadhar Card No. 123456789012 (12-digit)
    r'PAN Card No\. [A-Z]{5}\d{4}[A-Z]', # PAN Card No. ABCDE1234F
    r'Voter ID No\. [A-Z0-9]{10}',       # Voter ID No. ABCDE1234F
    r'Passport No\. [A-Z0-9]+\d+',       # Passport No. P12345678 (Alphanumeric)
    r'GSTIN No\. [0-9A-Z]{15}',          # GSTIN No. 12ABCDE3456F7G8 (15-character)
    r'EPIC No\. [A-Z0-9]{10}',           # EPIC No. ABCDE1234F (10-character)
    r'IFSC Code: [A-Z]{4}\d{7}',         # IFSC Code: ABCD1234567
    r'MICR Code: \d{9}',                 # MICR Code: 123456789
    r'UPC Code: [0-9A-F]+',              # UPC Code: 0A1B2C3D
    r'ESIC No\. \d{10}',                 # ESIC No. 1234567890 (10-digit)
    r'TIN No\. [A-Z]{2}\d{11}',          # TIN No. AB123456789012 (13-character)
    r'UAN No\. \d{12}',                  # UAN No. 123456789012 (12-digit)
    r'IMEI No\. \d{15}',                 # IMEI No. 123456789012345 (15-digit)
    r'SIM No\. \d{20}',                  # SIM No. 12345678901234567890 (20-digit)
    # r'[A-Z0-9]{9}', # Matches 9-character alphanumeric election numbers remember it
# Add more patterns as needed
    

]

def classify_text_segments(text_segments):
    classified_segments = []
    for text_segment in text_segments:
        tfidf_vector = vectorizer.transform([text_segment])
        classification = svm_classifier.predict(tfidf_vector)[0]
        classified_segments.append({"text": text_segment, "classification": classification})
    return classified_segments

def extract_identification_numbers(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    custom_config = r'--oem 3 --psm 6'
    detected_text = pytesseract.image_to_string(img, config=custom_config)

    # Extract text using regular expressions
    extracted_text = ""
    for pattern in identification_patterns:
        identification_numbers = re.findall(pattern, detected_text)
        extracted_text += "\n".join(identification_numbers) + "\n"

    # Classify each extracted text segment using the SVM classifier
    classified_identification_numbers = classify_text_segments(identification_numbers)

    return detected_text, extracted_text, classified_identification_numbers

def main():
    st.title("Text and Identification Number Extractor with Patterns")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Extracted Text:")
        st.write("Please wait while we process the image...")

        # Save the uploaded image to a temporary file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img.write(uploaded_image.read())
        temp_img_path = temp_img.name

        # Close the temporary file to release the resource
        temp_img.close()

        # Extract text, patterns, and classify identification numbers when the user uploads an image
        text, extracted_text, classified_identification_numbers = extract_identification_numbers(temp_img_path)

        # Display the extracted text
        st.write(text)

        # Display the extracted identification numbers
        st.write("Extracted Identification Numbers:")
        st.write(extracted_text)

        # Display the classified identification numbers
        st.write("Classified Identification Numbers:")
        for i, segment in enumerate(classified_identification_numbers, 1):
            st.write(f"{i}. Text: {segment['text']}, Classification: {segment['classification']}")

        # Create a button to download the extracted text and classified numbers as a PDF file
        if st.button("Download as PDF"):
            pdf_filename = "extracted_data.pdf"
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            style = styles["Normal"]
            elements = []

            # Add detected sentences to the PDF
            sentences = text.split('\n\n')
            for sentence in sentences:
                p = Paragraph(sentence, style)
                elements.append(p)

            # Add classified identification numbers to the PDF
            for segment in classified_identification_numbers:
                p = Paragraph(f"Text: {segment['text']}, Classification: {segment['classification']}", style)
                elements.append(p)

            doc.build(elements)
            st.success(f"[Download PDF]({pdf_filename})")

        # Remove the temporary image file after closing it
        os.remove(temp_img_path)

if _name_ == "_main_":
    main()