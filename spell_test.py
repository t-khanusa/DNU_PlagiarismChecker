import re
import fitz
import nltk
import numpy as np
from pyvi import ViTokenizer as tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer

MAX_LENGTH = 512
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2", device=0)

def check_box(doc, sentences):    
    # Mapping of sentences to their locations
    sentence_locations = {}

    # Track which sentences we've found
    found_sentences = set()
    
    # First pass: try direct search which is faster
    for page_num, page in enumerate(doc):
        print(page_num)
        print("--------------------------------")
        print(page)
        for sentence in sentences:
            if sentence in found_sentences:
                continue
    
            # Direct search for exact matches
            text_instances = page.search_for(sentence)
            if text_instances:
                if sentence not in sentence_locations:
                    sentence_locations[sentence] = []
                
                # Format the rectangles
                rects = [[float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)] 
                        for rect in text_instances]
                
                sentence_locations[sentence].append({
                    "page_num": page_num,
                    "rects": rects
                })
                
                found_sentences.add(sentence)
    return sentence_locations

def clean_text(text: str) -> str:
    """Clean whitespace from text."""
    return re.sub(r'\s+', ' ', text).strip()

def should_exclude_sentence(sent: str) -> bool:
    """Determine if a sentence should be excluded from processing."""
    sent = clean_text(sent)
    
    # Exclude patterns for non-content text
    exclude_patterns = [
        'CHƯƠNG', 'Hình', 'Bảng', 'Use Case', '    ',
        'em', 'Em', 'cảm ơn', 'Th.S', 'ThS', 'TS', 'Thầy', 'thầy', 'Cô'
    ]
    
    # Exclude short sentences or section headers
    if len(sent.split()) < 10:
        return True
        
    # Exclude section numbers
    if any(sent.strip().startswith(f"{i}.") for i in range(1, 100)):
        return True
    
    return any(pattern in sent for pattern in exclude_patterns)

def read_pdf(file_path: str):
    """Extract and clean text from a PDF file."""
    try:
        print("filepath", file_path)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]  # Get the page
        
        return " ".join(clean_text(page.get_text()) for page in doc).strip(), doc
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def split_into_sentences(text: str):
    """Split text into sentences and filter out unwanted ones."""
    text = clean_text(text)
    sentences = nltk.sent_tokenize(text)
    raw_sentences = []
    for sent in sentences:
        if not should_exclude_sentence(sent):
            raw_sentences.append(sent)

    return raw_sentences


file_path = "1903.pdf"
content, document = read_pdf(file_path)
raw_sentences = split_into_sentences(content)
# print(raw_sentences)

# Batch prediction
predictions = corrector(raw_sentences, max_length=MAX_LENGTH)
# Print predictions
matched_sentences = []
for text, pred in zip(raw_sentences, predictions):
    if text != pred['generated_text']:
        matched_sentences.append(text)
        print(text)
        print(pred['generated_text'])
        print("--------------------------------")

sentence_locations = check_box(document, matched_sentences)
page_sentences = {}
for sentence, locations in sentence_locations.items():
    if not locations:
        continue
    for location in locations:
        page_num = location["page_num"]
        
        if page_num not in page_sentences:
            page_sentences[page_num] = []
        found = False
        for existing in page_sentences[page_num]:
            if existing['content'] == sentence:
                found = True
                break
        
        if not found:
            page_sentences[page_num].append({
                'content': sentence,
                'rects': location["rects"]
            })

similarity_box_sentences = []
for page_num, sentences in sorted(page_sentences.items()):
    similarity_box_sentences.append({
        'pageNumber': page_num,
        'similarity_content': sentences
    })
print(similarity_box_sentences)
# document = fitz.open(file_path)
# # Process each sentence that matches with this document
# sentence_locations = check_box(document, matched_sentences)

# # Initialize page_sentences dictionary to store sentences by page number
# page_sentences = {}

# # Process each matched sentence and its locations
# for sentence, locations in sentence_locations.items():
#     # Skip if no locations found for this sentence
#     if not locations:
#         continue
        
#     # Process each location where this sentence was found
#     for location in locations:
#         page_num = location["page_num"]
        
#         if page_num not in page_sentences:
#             page_sentences[page_num] = []
        
#         # Check if this sentence is already added to this page
#         found = False
#         for existing in page_sentences[page_num]:
#             if existing['content'] == sentence:
#                 found = True
#                 break
        
#         if not found:
#             page_sentences[page_num].append({
#                 'content': sentence,
#                 'rects': location["rects"]
#             })

# # Format page sentences for output
# similarity_box_sentences = []
# for page_num, sentences in sorted(page_sentences.items()):
#     similarity_box_sentences.append({
#         'pageNumber': page_num,
#         'similarity_content': sentences
#     })




# # Calculate similarity percentage based on number of matched sentences
# similarity_value = 0
# if total_sentences > 0:
#     # Count total sentences found in this document
#     total_matched = len(matched_sentences)
#     similarity_value = int((total_matched / total_sentences) * 100)
#     # Cap at 100%
#     similarity_value = min(similarity_value, 100)

# # Add document to similarity documents
# similarity_documents.append({
#     'name': filename,
#     'similarity_value': similarity_value,
#     'similarity_box_sentences': similarity_box_sentences
# })

# # Get page size from the document
# page_size = {"width": 595.0, "height": 842.0}  # Default A4 size
# if document and hasattr(document, 'mediabox'):
# page_size = {
#     "width": document.mediabox[2],
#     "height": document.mediabox[3]
# }

# # Format final output
# result = {
# "data": {
#     "total_percent": len(matched_sentence_set) / total_sentences * 100,  # Always 100% for the document being checked
#     "size_page": page_size,
#     "similarity_documents": similarity_documents
# }
# }