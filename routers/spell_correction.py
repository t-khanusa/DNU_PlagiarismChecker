from config.conf import BASEURL, COOKIE
from transformers import pipeline
import requests
import json
import nltk
import fitz
import re
from datetime import datetime, timezone

baseURL = BASEURL
MAX_LENGTH = 512
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2", device=0)

def check_box(doc, sentences):    
    # Mapping of sentences to their locations
    sentence_locations = {}

    # Track which sentences we've found
    found_sentences = set()
    
    # First pass: try direct search which is faster
    for page_num, page in enumerate(doc):
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
        
        return " ".join(clean_text(page.get_text()) for page in doc).strip()
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

async def spell_checker(file_path, file_check_id, time_start):
    url = f"{baseURL}/api/checker/files/{file_check_id}/grammar"
    document = fitz.open(file_path)
    content = read_pdf(file_path)
    raw_sentences = split_into_sentences(content)
    if not raw_sentences:
        return {"error": "No valid raw_sentences found in document"}    
    print(f"Number of raw_sentences processed: {len(raw_sentences)}")


    # Batch prediction
    predictions = corrector(raw_sentences, max_length=MAX_LENGTH)
    # Print predictions
    matched_sentences = []
    for text, pred in zip(raw_sentences, predictions):
        if text != pred['generated_text']:
            matched_sentences.append(text)

    sentence_locations = check_box(document, matched_sentences)
    page_sentences = {}
    for sentence, locations in sentence_locations.items():
        if not locations:
            continue
        for location in locations:
            page_num = location["page_num"] + 1
            # print(f"type page_num: {type(page_num)}")
            
            if page_num not in page_sentences:
                page_sentences[page_num] = []
            found = False
            for existing in page_sentences[page_num]:
                if existing['sentence'] == sentence:
                    found = True
                    break
            
            if not found:
                # Lambda function to merge rectangles on the same line
                merge_rects = lambda rects: [
                    [min(group, key=lambda r: r[0])[0],  # min x0
                        min(group, key=lambda r: r[1])[1],  # min y0  
                        max(group, key=lambda r: r[2])[2],  # max x1
                        max(group, key=lambda r: r[3])[3]]  # max y1
                    for group in [
                        [rect for rect in rects if abs(rect[1] - y) < 5]  # Group by similar y-coordinate (within 5 pixels)
                        for y in sorted(set(rect[1] for rect in rects))
                    ] if group
                ]
                
                merged_rects = merge_rects(location["rects"])                
                page_sentences[page_num].append({
                    'sentence': sentence,
                    "suggestion": "suggestion",
                    "text":"text",
                    'rects': merged_rects
                })

    similarity_box_sentences = []
    for page_num, sentences in sorted(page_sentences.items()):
        similarity_box_sentences.append({
            'page_number': page_num,
            'spelling_error': sentences
        })

    result = {
        "file_check_id": file_check_id,
        "page_result": similarity_box_sentences,
        "spell_value": len(matched_sentences),
        "create_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    output_file = f"spell_check_{file_check_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Spell check results saved to {output_file}")
    
    headers_auth = {
        'Content-Type': 'application/json',
        'Cookie': 'jwt='+COOKIE
    }
    response2 = requests.request("POST", url, headers=headers_auth, data=json.dumps(result))
    # print("Spell check thanh cong")
    return {"message": "Thành công"}

