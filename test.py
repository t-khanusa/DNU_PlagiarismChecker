# import json
# import fitz
# from sentence_transformers import SentenceTransformer
# from pyvi.ViTokenizer import tokenize
# import torch
# import numpy as np
# import nltk

# # if (len(sent.split(' '))) > 15:
# #     check_1 = True
# #     if ('..' in sent) or ('\n \n' in sent) or ('•' in sent) or \
# #         ('*' in sent) or ('−' in sent) or ('' in sent) or ('➢' in sent) or \
# #         (':' in sent) or ('+' in sent) or ('_' in sent) or ('𝑓' in sent) or ('❖' in sent) \
# #         or ('-' in sent) or ('o' in sent) or ('a)' in sent) or ('[' in sent) \
# #         or ('(' in sent) or (')' in sent) or ('▪' in sent) or ('<' in sent) or ('%' in sent)\
# #         or ('"' in sent) or ('"' in sent) or ('?' in sent) \
# #         or ('Th.S' in sent) or ('ThS' in sent) or ('TS' in sent) or ('Thầy' in sent) or ('Cô' in sent) \
# #         or ('thầy' in sent) or ('cô' in sent) or ('cảm ơn' in sent) or ('' in sent) or ('–' in sent) or ('    ' in sent):

# def read_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def split_into_sentences(documents):
#     sentence_lists = []
#     sentences = nltk.sent_tokenize(documents)
#     for sent in sentences:
#         if len(sent.split(" ")) > 10:
#             if ('..' in sent) or ('\n \n' in sent) or ('•' in sent) or \
#                 ('*' in sent) or ('−' in sent) or ('' in sent) or ('➢' in sent) or \
#                 (':' in sent) or ('+' in sent) or ('_' in sent) or ('𝑓' in sent) or ('❖' in sent) \
#                 or ('-' in sent) or ('o' in sent) or ('a)' in sent) or ('[' in sent) \
#                 or ('(' in sent) or (')' in sent) or ('▪' in sent) or ('<' in sent) or ('%' in sent)\
#                 or ('"' in sent) or ('"' in sent) or ('?' in sent) or ('... ...' in sent)\
#                 or ('Th.S' in sent) or ('ThS' in sent) or ('TS' in sent) or ('Thầy' in sent) or ('Cô' in sent) \
#                 or ('thầy' in sent) or ('cô' in sent) or ('cảm ơn' in sent) or ('' in sent) or ('–' in sent) or ('    ' in sent):
#                     tokenizer_sent = tokenize(sent)
#                     sentence_lists.append(tokenizer_sent)
#     sentence_lists = np.array(sentence_lists)
#     return sentence_lists, sentences

# file_path = '/hdd1/similarity/CheckSimilarity/database/IT/1451020083_NguyenTrongHieu_BaoCao_DATN.pdf'
# content = read_pdf(file_path)
# sentence_lists, sentences = split_into_sentences(content)

# print(sentence_lists)

# import fitz
# import nltk
# import numpy as np
# import re
# from typing import List, Tuple
# from pyvi.ViTokenizer import tokenize
# from sentence_transformers import SentenceTransformer


# def clean_whitespace(text):
#     # Remove extra whitespace and indentation
#     # Replace multiple spaces with single space
#     text = re.sub(r'\s+', ' ', text)
#     # Remove leading/trailing whitespace
#     return text.strip()

# def should_exclude_sentence(sent):
#     # Clean whitespace before checking patterns
#     sent = clean_whitespace(sent)
    
#     # Only exclude formatting-related patterns and non-content markers
#     exclude_patterns = [
#         # Headers and formatting
#         # '\n \n', '•', '➢', '❖', '▪',
#         # Chapter markers and numbering
#         'CHƯƠNG',
#         # References to figures and tables
#         'Hình', 'Bảng',
#         # Document structure markers
#         'Use Case',
#         # Empty space patterns
#         '    ',
#         # Acknowledgments section (usually not relevant for plagiarism)
#         'cảm ơn', 'Th.S', 'ThS', 'TS', 'Thầy', 'thầy', 'Cô'
#     ]

#     # Check for very short sentences or those that are likely headers
#     if len(sent.split()) < 10:
#         return True
        
#     # Check for sentences that are likely section numbers
#     if any(sent.strip().startswith(str(i) + '.') for i in range(1, 100)):
#         return True
    
#     return any(pattern in sent for pattern in exclude_patterns)

# def read_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         # Clean whitespace for each page
#         page_text = clean_whitespace(page.get_text())
#         text += page_text + " "
#     return text.strip()

# def split_into_sentences(documents):
#     sentence_lists = []
#     # Clean whitespace before sentence tokenization
#     documents = clean_whitespace(documents)
#     sentences = nltk.sent_tokenize(documents)
    
#     for sent in sentences:
#         # Keep sentences that have meaningful content
#         if not should_exclude_sentence(sent):
#             # Keep punctuation and parentheses as they might be part of quoted text
#             tokenizer_sent = tokenize(sent)
#             sentence_lists.append(tokenizer_sent)
    
#     return np.array(sentence_lists), sentences

# def find_sentence_positions(text: str, sentences: List[str]) -> List[Tuple[int, int]]:
#     """Find the start and end character positions for each sentence"""
#     positions = []
#     current_pos = 0
    
#     for sentence in sentences:
#         # Find the sentence in the text starting from the current position
#         start_pos = text.find(sentence.strip(), current_pos)
#         if start_pos != -1:
#             end_pos = start_pos + len(sentence)
#             positions.append((start_pos, end_pos))
#             current_pos = end_pos
#         else:
#             # If sentence not found (due to whitespace differences), use approximate position
#             positions.append((current_pos, current_pos + len(sentence)))
#             current_pos += len(sentence)
    
#     return positions

# model = SentenceTransformer("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")
# file_path = '/hdd1/similarity/CheckSimilarity/database/IT/1451020083_NguyenTrongHieu_BaoCao_DATN.pdf'
# content = read_pdf(file_path)

# sentence_lists, sentences = split_into_sentences(content)
# # number_of_sentences = len(sentence_lists)
# embedding_vector = model.encode(sentence_lists, batch_size=256 ,device="cuda:0", show_progress_bar=True)

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from create_milvus_db import collection
# from create_corpus import CorpusCreator
# Connect to Milvus
conn = connections.connect(host="localhost", port="19530", db_name="vector_database")
# Define the schema for the collection

# fields = [
#     # Primary key from PostgreSQL - required for direct lookups and joins
#     FieldSchema(name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    
#     # Document context - required for filtering and ordering
#     FieldSchema(name="pdf_id", dtype=DataType.INT64, description="Foreign key to PostgreSQL pdf_files.pdf_id"),
#     FieldSchema(name="sentence_index", dtype=DataType.INT64, description="Position in document"),
    
#     # Vector data
#     FieldSchema(name="sentence_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
# ]
# schema = CollectionSchema(fields, "Sentence similarity collection with PostgreSQL alignment")

# # Create or recreate the collection
# collection_name = "sentence_similarity"
# # if utility.has_collection(collection_name):
# #     utility.drop_collection(collection_name)
# collection = Collection(name=collection_name, schema=schema)

# Check collection size
num_vectors = collection.num_entities
print(f"Number of vectors in collection: {num_vectors}")