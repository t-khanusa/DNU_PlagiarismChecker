import fitz  # PyMuPDF
import nltk
# from sentence_transformers import SentenceTransformer
# from pyvi.ViTokenizer import tokenize
# import numpy as np
import os
# import time
import json

nltk.download('punkt')

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_into_sentences(documents):
    # sentence_lists = []
    sentence_sets = set()
    sentences = nltk.sent_tokenize(documents)
    for sent in sentences:
        if len(sent.split(" ")) > 6:
            # tokenizer_sent = tokenize(sent)
            # sentence_lists.append(sent)
            sentence_sets.add(sent)
    # sentence_lists = np.array(sentence_lists)
    
    # return sentence_lists, sentences
    return sentence_sets
    
set_final = set()
fold_path = "../database/IT"
# for fold in os.listdir(fold_path):
#     file_path = os.path.join(fold_path, fold)
#     print(file_path)
for files in os.listdir(fold_path):
    pdf_path = os.path.join(fold_path, files)
    # print(pdf_path)
    content = read_pdf(pdf_path)
    sentence_sets = split_into_sentences(content)
    set_final.update(sentence_sets)

with open("all_sent_it.json", "w") as file:
    json.dump(list(set_final), file, ensure_ascii=False, indent=4)