import fitz  
import nltk
import numpy as np
import time
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from nlp_func import read_pdf, split_into_sentences, highlight_and_sumary_pdf, get_similar_sentences
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI()

@app.get("/home/")
async def test():
    corpus = np.load("./Corpus/Corpus.npy", allow_pickle=True)
    corpus_name = np.load("./Corpus/Corpus_name.npy")


    model = SentenceTransformer("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")
    model.max_seq_length=128
    file_path = "./test_similarity.pdf"

    content = read_pdf(file_path)
    sentence_lists, sentences = split_into_sentences(content)
    number_of_sentences = len(sentence_lists)
    embedding_vector = model.encode(sentence_lists)

    check_percent_similarity, similarity_sentences_dict = get_similar_sentences(model, corpus_name, corpus, embedding_vector, sentences)

    sorted_indices = np.argsort(check_percent_similarity)[::-1]

    top5_similarity_values = check_percent_similarity[sorted_indices][:5]
    top5_similarity_docs = corpus_name[sorted_indices][:5]
    top5_similarity_sentences_dict = {}

    Number_of_Similarity_sentences = []
    for similarity_doc in similarity_sentences_dict:
        for sentences in similarity_sentences_dict[similarity_doc]:
            if sentences not in Number_of_Similarity_sentences:
                Number_of_Similarity_sentences.append(sentences)

    for similarity_doc in top5_similarity_docs:
        if similarity_doc in similarity_sentences_dict:
            top5_similarity_sentences_dict[similarity_doc] = similarity_sentences_dict[similarity_doc]


    print(top5_similarity_docs)
    print(top5_similarity_values)
    Total_percent = int(round(len(Number_of_Similarity_sentences)/number_of_sentences,2)*100)


    return JSONResponse(content={
                "data":{
                    "total_similarity_percent": Total_percent,
                    "top_similarity_documents": top5_similarity_docs.tolist(),
                    "top_similarity_values": top5_similarity_values.tolist(),
                    "similarity_sentences": similarity_sentences_dict
                }
            }, status_code = 200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)