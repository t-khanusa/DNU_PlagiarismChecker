import fitz  
import nltk
import numpy as np
import time
import os, glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from nlp_func import read_pdf, split_into_sentences, get_similar_sentences
nltk.download('punkt')

corpus = np.load("./Corpus/Corpus.npy", allow_pickle=True)
corpus_name = np.load("./Corpus/Corpus_name.npy")


model = SentenceTransformer("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")

bamuoi = []
saumuoi = []
chinmuoi = []


for file_path in glob.glob("/hdd1/similarity/CheckSimilarity/create_data_test/data_test_it/*.pdf"):
    content = read_pdf(file_path)
    sentence_lists, sentences = split_into_sentences(content)
    number_of_sentences = len(sentence_lists)
    embedding_vector = model.encode(sentence_lists, batch_size=256 ,device="cuda:0", show_progress_bar=True)
    print("Done!")
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

    Total_percent = int(round(len(Number_of_Similarity_sentences)/number_of_sentences,2)*100)


# summary_text = {
#     "file_name": file_path,
#     "Total_percent": Total_percent,
#     "sim_name1": top5_similarity_docs[0],
#     "sim1": top5_similarity_values[0],
#     "sim_name2": top5_similarity_docs[1],
#     "sim2": top5_similarity_values[1],
#     "sim_name3": top5_similarity_docs[2],
#     "sim3": top5_similarity_values[2],
#     "sim_name4": top5_similarity_docs[3],
#     "sim4": top5_similarity_values[3],
#     "sim_name5": top5_similarity_docs[4],
#     "sim5": top5_similarity_values[4],
# }

# t0 = time.time()
# highlight_and_sumary_pdf('./test_similarity.pdf', top5_similarity_sentences_dict, summary_text)
# t1 = time.time()
# print(f"Percent_of_similarity_sentences: {round(len(Number_of_Similarity_sentences)/number_of_sentences,2)*100}%")