import json
from collections import defaultdict
from gensim import corpora, models, similarities
import pprint
import fitz  # PyMuPDF
import nltk
import os, glob
# from nltk.tokenize import word_tokenize

nltk.download('punkt')

with open("../create_data_test/all_sent_it.json", "r") as file3:
    text_corpus = json.load(file3)  

with open("statistic_word.json", "r") as file3:
    data = json.load(file3)    

stoplist = set(list(sorted(data.items(), key=lambda item: item[1], reverse=True)[:50])) #lấy n cái đầu tiên

texts = [[word for word in document.lower().split() if word not in stoplist]
                for document in text_corpus]


frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
tfidf = models.TfidfModel(bow_corpus)
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))

bamuoi = []
saumuoi = []
chinmuoi = []
i = 0
for file_path in glob.glob("/hdd1/similarity/CheckSimilarity/create_data_test/data_test_it/*.pdf"):
    print(i)
    i += 1
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    sentences = nltk.sent_tokenize(text)
    num_sent_sim = 0
    for sent in sentences:
        query_document = sent.lower().split()
        query_bow = dictionary.doc2bow(query_document)
        sims = index[tfidf[query_bow]]
        for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
            if score >= 0.9:
                num_sent_sim += 1
            break
    Total_percent = round(num_sent_sim/len(sentences),2) * 100
    # Total_percent = num_sent_sim
    if Total_percent <= 30:
        bamuoi.append(Total_percent)
    elif 30<Total_percent <= 60:
        saumuoi.append(Total_percent)
    elif 60<Total_percent <= 90:
        chinmuoi.append(Total_percent)
    print("Total percent: ", Total_percent)

print("30: ", sum(bamuoi)/(30*len(bamuoi)))
print("60: ", sum(saumuoi)/(60*len(saumuoi)))
print("90: ", sum(chinmuoi)/(90*len(chinmuoi)))



# def read_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text


# def split_into_sentences(documents):
#     words = word_tokenize(documents)
#     # Chuyển tất cả các từ thành chữ thường
#     words_lower = [word.lower() for word in words]
#     return words_lower