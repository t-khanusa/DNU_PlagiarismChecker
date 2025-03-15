import json
from collections import defaultdict
from gensim import corpora, models, similarities
import fitz  # PyMuPDF
import nltk
import os

nltk.download('punkt')

# Load dữ liệu
with open("../create_data_test/all_sent_it.json", "r") as file3:
    text_corpus = json.load(file3)

with open("statistic_word.json", "r") as file3:
    data = json.load(file3)

# Tạo stoplist từ dữ liệu thống kê, giữ lại n từ xuất hiện nhiều nhất
stoplist = set(word for word, _ in sorted(data.items(), key=lambda item: item[1], reverse=True)[:40])

# Tiền xử lý văn bản và lọc từ trong stoplist
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Đếm tần số từ và chỉ giữ từ xuất hiện nhiều hơn một lần
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

# Tạo dictionary và corpus Bag-of-Words
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

# Khởi tạo mô hình LDA
lda_model = models.LdaModel(bow_corpus, num_topics=20, id2word=dictionary, passes=15)

# Chuyển đổi corpus sang vector của các chủ đề bằng LDA
lda_corpus = lda_model[bow_corpus]

# Tạo ma trận tương tự (similarity matrix) giữa các từ
tfidf = models.TfidfModel(bow_corpus)
similarity_matrix = similarities.SparseTermSimilarityMatrix(tfidf, dictionary)

# Khởi tạo TermSimilarityIndex từ ma trận tương tự
term_index = similarities.TermSimilarityIndex(similarity_matrix)

# Tạo chỉ mục dựa trên các chủ đề từ LDA với SoftCosineSimilarity
index = similarities.SoftCosineSimilarity(lda_corpus, term_index, num_features=lda_model.num_topics)

# Đọc nội dung file PDF
doc = fitz.open("../create_data_test/data_test_it/60_1_it_test_.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Chia nội dung thành các câu và tính độ tương đồng
sentences = nltk.sent_tokenize(text)
num_sent_sim = 0
i = 1
for sent in sentences:
    print(i)
    i += 1
    query_document = sent.lower().split()
    query_bow = dictionary.doc2bow(query_document)
    query_lda = lda_model[query_bow]  # Chuyển câu truy vấn sang vector chủ đề LDA
    sims = index[query_lda]  # Tính độ tương đồng với các câu trong corpus
    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        if score >= 0.8:
            num_sent_sim += 1
        break

# Tính toán phần trăm câu có độ tương đồng cao
print("per_sim: ", num_sent_sim)
per_sim = num_sent_sim / len(sentences)
print("per_sim: ", per_sim)
