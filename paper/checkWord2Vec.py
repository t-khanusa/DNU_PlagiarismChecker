import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from distutils.version import LooseVersion, StrictVersion
import gensim


def preprocess_text(text):
    # Loại bỏ dấu câu và chuyển về chữ thường
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    return tokens


# Huấn luyện mô hình Word2Vec
def train_word2vec_model(documents):
    tokenized_docs = [preprocess_text(doc) for doc in documents]
    model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
    return model


def get_average_vector(text, model):
    tokens = preprocess_text(text)
    vector = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    return vector

def calculate_similarity(doc1, doc2, model):
    vector1 = get_average_vector(doc1, model).reshape(1, -1)
    vector2 = get_average_vector(doc2, model).reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]



from gensim.models import Word2Vec

# Tải mô hình tiền huấn luyện
model = Word2Vec.load('word2vec.model')

# Ví dụ biểu diễn từ
word_vector = model.wv['học']
print(word_vector)
