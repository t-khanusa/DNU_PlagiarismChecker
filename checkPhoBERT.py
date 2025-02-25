import fitz  
import nltk
import numpy as np
import time
import os, glob
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_sentences(documents):
    sentence_lists = nltk.sent_tokenize(documents)
    return sentence_lists


def get_embeddings(sentences, model, tokenizer, batch_size=128):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        # print("Input: ", inputs.shape)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            # print("cls shape", cls_embeddings.shape)
        all_embeddings.append(cls_embeddings)
    # print(len(all_embeddings))
    return torch.cat(all_embeddings, dim=0)


def calculate_sentence_similarity(corpus, sentences, corpus_name, embedding_vector):

    number_of_sentences = len(sentences)
    check_percent_similarity = []
    similarity_sentences_dict = {}
    # print("embedding_vector.shape: ", np.array(embedding_vector).shape)

    for i in range(len(corpus)):
        # print("corpus.shape: ", np.array(corpus[i]).shape)
        cosine_between_doc = cosine_similarity(np.array(corpus[i]), np.array(embedding_vector))
        max_in_columns = np.max(cosine_between_doc, axis=0)
        mask = np.zeros_like(cosine_between_doc, dtype=bool)

        # Iterate over each column
        for col in range(cosine_between_doc.shape[1]):
            max_indices = np.where(cosine_between_doc[:, col] == max_in_columns[col])[0]
            
            if max_in_columns[col] >= 0.9:
                if max_indices.size > 0:
                    mask[max_indices[0], col] = True

        result = np.where(mask, cosine_between_doc, 0)
        
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                if result[x,y] >= 0.9: 
                    if corpus_name[i] not in similarity_sentences_dict:
                        similarity_sentences_dict[corpus_name[i]] = []
                        similarity_sentences_dict[corpus_name[i]].append(sentences[y])
                    else: 
                        if sentences[y] not in similarity_sentences_dict[corpus_name[i]]:
                            similarity_sentences_dict[corpus_name[i]].append(sentences[y])

        non_zero_count = np.count_nonzero(result)
        check_percent_similarity.append(int(non_zero_count*100/number_of_sentences))
    check_percent_similarity = np.array(check_percent_similarity)

    return check_percent_similarity, similarity_sentences_dict


corpus = np.load("./Corpus/Corpus_old.npy", allow_pickle=True)
corpus_name = np.load("./Corpus/Corpus_name_old.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

bamuoi = []
saumuoi = []
chinmuoi = []

# corpus = []
# corpus_name = []
# for file_path in glob.glob("/hdd1/similarity/CheckSimilarity/database/IT/*.pdf"):
#     content = read_pdf(file_path)
#     print("read_pdf done")
#     sentence_lists = split_into_sentences(content)
#     print("split_into_sentences done")
#     number_of_sentences = len(sentence_lists)
#     print("preprocess_sentences done")
#     embedding_vector = get_embeddings(sentence_lists, model, tokenizer)
#     corpus.append(embedding_vector)
#     corpus_name.append(file_path.split("/")[-1])
# corpus = np.array(corpus, dtype="object")
# corpus_name = np.array(corpus_name)

# np.save("./Corpus/Corpus_old.npy", corpus)
# np.save("./Corpus/Corpus_name_old.npy", corpus_name)

for file_path in glob.glob("/hdd1/similarity/CheckSimilarity/create_data_test/data_test_it/*.pdf"):
    print("File path: ", file_path)
    content = read_pdf(file_path)
    # print("read_pdf done")
    sentence_lists = split_into_sentences(content)
    # print("split_into_sentences done")
    number_of_sentences = len(sentence_lists)
    # print("preprocess_sentences done")
    embedding_vector = get_embeddings(sentence_lists, model, tokenizer)
    # print("get_sentence_embeddings done")
    check_percent_similarity, similarity_sentences_dict = calculate_sentence_similarity(corpus, sentence_lists, corpus_name, embedding_vector)

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