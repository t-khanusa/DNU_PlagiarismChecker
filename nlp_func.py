import fitz  # PyMuPDF
import nltk
from pyvi.ViTokenizer import tokenize
import torch
# from docxtpl import DocxTemplate
import os
# from docx2pdf import convert
import numpy as np

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_into_sentences(documents):
    sentence_lists = []
    sentences = nltk.sent_tokenize(documents)
    MAX_LENGTH = 450  # Setting a safe limit below the model's max of 512
    
    for sent in sentences:
        if len(sent.split(" ")) > 6:
            tokenizer_sent = tokenize(sent)
            # Split long sentences into smaller chunks
            words = tokenizer_sent.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_len = len(word.split('_'))  # Count the number of subtokens
                if current_length + word_len > MAX_LENGTH:
                    if current_chunk:  # Only add if we have something
                        sentence_lists.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_len
                else:
                    current_chunk.append(word)
                    current_length += word_len
            
            if current_chunk:  # Add the last chunk if it exists
                sentence_lists.append(' '.join(current_chunk))
                
    sentence_lists = np.array(sentence_lists)
    return sentence_lists, sentences


def preprocess_sentences(sentence_lists, tokenizer):
    tokenized_sentences = []
    for sentences in sentence_lists:
        tokenized_doc_sentences = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
            tokenized_doc_sentences.append(inputs)
        tokenized_sentences.append(tokenized_doc_sentences)
    return tokenized_sentences


def get_sentence_embeddings(tokenized_sentences):
    sentence_embeddings = []
    for doc_sentences in tokenized_sentences:
        doc_embeddings = []
        for inputs in doc_sentences:
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                doc_embeddings.append(cls_embeddings)
        sentence_embeddings.append(doc_embeddings)
    return sentence_embeddings

def get_similar_sentences(model, corpus_name, corpus, embedding_vector, sentences):
    
    # print("number_of_sentences: ", number_of_sentences)
    number_of_sentences = len(sentences)
    check_percent_similarity = []
    similarity_sentences_dict = {}

    for i in range(len(corpus)):
        cosine_between_doc = model.similarity(np.array(corpus[i]), embedding_vector).numpy()
        max_in_columns = np.max(cosine_between_doc, axis=0)
        mask = np.zeros_like(cosine_between_doc, dtype=bool)

        # Iterate over each column
        for col in range(cosine_between_doc.shape[1]):
            max_indices = np.where(cosine_between_doc[:, col] == max_in_columns[col])[0]
            
            if max_in_columns[col] >= 0.8:
                if max_indices.size > 0:
                    mask[max_indices[0], col] = True

        result = np.where(mask, cosine_between_doc, 0)
        
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                if result[x,y] >= 0.8: 
                    if corpus_name[i] not in similarity_sentences_dict:
                        similarity_sentences_dict[corpus_name[i]] = []
                        similarity_sentences_dict[corpus_name[i]].append(sentences[y])
                    else: 
                        if sentences[y] not in similarity_sentences_dict[corpus_name[i]]:
                            similarity_sentences_dict[corpus_name[i]].append(sentences[y])

        non_zero_count = np.count_nonzero(result)
        # print("non_zero_count: ", non_zero_count)
        check_percent_similarity.append(int(non_zero_count*100/number_of_sentences))
    check_percent_similarity = np.array(check_percent_similarity)

    return check_percent_similarity, similarity_sentences_dict


def draw_number_in_box(page, number, x, y):
    # Define the size and position of the box
    box_width = 20
    box_height = 15

    # Draw the cyan-colored box
    rect = fitz.Rect(x - box_width, y - box_height / 2, x, y + box_height / 2)
    page.draw_rect(rect, color=(0, 1, 1), fill=(0, 1, 1))  # Cyan color

    # Calculate the position to center the number within the box
    number_width = 7  # Approximate width of a number character
    number_height = 6  # Approximate height of a number character
    text_x = x - box_width / 2 - number_width / 2
    text_y = y - number_height / 2

    # Add the number inside the box
    page.insert_text((text_x, text_y), str(number), fontsize=12, color=(1, 1, 1))


def merge_rects(rects):
    """Merge rectangles that are on the same line."""
    if not rects:
        return rects

    merged = []
    current = rects[0]

    for rect in rects[1:]:
        if abs(current.y1 - rect.y1) < 2:
            current |= rect  
        else:
            merged.append(current)
            current = rect

    merged.append(current)
    return merged


def highlight_duplicates_in_pdf(pdf_path, duplicate_sentences_dict, colors):
    doc = fitz.open(pdf_path)
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        number = 0
        highlighted = []
        for doc_file in duplicate_sentences_dict:
            duplicate_sentences = duplicate_sentences_dict[doc_file]
            for sentence in duplicate_sentences:
                if sentence in highlighted:
                    continue         
                text_instances = page.search_for(sentence)

                text_instances.sort(key=lambda r: (r.y1, r.x0))
                merged_instances = merge_rects(text_instances)

                for inst in merged_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=colors[number])  # Red color
                    highlight.update()
                highlighted.append(sentence)

                # x, y = inst[:2]
                # page.insert_text((x - 20, y), str(number), fontsize=12, color=colors[number])
                # draw_number_in_box(page, number, x - 10, y)
            number += 1
    return doc

def highlight_and_sumary_pdf(pdf_path, duplicate_sentences, summary_text):
    colors = [    
        (1.0, 0.8, 0.8), # đỏ
        (1.0, 0.8, 1.0), # tím
        (0.6, 0.8, 1), # xanh nước biển
        (0.6, 1.0, 0.8), # xanh lá cây
        (1.0, 0.8, 0.4), # vàng  
    ]

    doc = highlight_duplicates_in_pdf(pdf_path, duplicate_sentences, colors)
    docx = DocxTemplate('./docxtemplate/sim_rp.docx')

    context = {
        "file_name": summary_text['file_name'],
        "sim": summary_text['Total_percent'],
        "sim_name1": summary_text['sim_name1'],
        "sim1": summary_text['sim1'],
        "sim_name2": summary_text['sim_name2'],
        "sim2": summary_text['sim2'],
        "sim_name3": summary_text['sim_name3'],
        "sim3": summary_text['sim3'],
        "sim_name4": summary_text['sim_name4'],
        "sim4": summary_text['sim4'],
        "sim_name5": summary_text['sim_name5'],
        "sim5": summary_text['sim5'],
    }
    docx.render(context)
    docx_save_path = "./docxtemplate/test.docx"
    docx.save(docx_save_path)

    # current_directory = os.path.abspath(os.getcwd())
    # docx_path = f'{current_directory}/docxtemplate/test.docx'

    # absolute_path = os.path.abspath(docx_path)
    # pdf_folder = f'{os.path.abspath(current_directory)}/docxtemplate'

    # convert_to_pdf = f"libreoffice --headless --convert-to pdf {absolute_path} --outdir {pdf_folder}"
    # subprocess.run(convert_to_pdf, shell=True)
    import time
    t0 = time.time()
    convert(docx_save_path)
    t1 = time.time()
    print("Convert time: ", t1-t0)
    pdf_document = fitz.open('./docxtemplate/test.pdf')
    doc.insert_pdf(pdf_document)
    output_path = "final_output_with_summary.pdf"
    doc.save(output_path)
    # if os.path.exists(absolute_path):
    #     os.remove(absolute_path)
    # if os.path.exists('./docxtemplate/test.pdf'):
    #     os.remove('./docxtemplate/test.pdf')