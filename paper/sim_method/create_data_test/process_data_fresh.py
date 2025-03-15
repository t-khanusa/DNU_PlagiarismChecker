import os
import json

with open("all_sent_it_fresh.json", "r") as file:
    all_sents = json.load(file)
    list_result = []
    for sent in all_sents:
        if (len(sent.split(' '))) > 15:
            check_1 = True
            if ('..' in sent) or ('\n \n' in sent) or ('•' in sent) or \
                ('*' in sent) or ('−' in sent) or ('' in sent) or ('➢' in sent) or \
                (':' in sent) or ('+' in sent) or ('_' in sent) or ('𝑓' in sent) or ('❖' in sent) \
                or ('-' in sent) or ('o' in sent) or ('a)' in sent) or ('[' in sent) or (']' in sent)\
                or ('(' in sent) or (')' in sent) or ('▪' in sent) or ('<' in sent) or ('%' in sent)\
                or ('“' in sent) or ('”' in sent) or ('?' in sent) \
                or ('Th.S' in sent) or ('ThS' in sent) or ('TS' in sent) or ('Thầy' in sent) or ('Cô' in sent) \
                or ('thầy' in sent) or ('cô' in sent) or ('cảm ơn' in sent) or ('' in sent) or ('–' in sent) or ('    ' in sent):
                check_1 = False
            if check_1:
                check_2 = True
                for i in range(50):
                    if str(i) in sent:
                        check_2 = False
                if check_2:
                    cleaned_text = sent.replace('\n', ' ')
                    list_result.append(cleaned_text)
    print(len(list_result))    

    with open("all_sent_it_fresh_filter.json", "w") as file1:
        json.dump(list_result, file1, ensure_ascii=False, indent=4)
    # total_sentences = len(list_result)
    # sentences_per_file = total_sentences // 10
    # remainder = total_sentences % 10
    # for i in range(10):
    #     start_index = i * sentences_per_file + min(i, remainder)
    #     end_index = start_index + sentences_per_file + (1 if i < remainder else 0)
    #     part = list_result[start_index:end_index]

    #     with open(f"gen_data_close_meaning_part_{i+1}.json", "w") as file1:
    #         json.dump(part, file1, ensure_ascii=False, indent=4)