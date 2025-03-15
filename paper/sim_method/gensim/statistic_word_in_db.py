import json

dict_result = {}
with open("../create_data_test/all_sent_it_in_db_filter.json", "r") as file3:
    all_sents = json.load(file3)
    for sent in all_sents:
        for word in sent.split(' '):
            if word not in dict_result:
                dict_result[word] = 0
            else:
                dict_result[word] += 1
sorted_data = dict(sorted(dict_result.items(), key=lambda item: item[1], reverse=True))
with open("statistic_word.json", "w") as file:
    json.dump(sorted_data, file, ensure_ascii=False, indent=4)