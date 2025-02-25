import random
import json
from fpdf import FPDF

AVERAGE_SENT = 400 # THAY SỐ CÂU TRUNG BÌNH Ở ĐÂY

# for i in [30, 60, 90]:
PER_SIM = 90 # THAY PHẦN TRĂM TRÙNG Ở ĐÂY

NUMBER_SENT_IN_DB = int(AVERAGE_SENT * PER_SIM / 100 / 2)

NUMBER_CLOSE_MEANING_SENT = int(AVERAGE_SENT * PER_SIM / 100 / 2)

NUMBER_SENT_FRESH = int(AVERAGE_SENT * (100 - PER_SIM) / 100)

print("NUMBER_SENT_IN_DB: ", NUMBER_SENT_IN_DB)
print("NUMBER_CLOSE_MEANING_SENT: ", NUMBER_CLOSE_MEANING_SENT)
print("NUMBER_SENT_FRESH: ", NUMBER_SENT_FRESH)
for i in range(10):
    # DANH SÁCH FRESH IT
    with open("all_sent_it_fresh_filter.json", "r") as file1:
        all_sents_fresh = json.load(file1)
        list_sent_fresh = random.sample(all_sents_fresh, NUMBER_SENT_FRESH)
        # print(list_sent_fresh)

    # DANH SÁCH CÁC CÂU GẦN NGHĨA
    with open("close_meaning_sent.json", "r") as file2:
        all_sents_close_meaning = json.load(file2)
        list_sent_close_meaning = random.sample(all_sents_close_meaning, NUMBER_CLOSE_MEANING_SENT)

    # DANH SÁCH CÁC CÂU DÙNG ĐỂ SINH GẦN NGHĨA
    with open("gen_data_close_meaning.json", "r") as file4:
        list_gen_data_close_meaning = json.load(file4)

    # DANH SÁCH CÁC CÂU TRONG DB: Ở ĐÂY THỰC HIỆN LẤY CÁC CÂU MÀ KHÔNG NẰM TRONG DANH SÁCH CÂU ĐỂ SINH GẦN NGHĨA
    with open("all_sent_it_in_db_filter.json", "r") as file3:
        all_sents_it = json.load(file3)
        danh_sach_hop_le = [cau for cau in all_sents_it if cau not in list_gen_data_close_meaning] # không random vào câu trong danh sách sinh câu gần nghĩa
        list_sent_in_db = random.sample(danh_sach_hop_le, NUMBER_SENT_IN_DB)
    list_sent_in_new_file = list_sent_in_db + list_sent_close_meaning + list_sent_fresh

    pdf = FPDF()
    pdf.add_font('Arial', '', './Arial.ttf', uni=True)
    pdf.set_font('Arial', '', 12)

    pdf.add_page()

    margin_bottom = 20  
    text = ' '.join(list_sent_in_new_file)
    pdf.multi_cell(0, 10, text)

    if pdf.get_y() > pdf.h - margin_bottom:
        pdf.add_page()  

    pdf_output_path = f"./data_test_it/{PER_SIM}_{i+1}_it_test_.pdf"
    pdf.output(pdf_output_path)

    print(f"PDF file created at: {pdf_output_path}")

