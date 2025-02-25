from postgre_database import SessionLocal, PDFFile, Sentence

session = SessionLocal()

# Fetch PDF by filename
pdf = session.query(PDFFile).filter_by(filename="thesis_ai.pdf").first()
if pdf:
    print(f"PDF: {pdf.filename}")
    for sentence in pdf.sentences:
        print(f"Sentence: {sentence.sentence}")

session.close()
