from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from postgreSQL.postgre_database import PDFFile, Sentence, SessionLocal

# Create a new session
session = SessionLocal()

# Function to insert data into tables
def insert_data(filename, sentences_list):
    try:
        # Insert into PDFFile table
        pdf_file = PDFFile(filename=filename)
        session.add(pdf_file)
        session.commit()  # Commit to get the generated pdf_id

        # Insert sentences linked to the PDF
        for sentence_text in sentences_list:
            sentence = Sentence(pdf_id=pdf_file.pdf_id, sentence=sentence_text)
            session.add(sentence)

        # Commit all sentences
        session.commit()
        print(f"Inserted {len(sentences_list)} sentences for {filename}")

    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()
        
    return pdf_file.pdf_id
