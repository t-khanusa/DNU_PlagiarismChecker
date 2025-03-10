from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence, engine
from create_milvus_db import insert_vectors, collection
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
from pyvi.ViTokenizer import tokenize
from sqlalchemy.orm import Session
from tqdm import tqdm
import numpy as np
import traceback
import glob
import fitz
import nltk
import time
import re
import os



class CorpusCreator:
    """Creates and manages a corpus of document embeddings for similarity search."""
    
    def __init__(self, model_path="./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338"):
        """Initialize the corpus creator with a sentence transformer model."""
        self.model = SentenceTransformer(model_path)
        self.model.max_seq_length = 256
        self.batch_size = 512
        self.collection = collection
        self.engine = engine

    def get_session(self):
        """Create a new database session."""
        return SessionLocal()

    def clean_text(self, text: str) -> str:
        """Clean whitespace from text."""
        return re.sub(r'\s+', ' ', text).strip()

    def should_exclude_sentence(self, sent: str) -> bool:
        """Determine if a sentence should be excluded from processing."""
        sent = self.clean_text(sent)
        
        # Exclude patterns for non-content text
        exclude_patterns = [
            'CHƯƠNG', 'Hình', 'Bảng', 'Use Case', '    ',
            'em', 'Em', 'cảm ơn', 'Th.S', 'ThS', 'TS', 'Thầy', 'thầy', 'Cô'
        ]
        
        # Exclude short sentences or section headers
        if len(sent.split()) < 10:
            return True
            
        # Exclude section numbers
        if any(sent.strip().startswith(f"{i}.") for i in range(1, 100)):
            return True
        
        return any(pattern in sent for pattern in exclude_patterns)

    def read_pdf(self, file_path: str) -> str:
        """Extract and clean text from a PDF file."""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]  # Get the page
            
            return " ".join(self.clean_text(page.get_text()) for page in doc).strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences and filter out unwanted ones."""
        text = self.clean_text(text)
        raw_sentences = nltk.sent_tokenize(text)
        
        return [tokenize(sent) for sent in raw_sentences 
                if not self.should_exclude_sentence(sent)]

    def process_document(self, file_path: str) -> Tuple[List[str], np.ndarray]:
        """Process a document: extract text, split into sentences, and generate embeddings."""
        start_time = time.time()
        
        # Read and process text
        content = self.read_pdf(file_path)
        read_time = time.time() - start_time
        print(f"Read {file_path} in {read_time:.2f}s")
        if not content:
            return [], np.array([])

        # Split into sentences
        start_split = time.time()
        sentences = self.split_into_sentences(content)
        split_time = time.time() - start_split
        print(f"Split {file_path} in {split_time:.2f}s")
        if not sentences:
            return [], np.array([])

        # Generate embeddings
        start_encode = time.time()
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            device="cuda:0",
            show_progress_bar=False
        )
        encode_time = time.time() - start_encode
        print(f"Encode {file_path} in {encode_time:.2f}s")  
        
        return sentences, np.array(embeddings)

    def insert_data(self, filename: str, file_path: str, sentences: List[str], 
                   embeddings: Optional[np.ndarray] = None) -> Tuple[int, List[int]]:
        """Insert document data into PostgreSQL and Milvus databases."""
        start_time = time.time()
        session = self.get_session()
        
        try:
            # Insert PDF file record
            pdf_file = PDFFile(
                filename=filename,
                file_path=file_path,
                total_sentences=len(sentences)
            )
            session.add(pdf_file)
            session.flush()

            # Insert sentences in batches
            sentence_objects = []
            batch_size = 1000
            for i in range(0, len(sentences), batch_size):
                batch_end = min(i + batch_size, len(sentences))
                batch_sentences = sentences[i:batch_end]
                
                for idx, sentence_text in enumerate(batch_sentences, start=i):
                    sentence = Sentence(
                        pdf_id=pdf_file.pdf_id,
                        sentence=sentence_text,
                        sentence_index=idx
                    )
                    session.add(sentence)
                    sentence_objects.append(sentence)
                
                session.flush()
            
            # Get sentence IDs
            sentence_ids = np.array([s.id for s in sentence_objects], dtype=np.int64)
            session.commit()
            
            # Insert embeddings into Milvus if provided
            if embeddings is not None:
                pdf_ids = np.full(len(sentences), pdf_file.pdf_id, dtype=np.int64)
                sentence_indices = np.arange(len(sentences), dtype=np.int64)
                
                # Ensure correct data type
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                
                # Insert vectors in batches
                chunk_size = 2000
                for i in range(0, len(sentence_ids), chunk_size):
                    chunk_end = min(i + chunk_size, len(sentence_ids))
                    insert_vectors(
                        sentence_ids[i:chunk_end].tolist(),
                        pdf_ids[i:chunk_end].tolist(),
                        sentence_indices[i:chunk_end].tolist(),
                        embeddings[i:chunk_end]
                    )

            total_time = time.time() - start_time
            print(f"Inserted {filename} in {total_time:.2f}s")
            return pdf_file.pdf_id, sentence_ids.tolist()

        except Exception as e:
            session.rollback()
            print(f"Error inserting data for {filename}: {str(e)}")
            raise
        finally:
            session.close()

    def get_sentence_by_id(self, sentence_id: int) -> Optional[Sentence]:
        """Retrieve a sentence by its ID."""
        with self.get_session() as session:
            return session.query(Sentence).filter(Sentence.id == sentence_id).first()

    def get_pdf_sentences(self, pdf_id: int) -> List[Sentence]:
        """Retrieve all sentences for a given PDF ID."""
        with self.get_session() as session:
            return session.query(Sentence).filter(
                Sentence.pdf_id == pdf_id
            ).order_by(Sentence.sentence_index).all()

    def get_pdf_by_filename(self, filename: str) -> Optional[PDFFile]:
        """Retrieve PDF file information by filename."""
        with self.get_session() as session:
            return session.query(PDFFile).filter(
                PDFFile.filename == filename
            ).first()

    def save_to_databases(self, file_path: str, sentences: List[str], 
                         embeddings: np.ndarray, db: Session) -> bool:
        """Save processed document data to databases, checking for duplicates."""
        try:
            # Check if file already exists
            filename = os.path.basename(file_path)
            existing_file = db.query(PDFFile).filter(
                PDFFile.filename == filename
            ).first()
            
            if existing_file:
                print(f"File {filename} already exists in database, skipping...")
                return False

            # Insert data
            self.insert_data(filename, file_path, sentences, embeddings)
            return True
                
        except Exception as e:
            print(f"Error saving document {file_path}: {str(e)}")
            print(traceback.format_exc())
            return False

    def process_pdf_batch(self, pdf_files: List[str]) -> None:
        """Process a batch of PDF files in parallel."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for pdf_file in pdf_files:
                futures.append((pdf_file, executor.submit(self.process_document, pdf_file)))
            
            for pdf_file, future in futures:
                try:
                    sentences, embeddings = future.result()
                    if sentences:
                        with self.get_session() as session:
                            self.save_to_databases(pdf_file, sentences, embeddings, session)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")

    def create_corpus(self, folder_path: str):
        """Process all PDF files in the specified folder."""
        # Find PDF files
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return

        print(f"Found {len(pdf_files)} PDF files")
        
        # Test database connection
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
                print("Successfully connected to PostgreSQL database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return

        try:
            # Process files in batches
            batch_size = 4
            with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
                for i in range(0, len(pdf_files), batch_size):
                    batch = pdf_files[i:i + batch_size]
                    self.process_pdf_batch(batch)
                    pbar.update(len(batch))

            print("\nCorpus creation completed!")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Cleaning up...")
        except Exception as e:
            print(f"\nError during corpus creation: {str(e)}")
            print(traceback.format_exc())

# if __name__ == "__main__":
#     # Initialize corpus creator
#     creator = CorpusCreator()
    
#     # Process folder
#     folder_path = "/hdd1/similarity/CheckSimilarity/audio-test"
#     start = time.time()
#     creator.create_corpus(folder_path)
    
#     print("Corpus creation completed!")
#     print(f"Total time: {time.time()-start:.2f}s")

#     # Print collection stats
#     from pymilvus import connections, Collection
#     connections.connect(host="localhost", port="19530", db_name="vector_database")
#     collection = Collection(name="sentence_similarity")
#     print(f"Number of vectors in collection: {collection.num_entities}")