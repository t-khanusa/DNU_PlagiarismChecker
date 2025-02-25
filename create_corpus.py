import fitz
import nltk
import numpy as np
import os
import glob
import re
import traceback
from tqdm import tqdm
from typing import List, Tuple, Dict
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence, engine
from create_milvus_db import insert_vectors, collection

class CorpusCreator:
    def __init__(self, model_path="./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338"):
        self.model = SentenceTransformer(model_path)
        self.model.max_seq_length = 256
        self.batch_size = 512  # Increased batch size for better performance
        self.collection = collection
        # Use the engine from postgre_database.py
        self.engine = engine
        # nltk.download('punkt', quiet=True)

    def get_session(self):
        """Create a new session with the engine"""
        return SessionLocal()

    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and indentation"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()

    def should_exclude_sentence(self, sent: str) -> bool:
        """Check if sentence should be excluded based on patterns"""
        # Clean whitespace before checking patterns
        sent = self.clean_whitespace(sent)
        
        # Only exclude formatting-related patterns and non-content markers
        exclude_patterns = [
            # Chapter markers and numbering
            'CHƯƠNG',
            # References to figures and tables
            'Hình', 'Bảng',
            # Document structure markers
            'Use Case',
            # Empty space patterns
            '    ',
            # Acknowledgments section
            'em', 'Em','cảm ơn', 'Th.S', 'ThS', 'TS', 'Thầy', 'thầy', 'Cô'
        ]
        
        # Check for very short sentences or those that are likely headers
        if len(sent.split()) < 10:
            return True
            
        # Check for sentences that are likely section numbers
        if any(sent.strip().startswith(str(i) + '.') for i in range(1, 100)):
            return True
        
        return any(pattern in sent for pattern in exclude_patterns)

    def read_pdf(self, file_path: str) -> str:
        """Read PDF content and clean whitespace"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                # Clean whitespace for each page
                page_text = self.clean_whitespace(page.get_text())
                text += page_text + " "
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def split_into_sentences(self, documents: str) -> Tuple[List[str], List[str]]:
        """Split text into sentences and filter out unwanted ones"""
        sentence_lists = []
        # Clean whitespace before sentence tokenization
        documents = self.clean_whitespace(documents)
        sentences = nltk.sent_tokenize(documents)
        
        for sent in sentences:
            # Keep sentences that have meaningful content
            if not self.should_exclude_sentence(sent):
                # Keep punctuation and parentheses as they might be part of quoted text
                tokenizer_sent = tokenize(sent)
                sentence_lists.append(tokenizer_sent)
        
        return sentence_lists, sentences

    def process_document(self, file_path: str) -> Tuple[List[str], np.ndarray]:
        """Process a single document: read, tokenize, and generate embeddings"""
        start_time = time.time()
        
        # Read and preprocess text
        content = self.read_pdf(file_path)
        if not content:
            return [], np.array([])

        # Split into sentences
        sentences, _ = self.split_into_sentences(content)
        if len(sentences) == 0:
            return [], np.array([])

        # Generate embeddings with larger batch size
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            device="cuda:0",
            show_progress_bar=False
        )
        
        process_time = time.time() - start_time
        print(f"Processing time for {os.path.basename(file_path)}: {process_time:.2f}s")
        print("Number of sentences: ", len(sentences))
        return sentences, np.array(embeddings)

    def insert_data(self, filename: str, file_path: str, sentences_list: List[str], 
                   embeddings: np.ndarray = None) -> Tuple[int, List[int]]:
        """Insert data into both PostgreSQL and Milvus databases with batch processing"""
        start_time = time.time()
        session = self.get_session()
        
        try:
            # Insert into PDFFile table
            pdf_file = PDFFile(
                filename=filename,
                file_path=file_path,
                total_sentences=len(sentences_list)
            )
            session.add(pdf_file)
            session.flush()

            # Insert sentences in smaller batches
            sentence_objects = []
            batch_size = 1000  # Process sentences in smaller batches
            for i in range(0, len(sentences_list), batch_size):
                batch_end = min(i + batch_size, len(sentences_list))
                batch_sentences = sentences_list[i:batch_end]
                
                for idx, sentence_text in enumerate(batch_sentences, start=i):
                    sentence = Sentence(
                        pdf_id=pdf_file.pdf_id,
                        sentence=sentence_text,
                        sentence_index=idx
                    )
                    session.add(sentence)
                    sentence_objects.append(sentence)
                
                session.flush()  # Flush each batch
            
            # Get sentence IDs
            sentence_ids = np.array([s.id for s in sentence_objects], dtype=np.int64)
            
            # Commit PostgreSQL changes
            session.commit()
            postgres_time = time.time() - start_time
            print(f"PostgreSQL insertion time for {filename}: {postgres_time:.2f}s")
            
            # If embeddings are provided, insert into Milvus
            if embeddings is not None:
                milvus_start = time.time()
                pdf_ids = np.array([pdf_file.pdf_id] * len(sentences_list), dtype=np.int64)
                sentence_indices = np.array(range(len(sentences_list)), dtype=np.int64)
                
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                
                try:
                    print("Embedding vector shape: ", embeddings.shape)
                    print(embeddings[0])
                    # Insert vectors into Milvus in smaller batches
                    chunk_size = 2000  # Reduced chunk size for better stability
                    for i in range(0, len(sentence_ids), chunk_size):
                        chunk_end = min(i + chunk_size, len(sentence_ids))
                        insert_vectors(
                            sentence_ids[i:chunk_end].tolist(),
                            pdf_ids[i:chunk_end].tolist(),
                            sentence_indices[i:chunk_end].tolist(),
                            embeddings[i:chunk_end]
                        )
                    
                    milvus_time = time.time() - milvus_start
                    print(f"Milvus insertion time for {filename}: {milvus_time:.2f}s")
                except Exception as e:
                    print(f"Error inserting into Milvus: {str(e)}")
                    raise

            total_time = time.time() - start_time
            print(f"Total insertion time for {filename}: {total_time:.2f}s")
            return pdf_file.pdf_id, sentence_ids.tolist()

        except Exception as e:
            session.rollback()
            print(f"Error inserting data for {filename}: {str(e)}")
            raise
        finally:
            session.close()

    def get_sentence_by_id(self, sentence_id: int) -> Sentence:
        """Retrieve a sentence by its ID"""
        session = self.get_session()
        try:
            sentence = session.query(Sentence).filter(Sentence.id == sentence_id).first()
            return sentence
        finally:
            session.close()

    def get_pdf_sentences(self, pdf_id: int) -> List[Sentence]:
        """Retrieve all sentences for a given PDF ID"""
        session = self.get_session()
        try:
            sentences = session.query(Sentence).filter(
                Sentence.pdf_id == pdf_id
            ).order_by(Sentence.sentence_index).all()
            return sentences
        finally:
            session.close()

    def get_pdf_by_filename(self, filename: str) -> PDFFile:
        """Retrieve PDF file information by filename"""
        session = self.get_session()
        try:
            pdf_file = session.query(PDFFile).filter(
                PDFFile.filename == filename
            ).first()
            return pdf_file
        finally:
            session.close()

    def save_to_databases(self, file_path: str, sentences: List[str], 
                         embeddings: np.ndarray, db: Session) -> bool:
        """Save processed document data to both PostgreSQL and Milvus"""
        try:
            # First check if file already exists in database
            existing_file = db.query(PDFFile).filter(
                PDFFile.filename == os.path.basename(file_path)
            ).first()
            
            if existing_file:
                print(f"File {os.path.basename(file_path)} already exists in database, skipping...")
                return False

            try:
                # Use insert_data method to handle both PostgreSQL and Milvus insertions
                pdf_id, sentence_ids = self.insert_data(
                    filename=os.path.basename(file_path),
                    file_path=file_path,
                    sentences_list=sentences,
                    embeddings=embeddings
                )
                print(f"Successfully processed file with PDF ID: {pdf_id} and {len(sentence_ids)} sentences")
                return True
                
            except Exception as e:
                print(f"Error during database operations: {str(e)}")
                print(traceback.format_exc())
                return False
            
        except Exception as e:
            print(f"Error saving document {file_path}:")
            print(traceback.format_exc())
            return False

    def process_pdf_batch(self, pdf_files: List[str]) -> None:
        """Process a batch of PDF files in parallel with better connection handling"""
        try:
            # Reduce max_workers to prevent database connection exhaustion
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for pdf_file in pdf_files:
                    future = executor.submit(self.process_document, pdf_file)
                    futures.append((pdf_file, future))
                
                for pdf_file, future in futures:
                    try:
                        sentences, embeddings = future.result()
                        if len(sentences) > 0:
                            # Create a new session for each file processing
                            with self.get_session() as session:
                                self.save_to_databases(pdf_file, sentences, embeddings, session)
                    except Exception as e:
                        print(f"Error processing {pdf_file}: {str(e)}")
                        print(traceback.format_exc())
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
            return

    def create_corpus(self, folder_path: str):
        """Process all PDF files in the specified folder with parallel processing"""
        # Get list of PDF files
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return

        print(f"Found {len(pdf_files)} PDF files")
        
        # Test database connection
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("Successfully connected to PostgreSQL database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return

        try:
            # Process files in smaller batches to prevent connection pool exhaustion
            batch_size = 4  # Reduced batch size
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

if __name__ == "__main__":
    # Initialize corpus creator
    creator = CorpusCreator()
    
    # Process IT folder
    it_folder = "/hdd1/similarity/CheckSimilarity/audio-test"
    start = time.time()
    creator.create_corpus(it_folder)
    
    print("Corpus creation completed!")
    print("Stored time: ", time.time()-start)

    from pymilvus import connections, Collection, utility
    # Connect to Milvus
    connections.connect(host="localhost", port="19530", db_name="vector_database")
    # Load the collection
    collection_name = "sentence_similarity"
    collection = Collection(name=collection_name)

    # # Check collection size
    num_vectors = collection.num_entities
    print(f"Number of vectors in collection: {num_vectors}")