from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple
import numpy as np
import fitz
import time
from tqdm import tqdm
import os
import re
import nltk
from pyvi.ViTokenizer import tokenize

from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence, engine
from create_milvus_db import collection
from create_corpus import CorpusCreator
from sklearn.metrics.pairwise import cosine_similarity

checker = CorpusCreator("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")

def check_plagiarism(file_path: str, min_similarity: float = 0.9) -> Dict:
    """
    Check document for plagiarism against Milvus database
    Args:
        file_path: Path to the PDF file to check
        min_similarity: Minimum similarity threshold (default 0.9)
    Returns: Dict with plagiarism information:
    {
        "total_similarity_percent": float,
        "top_similarity_documents": List[str],
        "top_similarity_values": List[float],
        "similarity_sentences": Dict[str, List[Dict]]
    }
    """
    start_time = time.time()
    print(f"Processing document: {os.path.basename(file_path)}")

    # Process document to get sentences and embeddings
    sentences, embeddings = checker.process_document(file_path)
    
    # # Debug: Check if vectors are identical when processing same file twice
    # _, embeddings2 = checker.process_document(file_path)
    # cos = cosine_similarity(embeddings2, embeddings)
    # print(cos.shape)
    # # Count values > 0.9 in each column
    # # Get max value in each column
    # max_values = np.max(cos, axis=0)

    # # Create mask of same shape as cos, initialized to zeros
    # mask = np.zeros_like(cos)

    # # For each column, set True only for the max value position
    # for col in range(cos.shape[1]):
    #     max_idx = np.argmax(cos[:, col])
    #     mask[max_idx, col] = True

    # # Apply mask to keep only max values, rest become 0
    # filtered_cos = np.where(mask, cos, 0)

    # # Count values > 0.9 in each column
    # high_similarity_counts = np.sum(filtered_cos > 0.9, axis=0)

    # # print("\nNumber of high similarity matches (>0.9) per sentence in document 2:")
    # # for i, count in enumerate(high_similarity_counts):
    # #     if count > 0:
    # #         print(f"Sentence {i+1}: {count} matches")

    # total_matches = np.sum(high_similarity_counts)
    # print(f"\nTotal number of high similarity matches: {total_matches}")
    # print(f"Average matches per sentence: {total_matches/len(high_similarity_counts):.2f}")
    
    if not sentences:
        return {"error": "No valid sentences found in document"}
    print(f"Number of sentences processed: {len(sentences)}")

    # Initialize results
    similarity_sentences_dict = {}
    document_matches = {}  # Track matches per document

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10,  # Increase from 10 to search more segments
            # "ef": 100      # Add ef parameter to improve recall
        }
    }
    batches = 64
    # Process in batches with progress bar
    with tqdm(total=len(sentences), desc="Checking for plagiarism") as pbar:
        for i in range(0, len(sentences), batches):
            batch_end = min(i + batches, len(sentences))
            batch_vectors = embeddings[i:batch_end].tolist()
            
            # Search Milvus database
            search_results = collection.search(
                data=batch_vectors,
                anns_field="sentence_vector",
                param=search_params,
                limit=1,
                output_fields=["pdf_id", "sentence_id"]
            )

            # Process results
            with SessionLocal() as db:
                for idx, hits in enumerate(search_results):
                    sentence_idx = i + idx
                    current_sentence = sentences[sentence_idx]
                    
                    # # Debug print for each search result
                    # print(f"\nSearching sentence {sentence_idx + 1}/{len(sentences)}")
                    # print(f"Original sentence: {current_sentence[:100]}...")
                    # print(f"Search hits: {hits}")
                    
                    for hit in hits:
                        similarity_score = float(hit.distance)
                        # print(f"Similarity score: {similarity_score}")
                        
                        if similarity_score >= min_similarity:
                            pdf_id = hit.entity.get('pdf_id')
                            sentence_id = hit.entity.get('sentence_id')
                            # print(f"Match found - PDF ID: {pdf_id}, Sentence ID: {sentence_id}")
                            
                            pdf_file = db.query(PDFFile).filter(PDFFile.pdf_id == pdf_id).first()
                            sentence = db.query(Sentence).filter(Sentence.id == sentence_id).first()
                            
                            if pdf_file and sentence:
                                # Track document-level matches
                                if pdf_file.filename not in document_matches:
                                    document_matches[pdf_file.filename] = []
                                document_matches[pdf_file.filename].append(similarity_score)
                                
                                # Add to sentence-level matches
                                if current_sentence not in similarity_sentences_dict:
                                    similarity_sentences_dict[current_sentence] = []
                                
                                similarity_sentences_dict[current_sentence].append({
                                    'document': pdf_file.filename,
                                    'similarity_score': similarity_score,
                                    'matched_text': sentence.sentence
                                })
                        else:
                            print(f"Score {similarity_score} below threshold {min_similarity}")
            
            pbar.update(batch_end - i)

    # Calculate document-level statistics
    doc_similarity_percentages = {}
    for doc, scores in document_matches.items():
        # Calculate percentage of matching sentences for each document
        matching_sentences = len(scores)
        percentage = (matching_sentences / len(sentences)) * 100
        doc_similarity_percentages[doc] = {
            'percentage': percentage,
            'matching_sentences': matching_sentences,
            'total_sentences': len(sentences)
        }

    # Get top 5 similar documents based on percentage of matching sentences
    top_docs = sorted(doc_similarity_percentages.items(), 
                     key=lambda x: x[1]['percentage'], 
                     reverse=True)[:5]
    
    top_similarity_documents = [doc for doc, _ in top_docs]
    top_similarity_values = [stats['percentage'] for _, stats in top_docs]

    # Calculate total similarity percentage (number of matched sentences / total sentences)
    total_similarity_percent = (len(similarity_sentences_dict) / len(sentences)) * 100

    # Print summary statistics
    print(f"\nPlagiarism Check Summary:")
    print(f"Total sentences in document: {len(sentences)}")
    print(f"Sentences with matches above {min_similarity}: {len(similarity_sentences_dict)}")
    print(f"Total similarity percentage: {total_similarity_percent:.2f}%")
    print(f"\nTop {len(top_docs)} similar documents (by percentage of matching sentences):")
    for doc, stats in top_docs:
        print(f"- {doc}: {stats['percentage']:.2f}% "
              f"({stats['matching_sentences']} matching sentences out of {stats['total_sentences']} total)")

    # If less than 5 documents were found, print a note
    if len(top_docs) < 5:
        print(f"\nNote: Only {len(top_docs)} documents had matches above the similarity threshold ({min_similarity})")
    
    return {
        "total_similarity_percent": total_similarity_percent,
        "top_similarity_documents": top_similarity_documents,
        "top_similarity_values": top_similarity_values,  # These are now percentages
        "similarity_sentences": similarity_sentences_dict,
        "processing_time": time.time() - start_time
    }

def main():
    """Main function to run plagiarism checks"""
    # try:
    # Example usage
    file_to_check = "/hdd1/similarity/CheckSimilarity/database/IT/PhamQuangThanh__DATN.pdf"
    results = check_plagiarism(file_to_check)
    
    # Print summary
    print("\nPlagiarism Check Summary:")
    print(f"Total Similarity Percentage: {results['total_similarity_percent']:.2f}%")
    
    print("\nTop Similar Documents:")
    for doc, score in zip(results['top_similarity_documents'], results['top_similarity_values']):
        print(f"- {doc}: {score:.2f}")
    
    # print("\nDetailed Sentence Matches:")
    # for sentence, matches in results['similarity_sentences'].items():
    #     print(f"\nOriginal: {sentence}")
    #     for match in matches:
    #         print(f"- Found in: {match['document']}")
    #         print(f"  Similarity: {match['similarity_score']:.2f}")
    #         print(f"  Text: {match['matched_text']}")
    
    print(f"\nProcessing Time: {results['processing_time']:.2f} seconds")

    # except:
    #     print("Error")

if __name__ == "__main__":
    main() 