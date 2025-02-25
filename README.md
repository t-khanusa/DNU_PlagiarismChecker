# DaiNam University Plagiarism Detection System 🚀

<p align="center">
  <img src="docs/images/logo.png" alt="DaiNam University Logo" width="200"/>
</p>

<p align="center">
  <strong>An advanced plagiarism detection system for the Faculty of Information Technology</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#license">License</a>
</p>

## 📋 Overview

This system was developed for the Faculty of Information Technology at DaiNam University to detect plagiarism in students' graduation projects. It uses state-of-the-art vector similarity search technology to identify potential instances of plagiarism by comparing submitted documents against a comprehensive database of previous works.

## ✨ Features

- **🔍 High-Precision Detection**: Utilizes sentence-level semantic embeddings to detect plagiarism even when text has been paraphrased
- **🇻🇳 Vietnamese Language Support**: Optimized for Vietnamese text processing with specialized tokenization
- **📈 Scalable Architecture**: Built on Milvus vector database for efficient similarity search across thousands of documents
- **📊 Detailed Reporting**: Provides comprehensive reports showing similarity percentages and exact matching passages
- **⚡ Batch Processing**: Efficiently processes large documents through optimized batch operations
- **🎛️ Configurable Thresholds**: Adjustable similarity thresholds to control detection sensitivity

## 🏗️ Architecture

The system employs a three-tier architecture:

1. **📄 Document Processing Layer**: Extracts text from PDFs, segments into sentences, and generates embeddings
2. **💾 Storage Layer**: Stores document metadata in PostgreSQL and vector embeddings in Milvus
3. **🔎 Search Layer**: Performs high-performance similarity searches and generates detailed reports

<p align="center">
  <img src="docs/images/architecture.jpg" alt="System Architecture" width="600"/>
</p>

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Milvus 2.x
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/drkhanusa/PlagiarismChecker.git
cd PlagiarismChecker
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure database connections in `config.yaml`

5. Initialize the database:
```bash
python setup_database.py
```

## 🔧 Usage

### Adding Documents to the Database

```bash
python add_document.py --path /path/to/documents/folder
```

### Checking a Document for Plagiarism

```python
from plagiarism_checker import check_plagiarism

results = check_plagiarism("/path/to/document.pdf", min_similarity=0.9)
print(f"Similarity: {results['total_similarity_percent']}%")

# Print top similar documents
for doc, score in zip(results['top_similarity_documents'], results['top_similarity_values']):
    print(f"- {doc}: {score:.2f}%")
```

### Running the Web Interface

```bash
python app.py
```

Then access the web interface at http://localhost:5000

## ⚙️ Configuration

The system can be configured through the `config.yaml` file:

```yaml
database:
  postgres:
    host: localhost
    port: 5432
    user: postgres
    password: password
    database: plagiarism_db
  
  milvus:
    host: localhost
    port: 19530
    collection: sentence_vectors

processing:
  batch_size: 64
  min_similarity: 0.9
  embedding_model: "./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338"
```

## 📝 License

© 2023 Faculty of Information Technology, DaiNam University. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.
