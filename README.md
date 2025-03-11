# 🎓 DaiNam University Plagiarism Detection System

<div align="center">

<p align="center">
  <img src="docs/images/logo.png" alt="DaiNam University Logo" width="200"/>
  <img src="docs/images/AIoTLab_logo.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://fit.dainam.edu.vn)
[![Faculty of IT](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-green?style=for-the-badge)](https://fit.dainam.edu.vn)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)


</div>

<h3 align="center">🔬 Advanced Academic Integrity Through AI Innovation</h3>

<p align="center">
  <strong>A Next-Generation Plagiarism Detection System Powered by Deep Learning and Vector Search Technology</strong>
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-documentation">Docs</a>
</p>

## 🏗️ Architecture

<p align="center">
  <img src="docs/images/architecture.JPG" alt="System Architecture" width="800"/>
</p>

The system employs a three-tier architecture:

1. **📄 Document Processing Layer**: Extracts text from PDFs, segments into sentences, and generates embeddings
2. **💾 Storage Layer**: Stores document metadata in PostgreSQL and vector embeddings in Milvus
3. **🔎 Search Layer**: Performs high-performance similarity searches and generates detailed reports

## ✨ Key Features

### 🧠 Advanced AI Technology
- **Semantic Analysis Engine**: Powered by state-of-the-art transformer models
- **Multi-lingual Support**: Optimized for Vietnamese and English content
- **Context-Aware Detection**: Understanding beyond simple text matching

### ⚡ High-Performance Architecture
- **Vector Search Technology**: Using Milvus for lightning-fast similarity search
- **Parallel Processing**: Efficient handling of large document collections
- **Scalable Infrastructure**: Designed for institutional deployment

### 📊 Comprehensive Analysis
- **Visual Results**: Interactive visualization of matched content
- **Detailed Reports**: Page-by-page similarity analysis
- **Evidence Mapping**: Precise location of potential matches

## 🔧 Tech Stack

<div align="center">

### Core Technologies
[![Docker](https://img.shields.io/badge/Docker-9ae5ff?style=for-the-badge&logo=docker&logoColor=blue)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-19354c?style=for-the-badge&logo=HuggingFace&logoColor=ffbf00)](https://huggingface.co/sentence-transformers)
### Database Systems
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Milvus](https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=milvus&logoColor=white)](https://milvus.io/)

</div>

## 📥 Installation

### 🛠️ Prerequisites

- 🐍 **Python** `3.8+` - Core programming language
- 🐘 **PostgreSQL** `12+` - Relational database for metadata
- 🔍 **Milvus** `2.x` - Vector database for similarity search
- 🐳 **Docker & Docker Compose** - Container management
- 💾 **RAM** `8GB+` - Recommended for optimal performance
- 💻 **CPU** `4+ cores` - For parallel processing
- 🖴 **Storage** `10GB+` - For document storage and embeddings

### 🗃️ Database Setup

1. 🐘 **PostgreSQL Setup**
   ```bash
   # Start PostgreSQL service
   docker run -d \
     --name postgres \
     -e POSTGRES_USER=similarity \
     -e POSTGRES_PASSWORD=123456 \
     -e POSTGRES_DB=Sentence_Similarity \
     -p 5434:5432 \
     postgres:12
   ```

2. 🔍 **Milvus Setup**
   ```bash
   # Download Milvus docker-compose file
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml

   # Start Milvus
   docker-compose up -d
   ```

### ⚙️ Project Setup

1. 📦 **Clone Repository**
   ```bash
   git clone https://github.com/drkhanusa/DNU_PlagiarismChecker.git
   cd DNU_PlagiarismChecker
   ```

2. 🌟 **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 📚 **Install Dependencies**
   ```bash
   pip install -e .
   ```

4. ⚡ **Environment Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env with your settings
   # Example configuration:
   DATABASE_URL=postgresql://similarity:123456@localhost:5434/Sentence_Similarity
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   ```

5. 🔄 **Initialize Database**
   ```bash
   # Create database tables
   python setup_database.py

   # Initialize Milvus collection
   python create_milvus_db.py
   ```

## 🚀 Getting Started

### ⚡ Quick Start
```python
from plagiarism_checker import check_plagiarism_details

# Check a document
results = check_plagiarism_details(
    file_path="path/to/document.pdf",
    min_similarity=0.9
)

# View results
print(f"Overall Similarity: {results['data']['total_percent']}%")
for doc in results['data']['similarity_documents']:
    print(f"Match: {doc['name']} - {doc['similarity_value']}%")
```

### 📥 Adding Documents to Database
```python
from create_corpus import CorpusCreator

creator = CorpusCreator()
creator.process_document("path/to/document.pdf")
```

## 📚 Documentation

For detailed documentation, please visit our [Wiki](https://github.com/drkhanusa/DNU_PlagiarismChecker/wiki) or refer to the following sections:
- 📖 [Installation Guide](docs/installation.md)
- 👥 [User Manual](docs/user-manual.md)
- 🔧 [API Reference](docs/api-reference.md)
- 🤝 [Contributing Guidelines](docs/contributing.md)

## 📝 License

© 2024 AIoTLab, Faculty of Information Technology, DaiNam University. All rights reserved.

---

<div align="center">

### Made with 💻 by AIoTLab at DaiNam University

[Website](https://fit.dainam.edu.vn) • [GitHub](https://github.com/drkhanusa) • [Contact Us](mailto:contact@dainam.edu.vn)

</div>
