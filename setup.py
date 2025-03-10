from setuptools import setup, find_packages

setup(
    name="dnu-plagiarism-checker",
    version="0.1.0",
    description="A Vietnamese-optimized plagiarism detection system for DaiNam University",
    author="Khanh Nguyen",
    author_email="khanhnguyenthai.5@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pymilvus>=2.2.8",
        "sentence-transformers>=2.2.2",
        "sqlalchemy>=2.0.19",
        "psycopg2-binary>=2.9.6",
        "numpy>=1.24.3",
        "tqdm>=4.65.0",
        "PyMuPDF>=1.22.3",
        "pyvi>=0.1.1",
        "nltk>=3.8.1",
        "flask>=2.3.2",
        "flask-wtf>=1.1.1",
        "werkzeug>=2.3.6"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Text Processing :: Linguistic",
    ],
) 