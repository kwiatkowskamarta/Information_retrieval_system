# Biomedical Information Retrieval System: Polyphenol Literature Classifier

**Authors:** Marta Kwiatkowska, Weronika Pędzimąż  
**Institution:** Universidad Politécnica de Madrid  
**Date:** October 2025

## Project Overview
This project presents the design, implementation, and evaluation of a biomedical information retrieval system aimed at identifying scientific publications related to **polyphenols**. Given the growing volume of literature on polyphenols (compounds with antioxidant and anti-inflammatory properties), manual retrieval is inefficient.

We developed an automated system to retrieve, classify, and rank abstracts using:
1. **NCBI E-utilities API** for data acquisition.
2. A balanced dataset of **2,616 abstracts** (1,308 relevant, 1,308 non-relevant).
3. A comparison of lexical (TF-IDF) and semantic (BioBERT) approaches.

## Key Features
- **Automated Data Collection:** Scripts to fetch relevant and non-relevant abstracts directly from PubMed.
- **Reproducible Pipeline:** Fixed random seeds and stratified sampling.
- **Model Comparison:** Implementation of four distinct approaches:
    - TF-IDF + Logistic Regression
    - Frozen BioBERT embeddings + Logistic Regression
    - Frozen BioBERT embeddings + Cosine Similarity
    - Fine-tuned BioBERT for sequence classification

## Project Structure
```text
- abstract_import.py       # Fetches relevant abstracts using NCBI API
- non-relevant.py          # Generates non-relevant dataset from diverse topics
- approach_comparison.py   # Main script: trains models and evaluates performance
- pubmed_abstracts.csv     # Collected dataset (relevant)
- non_relevant.csv         # Collected dataset (non-relevant)
- publications.xlsx        # Input list of relevant titles
- requirements.txt         # Python dependencies
- README.md                # Documentation
