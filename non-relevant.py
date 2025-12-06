import requests
import pandas as pd
import random
import re

# Read the file from task 1
relevant_df = pd.read_csv("pubmed_abstracts.csv")
n_needed = len(relevant_df)

# Generic topics for non-relevant articles
non_relevant_terms = [
    "machine learning",
    "cancer therapy",
    "cardiology",
    "neuroscience"
]

results_non = []
print(f"Retrieving {n_needed} non-relevant abstracts from PubMed...")

while len(results_non) < n_needed:
    term = random.choice(non_relevant_terms)

    # ESearch
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 10,
        "retstart": random.randint(0, 10000)
    }
    r = requests.get(esearch_url, params=esearch_params)
    data = r.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])

    # EFetch
    for pmid in pmids:
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "text",
            "rettype": "medline"
        }
        response = requests.get(efetch_url, params=efetch_params)
        text = response.text

        # Extract title and abstract using regex
        title_match = re.search(r"TI  - (.+?)(?:\n[A-Z]{2}  -|\n\n)", text, re.DOTALL)
        abstract_match = re.search(r"AB  - (.+?)(?:\n[A-Z]{2}  -|\n\n)", text, re.DOTALL)

        title = title_match.group(1).replace("\n", " ").strip() if title_match else ""
        abstract = abstract_match.group(1).replace("\n", " ").strip() if abstract_match else ""

        if title and abstract:
            results_non.append({
                "PMID": pmid,
                "QueryTerm": term,
                "Title": title,
                "Abstract": abstract
            })

        if len(results_non) >= n_needed:
            break

# Save all the results
pd.DataFrame(results_non).to_csv("non_relevant.csv", index=False, encoding="utf-8")