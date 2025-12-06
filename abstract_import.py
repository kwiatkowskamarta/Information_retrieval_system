import requests
import pandas as pd

# load Excel file (must be in the same folder as this script)
df = pd.read_excel("publications.xlsx")
titles = df["title"].tolist()

results = []

# find PMID for each article and get the abstract
for title in titles:
    # ESearch
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": f"{title}[Title]",
        "retmode": "json"
    }
    esearch_response = requests.get(esearch_url, params=esearch_params)
    data = esearch_response.json()

    # check if PubMed returned any ID
    if "esearchresult" in data and data["esearchresult"]["idlist"]:
        pmid = data["esearchresult"]["idlist"][0]

        # EFetch
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "text",
            "rettype": "abstract"
        }
        abstract_response = requests.get(efetch_url, params=efetch_params)
        abstract_text = abstract_response.text
    else:
        pmid = None
        abstract_text = None

    # store the results
    results.append({
        "Title": title,
        "PMID": pmid,
        "Abstract": abstract_text
    })

# save all results to a CSV file
pd.DataFrame(results).to_csv("pubmed_abstracts.csv", index=False, encoding="utf-8")
