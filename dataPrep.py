import csv
import requests
from bs4 import BeautifulSoup
import json

# Function to fetch abstract from PubMed
def fetch_abstract(pmid):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        abstract_tag = soup.find("div", class_="abstract-content")
        if abstract_tag:
            # Join paragraphs into a single string, removing leading/trailing whitespace
            abstract_text = " ".join([p.text.strip() for p in abstract_tag.find_all("p")])
            return abstract_text
    return None

# Reading data from JSON file
with open("/Users/pjihae/Documents/GitHub/URECA/RELISH_v1.json", "r") as file:
    data = json.load(file)

# Writing to CSV
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["user input", "relevant"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for entry in data:
        user_input = fetch_abstract(entry["pmid"])
        
        if user_input:
            relevant_abstracts = []
            for pmid in entry["response"]["relevant"]:
                abstract = fetch_abstract(pmid)
                if abstract:
                    relevant_abstracts.append(abstract)
            relevant_abstracts_combined = " ".join(relevant_abstracts)
            writer.writerow({"user input": user_input, "relevant": relevant_abstracts_combined})
            