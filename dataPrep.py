import requests
from bs4 import BeautifulSoup
import json

# Load the JSON file
with open('RELISH_v1.json', 'r') as f:
    data = json.load(f)

# Create/open a file to save the abstracts
with open('abstracts.txt', 'w', encoding='utf-8') as abstracts_file:
    # Iterate through each set of data
    for item in data:
        # Extract the "pmid" value
        pmid = item['pmid']

        # Extract the list of "relevant" values
        relevant_values = item['response']['relevant']

        # Generate URL for each "pmid"
        pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid.strip()}/"
        response = requests.get(pmid_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        abs_section = soup.find('div', class_='abstract-content selected')
        abs_text = ''
        if abs_section:
            abs_text += abs_section.get_text()

        # Write the abstract text to the file
        abstracts_file.write(abs_text + '\n')

        # Generate URLs for each "relevant" value
        for relevant_value in relevant_values:
            relevant_url = f"https://pubmed.ncbi.nlm.nih.gov/{relevant_value.strip()}/"
            response = requests.get(relevant_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            abs_section = soup.find('div', class_='abstract-content')
            abs_text = ''
            if abs_section:
                abs_text += abs_section.get_text()

            # Write the abstract text to the file
            abstracts_file.write(abs_text)
