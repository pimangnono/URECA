import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read contents of a file
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Define file paths
paper1_path = "/Users/pjihae/Documents/GitHub/URECA/paper1.txt"
paper1_gpt_path = "/Users/pjihae/Documents/GitHub/URECA/paper1_gpt.txt"
paper1_se_path = "/Users/pjihae/Documents/GitHub/URECA/paper1_se.txt"

# Read contents of the files
paper1 = read_file(paper1_path)
paper1_gpt = read_file(paper1_gpt_path)
paper1_se = read_file(paper1_se_path)

# Tokenize and vectorize the texts
vectorizer = CountVectorizer().fit([paper1, paper1_gpt, paper1_se])
paper1_vector = vectorizer.transform([paper1])
paper1_gpt_vector = vectorizer.transform([paper1_gpt])
paper1_se_vector = vectorizer.transform([paper1_se])

# Compute cosine similarity
cos_sim1 = cosine_similarity(paper1_vector, paper1_gpt_vector)[0][0]
cos_sim2 = cosine_similarity(paper1_vector, paper1_se_vector)[0][0]

# Write results to CSV file
with open('cosine_similarity_results.csv', 'a', newline='') as csvfile:
    fieldnames = ['ResearchGPT', 'PubMed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header if file is empty
    if csvfile.tell() == 0:
        writer.writeheader()
    
    # Write cosine similarity values
    writer.writerow({'ResearchGPT': cos_sim1, 'PubMed': cos_sim2})
