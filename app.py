import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re

load_dotenv()

# 1. Vectorize the abstracts data
loader = CSVLoader("/Users/pjihae/Documents/GitHub/output.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)  # Returns 3 top results similar to the query
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4")


#hyperlink not currently matching title of the recommended paper
#need to work on this



template = """
you are a world class biomedical researcher.
you will receive an abstract of literature paper from user.
your job is to give the most relevant literature paper's title  as a recommendation to the query, along with 2~3 sentences of highlights. 
you will follow all of the rules below : 


1/ give the exact title of the paper with author
2/ do not recommend the same paper as the user query
3/ if the user query is not found in the list, search on google scholar and find one
4/ you should not make up the title 
5/ you should only recommend paper that exists and can be searched on google scholar

below is a query for a paper you will get:
{message}

here is the list of most relevant papers in response:
{best_practice}

please suggest the best paper matched to the query with its title,  and 2~3 sentences highlighting insights that match with the query, matching keywords:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Function to limit message length to 4000 tokens
def limit_message_length(message, max_tokens=4000):
    # Split message into tokens
    tokens = re.findall(r"\b\w+\b", message)
    # Limit tokens to max_tokens
    truncated_message = " ".join(tokens[:max_tokens])
    return truncated_message

# Truncate message to ensure total length <= 4000 tokens
def truncate_documents(document, max_tokens=4000):
    truncated_documents = []
    total_tokens = 0
    for doc in document:
        tokens = re.findall(r"\b\w+\b", doc)
        if total_tokens + len(tokens) <= max_tokens:
            truncated_documents.append(doc)
            total_tokens += len(tokens)
        else:
            break
    return truncated_documents

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    # Truncate message and relevant documents
    truncated_message = limit_message_length(message)
    truncated_documents = truncate_documents(best_practice)
    response = chain.run(message=truncated_message, best_practice=truncated_documents)
    return response

message = "Gene selection is important for cancer classification based on gene expression data, because of high dimensionality and small sample size. In this paper, we present a new gene selection method based on clustering, in which dissimilarity measures are obtained through kernel functions. It searches for best weights of genes iteratively at the same time to optimize the clustering objective function. Adaptive distance is used in the process, which is suitable to learn the weights of genes during the clustering process, improving the performance of the algorithm. The proposed algorithm is simple and does not require any modification or parameter optimization for each dataset. We tested it on eight publicly available datasets, using two classifiers (support vector machine, k-nearest neighbor), compared with other six competitive feature selectors. The results show that the proposed algorithm is capable of achieving better accuracies and may be an efficient tool for finding possible biomarkers from gene expression data."


truncated_message = limit_message_length(message)
result = generate_response(truncated_message)
print(result)

'''
# 5. Deploy with Streamlit
def main():
    st.set_page_config(
        page_title="ResearchGPT", page_icon="🎓"
    )

    st.header("ResearchGPT 🎓")

    message = st.text_area("Enter queries")

    if message:
        st.write("Finding a perfect paper for you...")

        # Limit token count of message to 4000 tokens
        truncated_message = limit_message_length(message)
        result = generate_response(truncated_message)
        st.info(result)


if __name__ == '__main__':
    main()
'''

# max_token=8192 is for both input+output(response)