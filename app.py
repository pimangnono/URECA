import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. vectorise the abstracts data
loader = TextLoader("/Users/pjihae/Documents/GitHub/URECA/abstracts.txt", encoding = 'UTF-8')
documents = loader.load_and_split()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3) # returns 3 top results similar to query

    page_contents_array = [doc.page_content for doc in similar_response]

    print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts

llm = ChatOpenAI(temperature=0, model="gpt-4")

template = """
you are a world class biomedical researcher.
you will receive a query from user.
your job is to give the most relevant literature paper's title and doi link as a recommendation to the query, along with 2~3 sentences of highlights. 
you will also evaluate the percentage relevance of two articles.
you will follow all of the rules below : 

1/ you will ask for the range of release date of the paper that the user wants, and recommend paper accordingly

2/ you will use cosine similarities to find relevance in two papers

below is a query for a paper you will get:
{message}

here is the list of most relevant papers in response:
{best_practice}

please suggest the best paper matched to the query with its title, doi link and 2~3 sentences highlighting insights that match with the query:
"""

prompt = PromptTemplate(
    input_variables= ["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 5. Deploy with streamlit
def main():
    st.set_page_config(
        page_title= "ResearchGPT", page_icon="🎓"
    )
    
    st.header("ResearchGPT 🎓")

    message = st.text_area("Enter queries")

    if message:
        st.write("Finding a perfect paper for you...")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()

    

