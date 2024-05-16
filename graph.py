import os

os.environ['PINECONE_API_KEY'] = 'Type your key here'
os.environ['OPENAI_API_KEY'] = 'Type your key here'

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
from langchain_pinecone import PineconeVectorStore
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#data lodaers
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import streamlit as st

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("nlpproject")
# index.describe_index_stats()
vectorstore = PineconeVectorStore(index,embeddings)
llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo',temperature=0.0)
llm_m = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo',temperature=0.0)

qa = RetrievalQA.from_chain_type(llm=llm_m,chain_type="stuff",retriever=vectorstore.as_retriever(k=10))


import streamlit as st
def chatbot_response(query):

    wo = llm.invoke(query)
    response_without_index =  wo.content
    wi = qa.invoke(query)
    response_with_index = wi["result"]
    

    return response_with_index, response_without_index

def main():
    st.title("Chatbot with RAGs")
    st.write("This chatbot provides responses with and without indices to the source information.")

    # User input
    query = st.text_area("Type your query here...", height=150)

    # Button to generate response
    if st.button("Submit"):
        response_with_index, response_without_index = chatbot_response(query)
        
        # Displaying responses
        st.write("### Response with Index")
        st.write(response_with_index)
        
        st.write("### Response without Index")
        st.write(response_without_index)

if __name__ == "__main__":
    main()