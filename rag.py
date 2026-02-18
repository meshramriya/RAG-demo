import streamlit as st
import os

#Load environment variables from .env file
#like apl keys and other sensitive information
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#step1: page configuration
st.set_page_config(page_title="c++ RAG ChatBot",page_icon="💭")
st.title("💭 C++ RAG ChatBot")
st.write("Ask question about C++ programming")

#step2: load environment variables
load_dotenv()

#step3: use cache(important for performance) to load and process documents
@st.cache_resource
#streamlit decorator- means run this functions only once, not every refresh
#this is very important for embeddings + FAISS speed


#MAIN RAG PIPELINE
def load_vectorstore():
    
    #load documents
    loader = TextLoader("C++_Introduction.txt",encoding="utf-8")
    documents = loader.load()

    #split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    final_docs = text_splitter.split_documents(documents)

    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #create FAISS vectorstore
    db = FAISS.from_documents(final_docs,embeddings)
    return db

#step4: load the vector db(only once) 

db = load_vectorstore()

#step5: user input
query = st.text_input("Ask a question about C++ programming:")

if query:
    #condition- find top 3 most similar chunks from FAISS
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrieved context: 📒")
    for i, doc in enumerate(docs):
        st.markdown(f"**result {i+1} :**")
        st.write(doc.page_content)

