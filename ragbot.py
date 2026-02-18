import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama 

#step1: page configuration
st.set_page_config(page_title="c++ RAG ChatBot", layout = "wide")
st.title("💭 C++ RAG ChatBot using Ollama")

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

db = load_vectorstore()

#extra step - load LLM(ollama)
llm = Ollama(model="gemma2:2b")

#last step- chat interface
user_question = st.text_input("Ask the question about c++:")

if user_question :
    with st.spinner("Thinking....."):
        docs = db.similarity_search(user_question)

        #combine context (construction)
        #1. Extracts text from retrieved documents
        #2. joins them into single string
        #3. This becomes context of LLM

        context = "\n".join([doc.page_content for doc in docs])

        #prompt engineering -
        Prompt = f"""
        Answer the question using only the context below

        Context :
        {context}

        Question :
        {user_question}

        Answer :
        """

        #created structured promps
        #1. provide context
        #2. provide question
        #3. ask for Answer
        
        #this is how hallucination effect is reduced

        response = llm.invoke(Prompt)
        st.subheader("Answer :")
        st.write(response)

