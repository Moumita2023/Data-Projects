#Load PDF, Chunk, Embed, Store
import os
import chromadb
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("AIzaSyDQjEWeProLWbE0dgDQScvRAvL23lFztmM")

def load_and_embed(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    from langchain.vectorstores import Chroma
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./vectorstore")
    vectorstore.persist()

    return vectorstore
