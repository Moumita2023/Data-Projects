#Retrieval-Augmented QA Logic
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def get_qa_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain

def get_answer(question):
    chain = get_qa_chain()
    return chain.run(question)
