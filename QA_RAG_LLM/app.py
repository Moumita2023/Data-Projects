# Streamlit UI code for a Retrieval-Augmented Generative AI Chatbot
import streamlit as st
import os
from retriever import load_and_embed
from chatbot import get_answer

st.set_page_config(page_title="RAG Chatbot")
st.title("📚 RAG-based Chatbot with Gemini Pro")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded successfully!")
    load_and_embed(file_path)
    st.info("Embedding and vector store created!")

st.markdown("---")

question = st.text_input("Ask a question based on your PDF:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = get_answer(question)
        st.success(answer)