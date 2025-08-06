# Streamlit UI code for a Retrieval-Augmented Generative AI Chatbot
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
import streamlit as st
import os
from retriever import load_and_embed
from chatbot import get_answer

st.set_page_config(page_title="RAG Chatbot")
st.title("📚 RAG-based Question Answer Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = os.path.join("data", uploaded_file.name)
    # Remove the file if it already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded successfully!")
    load_and_embed(file_path)

st.markdown("---")

question = st.text_input("Ask a question based on your PDF:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = get_answer(question)
        st.success(answer)