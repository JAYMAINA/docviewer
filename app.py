import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import faiss
import numpy as np
import tiktoken
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use environment variable for API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load and extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Split text into chunks (by tokens)
def split_text(text, max_tokens=800):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = enc.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

# Embed text using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

# Create FAISS index
def create_faiss_index(chunks):
    dimension = len(get_embedding(chunks[0]))
    index = faiss.IndexFlatL2(dimension)
    vectors = [get_embedding(chunk) for chunk in chunks]
    index.add(np.array(vectors))
    return index, vectors

# RAG-based answer using GPT-4o-mini
def answer_question(question, chunks, index, vectors):
    q_embedding = get_embedding(question)
    D, I = index.search(np.array([q_embedding]), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n---\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Summarize entire document
def summarize_document(chunks):
    joined = "\n".join(chunks[:10])  # use first N chunks
    prompt = f"Summarize the following document:\n{joined}\n\nSummary:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with GPT-4o-mini", layout="wide")
st.title("📄 Chat with your PDF (GPT-4o-mini + RAG)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_text(text)

    with st.spinner("Generating embeddings and building index..."):
        index, vectors = create_faiss_index(chunks)

    st.success("PDF processed! You can now ask questions.")

    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Getting answer from GPT-4o-mini..."):
            answer = answer_question(question, chunks, index, vectors)
            st.markdown(f"**Answer:** {answer}")

    if st.button("Summarize Document"):
        with st.spinner("Summarizing with GPT-4o-mini..."):
            summary = summarize_document(chunks)
            st.markdown("### 📌 Summary")
            st.write(summary)
