import streamlit as st
import fitz  # PyMuPDF
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from joblib import Parallel, delayed
import zipfile
import shutil

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to create vector database
def create_vector_database(pdf_folder):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = []
    filenames = []
    filepaths = []

    # List all PDF files
    pdf_files = [os.path.join(pdf_folder, filename) for filename in os.listdir(pdf_folder) if filename.endswith(".pdf")]

    # Extract text in parallel
    texts = Parallel(n_jobs=-1)(delayed(extract_text_from_pdf)(pdf) for pdf in pdf_files)
    filenames = [os.path.basename(pdf) for pdf in pdf_files]
    filepaths = pdf_files

    # Encode text
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)
    vector_dim = vectors.shape[1]

    index = faiss.IndexFlatL2(vector_dim)
    index.add(vectors)

    vector_database = {
        'index': index,
        'texts': texts,
        'filenames': filenames,
        'filepaths': filepaths
    }

    return vector_database

# Function to save vector database to a file
def save_vector_database(vector_database, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vector_database, f)

# Streamlit app interface
st.title("PDF to Vector Database")
st.write("Enter the path to the folder of PDFs to extract text and create a vector database.")

folder_path = st.text_input("Folder Path")

if folder_path:
    if os.path.isdir(folder_path):
        st.write("Creating vector database...")
        vector_database = create_vector_database(folder_path)
        save_vector_database(vector_database, "vector_database.pkl")

        st.write("Vector database created and saved.")
        st.download_button(
            label="Download Vector Database",
            data=open("vector_database.pkl", "rb").read(),
            file_name="vector_database.pkl",
            mime="application/octet-stream"
        )
    else:
        st.error("The provided path is not a valid directory.")

st.write("Ready to create your vector database!")