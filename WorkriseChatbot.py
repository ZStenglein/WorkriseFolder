import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Load OpenAI API key
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]["api_key"]
)

# Load the vector database
with open('D:/Downloads/vector_database.pkl', 'rb') as f:
    vector_database = pickle.load(f)

index = vector_database['index']
texts = vector_database['texts']
filenames = vector_database['filenames']

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the top N most similar vectors
# Try summarizing and verifying approach
# async.io
# How to make async call to openai
# Parallel processing of summarization
def find_most_similar_vectors(question, index, top_n=10):
    question_vector = model.encode([question])
    D, I = index.search(question_vector, top_n)
    return I[0]


# Function to get the OpenAI GPT-4 Turbo response
def get_response(question, context):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to an employer looking for candidates. Answer the question based only on the context below."},
            {"role": "user", "content": f"Context: {context} \nQuestion: {question}"}
        ],
    )
    return response.choices[0].message.content

# Streamlit app interface
st.title("Resume Search Chatbot")
st.write("Ask a question and the chatbot will find the most relevant resumes and answer your question.")

question = st.text_input("Enter your question")

if question:
    st.write("Finding the most relevant resumes...")
    top_indices = find_most_similar_vectors(question, index)
    
    context = "\n\n".join([texts[i] for i in top_indices])
    
    st.write("Getting the response...")
    answer = get_response(question, context)
    
    st.write("Response:")
    st.write(answer)