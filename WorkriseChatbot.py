import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os

# Load OpenAI API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]["api_key"])

# Load the vector database
with open('D:/Downloads/VectorDatabase.pkl', 'rb') as f:
    vector_database = pickle.load(f)

index = vector_database['index']
texts = vector_database['texts']
filenames = vector_database['filenames']
filepaths = vector_database['filepaths']
# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the top N most similar vectors
def find_most_similar_vectors(question, index, top_n=10):
    question_vector = model.encode([question])
    D, I = index.search(question_vector, top_n)
    return I[0]

# Function to classify the question and detect the number of candidates
def classify_question(question, chat_history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Determine if the following question is asking for new candidates in a job search or rather asking for elaboration on previous candidates or a different question type altogether. Additionally, try to detect the number of candidates the employer is looking for based on the question. You may use chat history when determining these things. Return 'yes' if the employer is asking for new candidates, and include the number of candidates if specified (e.g., 'yes, 5'). If the user does not specify the number of candidates, just include your yes or no answer. Be lenient for the yes/no answer (Assume yes unless you are very sure no) because the target user for this application is an employer looking for candidates."},
    ]
    for chat in chat_history:
        if "question" in chat and "answer" in chat:
            messages.append({"role": "user", "content": chat["question"]})
            messages.append({"role": "assistant", "content": chat["answer"]})
    messages.append({"role": "user", "content": f"Question: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    classification = response.choices[0].message.content.lower()
    
    is_asking_for_candidates = 'yes' in classification
    num_candidates = 10  # Default value
    
    # Extract number of candidates if specified
    if 'yes' in classification:
        words = classification.split()
        for i, word in enumerate(words):
            if word.isdigit():
                num_candidates = int(word)
                break
    return is_asking_for_candidates, num_candidates

# Function to get the OpenAI GPT-4 Turbo response when a search was performed
# Match the candidate name to the file path
# Unique ID for each candidate
# Test job searches from LinkedIn
def get_response_with_search(question, context, chat_history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant to an employer looking for candidates. Answer the question based only on the context below and chat history (Assume that the user is asking for candidate summaries even if you're not sure). Provide a summary of **each resume** given to you in the context only once (for example, if there are ten resumes in the context, output ten summaries, each one unique and using a different resume). Include each resume even if you think they don't adequately answer the question. When using all the resumes, however, create the summary based on what's relevant to the user question. Include both the candidate name and candidate ID in the title of your summary. **DO NOT SUMMARIZE TWO CANDIDATES WITH THE SAME ID. IF YOU ARE GIVEN TEN RESUMES, USE TEN RESUMES TO MAKE TEN SUMMARIES.** Order the summaries from most useful to least useful."},
    ]
    for chat in chat_history:
        if "question" in chat and "answer" in chat:
            messages.append({"role": "user", "content": chat["question"]})
            messages.append({"role": "assistant", "content": chat["answer"]})
    # Include the current context
    messages.append({"role": "user", "content": f"Context: {context} \nQuestion: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    
    return response.choices[0].message.content

# Function to get the OpenAI GPT-4 Turbo response when no search was performed
def get_response_without_search(question, chat_history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant to an employer looking for candidates. Answer the question based on the most recent context given to you as well as chat history. If it is the first question of the conversation and you have no context about resumes to work with, don't answer the question and inform the user that this app is designed for asking about candidates. Often the question will be something about elaborating on previous candidates or asking for the full resume of a candidate. You are allowed to answer these questions and give full resumes because these are public documents and the employers are allowed to see them."},
    ]
    for chat in chat_history:
        if "question" in chat and "answer" in chat:
            messages.append({"role": "user", "content": chat["question"]})
            messages.append({"role": "assistant", "content": chat["answer"]})
    current_context = st.session_state.get("current_context", "")
    if current_context:
        messages.append({"role": "user", "content": f"Context: {current_context} \nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": f"Question: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_chat_history" not in st.session_state:
    st.session_state.api_chat_history = []

# Initialize current context
if "current_context" not in st.session_state:
    st.session_state.current_context = []

if "buttonlist" not in st.session_state:
    st.session_state.buttonlist = []

if "searched" not in st.session_state: 
    st.session_state.searched = False

def handle_question():
    st.session_state.buttonlist = []
    question = st.session_state.question
    if question:
        # Classify the question and get the number of candidates
        is_asking_for_candidates, num_candidates = classify_question(question, st.session_state.api_chat_history)
        
        if is_asking_for_candidates:
            # Find the most relevant resumes
            top_indices = find_most_similar_vectors(question, index, num_candidates)
            # Prepare current context with resume texts and file paths
            current_context = [(texts[i], filepaths[i], i + 1000000) for i in top_indices]
            st.session_state.current_context = current_context
            # Get the response with search
            answer = get_response_with_search(question, current_context, st.session_state.api_chat_history, num_candidates)
            st.session_state.searched = True
            
            
            # Display the relevant resumes with download buttons
            st.session_state.chat_history.append({"question": question, "answer": answer})
            for word in answer.split():
                for item in current_context: 
                    if word == str(item[2]):
                        st.session_state.buttonlist.append([item[1], word])
                        st.session_state.chat_history.append({"File path": item[1], "ID": word})
                        

        else:
            # Get the response without search
            answer = get_response_without_search(question, st.session_state.api_chat_history)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.session_state.searched = False
        
        # Update API chat history
        st.session_state.api_chat_history.append({"question": question, "answer": answer})
        
        # Truncate API chat history to the last 10 messages
        max_history_length = 10
        st.session_state.api_chat_history = st.session_state.api_chat_history[-max_history_length:]
        
        # Clear the input field after processing
        st.session_state.question = ""

# Streamlit app interface
st.title("Resume Search Chatbot")
st.write("Ask a question and the chatbot will find the most relevant resumes and answer your question.")

# Display chat history to the user
for chat in st.session_state.chat_history:
    if "question" in chat: 
        st.write(f"**You:** {chat['question']}")
        if st.session_state.searched: 
            st.write(f"**Bot:** \n{chat['answer']}")
        else: 
            st.write(f"**Bot:** {chat['answer']}")
        st.write("---")
    if "File path" in chat:
        with open(chat["File path"], "rb") as file:
                st.download_button(
                    label="Download candidate resume for Candidate " + chat["ID"],
                    data=file,
                    file_name=chat["File path"],
                    mime="application/pdf"
            )
    

    
# Button to clear API chat history
if st.button("Reset Conversation History"):
    st.session_state.current_context = ""  # Clear the context for the API
    st.session_state.api_chat_history = []  # Clear the API's chat history
    st.write("API chat history cleared.")

# Input for new question at the bottom with Enter to submit
st.text_input("Enter your question", key="question", on_change=handle_question)