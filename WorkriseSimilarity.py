import streamlit as st
import openai
from openai import OpenAI
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from io import BytesIO

nltk.download('punkt')

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]["api_key"]
)

# Load pre-trained BERT model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "summaries_generated" not in st.session_state:
    st.session_state.summaries_generated = False

if "final_message_displayed" not in st.session_state:
    st.session_state.final_message_displayed = False

if "success_message" not in st.session_state:
    st.session_state.success_message = ""

if "success_message_time" not in st.session_state:
    st.session_state.success_message_time = None

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "clarification_message" not in st.session_state:
    st.session_state.clarification_message = ""

if "original_summary" not in st.session_state:
    st.session_state.original_summary = ""

if "new_summary" not in st.session_state:
    st.session_state.new_summary = ""

if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = []

if "inputs_collected" not in st.session_state:
    st.session_state.inputs_collected = False

if "summaries" not in st.session_state:
    st.session_state.summaries = []

if "candidate_names" not in st.session_state:
    st.session_state.candidate_names = []

# Extract text from the PDF files
def extract_text_from_pdf(file_path):
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Load data from spreadsheet
def load_data_from_spreadsheet(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file, header=None)
    else:
        df = pd.read_csv(file, header=None)
    return df

# Collect data from the spreadsheet
def collect_data_from_spreadsheet(file):
    df = load_data_from_spreadsheet(file)
    candidate_info = []

    for i, row in df.iterrows():
        # Check if the row is empty and stop processing if it is
        if row.isnull().all():
            break

        pdf_path = row[0]
        company_name = row[1]
        role_description = row[2]
        recipient_name = row[3]
        desired_pay_rate = row[4]
        availability = row[5]
        human_summary = row[6]
        acceptability = row[7] if len(row) > 7 else "0/0"

        # Check if PDF path is valid and file exists
        if not pdf_path or not pdf_path.strip():
            continue

        # Read PDF file
        try:
            resume_text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            continue

        candidate_info.append({
            "pdf_path": pdf_path,
            "company_name": company_name,
            "role_description": role_description,
            "recipient_name": recipient_name,
            "desired_pay_rate": desired_pay_rate,
            "availability": availability,
            "resume_text": resume_text,
            "human_summary": human_summary,
            "acceptability": acceptability,
            "summary": ""  # Initialize the summary field
        })
    
    return candidate_info

# Function to process and generate summaries
def generate_summaries():
    summaries = []
    for candidate in st.session_state.candidate_info:
        st.session_state.messages.append({
            "role": "user",
            "content": f"Company Name: {candidate['company_name']}\nRole Description: {candidate['role_description']}\n\nResume Text: {candidate['resume_text']}\n\nIdentify key skills and experience relevant to the role description and company name. Include six to seven bullet points (not including desired pay rate and availability) with no more than twenty words each. The first bullet point should include years of experience in relevant areas, the next few bullet points should include more key skills relevant to the company name and role description, ***the last bullet point (before desired pay rate and availability) should include specific technologies or tools the candidate is familiar with (for example Microsoft Office Suite (Word, Excel, Outlook, etc))***, followed by:\n- Desired Pay Rate: {candidate['desired_pay_rate']}\n- Availability: {candidate['availability']}. Remember to consider the metrics of Relevance: Evaluates if the summary includes only important information and excludes redundancies. Coherence: Assesses the logical flow and organization of the summary. Consistency: Checks if the summary aligns with the facts in the source document. Fluency: Rates the grammar and readability of the summary. while writing the summary Do not output anything about these metrics, just use them while writing the summary."
        })

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=False,
        )

        response_text = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        candidate['summary'] = response_text  # Update the candidate's summary
        summaries.append(candidate)
    return summaries

# Display the summaries and chat bar for editing
def display_summaries():
    for candidate in st.session_state.summaries:
        st.write(f"### Key Skills and Experience for {candidate['pdf_path']}")
        st.markdown(candidate['summary'])
        st.write(f"**Human-made Summary:** {candidate['human_summary']}")

# Define the function to get similarity score from the API
def get_similarity_score(human_summary, ai_summary):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Compare the following summaries. You may use the metrics of Relevance: Evaluates if the summary includes only important information and excludes redundancies. Coherence: Assesses the logical flow and organization of the summary. Consistency: Checks if the summary aligns with the facts in the source document. Fluency: Rates the grammar and readability of the summary. Provide a similarity score from 1 to 10 between the two summaries and three to four concise reasons for this score in bullet point format, each reason no more than twenty words."},
            {"role": "user", "content": f"Human Summary: {human_summary}\nAI Summary: {ai_summary}"}
        ]
    )
    score_and_reason = response.choices[0].message.content
    score = re.search(r'\d+', score_and_reason).group()
    return int(score), score_and_reason

# Define the function to get accuracy score from the API
def get_accuracy_score(pdf_text, company_name, role_description, ai_summary):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Review the following resume content, company name, and role description, and provide an accuracy score for the AI summary from 1-10 with three to four concise reasons in bullet point format, each reason no more than twenty words. Base the score off of how well the summary matches the resume (whether there's irrelevant points or key points were missed) and the skills needed for the specific company and role description. Don't worry about desired pay rate and availability while scoring, these were added by the user and were not in the resume."},
            {"role": "user", "content": f"Resume Content: {pdf_text}\nCompany Name: {company_name}\nRole Description: {role_description}\nAI Summary: {ai_summary}"}
        ]
    )
    score_and_reason = response.choices[0].message.content
    score = re.search(r'\d+', score_and_reason).group()
    return int(score), score_and_reason

# Calculate similarity score using BERT-based embeddings
def calculate_sentence_similarity(human_summary, ai_summary):
    human_sentences = sent_tokenize(human_summary)
    ai_sentences = sent_tokenize(ai_summary)

    total_similarity = 0

    for ai_sentence in ai_sentences:
        max_similarity = 0
        for human_sentence in human_sentences:
            # Encode sentences
            embeddings1 = model.encode(ai_sentence, convert_to_tensor=True)
            embeddings2 = model.encode(human_sentence, convert_to_tensor=True)

            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
            if similarity > max_similarity:
                max_similarity = similarity
        total_similarity += max_similarity

    average_similarity = total_similarity / len(ai_sentences)
    return average_similarity

# Update acceptability in spreadsheet
def update_acceptability(candidate_info, df):
    for i, candidate in enumerate(candidate_info):
        if candidate['acceptability']:
            df.at[i, 7] = candidate['acceptability']
    return df

# Main application logic
def main():
    st.title("Summarization Tester")

    # Upload spreadsheet file
    uploaded_spreadsheet = st.file_uploader("Upload Spreadsheet", type=["xlsx", "csv"])

    # Process spreadsheet data and generate summaries
    if uploaded_spreadsheet and st.button("Confirm and Process Spreadsheet"):
        st.session_state.candidate_info = collect_data_from_spreadsheet(uploaded_spreadsheet)
        st.session_state.inputs_collected = True

    if st.session_state.inputs_collected and not st.session_state.summaries_generated:
        st.session_state.summaries = generate_summaries()
        st.session_state.summaries_generated = True

    if st.session_state.summaries_generated:
        display_summaries()

        # Automatically run similarity and accuracy tests
        for idx, candidate in enumerate(st.session_state.candidate_info):
            human_summary = candidate['human_summary']
            ai_summary = candidate['summary']

            # Calculate sentence-level similarity
            sentence_similarity_score = calculate_sentence_similarity(human_summary, ai_summary)

            # Get similarity score
            similarity_score, similarity_reason = get_similarity_score(human_summary, ai_summary)

            # Get accuracy score
            accuracy_score, accuracy_reason = get_accuracy_score(candidate['resume_text'], candidate['company_name'], candidate['role_description'], ai_summary)

            # Total score
            total_score = sentence_similarity_score + similarity_score + accuracy_score

            st.write(f"### Scores for {candidate['pdf_path']}:")
            st.write(f"Sentence Similarity Score: {sentence_similarity_score:.2f}")
            st.write(f"Similarity Score: {similarity_score} - {similarity_reason}")
            st.write(f"Accuracy Score: {accuracy_score} - {accuracy_reason}")
            st.write(f"Total Score: {total_score:.2f}")

            acceptability_count = list(map(int, candidate['acceptability'].split('/')))
            if sentence_similarity_score > 50 and similarity_score > 5 and accuracy_score > 5:
                st.write(f"AI-made summary for {candidate['pdf_path']} is acceptable.")
                acceptability_count[0] += 1
            else:
                st.write(f"AI-made summary for {candidate['pdf_path']} is not acceptable.")
                acceptability_count[1] += 1
            candidate['acceptability'] = f"{acceptability_count[0]}/{acceptability_count[1]}"

        # Update the spreadsheet with the new acceptability counts
        df = load_data_from_spreadsheet(uploaded_spreadsheet)
        updated_df = update_acceptability(st.session_state.candidate_info, df)

        # Save updated file
        output = BytesIO()
        if uploaded_spreadsheet.name.endswith('.csv'):
            updated_df.to_csv(output, index=False, header=False)
            mime_type = 'text/csv'
        else:
            updated_df.to_excel(output, index=False, header=False)
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        st.download_button(label="Download Updated Spreadsheet", data=output.getvalue(), file_name=uploaded_spreadsheet.name, mime=mime_type)

if __name__ == "__main__":
    main()