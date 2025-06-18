import streamlit as st
from groq import Groq, InternalServerError
from sentence_transformers import SentenceTransformer
import chromadb
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it up.")
    st.stop()
groq_client = Groq(api_key=api_key)

# Initialize Chroma and SentenceTransformer
client = chromadb.PersistentClient(path="chroma_store")
collection_name = "insurance_claims"
try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(name=collection_name)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Auto Insurance Analysis", page_icon="🚗", layout="centered")
st.title("Auto Insurance Analysis")
st.markdown("**Powered by RAG: Retrieval-Augmented Generation**")
st.write("Enter your insurance query below to get data-driven insights.")

# Input form
with st.form(key="query_form"):
    query = st.text_area(
        "Your Query:",
        value="What incidents involved bodily injury in Arlington?",
        height=100,
        placeholder="e.g., What incidents involved bodily injury in Arlington?"
    )
    submit_button = st.form_submit_button("Analyze")

# Placeholder for results and errors
result_placeholder = st.empty()
error_placeholder = st.empty()

# Custom HTML/CSS/JavaScript for LinkedIn-worthy polish
st.html("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .stTextArea textarea { border-radius: 5px; }
        .stButton>button { width: 100%; background-color: #0a66c2; border-color: #0a66c2; }
        .stButton>button:hover { background-color: #004182; border-color: #004182; }
        .result-box, .error-box { background-color: #ffffff; padding: 15px; border-radius: 5px; margin-top: 20px; min-height: 100px; display: none; }
        .error-box { background-color: #f8d7da; color: #721c24; }
        .spinner-border-sm { margin-right: 5px; }
    </style>
    <script>
        /**
         * Auto Insurance Analysis App - Frontend Logic
         * Author: RAG (Retrieval-Augmented Generation) Bot
         */
        document.addEventListener('DOMContentLoaded', () => {
            const form = document.querySelector('form');
            const submitButton = form?.querySelector('button[kind="formSubmit"]');
            const textArea = form?.querySelector('textarea');

            if (!form || !submitButton || !textArea) return;

            form.addEventListener('submit', (event) => {
                const query = textArea.value.trim();
                if (!query) {
                    event.preventDefault();
                    event.stopPropagation();
                    alert('Please enter a query to analyze.');
                    return false;
                }
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            });
        });
    </script>
""")

if submit_button and query:
    with st.spinner("Analyzing with RAG..."):
        try:
            # Embed the query
            query_embedding = embed_model.encode([query]).tolist()[0]

            # Retrieve top 5 similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas"]
            )
            retrieved_docs = results["documents"][0]
            retrieved_metadata = results["metadatas"][0] if results["metadatas"] else []

            # Build context with metadata
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                metadata = retrieved_metadata[i] if i < len(retrieved_metadata) else {}
                context_parts.append(f"Case {i+1}: {doc}")
                if metadata:
                    context_parts.append(
                        f"Additional Info: Premium: ${metadata.get('policy_annual_premium', 'N/A')}, "
                        f"Deductible: ${metadata.get('policy_deductable', 'N/A')}, "
                        f"State: {metadata.get('policy_state', 'N/A')}"
                    )
            context = "\n\n".join(context_parts)

            # Auto Insurance Specialized Prompt
            prompt = f"""You are an expert AUTO INSURANCE analyst and advisor.
Analyze the following auto insurance claims and customer data to provide insights.

IMPORTANT GUIDELINES:
- Focus specifically on AUTO INSURANCE factors (age, vehicle type, driving history, location)
- Consider risk assessment, premium calculation factors, and claims patterns
- Provide data-driven insights based on the retrieved cases
- Include relevant statistics if patterns emerge
- Mention limitations of the data when giving advice

RETRIEVED AUTO INSURANCE DATA:
---------------------
{context}
---------------------

QUESTION: {query}

Provide a comprehensive analysis covering:
1. Direct answer based on the data
2. Key patterns or trends observed
3. Risk factors that influence premiums/claims
4. Any limitations or caveats

ANSWER:"""

            # Groq API call with retry logic
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized AUTO INSURANCE data analyst. Provide accurate, data-driven insights about auto insurance premiums, claims, and risk factors. Always mention when data is limited or when general recommendations should be verified with insurance professionals."
                    },
                    {"role": "user", "content": prompt}
                ],
                "model": "llama3-8b-8192",
                "temperature": 0.3,
                "max_tokens": 1000
            }

            max_retries = 3
            retry_delay = 5
            for attempt in range(max_retries):
                try:
                    response = groq_client.chat.completions.create(**payload)
                    result = response.choices[0].message.content
                    result += f"\n\n📊 Based on {len(retrieved_docs)} similar cases from the database"
                    result_placeholder.markdown(f"### Analysis Result\n{result}", unsafe_allow_html=True)
                    break
                except InternalServerError:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        error_placeholder.error("Service unavailable after multiple attempts. Please try again later.")
        except Exception as e:
            error_placeholder.error(f"An error occurred: {str(e)}")