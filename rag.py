from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()
import os
# Step 1: Load Chroma and SentenceTransformer
client = chromadb.PersistentClient(path="chroma_store")
collection_name = "insurance_claims"

try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(name=collection_name)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Auto Insurance Specific Query
query = """What incidents involved bodily injury in Arlington?"""

# Step 3: Embed the query
query_embedding = embed_model.encode([query]).tolist()[0]

# Step 4: Retrieve top 5 similar chunks (more context for insurance analysis)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=['documents', 'metadatas']  # Include metadata for richer context
)

retrieved_docs = results['documents'][0]
retrieved_metadata = results['metadatas'][0] if results['metadatas'] else []

# Enhanced context with metadata
context_parts = []
for i, doc in enumerate(retrieved_docs):
    metadata = retrieved_metadata[i] if i < len(retrieved_metadata) else {}
    context_parts.append(f"Case {i+1}: {doc}")
    if metadata:
        context_parts.append(f"Additional Info: Premium: ${metadata.get('policy_annual_premium', 'N/A')}, "
                           f"Deductible: ${metadata.get('policy_deductable', 'N/A')}, "
                           f"State: {metadata.get('policy_state', 'N/A')}")

context = "\n\n".join(context_parts)
api_key = os.getenv("GROQ_API_KEY")


groq_client = Groq(api_key=api_key)
# Step 5: Auto Insurance Specialized Prompt

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

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a specialized AUTO INSURANCE data analyst. Provide accurate, data-driven insights about auto insurance premiums, claims, and risk factors. Always mention when data is limited or when general recommendations should be verified with insurance professionals."
        },
        {"role": "user", "content": prompt}
    ],
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 0.3,  # Lower temperature for more consistent analysis
    "max_tokens": 1000
}

response = groq_client.chat.completions.create(**payload)

print("🚗 AUTO INSURANCE ANALYSIS:")
print("="*50)
print(response.choices[0].message.content)
print("="*50)
print(f"📊 Based on {len(retrieved_docs)} similar cases from the database")