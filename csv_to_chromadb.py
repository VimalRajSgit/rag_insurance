import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# 1. Load CSV
df = pd.read_csv("insurance_claims.csv")

# 2. Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Initialize ChromaDB Persistent Client
client = PersistentClient(path="chroma_store")
collection = client.get_or_create_collection(name="insurance_claims")

# 4. Prepare Chunks and Embeddings
all_chunks = []
all_metadatas = []
all_ids = []
all_embeddings = []
for idx, row in df.iterrows():
    # Chunk 1: Policy & Customer Details
    chunk1 = (
        f"Policy number {row['policy_number']} issued in state {row['policy_state']} with annual premium ${row['policy_annual_premium']}. "
        f"Customer is a {row['age']}-year-old {row['insured_sex']} working as {row['insured_occupation']}, education level {row['insured_education_level']}, "
        f"hobbies include {row['insured_hobbies']}. Policy started on {row['policy_bind_date']} and includes umbrella limit of ${row['umbrella_limit']}."
    )

    # Chunk 2: Incident Info
    chunk2 = (
        f"Incident occurred on {row['incident_date']} at {row['incident_hour_of_the_day']}:00 in {row['incident_city']}, {row['incident_state']}. "
        f"It involved a {row['incident_type']} with {row['incident_severity']} severity. {row['number_of_vehicles_involved']} vehicle(s) involved. "
        f"Authorities contacted: {row['authorities_contacted']}. Witnesses: {row['witnesses']}. Police report: {row['police_report_available']}."
    )

    # Chunk 3: Claim and Vehicle Info
    chunk3 = (
        f"Claim filed for total amount ${row['total_claim_amount']}, including ${row['injury_claim']} injury claim, ${row['property_claim']} property claim, "
        f"and ${row['vehicle_claim']} vehicle claim. Vehicle involved: {row['auto_make']} {row['auto_model']} {row['auto_year']}. "
        f"Suspected fraud: {row['fraud_reported']}."
    )

    chunks = [chunk1, chunk2, chunk3]

    # Metadata for filtering/search
    metadata = {
        "policy_number": str(row["policy_number"]),
        "incident_type": str(row["incident_type"]),
        "incident_city": str(row["incident_city"]),
        "age": int(row["age"]),
        "fraud_reported": str(row["fraud_reported"])
    }

    # Encode text to vector embeddings
    embeddings = model.encode(chunks).tolist()

    all_embeddings.extend(embeddings)
    # Collect for ChromaDB
    all_chunks.extend(chunks)
    all_metadatas.extend([metadata] * 3)
    all_ids.extend([f"{idx}_chunk{i+1}" for i in range(3)])
print(len(all_chunks), len(all_embeddings), len(all_metadatas), len(all_ids))

# 5. Store in ChromaDB
collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids
)

print(f"✅ Stored {len(all_chunks)} chunks in ChromaDB")
