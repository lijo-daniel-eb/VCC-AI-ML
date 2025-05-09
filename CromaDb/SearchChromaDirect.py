from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="C:\\ChromaDb")

chroma_Collection = chroma_client.get_or_create_collection(name="AuditLog")

# Fetch all records from the collection
# def fetch_audit_log_records():
#     records = chroma_Collection.get(include=['embeddings', 'documents', 'metadatas'])    
#     print(f"Fetched {len(records['documents'])} records from ChromaDB.")
#     return records

# results = fetch_audit_log_records()

# Initialize a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

query = "Security!Shooting"
query_embedding = model.encode([query]).tolist()
results = chroma_Collection.query(
 query_embeddings=query_embedding,
 n_results=20
)

print("Query Results:")
for i, doc in enumerate(results['documents']):
    print(f"Document {i+1}: {doc}")
    #add new line for every document
    print("\n")