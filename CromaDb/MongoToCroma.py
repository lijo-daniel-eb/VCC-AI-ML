from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize ChromaDB client - In memory mode
#chroma_client = chromadb.Client()

# for client server - refer https://docs.trychroma.com/docs/run-chroma/client-server

## Initialize ChromaDB client - Persisnatnt storage mode
chroma_client = chromadb.PersistentClient(path="C:\\ChromaDb")

chroma_Collection = chroma_client.get_or_create_collection(name="AuditLog")

# Load a local pre-trained model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can choose other models from Hugging Face

client = MongoClient("mongodb://localhost:27017/")
db = client["VJournal"]
collection = db["AuditLog"]
documents = list(collection.find())

# Generate embeddings for each document
for doc in documents:
    content = f"{doc.get('AlertTitle', '')} {doc.get('ActionTaken', '')} {doc.get('RiskEventSource', '')}  {doc.get('RiskEventType', '')} {doc.get('RiskEventCategory', '')} {doc.get('RiskEventStatus', '')} {doc.get('RiskEventStartTime', '')}"
    if content:
        # Generate embedding locally
        embedding = model.encode(content).tolist()
        # Add document and embedding to ChromaDB
        chroma_Collection.add(
            documents=[content],  # The content of the document
            metadatas=[{"mongo_id": str(doc["_id"])}],  # Metadata (e.g., MongoDB ID)
            ids=[str(doc["_id"])],
            embeddings=embedding  # Unique ID for the document
        )
        print(f"Document ID: {doc['_id']}")