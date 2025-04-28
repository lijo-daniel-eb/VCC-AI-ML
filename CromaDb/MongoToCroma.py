from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client - In memory mode
#chroma_client = chromadb.Client()

# for client server - refer https://docs.trychroma.com/docs/run-chroma/client-server

## Initialize ChromaDB client - Persisnatnt storage mode
chroma_client = chromadb.PersistentClient(path="C:\\ChromaDb")

chroma_Collection = chroma_client.get_or_create_collection(name="chroma_collection")

# Clear the ChromaDB collection before inserting any records


# Load a local pre-trained model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can choose other models from Hugging Face

client = MongoClient("mongodb://localhost:27017/")
db = client["admin"]
collection = db["TestCollection"]
documents = list(collection.find())

# Ensure the OpenAI library is installed
# pip install openai

# Generate embeddings for each document
for doc in documents:
    content = doc.get("content", "")
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
        print(f"Document ID: {doc['_id']}, Embedding: {embedding}")