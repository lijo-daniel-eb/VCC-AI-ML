from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.schema import Document


# Load a local pre-trained model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can choose other models from Hugging Face

client = MongoClient("mongodb://localhost:27017/")
db = client["VJournal"]
collection = db["AuditLog"]

# Transform MongoDB documents into the required format
documents = [
    Document(
        page_content = f"{doc.get('AlertTitle', '')} {doc.get('ActionTaken', '')} {doc.get('RiskEventSource', '')}  {doc.get('RiskEventType', '')} {doc.get('RiskEventCategory', '')} {doc.get('RiskEventStatus', '')} {doc.get('RiskEventStartTime', '')}",  # Replace "content" with the key containing the text in your MongoDB documents
        metadata={key: value for key, value in doc.items() }  # Include other fields as metadata
    )
    for doc in collection.find().limit(500)
]

# Embed the document chunks and store them in ChromaDB
db = Chroma.from_texts(documents, model)

query = "Find records where AlertTitle is Security!Shooting: Shooting and RiskEventType"
results = db.similarity_search(query, k=10)  # k specifies the number of results to return