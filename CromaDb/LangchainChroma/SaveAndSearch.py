from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings

collectionName = "AuditLog"
client = MongoClient("mongodb://localhost:27017/")
db = client["VJournal"]
collection = db[collectionName]

# Transform MongoDB documents into the required format
documents = [
    Document(
        page_content = f"AlertCreatedTime: {doc.get('AlertCreatedTime', '')} AlertTitle: {doc.get('AlertTitle', '')} ActionTaken: {doc.get('ActionTaken', '')} RiskEventSource: {doc.get('RiskEventSource', '')}  RiskEventType: {doc.get('RiskEventType', '')} RiskEventCategory: {doc.get('RiskEventCategory', '')} RiskEventStatus: {doc.get('RiskEventStatus', '')} RiskEventStartTime: {doc.get('RiskEventStartTime', '')} RiskEventEndTime: {doc.get('RiskEventEndTime', '')} RiskEventPublishedTime: {doc.get('RiskEventPublishedTime', '')}"
    )
    for doc in collection.find().limit(150)
]

# Load a local pre-trained model for embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 

# Embed the document chunks and store them in ChromaDB
db = Chroma.from_documents(documents,collection_name=collectionName, embedding=embedding_model, persist_directory="C:\ChromaDbLangchain")


query = "Find records where AlertTitle is Security!Shooting: Shooting and RiskEventType"
results = db.similarity_search(query, k=2)  # k specifies the number of results to return
#display the results line by line
for i, doc in enumerate(results):
    print(f"Document {i+1}: {doc.page_content}")
    #add new line for every document
    print("\n")
input("Press Enter to continue...")