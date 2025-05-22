from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
import json
import os

collectionName = "AuditLog"
client = MongoClient("mongodb://localhost:27017/")
db = client["VJournal"]
collection = db[collectionName]

# Use the file from the current working directory
template_file = os.path.join(os.getcwd(), "RiskEventTemplate.json")
if not os.path.exists(template_file):
    print(f"File not found: {template_file}")
with open(template_file, "r") as file:
    template = json.load(file)

# Transform MongoDB documents into the required format
documents = [
    Document(
        page_content=(
            template["text"].format(
                **{key: doc.get(key, "") for key in template["text"].split("{") if "}" in key}
            ) + (
                " ".join(
                    f"{prop}: {value}" for prop, value in doc.get("ExtendedPropertiesAccess", {}).items()
                ) if "ExtendedPropertiesAccess" in doc else ""
            )
        )
    )
    for doc in collection.find().limit(10)
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