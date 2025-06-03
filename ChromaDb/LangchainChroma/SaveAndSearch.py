from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
import json
import os
import re
from uuid import UUID

collectionName = "RiskEvent"
client = MongoClient("mongodb://localhost:27017/")
db = client["VJournal"]
collection = db[collectionName]

# Load the template from ChromaTemplate.json
with open("ChromaTemplate.json", "r") as file:
    template = json.load(file)

# Find the RiskEventTemplate object
risk_event_template = None
for obj in template:
    if "RiskEventTemplate" in obj:
        risk_event_template = obj["RiskEventTemplate"]
        break

# Transform MongoDB documents into the required format
pattern = re.compile(r"{(.*?)}")
batch_size = 100  # Adjust as needed

# Load a local pre-trained model for embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "C:\\ChromaDbLangchain"
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collectionName
)
print("Status: Loading data from MongoDB and processing batches...")
total_docs = collection.count_documents({})
for skip in range(0, total_docs, batch_size):
    percent = int(((skip + batch_size) / total_docs) * 100) if total_docs else 100
    percent = min(percent, 100)
    bar_length = 40
    filled_length = int(bar_length * percent // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProcessing: |{bar}| {percent}%", end="")
    documents = []
    batch_docs = collection.find().skip(skip).limit(batch_size)
    for doc in batch_docs:
        keys = pattern.findall(risk_event_template)
        page_content = risk_event_template
        for key in keys:
            page_content = page_content.replace(f"{{{key}}}", str(doc.get(key, "")))
        # If ExtendedPropertiesAccess exists and is a dict, append its properties
        if "ExtendedPropertiesAccess" in doc and isinstance(doc["ExtendedPropertiesAccess"], dict):
            ext_props = doc["ExtendedPropertiesAccess"]
            ext_props_str = " ".join(f"{k}: {v}" for k, v in ext_props.items())
            page_content += " " + ext_props_str
        documents.append(Document(page_content=page_content))
    # Embed the document chunks and store them in ChromaDB for this batch
    db.add_documents(documents)
print("\nStatus: Loading complete.")

# query = "Find records where AlertTitle is Security!Shooting: Shooting and RiskEventType"
# results = db.similarity_search(query, k=2)  # k specifies the number of results to return
# #display the results line by line
# for i, doc in enumerate(results):
#     print(f"Document {i+1}: {doc.page_content}")
#     #add new line for every document
#     print("\n")
input("Press Enter to continue...")