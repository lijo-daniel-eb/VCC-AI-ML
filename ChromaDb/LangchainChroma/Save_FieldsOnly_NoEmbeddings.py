from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os
import re
from uuid import UUID
import datetime
from datetime import timezone, timedelta

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
#embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "ChromaDbEmbeddings.py"
db = Chroma(
    persist_directory=persist_directory,
    collection_name=collectionName
)
print("Status: Loading data from MongoDB and processing batches...")
total_docs = collection.count_documents({})

def ticks_to_date(tick_array):
    try:
        ticks, offset_minutes = tick_array
        EPOCH_TICKS = 621355968000000000  # Ticks between 0001-01-01 and 1970-01-01
        TICKS_PER_SECOND = 10_000_000

        # Calculate total seconds since Unix epoch
        seconds_since_epoch = (int(ticks) - EPOCH_TICKS) / TICKS_PER_SECOND
        utc_datetime = datetime.datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds_since_epoch)

        # Apply timezone offset (in this case, it's 0 → UTC)
        tz_offset = timezone(timedelta(minutes=int(offset_minutes)))
        return utc_datetime.astimezone(tz_offset).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Exception converting ticks for value '{tick_array}': {e}")
        return str(tick_array)

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
            # Check for keys with '- date' suffix
            if key.strip().lower().endswith('- date'):
                base_key = key.rsplit('-', 1)[0].strip()
                ticks = doc.get(base_key, "")
                if ticks is not None and ticks != "":
                    try:
                        page_content = page_content.replace(f"{{{key}}}", ticks_to_date(ticks))
                    except Exception:
                        page_content = page_content.replace(f"{{{key}}}", str(ticks))
                else:
                    page_content = page_content.replace(f"{{{key}}}", "")
            else:
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