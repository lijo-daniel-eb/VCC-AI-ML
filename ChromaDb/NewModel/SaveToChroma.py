from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import random

# Sample data: 10 random records
sample_records = [
    {"text": f"This is sample document {i+1}", "meta": {"id": i+1, "category": random.choice(["A", "B", "C"])}}
    for i in range(10)
]

# Initialize the embedding model (with trust_remote_code for sentence-transformers)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"trust_remote_code": True}
)

# Prepare ChromaDB
persist_directory = "./chroma_test_bge"
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name="TestCollection"
)

# Convert records to LangChain Document objects
documents = [
    Document(page_content=rec["text"], metadata=rec["meta"]) for rec in sample_records
]

# Save to ChromaDB
ids = [str(rec["meta"]["id"]) for rec in sample_records]
db.add_documents(documents, ids=ids)
print("Saved 10 random records to ChromaDB with BAAI/bge-base-en-v1.5 embeddings.")
