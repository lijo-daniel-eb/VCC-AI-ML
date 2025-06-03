#load from Chromadb wrapper class and use it to search
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Load the Chroma wrapper class
collectionName = "AuditLog"
persist_directory = "C:\\ChromaDbLangchain"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function,
    collection_name=collectionName
)

# Perform a similarity search
query = "Security!Shooting: Shooting"
results = db.similarity_search(query, k=10)  # k specifies the number of results to return

# Print the results
print("Query Results:")
for i, doc in enumerate(results):
    print(f"Document {i+1}: {doc.page_content}")
    print("\n")
    input("Press Enter to continue...")
