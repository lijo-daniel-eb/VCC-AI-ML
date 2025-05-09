from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="C:\\ChromaDb",
    embedding_function=embedding_function
)

query = "Local PD"
results = db.similarity_search(query, k=10)  # k specifies the number of results to return

print("Query Results:")
# for i, doc in enumerate(results):
#     print(f"Document {i+1}: {doc}")
#     #add new line for every document
#     print("\n")

print(results)