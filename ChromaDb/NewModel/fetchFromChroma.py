from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model (must match the one used for saving)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"trust_remote_code": True}
)

persist_directory = "./chroma_test_bge"
collection_name = "TestCollection"

# Load the ChromaDB collection
chroma_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collection_name
)

while True:
    query = input("Enter your search query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting.")
        break
    # Embed and search
    results = chroma_db.similarity_search(query, k=5)
    print("\nTop results:")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print()
