import os
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
persist_directory = "C:\\FieldsOnlyAndMetadata"
collection_name = "RiskEvent"

# Initialize Chroma vectorstore
chroma_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function,
    collection_name=collection_name
)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the application.")
        break
    query_embedding = embedding_function.embed_query(query)
    where_document = None
    filter_input = input("Enter a metadata filter as key=value (or press Enter to skip): ")
    if filter_input.strip():
        key, value = filter_input.split('=', 1)
        where_document = {key.strip(): value.strip()}
    results = chroma_db.similarity_search_by_vector(query_embedding, k=5, where_document=where_document)
    print(f"Number of results: {len(results)}")
    print("Query Results:")
    for i, doc in enumerate(results):
        print(f"Document {i+1}:")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("\n")

