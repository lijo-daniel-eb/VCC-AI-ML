from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize the SentenceTransformerEmbeddings model
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Example vector data (text data)
vector_data = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document."
]

# Generate embeddings for the vector data
embeddings = [embedding_function.embed_query(data) for data in vector_data]

# Print the embeddings
for i, embedding in enumerate(embeddings):
    print(f"Embedding for document {i+1}: {embedding}")