from chromadb.config import Settings
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_embedding_distance(embedding1, embedding2):
    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    # Convert similarity to distance (1 - similarity)
    distance = 1 - similarity
    return distance

def search_chromadb(keyword):
    # Initialize ChromaDB client in persistent storage mode
    chroma_client = chromadb.Client(Settings(persist_directory="C:\\ChromaDb"))

    # Get the collection
    chroma_collection = chroma_client.get_or_create_collection(name="chroma_collection")

    # Perform the search
    results = chroma_collection.query(
        query_texts=[keyword],
        n_results=2,  # Number of relevant records to retrieve
        include_embeddings=True  # Include embeddings in the result
    )

    # Apply relevance threshold (e.g., 0.75)
    threshold = 0.75
    filtered_results = {
        'documents': [],
        'distances': [],
        'embeddings': []
    }

    for doc, score, embedding in zip(results['documents'], results['distances'], results['embeddings']):
        if score >= threshold:
            filtered_results['documents'].append(doc)
            filtered_results['distances'].append(score)
            filtered_results['embeddings'].append(embedding)

    # Calculate distances between the keyword and results
    if filtered_results['embeddings']:
        keyword_embedding = filtered_results['embeddings'][0]
        distances = []
        for embedding in filtered_results['embeddings']:
            distance = calculate_embedding_distance(keyword_embedding, embedding)
            distances.append(distance)

        # Update distances in the results
        filtered_results['distances'] = distances

    return filtered_results

if __name__ == "__main__":
    keyword = input("Enter a keyword to search: ")
    search_results = search_chromadb(keyword)
    print("Search Results:")
    for i, result in enumerate(search_results.get('documents', [])):
        print(f"Result: {result}, Distance: {search_results['distances'][i]}")