import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import chromadb

def visualize_embeddings_2d():
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="C:\\ChromaDb")

    # Get the collection
    chroma_collection = chroma_client.get_or_create_collection(name="chroma_collection")

    # Retrieve all embeddings and their metadata
    embeddings = []
    metadata = []
    existing_count = chroma_collection.count()
    batch_size = 10
    for i in range(0, existing_count, batch_size):
        batch = chroma_collection.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=i)
        embeddings.extend(batch["embeddings"])
        metadata.extend(batch["metadatas"])

    # Convert embeddings to a NumPy array
    embeddings = np.array(embeddings)

    # Reduce dimensions to 2D using t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot the embeddings in 2D
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

    # Annotate points with metadata if available
    for i, meta in enumerate(metadata):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1],
                 meta.get('label', str(i)), fontsize=8)

    plt.title("2D Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings_2d()