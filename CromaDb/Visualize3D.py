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

    # Reduce dimensions to 3D using PCA for visualization
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot the embeddings in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.7)

    # Annotate points with metadata if available
    for i, meta in enumerate(metadata):
        ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2],
                meta.get('label', str(i)), fontsize=8)

    ax.set_title("3D Visualization of Embeddings")
    ax.set_xlabel("Principal Component 1 ")
    ax.set_ylabel("Principal Component 2 ")
    ax.set_zlabel("Principal Component 3 ")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings_2d()