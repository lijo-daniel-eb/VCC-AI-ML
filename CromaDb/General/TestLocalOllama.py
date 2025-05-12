# use local ollama model to search generic text
from langchain.embeddings import OllamaEmbeddings

# Initialize the Ollama embeddings model
embedding_function = OllamaEmbeddings(model_name="local-ollama-model")

# Example query text
query = "Find information about the first document."

# Simulate a local Ollama LLM search (without embeddings)
def ollama_search(query):
    # Simulated response from the local Ollama model
    response = f"Simulated response for query: '{query}'"
    return response

# Perform the search
response = ollama_search(query)

# Print the response
print("Search Response:", response)