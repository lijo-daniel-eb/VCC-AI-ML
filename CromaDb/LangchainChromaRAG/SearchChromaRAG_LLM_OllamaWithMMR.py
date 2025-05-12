import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
# Define model info
model_name = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
local_model_path = "C:\\LLM_Model\\all-MiniLM-L6-v2"  # Change this to your preferred local path

# Check if model exists locally, if not download it
def ensure_model_is_local(model_name, local_path):
    if not os.path.exists(local_path) or len(os.listdir(local_path)) == 0:
        print(f"Model not found locally. Downloading {model_name} to {local_path}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path
        )
        print(f"Model downloaded successfully to {local_path}")
    else:
        print(f"Using cached model at {local_path}")
    return local_path

# Ensure model is available locally
local_model_path = ensure_model_is_local(model_name, local_model_path)

# 1. Initialize the embedding model. Use the local model path for embeddings
# Load the Chroma wrapper class
persist_directory = "C:\\ChromaDbLangchain"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create a LangChain Chroma wrapper that points to the existing collection
# This directly connects to your existing ChromaDB collection
collectionName = "AuditLog"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function,
    collection_name=collectionName
)

# 3. Initialize Ollama LLM
# Make sure Ollama is running locally with your chosen model
ollama_llm = Ollama(
    model="llama3",  # Or any model you have in Ollama: mistral, llava, gemma, etc.
    temperature=0.1  # Lower temperature for more factual responses
)


# 4. Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 20}
)

# 5. Create a RAG chain with the Ollama LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",  # Simple document compilation
    retriever=retriever,
    return_source_documents=True  # To see which documents were used
)

# Replace the static query with a user input loop
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the application.")
        break

    # Run the query through the RAG system
    result = qa_chain({"query": query})
    print(f"Answer: {result['result']}")

    # Display source documents
    # print("Source documents:")
    # for i, doc in enumerate(result['source_documents']):
    #     print(f"Document {i+1}: {doc.page_content[:100]}...")
    #     print(f"Metadata: {doc.metadata}")
    #     print("---")


# 7. Advanced retrieval with MMR for more diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 10, "lambda_mult": 0.7}
)

qa_chain_mmr = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True
)

# 8. Using metadata filters if your documents have metadata
metadata_filter = {"category": "important"}
filtered_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20, "filter": metadata_filter}
)

# 10. Get all documents from the collection if needed
collection = vectorstore._collection
all_results = collection.get()
print(f"Total documents in collection: {len(all_results['documents'])}")

