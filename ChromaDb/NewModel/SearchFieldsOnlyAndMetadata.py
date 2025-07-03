import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import spacy
# Define model info
embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
local_model_path = "C:\\Models\\all-MiniLM-L6-v2"  # Change this to your preferred local path

# Check if model exists locally, if not download it
# (Optional: If you want to ensure the model is downloaded, you can use huggingface_hub, but HuggingFaceEmbeddings will handle it if not present)

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

# Ensure model is available locally (optional, for explicit download)
local_model_path = ensure_model_is_local(embedding_model_name, local_model_path)

# 1. Initialize the embedding model. Use the local model path for embeddings
# Load the Chroma wrapper class
persist_directory = "C:\\FieldsOnlyAndMetadata"

# 2. Create a LangChain Chroma wrapper that points to the existing collection
# This directly connects to your existing ChromaDB collection
collectionName = "RiskEvent"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function,
    collection_name=collectionName
)

# 3. Initialize Ollama LLM
# Make sure Ollama is running locally with your chosen model
ollama_llm = OllamaLLM(
    model="llama3",  # Or any model you have in Ollama: mistral, llava, gemma, etc.
    temperature=0.1  # Lower temperature for more factual responses
)

nlp = spacy.load("en_core_web_sm")


# 7. Advanced retrieval with MMR for more diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 10, "lambda_mult": 0.5 }
)
def metadata_filter_retriever(query, docs):
    # Use Spacy to extract a metadata filter condition from the query
    doc_nlp = nlp(query)
    filter_key = None
    filter_value = None
    # Try to find a pattern like 'where <key> is <value>' or '<key> is <value>'
    for i, token in enumerate(doc_nlp):
        # Pattern: 'where <key> is <value>' or 'with <key> is <value>'
        if token.text.lower() in ("where", "with") and i+3 <= len(doc_nlp):
            if doc_nlp[i+1].pos_ == "NOUN" and doc_nlp[i+2].text.lower() in ("is", "=", "equals", "equal"):
                filter_key = doc_nlp[i+1].text
                filter_value = doc_nlp[i+3-1].text
                break
        # Pattern: '<key> is <value>' or '<key> = <value>'
        if token.text.lower() in ("is", "=", "equals", "equal") and i > 0 and i < len(doc_nlp)-1:
            if doc_nlp[i-1].pos_ == "NOUN":
                filter_key = doc_nlp[i-1].text
                filter_value = doc_nlp[i+1].text
                break
    if filter_key and filter_value:
        # Lowercase both for robust matching
        return [doc for doc in docs if str(doc.metadata.get(filter_key, "")).lower() == filter_value.lower()]
    return docs

qa_chain_mmr = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True,
    filter_retriever=metadata_filter_retriever
)

# 10. Get all documents from the collection if needed
# collection = vectorstore._collection
# all_results = collection.get()
# print(f"Total documents in collection: {len(all_results['documents'])}")

# Add an option to use filtered_retriever in the query loop
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the application.")
        break
    result = qa_chain_mmr.invoke({"query": query})
    print(f"Answer: {result['result']}")
    # Optionally, print source documents
    if 'source_documents' in result:
        print("\nSource Documents:")
        for i, doc in enumerate(result['source_documents']):
            print(f"Document {i+1}:")
            print(doc.page_content)
            print("Metadata:", doc.metadata)
            print()
        # Return a natural language summary of the most relevant document
        if result['source_documents']:
            top_doc = result['source_documents'][0]
            print("\nMost relevant document summary:")
            print(f"This answer is based on a document with the following content: '{top_doc.page_content}'. Metadata: {top_doc.metadata}")



