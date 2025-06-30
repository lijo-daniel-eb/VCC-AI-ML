import os
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Ensure spaCy model is available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. Model and vectorstore setup
collectionName = "RiskEvent"
embedding_model_name = "all-mpnet-base-v2"
local_model_path = "C:/Models/all-mpnet-base-v2"

if not os.path.exists(local_model_path) or not any(f.endswith((".bin", ".safetensors", ".h5", ".msgpack")) for f in os.listdir(local_model_path)):
    print(f"Downloading model to {local_model_path}...")
    snapshot_download(repo_id="sentence-transformers/all-mpnet-base-v2", local_dir=local_model_path)
    print(f"Model downloaded successfully to {local_model_path}")
else:
    print(f"Using cached model at {local_model_path}")

embedding_model = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"trust_remote_code": True}
)
persist_directory = "C:\\FieldsOnlyAndMetadata"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collectionName
)

# 3. Initialize Ollama LLM
ollama_llm = OllamaLLM(
    model="llama3",
    temperature=0.1
)

# 4. Advanced retrieval with MMR for more diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)
qa_chain_mmr = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True
)

# 5. Interactive query loop with placeholder for metadata/field-based search
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the application.")
        break
    # --- Placeholder for metadata/field-based search logic ---
    # If you want to support field-based (non-semantic) search, add logic here to detect fields in the query
    # and use vectorstore.similarity_search(query, where=...) or similar methods.
    # For now, always use semantic MMR search:
    result = qa_chain_mmr.invoke({"query": query})
    print(f"Answer: {result['result']}")
    if 'source_documents' in result:
        print("\nSource Documents:")
        for i, doc in enumerate(result['source_documents']):
            print(f"Document {i+1}:")
            print(doc.page_content)
            print("Metadata:", doc.metadata)
            print()
        if result['source_documents']:
            top_doc = result['source_documents'][0]
            print("\nMost relevant document summary:")
            print(f"This answer is based on a document with the following content: '{top_doc.page_content}'. Metadata: {top_doc.metadata}")



