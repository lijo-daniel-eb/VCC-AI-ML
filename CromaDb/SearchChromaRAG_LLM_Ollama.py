import os
from huggingface_hub import snapshot_download
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from chromadb import Client
# Define model info
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
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
# embeddings = HuggingFaceEmbeddings(model_name=local_model_path)


# 2. Create a LangChain Chroma wrapper that points to the existing collection
# This directly connects to your existing ChromaDB collection

vectorstore = Chroma(
    collection_name="AuditLog",  
    persist_directory="C:\\ChromaDb",
    # embedding_function=embeddings,
    client_settings={"_type": "chroma"} 
)

# 3. Initialize the local LLM
llm = Ollama(
    model="llama3",
    temperature=0.1,
    max_tokens=2000,
    context_window=4096,  # Updated from n_ctx to context_window
    top_p=1
)

# 4. Create a custom prompt template for RAG
prompt_template = """
Answer the question based only on the following context:

{context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 5. Create the RAG pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Options: "stuff", "map_reduce", "refine"
    retriever=vectorstore.as_retriever(
        search_type="similarity",  # Options: "similarity", "mmr"
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # Return the source documents with the answer
)

# 6. Function to query the RAG system
def answer_question(question):
    """
    Query the RAG system with a question
    
    Args:
        question: String containing the user's question
        
    Returns:
        Dictionary with answer and source documents
    """
    result = rag_chain.invoke({"query": question})  # Changed from call to invoke
    
    # Extract answer and sources
    answer = result.get("result", "")  # Using get() with default value
    sources = result.get("source_documents", [])  # Using get() with default value
    
    # Print results
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print("\nSources:")
    for i, doc in enumerate(sources):
        print(f"Source {i+1}: {doc.page_content[:150]}...")
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}\n")
    
    return result

# 7. Run in a loop to allow multiple questions
def main():
    print("RAG Question-Answering System")
    print("Type 'exit', 'quit', or 'q' to end the session")
    
    while True:
        # Get question from user input
        question = input("\nEnter your question: ")
        
        # Check if user wants to exit
        if question.lower() in ['exit', 'quit', 'q']:
            print("Exiting the program. Goodbye!")
            break
            
        # Process the question
        response = answer_question(question)

if __name__ == "__main__":
    main()