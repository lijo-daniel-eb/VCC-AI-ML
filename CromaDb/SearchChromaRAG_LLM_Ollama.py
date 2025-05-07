
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp  # For local LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms import Ollama
import chromadb

# 1. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 2. Connect to your existing ChromaDB collection
client = chromadb.Client(Settings(persist_directory="C:\\ChromaDb"))
# Option 1: List existing collections and use one if it exists
existing_collections = client.list_collections()
existing_collection = client.get_collection(name="chroma_collection")

# 3. Get all documents from your existing collection
raw_docs = existing_collection.get(include=["documents", "metadatas", "embeddings"])

# 4. Convert to LangChain documents
documents = []
for i, doc_text in enumerate(raw_docs["documents"]):
    metadata = raw_docs["metadatas"][i] if i < len(raw_docs["metadatas"]) else {}
    documents.append(Document(page_content=doc_text, metadata=metadata))

# 5. Create a LangChain Chroma wrapper that points to the existing collection
vectorstore = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory="C:\\ChromaDb"
)

# 6. Initialize the local LLM (adjust path and parameters to your model)
llm = Ollama(
    model="llama3",  # Update with your model path
    temperature=0.1,
    max_tokens=2000,
    n_ctx=4096,  # Adjust context window size based on your model
    top_p=1
)

# 7. Create a custom prompt template for RAG
prompt_template = """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 8. Create the RAG pipeline
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

# 9. Function to query the RAG system
def answer_question(question):
    """
    Query the RAG system with a question
    
    Args:
        question: String containing the user's question
        
    Returns:
        Dictionary with answer and source documents
    """
    result = rag_chain({"query": question})
    
    # Extract answer and sources
    answer = result["result"]
    sources = result["source_documents"]
    
    # Print results
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\nSources:")
    for i, doc in enumerate(sources):
        print(f"Source {i+1}: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}\n")
    
    return result

# Example usage
question = "What information do you have about X?"  # Replace with actual question
response = answer_question(question)