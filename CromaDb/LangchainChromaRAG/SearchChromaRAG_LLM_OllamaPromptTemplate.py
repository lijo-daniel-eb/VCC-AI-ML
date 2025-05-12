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

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
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
    search_kwargs={"k": 10}
)

# 5. Create a RAG chain with the Ollama LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",  # Simple document compilation
    retriever=retriever,
    return_source_documents=True  # To see which documents were used
)

# 6. Run a query through the RAG system
query = "display all records where Risk Event Source is Local PD occured on 2nd Feb 2025"

# 11. For better prompting with Ollama models, you might want to customize the prompt template
from langchain.prompts import PromptTemplate

# Create a custom prompt template specifically for RAG
template = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create a chain with the custom prompt
qa_chain_custom_prompt = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

result_custom = qa_chain_custom_prompt({"query": query})
print(f"Answer with custom prompt: {result_custom['result']}")