from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Initialize ChromaDB vector store
vector_store = Chroma(persist_directory="C:\\ChromaDb")

# Fetch documents from the ChromaDB collection
documents = vector_store.get_all_documents()

# Use the fetched documents to initialize the vector store
vector_store = vector_store.from_documents(documents[:10], embedding_function=None)

# Define the retriever
retriever = vector_store.as_retriever()

# Initialize the local HuggingFace model
local_llm_pipeline = pipeline("text-generation", model="llama3")
llm = HuggingFacePipeline(pipeline=local_llm_pipeline)

# Define a custom prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question accurately:
    Context: {context}
    Question: {question}
    Answer:
    """
)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=llm,
    chain_type_kwargs={"prompt": prompt_template}
)

# Example usage
if __name__ == "__main__": 
    question = "What is the role of music?"
    result = qa_chain.run(question)
    print("Answer:", result)