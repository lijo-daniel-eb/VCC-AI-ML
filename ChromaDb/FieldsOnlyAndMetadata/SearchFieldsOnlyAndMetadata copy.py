import os
import json
import re
import subprocess
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# === Step 1: Extract Metadata Field Names from Template ===
def extract_field_names_from_template(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        json_data = json.loads(content)
        # If the template is in a nested key, e.g., {"RiskEventTemplate": "..."},
        if isinstance(json_data, dict) and "RiskEventTemplate" in json_data:
            template_str = json_data["RiskEventTemplate"]
        else:
            template_str = content
    except Exception:
        template_str = content
    fields = re.findall(r"\{(.*?)\}", template_str)
    # Remove ' - date' suffix from any field names
    cleaned_fields = [f[:-6] if f.lower().endswith(' - date') else f for f in fields]
    return sorted(set(cleaned_fields))

TEMPLATE_PATH = "ChromaTemplate.json"
metadata_fields = extract_field_names_from_template(TEMPLATE_PATH)
print("Metadata Fields Available:", metadata_fields)

# === Step 2: Embedding Model Setup ===
collectionName = "RiskEvent"
local_model_path = "C:/Models/all-mpnet-base-v2"

if not os.path.exists(local_model_path) or not any(f.endswith((".bin", ".safetensors", ".h5", ".msgpack")) for f in os.listdir(local_model_path)):
    print(f"Downloading embedding model to {local_model_path}...")
    snapshot_download(repo_id="sentence-transformers/all-mpnet-base-v2", local_dir=local_model_path)
else:
    print(f"Using cached embedding model at {local_model_path}")

embedding_model = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"trust_remote_code": True}
)

# === Step 3: Initialize Chroma Vectorstore ===
persist_directory = "C:\\FieldsOnlyAndMetadata"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collectionName
)

# === Step 4: Setup Ollama LLM ===
ollama_llm = OllamaLLM(
    model="codellama",
    temperature=0.1
)

# === Step 5: Semantic Retrieval Setup ===
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def run_ollama_llm(prompt: str, model: str = "codellama") -> str:
    """Run Ollama LLM with the given prompt and return the output as a string."""
    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error running Ollama LLM: {e}"

# === Step 6: LLM-Based Metadata Filter Extractor ===
LLM_FILTER_EXTRACTION_MAX_RETRIES = 3

prompt_template = """
Extract metadata filters from the user query:"{query}"
You are a filter-building assistant for a ChromaDB vector search. Your task is to convert user-provided conditions into a valid JSON filter dictionary.

Only include filters using these fields: {fields}. The field names must be exact matches from the list above.

Return ONLY a valid JSON dictionary, with no explanation or extra text. The output must be in this format:
For single conditions use:
{{
  "ExternalID": "123432"
}}

For multiple conditions use:
{{
  "$and": [
    {{"ExternalID": {{"$eq": "112443"}}}},
    {{"Category": {{"$eq": "Critical"}}}},
    {{"Region": {{"$eq": "Asia"}}}}
  ]
}}

Include the appropriate operators like "$eq", "$ne", "$gt", "$lt", "$gte", "$lte", "$and", "$or" as needed.

All field names and values must be in double quotes. Do not include any other text or comments before or after the JSON.

Ignore any field not listed in the allowed fields. Do not add unwanted fields. Fields with no value must be omitted.
""".strip()

def extract_metadata_filters_with_ops(query: str, fields: list[str]) -> dict:
    import ast
    prompt = prompt_template.format(query=query, fields=', '.join(fields))
    last_exception = None
    for attempt in range(1, LLM_FILTER_EXTRACTION_MAX_RETRIES + 1):
        output = run_ollama_llm(prompt)
        try:
            # Try to extract the first valid JSON object from the output
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            # Remove any trailing/leading non-JSON characters
            json_str = json_str.strip()
            # Try to fix common issues: single quotes, trailing commas
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            filters = json.loads(json_str)
            return filters
        except Exception as e:
            print(f"Attempt {attempt} failed to extract filters (invalid JSON from LLM):", e)
            print("LLM Output:", output)
            last_exception = e
    print(f"All {LLM_FILTER_EXTRACTION_MAX_RETRIES} attempts failed. Falling back to semantic search.")
    return "__FALLBACK_TO_SEMANTIC__"

# === Step 7: Summarize result using Ollama ===
def summarize_with_ollama(context_docs: list, user_query: str) -> list:
    if not context_docs:
        return ["No relevant documents found."]
    summaries = []
    for doc in context_docs:
        context = doc.page_content
        prompt = f"""
You are an assistant responding to a user query based on internal documents.

Query: "{user_query}"

Document:
{context}

Based on this document, provide a short and clear answer to the user's question.
"""
        summary = run_ollama_llm(prompt)
        summaries.append(summary)
    return summaries

# === Step 8: Interactive Loop ===
def normalize_filter(filters):
    if filters is None:
        return None
    if len(filters) <= 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}

while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting.")
        break

    filters = extract_metadata_filters_with_ops(query, metadata_fields)

    if filters and filters != "__FALLBACK_TO_SEMANTIC__":
        print(f"\nExtracted filters: {json.dumps(filters, indent=2)}")
        try:
            normalized_filters = normalize_filter(filters)
            results = vectorstore.similarity_search(
                query=query,
                k=5,
                filter=normalized_filters
            )
            # If no results, fallback to semantic search
            if results:
                summaries = summarize_with_ollama(results, query)
                print("\nSummarized Results:")
                for idx, summary in enumerate(summaries, 1):
                    print(f"\nResult {idx}:")
                    print(summary)
            else:
                print("No documents matched these filters. Performing semantic search...")
                try:
                    result = qa_chain.invoke({"query": query})
                    summaries = summarize_with_ollama(result['source_documents'], query)
                    print("\nSummarized Results:")
                    for idx, summary in enumerate(summaries, 1):
                        print(f"\nResult {idx}:")
                        print(summary)
                except Exception as e:
                    print(f"Error during semantic search: {e}")
        except Exception as e:
            print(f"Error during similarity_search: {e}")
    else:
        print("No filters found or error extracting filters. Performing semantic search...")
        try:
            result = qa_chain.invoke({"query": query})
            summaries = summarize_with_ollama(result['source_documents'], query)
            print("\nSummarized Results:")
            for idx, summary in enumerate(summaries, 1):
                print(f"\nResult {idx}:")
                print(summary)
        except Exception as e:
            print(f"Error during semantic search: {e}")