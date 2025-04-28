from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyodbc

# Load the SQLCoder model
tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", torch_dtype=torch.float16, device_map="auto")

def fetch_data_from_prompt(prompt):
    """
    Generate SQL from natural language prompt using SQLCoder and execute it on MSSQL.
    """
    # Define schema if known (optional)
    schema = """
    Tables:
    - employee(id, name, department, hire_date, salary)
    """

    # Format prompt
    formatted_prompt = f"""### Postgres SQL tables, with their properties:
{schema}
### A query to answer: {prompt}
SELECT"""

    # Tokenize and generate SQL
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256, do_sample=False)

    # Decode output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_sql = "SELECT " + decoded.split("SELECT", 1)[-1].strip()

    print("Generated SQL:", generated_sql)

    # Connect to MSSQL
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=VCCDS;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()

    try:
        cursor.execute(generated_sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        conn.close()
        return f"Error executing query: {e}"

# Example usage
if __name__ == "__main__":
    result = fetch_data_from_prompt("How many records are there in the employee table?")
    print("Fetched Data:", result)
