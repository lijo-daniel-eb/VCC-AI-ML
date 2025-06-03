import subprocess


# This script runs a local LLM using the Ollama CLI and captures its output.
result = subprocess.run(
    ['ollama', 'run', 'llama3'],
    input='What is the capital of France?\n',
    capture_output=True,
    text=True
)


print(result.stdout)