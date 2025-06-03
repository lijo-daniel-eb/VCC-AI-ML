import subprocess
import json

#
data = {
    "model": "llama3",
    "messages": [
        {"role": "user", "content": "What's the capital of Japan?"}
    ]
}

command = [
    'curl',
    '-s',
    '-X', 'POST',
    'http://localhost:11434/api/chat',
    '-H', 'Content-Type: application/json',
    '-d', json.dumps(data)
]

# result as tokens
result = subprocess.run(command, capture_output=True, text=True)

# Print the result as JSON
print(result.stdout)

# Add another print method for human-readable simple output
try:
    # Split the response into individual JSON objects
    responses = result.stdout.strip().split('\n')
    human_readable_output = []

    for response in responses:
        try:
            response_json = json.loads(response)
            if isinstance(response_json, dict) and 'message' in response_json:
                human_readable_output.append(response_json['message'].get('content', ''))
        except json.JSONDecodeError:
            print("Failed to decode a chunk of the response:", response)

    if human_readable_output:
        print("Human-readable output:", " ".join(human_readable_output))
    else:
        print("No valid content found in the response.")

except Exception as e:
    print("An error occurred while processing the response:", str(e))
