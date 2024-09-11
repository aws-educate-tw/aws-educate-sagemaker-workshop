import boto3
import json
import time
from utils.prompt_helper import get_prompt

def call_claude3(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"
    session = boto3.Session(profile_name="cmd")
    session = boto3.Session()
    bedrock_runtime = session.client(service_name="bedrock-runtime")

    try:
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        results = response_body.get("content")[0].get("text")
        return results
    except Exception as e:
        print(f"Error calling Bedrock model: {e}")
        return None

def save_to_file(data, file_index):
    filename = f"generated_dataset_{file_index}.json"
    with open(filename, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

def generate_dataset(prompt, num_samples, delay=2):
    results = []
    batch_size = 10  
    num_batches = num_samples // batch_size
    file_index = 1

    for batch_num in range(num_batches):
        result = call_claude3(prompt)
        if result:
            results.extend(json.loads(result))
        else:
            print("Error occurred, skipping this batch.")


        time.sleep(delay)

        
        if len(results) >= 20:
            save_to_file(results, file_index)
            results = [] 
            file_index += 1

    
    if results:
        save_to_file(results, file_index)

if __name__ == "__main__":
    prompt = get_prompt("prompt/prompt.txt")
    num_samples = 1000 
    generate_dataset(prompt, num_samples)

