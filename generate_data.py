import boto3
import json
from util.prompt_helper import get_prompt
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
    bedrock_runtime = session.client(service_name="bedrock-runtime")

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    results = response_body.get("content")[0].get("text")
    
    return results

if __name__ == "__main__":
    prompt = get_prompt("prompt/prompt.txt")
    result = call_claude3(prompt)
    print(result)
