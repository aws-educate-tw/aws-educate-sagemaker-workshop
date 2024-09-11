import json
import os
import time
import boto3
from boto3 import client
from botocore.config import Config


def read_example_data(data, start_index, num_examples=2):
    end_index = start_index + num_examples
    if end_index > len(data):
        end_index = len(data)
        start_index = end_index - num_examples if end_index - num_examples >= 0 else 0
    return data[start_index:end_index]


def generate_prompt(examples):
    base_prompt = "你是aws占卜師, 你要產生user會遇到的情境並針對情境做assitant回答, 你需要用一些很好笑、有趣、聊天、朋友、諧音梗、有aws教育意義、神秘、微微暗示結果、時不時喵一下的口氣或是(＝^ω^＝)(=①ω①=)(=ＴェＴ=)(=ↀωↀ=)(=ΦωΦ=)(ΦзΦ)(^・ω・^ )(ฅ^•ﻌ•^ฅ)來回答user, 絕對不可以表現的調侃或是輕浮來回答user。請你每次回答我15筆資料, 以下是一些例子：\n"
    examples_text = str(examples)
    format_prompt = '''
請嚴格按照以下格式回答：
{
    "messages": [
        {
            "role": "user",
            "content": "<放入情境>"
        },
        {
            "role": "assistant",
            "content": "<放入回答>"
        }
    ]
}
請直接給我15個這樣的回應'''
    print(f"example_text: {examples_text}")
    return f"{base_prompt}{examples_text}{format_prompt}"


def call_claude3(prompt):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    session = boto3.Session(profile_name="workshop",
                            region_name="us-west-2")
    config = Config(read_timeout=1000)
    bedrock_runtime = session.client(
        service_name="bedrock-runtime", config=config)

    try:
        print("Calling Bedrock model")
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        results = response_body.get("content")
        if not results:
            print("Warning: Empty response from model")
            return None
        print("Received response from Bedrock model")
        return results
    except Exception as e:
        print(f"Error calling Bedrock model: {e}")
        return None


def parse_claude_response(response):
    parsed_data = []
    for item in response:
        if item['type'] == 'text':
            message_pairs = item['text'].split('\n\n')
            for pair in message_pairs:
                if '"messages":' in pair:
                    try:
                        messages = json.loads(pair)['messages']
                        if len(messages) == 2:
                            user_message = messages[0]['content']
                            assistant_message = messages[1]['content']
                            parsed_data.append(
                                {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": user_message
                                        },
                                        {
                                            "role": "assistant",
                                            "content": assistant_message
                                        }
                                    ]
                                }
                            )
                    except json.JSONDecodeError:
                        print(f"Failed to parse message pair: {pair}")
                        'raw message: xxx \n \n xxx'
    return parsed_data


def save_to_file(data, filename):
    print(f"Saving data to {filename}")
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved successfully to {filename}")


def generate_dataset(initial_file_path, target_samples=3500, examples_per_prompt=2, delay=15):
    print("Starting dataset generation")
    all_data = []

    total_generated = len(all_data)
    start_index = 60
    batch_number = 1
    with open(initial_file_path, 'r', encoding='utf-8') as f:
        initial_data = json.load(f)
        all_data = [entry for entry in initial_data]

    while total_generated < target_samples:
        current_examples = read_example_data(
            all_data, start_index, examples_per_prompt)
        prompt = generate_prompt(current_examples)

        print(f"Generating batch {
              batch_number} starting from index {start_index}")
        # print("--------------Current prompt used--------------")
        # print(prompt)
        # print("--------------------------------------------")
        # for example in current_examples:
        #     print(f"User: {example['user']}")
        #     print(f"Assistant: {example['assistant']}")
        #     print("--------------------------------------------")

        response = call_claude3(prompt)
        if response:
            print("Received valid response from Claude model")
            parsed_data = parse_claude_response(response)
            if parsed_data:
                all_data.extend(parsed_data)
                total_generated = len(all_data)
                print(f"Generated total: {total_generated}/{target_samples}")

                save_to_file(all_data, f"data/output.json")

                start_index += examples_per_prompt
                if start_index >= len(all_data):
                    print("Reached end of data, starting over.")
                    break

                batch_number += 1
            else:
                print("Failed to parse Claude's response")
        else:
            print("No valid response from Claude model, retrying after delay.")

        time.sleep(delay)

        if total_generated >= target_samples:
            print("Reached target sample limit, ending generation.")
            break

    print("Dataset generation completed")

    return all_data


if __name__ == "__main__":
    initial_file_path = "data/output.json"
    generated_data = generate_dataset(initial_file_path)
    print("Script completed")
