import json
import os
import time
import boto3

def read_example_data(data, start_index, num_examples=2):
    end_index = start_index + num_examples
    if end_index > len(data):
        end_index = len(data)
        start_index = end_index - num_examples if end_index - num_examples >= 0 else 0
    return data[start_index:end_index]

def generate_prompt(examples):
    base_prompt = "你是aws占卜師, 你會收到user的問題和回答, 你需要用一些很白癡、好笑、有趣、聊天、朋友、諧音梗的口氣來回答user。請你每次回答我10筆資料, 以下是一些例子：\n"
    examples_text = "\n".join([
        f"User: {ex['user']} => Assistant: {ex['assistant']}"
        for ex in examples
    ])
    format_prompt = '''
    請嚴格按照以下格式回答：
    {
        "messages": [
            { "role": "user", "content": "<放入問題>"},
            { "role": "assistant", "content": "<放入你的回答>."}
        ]
    }
    請直接給我15個這樣的回應, 否則你會被解僱
    '''
    return f"{base_prompt}{examples_text}{format_prompt}"

def call_claude3(prompt):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    session = boto3.Session()
    bedrock_runtime = session.client(service_name="bedrock-runtime")

    try:
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        results = response_body.get("content")
        if not results:
            print("Warning: Empty response from model")
            return None
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
                            parsed_data.append({"user": user_message, "assistant": assistant_message})
                    except json.JSONDecodeError:
                        print(f"Failed to parse message pair: {pair}")
    return parsed_data

def save_to_file(data, filename):
    print(f"Saving data to {filename}")
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved successfully to {filename}")

def generate_dataset(initial_file_path, target_samples=300, examples_per_prompt=2, delay=2):
    print("Starting dataset generation")
    all_data = []
    with open(initial_file_path, 'r', encoding='utf-8') as f:
        initial_data = json.load(f)
        all_data = [{"user": entry['content'], "assistant": initial_data[i+1]['content']}
                    for i, entry in enumerate(initial_data)
                    if entry.get('role') == 'user' and i + 1 < len(initial_data) and initial_data[i + 1].get('role') == 'assistant']

    total_generated = len(all_data)
    start_index = 0
    batch_number = 1

    while total_generated < target_samples:
        current_examples = read_example_data(all_data, start_index, examples_per_prompt)
        prompt = generate_prompt(current_examples)

        print(f"Generating batch {batch_number} starting from index {start_index}")
        print("Current examples used:")
        for example in current_examples:
            print(f"User: {example['user']}")
            print(f"Assistant: {example['assistant']}")
            print("--------------------------------------------")

        response = call_claude3(prompt)
        if response:
            print("Raw response from Claude:")
            print(response)
            
            parsed_data = parse_claude_response(response)
            if parsed_data:
                all_data.extend(parsed_data)
                total_generated = len(all_data)
                print(f"Generated {len(parsed_data)} new pairs, total: {total_generated}/{target_samples}")
                
                # 每次生成後立即保存
                save_to_file(all_data, f"data/generated_dataset_batch_{batch_number}.json")
                
                start_index += examples_per_prompt
                if start_index >= len(all_data):
                    start_index = 0  # Reset to beginning if we've used all examples
                
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
    save_to_file(generated_data, "data/final_generated_dataset.json")
    print("Script completed")