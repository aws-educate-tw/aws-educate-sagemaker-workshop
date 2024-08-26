import boto3
import json
import time

def read_existing_data(file_path):
    """讀取已有的數據文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    return existing_data

def generate_prompt(existing_data):
    """生成新的 prompt 基於已有的數據"""
    base_prompt = "你是aws占卜師, 你會收到user的問題和回答, 你需要用一些很白癡、好笑、有趣、聊天、朋友、諧音梗的口氣來回答user。以下是一些例子：\n"
    examples = "\n".join([f"User: {entry['content']} => Assistant: {entry['assistant']}" for entry in existing_data])
    return f"{base_prompt}{examples}"

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
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"成功保存 {len(data)} 筆數據至 {filename}")

def generate_dataset(file_path, num_samples, delay=2):
    existing_data = read_existing_data(file_path)
    results = []
    batch_size = 10  # 每次模型調用生成的數量
    num_batches = num_samples // batch_size
    file_index = 1

    prompt = generate_prompt(existing_data)

    for batch_num in range(num_batches):
        result = call_claude3(prompt)
        if result:
            new_entries = json.loads(result)  # 假設結果是JSON格式的字符串
            results.extend(new_entries)
        else:
            print("Error occurred, skipping this batch.")

        # 控制API速率，防止超過限制
        time.sleep(delay)

        # 每生成20筆資料（2個批次）就存成一個檔案
        if len(results) >= 20:
            save_to_file(results, file_index)
            results = []  # 清空列表以便存儲新的數據
            file_index += 1

    # 如果最後還有未存儲的數據
    if results:
        save_to_file(results, file_index)

if __name__ == "__main__":
    file_path = "data/output.json"  # 輸入已有數據文件的路徑
    num_samples = 1000  # 目標數量
    generate_dataset(file_path, num_samples)
