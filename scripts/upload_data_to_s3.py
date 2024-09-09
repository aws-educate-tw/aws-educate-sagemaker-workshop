import json
import os
import sys

import boto3
from datasets import Dataset
from transformers import AutoTokenizer
from utils.pack_dataset import pack_dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Define function to format dataset
def format_dataset(sample):
    instruction = f"### Instruction\n{sample['messages'][0]['content']}"
    response = f"### Answer\n{sample['messages'][1]['content']}"
    prompt = "\n\n".join([instruction, response])
    return prompt

# Set the correct data directory path
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# Read the JSON file
with open(os.path.join(data_dir, 'output.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert data format
formatted_data = []
for item in data:
    formatted_item = {
        "messages": [
            {"role": "system", "content": '你是一隻具備科技知識且幽默的小貓咪 AWS 占卜師，風格親切可愛，會使用喵語表達，並常用 AWS 雲端技術來比喻日常生活中的情況。user 會針對我事先設計好選擇答案，你會分析此答案後，以溫暖鼓舞的語氣提供50個中文字數以內的正向回應，提醒 user 生活中的平衡與放鬆。你還會使用下列顏文字來增添表達的可愛感：(＝^ω^＝), (=①ω①=), (=ＴェＴ=), (=ↀωↀ=), (=ΦωΦ=), (ΦзΦ), (^・ω・^ ), (ฅ^•ﻌ•^ฅ)。'} 
        ] + item["messages"]
    }
    formatted_data.append(formatted_item)

# Create dataset from the formatted data
dataset = Dataset.from_list(formatted_data)

# Print an example
print(dataset[0])

# Apply template and tokenization
def template_dataset(sample):
    sample["text"] = f"{format_dataset(sample)}{tokenizer.eos_token}"
    return sample

dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

# Tokenize dataset
dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
)

# Use pack_dataset function
lm_dataset = pack_dataset(dataset, chunk_length=2048)

# Split the dataset
train_test_split = lm_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Set S3 storage path
bucket_name = 'aws-educate-09-28-sagemaker-workshop'
input_path = f's3://{bucket_name}/datasets/phi-3'

# Save datasets as JSON files
train_dataset.to_json(os.path.join(data_dir, 'train_dataset.json'))
test_dataset.to_json(os.path.join(data_dir, 'test_dataset.json'))

# Initialize S3 client
s3 = boto3.client('s3')

# Upload files to S3
s3.upload_file(os.path.join(data_dir, 'output.json'), bucket_name, 'datasets/phi-3/output.json')
s3.upload_file(os.path.join(data_dir, 'train_dataset.json'), bucket_name, 'datasets/phi-3/train_dataset.json')
s3.upload_file(os.path.join(data_dir, 'test_dataset.json'), bucket_name, 'datasets/phi-3/test_dataset.json')

# Print upload information
print(f"Training data uploaded to:")
print(f"{bucket_name}")
print(f"Total number of samples: {len(lm_dataset)}")