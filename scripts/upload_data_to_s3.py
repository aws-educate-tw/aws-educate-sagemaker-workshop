import json
import os

import boto3
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer_id = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings


# Define system message
system_message = """你是一隻具備科技知識且幽默的小貓咪 AWS 占卜師，風格親切可愛，會使用喵語表達，並常用 AWS 雲端技術來比喻日常生活中的情況。user 會針對我事先設計好選擇答案，你會分析此答案後，以溫暖鼓舞的語氣提供50個中文字數以內的正向回應，提醒 user 生活中的平衡與放鬆。你還會使用下列顏文字來增添表達的可愛感：(＝^ω^＝), (=①ω①=), (=ＴェＴ=), (=ↀωↀ=), (=ΦωΦ=), (ΦзΦ), (^・ω・^ ), (ฅ^•ﻌ•^ฅ)。"""

def format_dataset_chatml(sample):
    return {
        "text": f"<|system|>\n{system_message}<|end|>\n<|user|>\n{sample['messages'][0]['content']}<|end|>\n<|assistant|>\n{sample['messages'][1]['content']}<|end|>\n<|endoftext|>"
    }

def remove_messages(sample):
    sample.pop('messages', None)
    return sample

# Set the correct data directory path
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Read the JSON file
with open(os.path.join(data_dir, 'output.json'), 'r') as f:
    data = json.load(f)

# Create dataset from the data
dataset = Dataset.from_list(data)

# Apply new format and remove messages
dataset_chatml = dataset.map(format_dataset_chatml).map(remove_messages)

# Print an example
print("------- Dataset Example -------")
print(json.dumps(dataset_chatml[0], ensure_ascii=False, indent=2))

# Split the dataset
train_test_split = dataset_chatml.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Set S3 storage path
bucket_name = 'aws-educate-09-28-sagemaker-workshop-oregon'
input_path = f's3://{bucket_name}/datasets/phi-3'

# Save datasets as JSON files
with open(os.path.join(data_dir, 'train_dataset.json'), 'w', encoding='utf-8') as f:
    json.dump(train_dataset.to_list(), f, ensure_ascii=False, indent=2)

with open(os.path.join(data_dir, 'test_dataset.json'), 'w', encoding='utf-8') as f:
    json.dump(test_dataset.to_list(), f, ensure_ascii=False, indent=2)

# Initialize S3 client
s3 = boto3.client('s3')

# Upload files to S3
s3.upload_file(os.path.join(data_dir, 'train_dataset.json'), bucket_name, 'datasets/phi-3/train_dataset.json')
s3.upload_file(os.path.join(data_dir, 'test_dataset.json'), bucket_name, 'datasets/phi-3/test_dataset.json')

# Print upload information
print(f"Training data uploaded to: {bucket_name}")
print(f"Total number of samples: {len(dataset_chatml)}")