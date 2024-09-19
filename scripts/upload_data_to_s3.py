import json
import os
from pathlib import Path

import boto3
from datasets import Dataset
from transformers import AutoTokenizer

# Constants
TOKENIZER_ID = "microsoft/Phi-3.5-mini-instruct"
BUCKET_NAME = 'aws-educate-09-28-sagemaker-workshop-oregon'
S3_PREFIX = 'datasets/phi-3.5-mini-instruct/workshop'

# Define system message
SYSTEM_MESSAGE = """你是一隻具備科技知識且幽默的小貓咪 AWS 占卜師，風格親切可愛，會使用喵語表達，並常用 AWS 雲端技術來比喻日常生活中的情況。user 會針對我事先設計好選擇答案，你會分析此答案後，以溫暖鼓舞的語氣提供50個中文字數以內的正向回應，提醒 user 生活中的平衡與放鬆。你還會使用下列顏文字來增添表達的可愛感：(＝^ω^＝), (=①ω①=), (=ＴェＴ=), (=ↀωↀ=), (=ΦωΦ=), (ΦзΦ), (^・ω・^ ), (ฅ^•ﻌ•^ฅ)。"""

def format_dataset_chatml(sample):
    return {
        "text": f"<|system|>\n{SYSTEM_MESSAGE}<|end|>\n<|user|>\n{sample['messages'][0]['content']}<|end|>\n<|assistant|>\n{sample['messages'][1]['content']}<|end|>\n<|endoftext|>"
    }

def process_dataset(data):
    dataset = Dataset.from_list(data)
    dataset_chatml = dataset.map(format_dataset_chatml).map(lambda x: {"text": x["text"]})
    return dataset_chatml

def save_and_upload(dataset, filename, s3_client):
    local_path = Path(data_dir) / filename
    with local_path.open('w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)
    s3_client.upload_file(str(local_path), BUCKET_NAME, f'{S3_PREFIX}/{filename}')

# Set up
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.padding_side = 'right'  # to prevent warnings

data_dir = Path(__file__).parent.parent / 'data'
with (data_dir / 'output.json').open('r') as f:
    data = json.load(f)

# Process dataset
dataset_chatml = process_dataset(data)

# Print an example
print("------- Dataset Example -------")
print(json.dumps(dataset_chatml[0], ensure_ascii=False, indent=2))

# Split the dataset
train_dataset, test_dataset = dataset_chatml.train_test_split(test_size=0.1).values()

# Initialize S3 client and upload
s3 = boto3.client('s3')
save_and_upload(train_dataset, 'train_dataset.json', s3)
save_and_upload(test_dataset, 'test_dataset.json', s3)

# Print upload information
print(f"Training data uploaded to: {BUCKET_NAME}")
print(f"Total number of samples: {len(dataset_chatml)}")