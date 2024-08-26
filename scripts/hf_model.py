import os
import sys

import boto3
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(dotenv_path="../.env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
print(HF_TOKEN)
s3_bucket = "aws-educate-09-28-sagemaker-workshop"
s3_prefix = "model/llama3.1/"

# 下載模型
print("開始下載模型...")
model_path = snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8b",
    revision="main",
    allow_patterns=["*.bin", "*.json", "*.model", "*.txt"],
)
print(f"模型下載到: {model_path}")

# 初始化 S3 客戶端
s3 = boto3.client('s3')

# 上傳文件到 S3
print("開始上傳文件到 S3...")
for root, dirs, files in os.walk(model_path):
    for file in files:
        local_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_path, model_path)
        s3_path = os.path.join(s3_prefix, relative_path)
        
        print(f"上傳 {local_path} 到 s3://{s3_bucket}/{s3_path}")
        s3.upload_file(local_path, s3_bucket, s3_path)

print("上傳完成！")

# 生成一個可以在訓練腳本中使用的 S3 URI
s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
print(f"模型已上傳到 S3。在訓練腳本中使用以下 URI: {s3_uri}")