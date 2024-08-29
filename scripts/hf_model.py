import os
import sys
import logging
import boto3
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from boto3.s3.transfer import TransferConfig

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Retrieve Hugging Face Token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
logger.info(f"Hugging Face Token: {HF_TOKEN}")

s3_bucket = "aws-educate-09-28-sagemaker-workshop"
s3_prefix = "model/mistral/"

# Download the model from Hugging Face to the specified directory
target_directory = "models/stablelm-2-zephyr-1_6b"
os.makedirs(target_directory, exist_ok=True)
logger.info("Starting download...")

model_path = snapshot_download(repo_id="stabilityai/stablelm-2-zephyr-1_6b", repo_type="model", local_dir=target_directory)
logger.info(f"Model downloaded to {model_path}")