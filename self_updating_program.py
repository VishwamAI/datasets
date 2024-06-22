import os
import requests
import argparse
from huggingface_hub import HfApi

# Function to authenticate with Hugging Face API
def authenticate_huggingface(api_key):
    api = HfApi()
    api.set_access_token(api_key)
    return api

# Function to check for dataset updates
def check_for_updates(api, dataset_name):
    # Get the latest dataset info from Hugging Face
    dataset_info = api.dataset_info(dataset_name)
    return dataset_info

# Function to download the latest dataset
def download_dataset(api, dataset_name, output_dir):
    # Download the dataset files
    api.download_dataset(dataset_name, output_dir=output_dir)

# Main function to handle the self-updating process
def self_update(api_key, dataset_name, output_dir):
    # Authenticate with Hugging Face API
    api = authenticate_huggingface(api_key)

    # Check for updates
    dataset_info = check_for_updates(api, dataset_name)

    # Download the latest dataset
    download_dataset(api, dataset_name, output_dir)
    print(f"Dataset {dataset_name} has been updated and downloaded to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-updating program for datasets.")
    parser.add_argument("--api_key", type=str, required=True, help="Hugging Face API key.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to update.")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Directory to save the updated dataset.")
    args = parser.parse_args()

    self_update(args.api_key, args.dataset_name, args.output_dir)
