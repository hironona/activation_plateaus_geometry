import argparse
import os
from huggingface_hub import HfApi

def upload_checkpoints(repo_id, checkpoint_dir, token=None):
    """
    Uploads the contents of the checkpoint directory to a Hugging Face Hub repository.
    """
    api = HfApi()

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return

    print(f"Uploading files from '{checkpoint_dir}' to repo '{repo_id}'...")
    
    try:
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Checkpoints to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g., username/repo_name)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token (optional if logged in via CLI)")
    
    args = parser.parse_args()

    upload_checkpoints(args.repo_id, args.checkpoint_dir, args.token)
