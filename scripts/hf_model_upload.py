import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload files to the Hugging Face Hub."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the local file or directory to upload",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path where the file(s) will be stored in the repo",
        default=None,
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="ID of the repository in the format 'username/repo-name'",
    )
    
    return parser.parse_args()


def upload_file_to_hub(args):
    """Upload file to Hugging Face Hub."""
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Set default path_in_repo if not provided
    if args.path is None:
        args.path = os.path.basename(args.file)

    print(f"Uploading {args.file} to {args.repo_id}...")
    
    # Upload the file
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=args.path,
        repo_id=args.repo_id
)
    
    print(f"Successfully uploaded to {args.repo_id}/{args.path}")


def main():
    """Main function to run the CLI."""
    args = parse_args()
    upload_file_to_hub(args)


if __name__ == "__main__":
    main()
