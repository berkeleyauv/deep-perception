import os
import argparse
from dotenv import load_dotenv
from deep_perception.data_download.roboflow_downloader import RoboflowDownloader

def main(args):
    # Load environment variables from .env file
    load_dotenv()

    # Read API key from environment variables
    api_key = os.getenv("ROBOFLOW_API_KEY")

    # Create RoboflowDownloader object
    downloader = RoboflowDownloader(api_key)

    dataset_info = {
        "workspace": args.workspace_name,
        "project": args.project,
        "version": args.version_number,
        "download_format": args.download_format
    }

    # Download file
    downloader.download_dataset(dataset_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_name", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--version_number", type=int, required=True)
    parser.add_argument("--download_format", type=str, required=True)
    args = parser.parse_args()
    main(args)
