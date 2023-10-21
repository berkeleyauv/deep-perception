from typing import Dict

from deep_perception.data_download.downloader import Downloader
from roboflow import Roboflow

class RoboflowDownloader(Downloader):
    """
    Download dataset from Roboflow.
    """
    def __init__(self, api_key: str):
        """
        Initialize the Roboflow downloader.
        """
        self.api_key = api_key
        self.roboflow_instance = Roboflow(api_key=self.api_key)

    def download_dataset(self, dataset_identifier: Dict):
        """
        Download dataset from the given identifier.
        """
        print("Downloading dataset from Roboflow")
        project = self.roboflow_instance.workspace(dataset_identifier["workspace"]).project(dataset_identifier["project"])
        dataset = project.version(dataset_identifier["version"]).download(dataset_identifier["download_format"], location="./data")