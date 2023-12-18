from pathlib import Path

class PathFactory:
    def __init__(self, dataset_name:str, bucket:str = "classification"):
        assert bucket in ("classification", "segmentation")

        self.path = Path.home() / "datasets" / dataset_name
        self.shards_path = Path.home() / "shards" / dataset_name

        self.url = f"s3://{bucket}/datasets/{dataset_name}"
        self.shards_url = f"s3://{bucket}/shards/{dataset_name}"
    
    #TODO: add functions to verify, report and create local directories
    #TODO: add function to check the exisitence of bucket in s3
