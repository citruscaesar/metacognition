from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS, EVAL_DATALOADERS
)

class DataLoaderFactory:
    def __init__(self, 
            batch_size: int, 
            num_workers: int) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self, dataset) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = dataset,  
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            pin_memory = True,
            shuffle = True) # type: ignore

    def eval_dataloader(self, dataset) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            num_workers = 1,
            shuffle = False
        )
    
    def streaming_train_dataloader(self, dataset) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )

    def streaming_eval_dataloader(self, dataset) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size
        )