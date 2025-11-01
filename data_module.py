from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from litdata import StreamingDataLoader, StreamingDataset


class MNIXDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        rank_zero_info("Setting up dataset...")
        # Initialize StreamingDataset from the optimized data directory
        # It's important that this is initialized here to work properly with DDP
        self.dataset = StreamingDataset("mnix_data")

    def train_dataloader(self):
        return StreamingDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
