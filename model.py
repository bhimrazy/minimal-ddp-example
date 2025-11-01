import torch
from torch import nn
from lightning import LightningModule


class MNIXModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(3 * 28 * 28, 10), nn.ReLU())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["class"]
        y_hat = self(x)  # Forward pass
        loss = self.criterion(y_hat, y)  # Compute loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    # Quick test to verify the model works with a sample batch
    from data_module import MNIXDataModule

    data_module = MNIXDataModule()
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    model = MNIXModel()
    loss = model.training_step(batch)

    print(f"Loss: {loss.item()}")
