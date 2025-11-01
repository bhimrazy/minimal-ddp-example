from lightning import Trainer
from data_module import MNIXDataModule
from model import MNIXModel


def train():
    data_module = MNIXDataModule()
    model = MNIXModel()
    trainer = Trainer(max_epochs=10, strategy="ddp", devices=2)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
