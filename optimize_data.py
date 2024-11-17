import os
import torch
from litdata import optimize, StreamingDataLoader, StreamingDataset


def optimize_fn(index):
    return {
        "index": index,
        "image": torch.rand(3, 28, 28),
        "class": torch.randint(0, 10, (1,)).item(),
    }


if __name__ == "__main__":
    optimize(
        fn=optimize_fn,
        inputs=list(range(1000)),
        output_dir="mnix_data",
        num_workers=4,
        chunk_bytes="64MB",
    )
    dataset = StreamingDataset(
        input_dir="mnix_data",
        shuffle=True,
        drop_last=True,
    )
    print(f"Number of samples: {len(dataset)}")
    train_dataloader = StreamingDataLoader(
        dataset,
        batch_size=4,
        pin_memory=True,
        num_workers=os.cpu_count() or 1,
    )
    print(f"Number of batches: {len(train_dataloader)}")

    first_batch = next(iter(train_dataloader))
    print(first_batch)
