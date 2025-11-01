# Minimal DDP Example with LitData and PyTorch Lightning

[![CI](https://github.com/bhimrazy/minimal-ddp-example/actions/workflows/ci.yml/badge.svg)](https://github.com/bhimrazy/minimal-ddp-example/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository provides a minimal example of using [LitData](https://github.com/Lightning-AI/litData) with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for distributed data parallel (DDP) training.

## Overview

LitData is a library for efficient data processing and streaming, particularly useful for large datasets. This example demonstrates:

- **Data Optimization**: Converting raw data into LitData's optimized streaming format
- **Streaming Dataset**: Loading data efficiently during training
- **DDP Training**: Distributed training across multiple GPUs using PyTorch Lightning

The example uses a synthetic "MNIX" dataset (MNIST-like) for simplicity.

## Prerequisites

- Python 3.10+
- PyTorch
- CUDA (optional, for GPU training)
- Multiple GPUs (for DDP demonstration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bhimrazy/minimal-ddp-example.git
cd minimal-ddp-example
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare the Data

First, optimize your raw data into LitData's format:

```bash
python optimize_data.py
```

This script generates synthetic image data and stores it in the `mnix_data/` directory as optimized chunks.

### 2. Train the Model

Run the training script with DDP:

```bash
# The script is configured for 2 devices by default
python train.py
```

## Contributing

Feel free to open issues or submit pull requests to improve this example!

## License

This project is open source and available under the [MIT License](LICENSE).
