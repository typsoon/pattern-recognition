import logging
import numpy as np
import torch
from pathlib import Path
import urllib.request
import pandas as pd
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,  # Replace notebook logging handlers so progress messages remain visible.
)

device = "cuda" if torch.cuda.is_available() else "cpu"  # Probe GPU access in this process.
# This starter keeps the full dataset on the GPU and expects GPU training.
assert device == "cuda", (
    "Switch to a GPU runtime: Runtime -> Change runtime type -> T4 GPU"
)

# CIFAR-10 channel statistics have shape (1, 3, 1, 1) to broadcast over NCHW batches.
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device)[None, :, None, None]
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device)[None, :, None, None]


def normalize(x):
    # Standardize each color channel using the dataset's fixed mean and deviation.
    return (x - MEAN) / STD


def load(split):
    # returns: images (N, 3, 32, 32) as floats in [0, 1], labels (N), both on the GPU
    fname = f"{split}-00000-of-00001.parquet"
    URL = f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{fname}"
    fpath = Path.cwd() / fname
    if not fpath.exists():  # Download only when a local copy is unavailable.
        logging.info("File %s missing. Starting the download", fpath)
        urllib.request.urlretrieve(URL, fpath)  # Save the requested split's parquet file.

    table = pd.read_parquet(fpath)  # Image cells contain encoded bytes rather than pixels.
    # Decode every image as RGB and stack the arrays into (N, 32, 32, 3).
    images = np.stack(
        [
            # Wrap encoded bytes as a file, decode them, and expose the RGB pixel array.
            np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
            for row in table["img"]
        ]
    )
    # Reorder NHWC to NCHW, map uint8 pixels into [0, 1], and move pixels and labels
    # to the same device. Channel standardization happens later in normalize().
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(
        device
    ), torch.tensor(table["label"].values).to(device)
