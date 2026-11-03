import logging
import numpy as np
import torch
import urllib
from pathlib import Path
import urllib.request
import pandas as pd
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", (
    "Switch to a GPU runtime: Runtime -> Change runtime type -> T4 GPU"
)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device)[None, :, None, None]
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device)[None, :, None, None]


def normalize(x):
    return (x - MEAN) / STD  # apply to every crop before it enters the network\n,


def load(split):
    # returns: images (N, 3, 32, 32) as floats in [0, 1], labels (N), both on the GPU
    fname = f"{split}-00000-of-00001.parquet"
    URL = f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{fname}"
    fpath = Path.cwd() / fname
    if not fpath.exists():
        logging.log(logging.INFO, f"File {fpath} missing. Starting the download")
        urllib.request.urlretrieve(URL, fname)
        # response = requests.get(URL, timeout=120)
        # response.raise_for_status()
        # table = pd.read_parquet(io.BytesIO(response.content))

    table = pd.read_parquet(fpath)
    images = np.stack(
        [
            np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
            for row in table["img"]
        ]
    )
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(
        device
    ), torch.tensor(table["label"].values).to(device)
