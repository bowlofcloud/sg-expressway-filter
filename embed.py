"""Step 2: embed every image with CLIP once and cache the vectors.

Usage: python embed.py
"""
import time

import numpy as np
import open_clip
import torch
from PIL import Image

from config import ARTIFACTS, EMBEDDINGS, FILENAMES, MODEL, WEIGHTS, list_images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=WEIGHTS, device=DEVICE)
    model.eval()
    return model, preprocess


@torch.no_grad()
def embed_images(paths, model, preprocess, batch_size=64):
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch_size]]).to(DEVICE)
        feats = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    paths = list_images()
    model, preprocess = load_model()
    t0 = time.perf_counter()
    X = embed_images(paths, model, preprocess)
    print(f"{len(paths)} images in {time.perf_counter() - t0:.1f}s on {DEVICE}, shape {X.shape}")
    ARTIFACTS.mkdir(exist_ok=True)
    np.save(EMBEDDINGS, X)
    FILENAMES.write_text("\n".join(p.name for p in paths))
