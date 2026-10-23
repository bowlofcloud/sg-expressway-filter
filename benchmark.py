"""Step 8: measure throughput, latency and memory on your machine.

Usage: python benchmark.py
"""
import os
import time

import joblib
import psutil
import torch

from config import CLASSIFIER, list_images
from embed import DEVICE, embed_images, load_model


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def main():
    paths = list_images()
    model, preprocess = load_model()
    embed_images(paths[:32], model, preprocess)               # warm-up
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(); t0 = time.perf_counter()
    X = embed_images(paths, model, preprocess, batch_size=64)
    sync(); secs = time.perf_counter() - t0

    sync(); t1 = time.perf_counter()
    embed_images(paths[:1], model, preprocess, batch_size=1)
    sync(); single = time.perf_counter() - t1

    clf = joblib.load(CLASSIFIER)["clf"]
    t2 = time.perf_counter(); clf.predict_proba(X); clf_secs = time.perf_counter() - t2

    print(f"device: {DEVICE}")
    print(f"throughput: {len(paths) / secs:.1f} img/s ({secs:.1f}s for {len(paths)})")
    print(f"latency, batch 1: {single * 1000:.0f} ms")
    print(f"classifier on all images: {clf_secs * 1000:.2f} ms")
    print(f"CLIP params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"process RAM: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")
    if DEVICE == "cuda":
        print(f"GPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"1M images at this rate: {1e6 / (len(paths) / secs) / 3600:.1f} h")


if __name__ == "__main__":
    main()
