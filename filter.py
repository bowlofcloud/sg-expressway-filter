"""Step 9: keep only Singapore-expressway images.

Usage: python filter.py data/images --out kept.txt
"""
import argparse
from pathlib import Path

import joblib

from config import CLASSIFIER, list_images
from embed import embed_images, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="kept.txt")
    args = ap.parse_args()

    paths = list_images(args.folder)
    model, preprocess = load_model()
    bundle = joblib.load(CLASSIFIER)
    probs = bundle["clf"].predict_proba(embed_images(paths, model, preprocess))[:, 1]
    kept = [p.name for p, s in zip(paths, probs) if s >= bundle["threshold"]]
    Path(args.out).write_text("\n".join(kept))
    print(f"kept {len(kept)} of {len(paths)} -> {args.out}")


if __name__ == "__main__":
    main()
