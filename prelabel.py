"""Step 3: rank all images, then set up review folders.

Usage: python prelabel.py [--top 250] [--spot 100]

Ranks every image by zero-shot CLIP score + similarity to your seeds, then:
  review/keep, review/reject  <- the top-ranked images, pre-sorted by CLIP
  review/spotcheck            <- a random sample of the low-ranked rest
Fix the top folders by dragging images between keep and reject. In
spotcheck, move any expressway into review/keep. Then run make_labels.py.
"""
import argparse
import random
import shutil

import numpy as np
import open_clip
import pandas as pd
import torch

from config import (EMBEDDINGS, FILENAMES, IMAGE_DIR, MODEL, NEG_PROMPTS,
                    POS_PROMPTS, REVIEW_DIR, read_seeds)
from embed import DEVICE, load_model


def zero_shot_scores(X):
    """P(expressway) = summed softmax probability of the positive prompts."""
    model, _ = load_model()
    tokenizer = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        T = model.encode_text(tokenizer(POS_PROMPTS + NEG_PROMPTS).to(DEVICE))
        T = torch.nn.functional.normalize(T, dim=-1).cpu().numpy()
    logits = 100 * X @ T.T
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return probs[:, :len(POS_PROMPTS)].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=250, help="top-ranked images to review")
    ap.add_argument("--spot", type=int, default=100, help="random low-ranked images to spot-check")
    ap.add_argument("--force", action="store_true", help="wipe existing review folders")
    args = ap.parse_args()

    X = np.load(EMBEDDINGS)
    names = FILENAMES.read_text().split("\n")
    seeds = set(read_seeds())
    seed_idx = [i for i, n in enumerate(names) if n in seeds]

    df = pd.DataFrame({"filename": names, "zs_score": zero_shot_scores(X)})
    df["seed_sim"] = (X @ X[seed_idx].T).max(1)
    # Average the two rankings so neither score's scale dominates.
    df["rank_score"] = (df.zs_score.rank(pct=True) + df.seed_sim.rank(pct=True)) / 2
    df = df.sort_values("rank_score", ascending=False)
    df.to_csv("zeroshot.csv", index=False)

    top = df.head(args.top)
    random.seed(42)
    rest = df.filename.iloc[args.top:].tolist()
    spot = random.sample(rest, min(args.spot, len(rest)))

    if (REVIEW_DIR / "keep").exists() and not args.force:
        raise SystemExit("review/ already has your labelling work. Re-run with --force "
                         "to wipe it, or use review_more.py to add images instead.")
    shutil.rmtree(REVIEW_DIR, ignore_errors=True)
    for sub in ("keep", "reject", "spotcheck"):
        (REVIEW_DIR / sub).mkdir(parents=True)
    for n, zs in zip(top.filename, top.zs_score):
        sub = "keep" if (n in seeds or zs >= 0.5) else "reject"
        shutil.copy2(IMAGE_DIR / n, REVIEW_DIR / sub / n)
    for n in spot:
        shutil.copy2(IMAGE_DIR / n, REVIEW_DIR / "spotcheck" / n)

    pd.DataFrame({"filename": list(top.filename) + spot,
                  "pool": ["top"] * len(top) + ["spot"] * len(spot)}) \
      .to_csv("review_manifest.csv", index=False)

    n_keep = len(list((REVIEW_DIR / "keep").iterdir()))
    print(f"top {len(top)}: {n_keep} pre-sorted to keep, {len(top) - n_keep} to reject")
    print(f"spotcheck: {len(spot)} random images from the other {len(rest)}")
    print("Now correct the folders by hand (time yourself), then run make_labels.py")


if __name__ == "__main__":
    main()
