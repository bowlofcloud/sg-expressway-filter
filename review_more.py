"""Step 3, round 2: find expressways the first ranking missed.

Usage: python review_more.py [--n 80]

Trains a quick classifier on the images you already reviewed, re-ranks the
unreviewed ones, and copies the top N into review/round2. Move any
expressway from review/round2 into review/keep, then run make_labels.py.
Your existing review folders are left untouched.
"""
import argparse
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import EMBEDDINGS, FILENAMES, IMAGE_DIR, REVIEW_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="unreviewed images to review")
    args = ap.parse_args()

    X = np.load(EMBEDDINGS)
    names = FILENAMES.read_text().split("\n")
    labels = pd.read_csv("labels.csv").set_index("filename").loc[names]
    reviewed = (labels.pool != "unreviewed").to_numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X[reviewed], labels.label.to_numpy()[reviewed])
    scores = clf.predict_proba(X[~reviewed])[:, 1]
    unrev = np.array(names)[~reviewed]
    picks = unrev[np.argsort(-scores)[:args.n]]

    out = REVIEW_DIR / "round2"
    out.mkdir(exist_ok=True)
    for n in picks:
        shutil.copy2(IMAGE_DIR / n, out / n)

    manifest = pd.read_csv("review_manifest.csv")
    new = pd.DataFrame({"filename": picks, "pool": "round2"})
    pd.concat([manifest[~manifest.filename.isin(picks)], new]) \
      .to_csv("review_manifest.csv", index=False)
    print(f"copied {len(picks)} images to {out}. Move expressways into review/keep, "
          "then run make_labels.py")


if __name__ == "__main__":
    main()
