"""Step 3 (end): turn the corrected review folders into labels.csv for all images.

Usage: python make_labels.py

review/keep = 1. Everything else = 0: reviewed rejects, spot-checked images
left in spotcheck, and the unreviewed low-ranked images. The spot-check
result estimates how many expressways that last group might be hiding.
"""
import pandas as pd

from config import FILENAMES, REVIEW_DIR

names = FILENAMES.read_text().split("\n")
keep = {p.name for p in (REVIEW_DIR / "keep").iterdir()}
manifest = pd.read_csv("review_manifest.csv").set_index("filename").pool

df = pd.DataFrame({"filename": names})
df["label"] = df.filename.isin(keep).astype(int)
df["pool"] = df.filename.map(manifest).fillna("unreviewed")
df.to_csv("labels.csv", index=False)

print(f"{len(df)} images labelled: {df.label.sum()} expressway, {(df.label == 0).sum()} reject")
for pool in ("top", "spot", "round2"):
    part = df[df.pool == pool]
    if len(part):
        print(f"  {pool:<7} {part.label.sum()} / {len(part)} expressway")
print(f"  unreviewed: {(df.pool == 'unreviewed').sum()} images, labelled 0")
spot = df[df.pool == "spot"]
if spot.label.sum() > 1 and not (df.pool == "round2").any():
    print("Spot-check found several expressways: run review_more.py "
          "(do NOT re-run prelabel.py: it resets the review folders).")
