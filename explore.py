"""Step 1: check the dataset and look at it.

Usage: python explore.py

Checks every image opens, summarises sizes, checks your seed list,
and saves two contact sheets to artifacts/ for you to look at.
"""
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")                    # save to file; works without a display
import matplotlib.pyplot as plt
from PIL import Image

from config import ARTIFACTS, IMAGE_DIR, list_images, read_seeds


def check_images(paths):
    """Return (readable, unreadable) lists and a Counter of image sizes."""
    ok, bad, sizes = [], [], Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:      # verify() closes the file; reopen for size
                sizes[im.size] += 1
            ok.append(p)
        except Exception as e:
            bad.append((p, e))
    return ok, bad, sizes


def contact_sheet(paths, out_file, title, cols=8):
    rows = max(1, -(-len(paths) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))
    for ax in axes.flat if hasattr(axes, "flat") else [axes]:
        ax.axis("off")
    for ax, p in zip(axes.flat if hasattr(axes, "flat") else [axes], paths):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(p.name[:18], fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_file, dpi=90)
    plt.close(fig)
    print(f"saved {out_file}")


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    paths = list_images()
    if not paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}. Copy your images there first.")

    ok, bad, sizes = check_images(paths)
    print(f"{len(paths)} images found, {len(ok)} readable, {len(bad)} unreadable")
    for p, e in bad:
        print(f"  unreadable: {p.name} ({e})")

    print("most common sizes (width x height):")
    for (w, h), n in sizes.most_common(5):
        print(f"  {w}x{h}: {n}")

    seeds = read_seeds()
    names = {p.name for p in ok}
    missing = [s for s in seeds if s not in names]
    print(f"{len(seeds)} seed expressway images listed in seeds.txt, {len(missing)} not found")
    for s in missing:
        print(f"  missing seed: {s} (copy it into data/images/ or fix the name)")

    random.seed(42)
    contact_sheet(random.sample(ok, min(48, len(ok))),
                  ARTIFACTS / "sample_grid.png", "Random sample of 48 images")
    seed_paths = [IMAGE_DIR / s for s in seeds if s in names]
    if seed_paths:
        contact_sheet(seed_paths, ARTIFACTS / "seed_grid.png", "Your known expressway images")

#    print("\nNext: open the grids, then fill in notes.md before Step 2.")


if __name__ == "__main__":
    main()
