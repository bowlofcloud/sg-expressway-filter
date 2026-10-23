"""Shared settings: paths, model choice and CLIP prompts."""
from pathlib import Path

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "data" / "images"
SEEDS_FILE = ROOT / "seeds.txt"          # your 13 known expressway filenames
ARTIFACTS = ROOT / "artifacts"
REVIEW_DIR = ROOT / "review"

EMBEDDINGS = ARTIFACTS / "embeddings.npy"
FILENAMES = ARTIFACTS / "filenames.txt"
CLASSIFIER = ARTIFACTS / "classifier.joblib"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Smallest common CLIP: fastest. Only upgrade if F1 < 0.80, and record the cost.
MODEL, WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"

POS_PROMPTS = [
    "a photo of a wide multi-lane expressway with road shoulders",
    "a photo of a highway with an overhead gantry and exit sign",
]
NEG_PROMPTS = [
    "a photo of a road junction with traffic lights",
    "a photo of a yellow box junction",
    "a photo of a bus stop by the road",
    "a photo of a narrow city street with shops and buildings",
    "a photo of a residential road with double yellow lines",
    "a photo that is not of a road",
]


def list_images(folder=IMAGE_DIR):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_seeds():
    if not SEEDS_FILE.exists():
        return []
    lines = (l.strip() for l in SEEDS_FILE.read_text().splitlines())
    return [l for l in lines if l and not l.startswith("#")]
