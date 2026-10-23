# Singapore Expressway Image Filter

Keeps only images of Singapore expressways from a noisy set of 999 road images.

**Approach:** frozen CLIP ViT-B-32 image embeddings + an L2-regularised logistic
regression (a "linear probe"), trained on labels gathered with two rounds of
active learning. Everything runs locally: no API calls, no images leave the machine.

| | Result |
| --- | --- |
| Test F1 (338 held-out images, 34 expressways) | **0.875** (precision 0.93, recall 0.82) |
| Cross-validated F1 (training set) | 0.757 |
| Throughput / latency (RTX 3050 Laptop GPU) | 55.7 img/s / 49 ms per image |
| GPU memory / RAM | 0.77 GB / 1.2 GB |
| Cost per image | $0 (local) |
| Human labelling | 430 images reviewed in 34 min, labels for all 999 |

---

## 1. Pipeline

```
image -> CLIP ViT-B-32 (frozen) -> 512-d embedding -> logistic regression -> keep if p >= 0.82
```

1. **Embed once.** Every image is encoded by CLIP once and cached (`embeddings.npy`).
   All later steps reuse the cache, so the only expensive computation runs once.
2. **Label with active learning** (34 min of human time, see section 3).
3. **Train a linear probe**: logistic regression (a binomial GLM, 513 parameters)
   with class-balanced weights. C and the decision threshold are chosen by 5-fold
   cross-validation on the training set only.
4. **Filter**: `filter.py` embeds a folder and keeps images above the threshold.

## 2. Why this model

Given that the brief rewards simple, fast solutions over complex ones with marginal gains, and the rejection rules (traffic lights, bus stops, yellow boxes, double yellow lines) are semantic, a general-purpose vision-language embedding captures most of them in a cost-effective manner.

| Option | Speed / cost | Decision |
| --- | --- | --- |
| Zero-shot CLIP (prompts only) | Same cost as chosen model, no labels needed | Tried as baseline: **F1 0.38**, not good enough |
| **CLIP + linear probe** | ~18 ms/img at batch 64; probe adds ~0.02 ms/img; $0 | **Chosen**: F1 0.88 test / 0.76 CV, same inference cost as zero-shot |
| Fine-tuning a CNN / CLIP | Hours of training, GPU-heavy, overfits on ~100 positives | Rejected: complexity for uncertain gain |
| Segmentation / lane / traffic-light detection | Heavier models plus hand-written rules on top | Rejected: many rules to engineer, slower |
| VLM (Gemini, GPT, Qwen-VL) on every image | Seconds per image; paid API or a large local GPU; data leaves machine if hosted | Rejected as primary model; possible fallback for uncertain cases only |

**Key result:** zero-shot classification and the linear probe share the same frozen CLIP encoder, which takes about 18 ms per image. The probe adds only about 0.02 ms. Training it therefore raised F1 from 0.38 to 0.88 without making filtering any slower or more expensive. The only added cost was a once-off 34 minutes of labelling.

## 3. Data and labelling (minimising human effort)

**Data findings (Step 1):** 999 images, all 860x480, mixed viewpoints (overhead
bridge, bus front camera with "Next Stop" overlay, ground level). Expressways are
rare (~10%). Filenames are sequential video frames, so neighbours are near-duplicates.
13 images were known expressways ("seeds").

Because positives are rare, a random sample would have yielded too few to learn from.
Instead, labelling was targeted:

| Round | How images were chosen | Reviewed | Time | Expressways found |
| --- | --- | --- | --- | --- |
| 1 | Top 250 by zero-shot score + similarity to seeds | 250 | 30 min (incl. spot-check) | 90 (36%) |
| 1 | Random spot-check of the remaining 749 | 100 | | 3 (3%) |
| 2 | Top 80 of the unreviewed, ranked by a probe trained on round-1 labels | 80 | 4 min | 10 (12.5%) |
| **Total** | | **430 (43%)** | **34 min** | **103** |

- The 569 never-reviewed images were labelled "reject". The round-2 ranking found
  expressways ~4x as often as random checking, which justifies stopping there.
- CLIP's zero-shot pre-sort agreed with human labels on only 54% of the top 250,
  mostly by over-keeping wide arterial roads (97 of 116 disagreements). This is
  further evidence that prompts alone cannot capture the rejection rules.

## 4. Evaluation

**Split:** train 661 / test 338 images, stratified and **grouped by blocks of 20
consecutive frames**, so near-duplicate frames never appear in both sets (avoiding
leakage). Cross-validation inside training uses the same grouping.

| Run | Method | Test F1 | Precision | Recall | CV F1 |
| --- | --- | --- | --- | --- | --- |
| All runs | Zero-shot CLIP | 0.38 | 0.32 | 0.47 | - |
| 1 | Probe, C grid 0.1-10 (best C=10, at grid edge) | 0.780 | 0.67 | 0.94 | 0.730 |
| 2 | Probe, C grid 1-300 (best C=30) | 0.784 | 0.73 | 0.85 | 0.754 |
| 3 | Probe, after 2 label corrections | **0.875** | **0.93** | **0.82** | 0.757 |

**Error review.** All 16 test errors from run 2 were re-checked at full size against
the labelling rules. 2 labels were wrong (both rushed in round 1) and were corrected:
one 0 -> 1, one 1 -> 0. A label was changed only when a specific reason could be named.

**Reading the results honestly:**
- The jump from 0.784 to 0.875 is larger than two corrections can explain directly.
  Most of it comes from the re-tuned threshold (0.68 -> 0.82). The label changes also
  slightly shifted the stratified group split. CV F1 barely moved (0.754 -> 0.757).
- With only 34 test positives, a few images move test F1 by ~0.05. **Expected
  performance on new footage is likely between the CV F1 (0.76) and the test F1 (0.88).**
- Only test errors were re-checked, so corrections could only help the score. This is
  reported for transparency.
- ~9 expressways (~1.6% of unreviewed images) are estimated to remain labelled
  "reject". They were ranked low by models similar to the final one, so the final
  model likely misses them too, and reported recall may be overstated slightly.

**Main failure mode: single vs double yellow edge lines.** In Singapore, a single
yellow edge line is normal on expressways, while double yellow lines mark ordinary
roads. 9 of 11 false keeps had double yellow lines, and the remaining misses were
expressways with single yellow edge lines.
At CLIP's 224x224 input, the gap between two thin lines is a few pixels, so the
embedding encodes "yellow edge line" without reliably telling single from double.

## 5. Speed, scalability and cost

Measured on an NVIDIA RTX 3050 Laptop GPU (4 GB), plugged in, after warm-up.

| Measure | Value |
| --- | --- |
| Throughput (batch 64) | 55.7 img/s (999 images in 17.9 s) |
| Latency (batch 1) | 49 ms |
| Linear probe | 21 ms for all 999 images (~0.02 ms/img) |
| GPU peak memory | 0.77 GB |
| Process RAM | 1.2 GB |
| Model size | CLIP 151M params (image encoder ~88M runs per image) + 513-param probe |
| API cost | $0 |
| 1M images | ~5 h on this laptop |

**Scaling:** throughput was the same with and without warm-up, so the bottleneck is
single-threaded image decoding on the CPU, not the GPU. Parallel data loading
(PyTorch `DataLoader` with 4-8 workers) and half-precision inference are the first
steps and should raise throughput several-fold with no change in accuracy. Embedding
is embarrassingly parallel, so it also scales across GPUs.

**Privacy:** all inference is local; no images are sent to third parties.

## 6. Limitations and next steps

- **Double yellow lines are the main source of errors.** The model often cannot tell
  a single yellow edge line from a double one, because the gap between the lines is
  only a few pixels at CLIP's 224x224 input. A cheap fix would be a conventional
  computer-vision check: crop the road edge from the full-resolution image, highlight
  yellow pixels with a colour mask, and count the parallel lines. This was not built
  or timed, but a check like this would likely add only a few milliseconds per image.
- **Borderline cases could go to a vision-language model.** Images with a probability
  close to the threshold could be sent to a VLM with a checklist prompt. This was not
  done: the probe already meets the target, and a VLM would add cost, latency and,
  if hosted externally, privacy risks.
- **The test set is small.** With only 34 expressways in the test set, a few images
  can shift F1 by about 0.05. More labelled footage, especially from new locations,
  would give a more reliable estimate.
---

## How to run

### Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Choose the PyTorch CUDA build to match your driver (see pytorch.org). On CPU-only
machines, plain `pip install -r requirements.txt` works, just more slowly.

Put images in `data/images/` and known expressway filenames in `seeds.txt`.

### Run order

| Step | Command | Output |
| --- | --- | --- |
| 1. Explore | `python explore.py` | image checks, `artifacts/sample_grid.png` |
| 2. Embed | `python embed.py` | `artifacts/embeddings.npy` |
| 3. Rank and review | `python prelabel.py`, then sort `review/` folders by hand | `review_manifest.csv` |
| 3. Labels | `python make_labels.py` | `labels.csv` |
| 3. Round 2 | `python review_more.py`, sort `review/round2`, re-run `make_labels.py` | updated `labels.csv` |
| 4-6. Train | `python train.py` | `split.csv`, `artifacts/classifier.joblib` |
| Errors | `python errors.py` | list of test-set mistakes |
| 8. Benchmark | `python benchmark.py` | speed and memory figures |
| 9. Filter | `python filter.py data/images --out kept.txt` | `kept.txt` (kept 100 of 999) |

### Project structure

```
config.py        paths, model choice, CLIP prompts
explore.py       Step 1: data checks and contact sheets
embed.py         Step 2: CLIP embeddings (cached)
prelabel.py      Step 3: rank images, create review folders
review_more.py   Step 3: second active-learning round
make_labels.py   Step 3: review folders -> labels.csv
train.py         Steps 4-6: group split, zero-shot baseline, linear probe
errors.py        list test-set errors for review
benchmark.py     Step 8: throughput, latency, memory
filter.py        Step 9: the filtering tool
notes.md         working notes and full error analysis
```
