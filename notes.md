# Working notes

## Setup
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), plugged in
- Software: Python 3.13.9, PyTorch 2.14.1+cu126, open_clip ViT-B-32 (laion2b_s34b_b79k)

## Step 1: what the data looks like
- 999 images, all readable, all 860x480 (single source, likely video frames)
- 13 known expressway seeds: road (209)-(215), (232)-(237); all overhead-bridge views
- Expressways look rare: ~3-5 of a 48-image random sample, so ~5-10% (~50-100 of 999)
- Common noise: signalised junctions, yellow box markings, bus stops, shophouse
  streets, roadworks, residential roads
- Mixed viewpoints: overhead bridge, bus front camera ("Next Stop" text overlay),
  ground level
- Filenames are sequential video frames: neighbours are near-duplicates

## Decisions from Step 1
- Positives are rare, so a random 300-image sample would give too few. Instead:
  rank all images by zero-shot + seed similarity, label the top ~250, and
  spot-check ~100 random from the rest
- Split train/test by groups of 20 consecutive frames to avoid leakage
- Single yellow edge lines are normal on SG expressways; only double yellow = reject
- If the random spot-check finds ~0-1 expressways, label the unchecked rest as 0
  and group-split all 999, so the test set matches the real class ratio
- Spot-check found 3/100 expressways (too many to ignore), so added a second
  active-learning round: a classifier trained on round-1 labels ranked the
  unreviewed images far better than zero-shot did

## Labelling rules
- Keep (1): 3-6 lanes per direction, shoulders, ramps, ERP gantries, expressway signs
- Reject (0): traffic lights, signalised junction, yellow box, bus stop/bay/lane,
  double yellow lines, clearly urban/local road, non-road
- Ambiguous: 0

## Numbers for the write-up
- Embedding time (Step 2): 17.5 s for 999 images on cuda (~57 img/s, first run,
  includes start-up; proper benchmark in Step 8)

### Labelling (human effort)
- Round 1: 350 images reviewed in 30 min (~5 s/image)
    - Positives found in top 250: 90; in random 100 of the rest: 3
      (3% -> ~19 estimated missed among the 649 unreviewed)
    - Zero-shot pre-labels flipped: 116 of top 250 (54% agreement at the 0.5
      cut-off); 97 keep->reject (CLIP over-keeps wide arterial roads), 19 reject->keep
- Round 2: 80 images reviewed in 4 min (~3 s/image)
    - 10 / 80 expressways (12.5% hit rate vs 3% random spot-check: ~4x)
- Total: 430 images reviewed in 34 min (~43% of data), labels for all 999
- Final labels: 103 expressway, 896 reject; remaining 569 unreviewed labelled 0
  (~9 expressways estimated still missed, ~1.6%; likely also missed by the final
  model, so reported recall may be slightly overstated, worst case ~8 points)

### Results (first run, C grid 0.1-10)
- Split: train 661 (69 pos) | test 338 (34 pos), grouped by 20-frame blocks
- Zero-shot test F1: 0.381 (precision 0.32, recall 0.47; 34 false keeps, 18 missed)
- Linear probe test F1: 0.780 (precision 0.67, recall 0.94; 16 false keeps, 2 missed)
    - best C = 10 (edge of grid), CV F1 0.730, threshold 0.572

### Results (second run, C grid 1-300)
- Same split: train 661 (69 pos) | test 338 (34 pos)
- Zero-shot test F1: 0.381 (unchanged; doesn't depend on C)
- Linear probe test F1: 0.784 (precision 0.73, recall 0.85; 11 false keeps, 5 missed)
    - best C = 30 (inside grid), CV F1 0.754 (up from 0.730), threshold 0.682
    - Widening the C grid cut false keeps 16 -> 11 but added 3 misses; test F1 +0.004
    - Still just below 0.80 -> reviewed the 16 test errors (next section)

## Error analysis (test set, second run: C=30, threshold 0.682)
Re-checked all 16 test errors at full size against the labelling rules.
A label was changed only when a specific reason could be named.

| Image | Label | Model | p | Verdict | Reason |
| --- | --- | --- | --- | --- | --- |
| road (11) | 0 | 1 | 0.99 | label correct | only one lane, and there are double yellow lines in the road beside on the right-hand side |
| road (523) | 0 | 1 | 0.89 | label correct | there are double yellow lines on the left side of the road |
| road (909) | 0 | 1 | 0.85 | label correct | there are only two lanes in the road |
| road (465) | 0 | 1 | 0.84 | label correct | there are double yellow lines on the left side of the road, although less visible at CLIP resolution |
| road (189) | 0 | 1 | 0.83 | label correct | there are double yellow lines on the left side of the road |
| road (13) | 0 | 1 | 0.81 | label correct | there are double yellow lines on the left side of the road |
| road (178) | 0 | 1 | 0.80 | label correct | there are double yellow lines on the left side of the road, although less visible at CLIP resolution |
| road (634) | 0 | 1 | 0.72 | label wrong -> 1 | 4-lanes expressway, no double yellow lines at the side of the road, no bus stops, no yellow box, no lights or junctions. Rushed during round 1 |
| road (56) | 0 | 1 | 0.72 | label correct | there are double yellow lines on the left side of the road |
| road (949) | 0 | 1 | 0.72 | label correct | there are double yellow lines on the left side of the road |
| road (636) | 0 | 1 | 0.68 | label correct | although it is a 4-lane road, there are double yellow lines on the left side of the road |
| road (172) | 1 | 0 | 0.68 | label correct | there are no double yellow lines on the left side of the road, only single yellow line. it is a 6-lanes expressway with no lights or junctions. |
| road (51) | 1 | 0 | 0.63 | label correct | there are no double yellow lines on the left side of the road. it is a 4-lanes expressway with no signal lights or junctions. |
| road (232) | 1 | 0 | 0.56 | label correct | there are no double yellow lines on the left side of the road, only single yellow line. it is a 3-lanes expressway with no lights or junctions. |
| road (476) | 1 | 0 | 0.51 | label wrong -> 0 | It is not an expressway as there are visible signal lights and junction. Rushed during round 1 |
| road (519) | 1 | 0 | 0.16 | label correct | there are no double yellow lines on the left side of the road, only single yellow line. it is a 3-lanes expressway with no lights or junctions. |

Verdict = "label correct" (genuine model error) or "label wrong" (corrected).
Summary: 2 of 16 labels corrected (road 634: 0 -> 1; road 476: 1 -> 0; both
rushed in round 1); remaining 14 are genuine model errors. The two corrections
cancel out, so the total stays at 103 expressways.
Only test errors were re-checked, so corrections could only help the score;
reported for transparency.
Common patterns: the model confuses single and double yellow edge lines
(9 of 11 false keeps had double yellow lines).
Expressways with a single yellow edge line were rejected (misses), and roads
with double yellow lines were kept (false keeps). At CLIP's 224x224 input,
the gap between two thin lines is only a few pixels, so the embedding likely
encodes "yellow edge line" without distinguishing single from double.

### Results (third run, after 2 label corrections from error review)
- Same split sizes: train 661 (69 pos) | test 338 (34 pos); the label changes
  can shift which frame blocks the stratified group split puts in train/test
- Zero-shot test F1: 0.382 (threshold re-tuned on the new training data)
- Linear probe test F1: 0.875 (precision 0.93, recall 0.82; 2 false keeps, 6 missed)
    - best C = 30, CV F1 0.757, threshold 0.820 (up from 0.682)
    - Test gain mostly from the stricter threshold, not the corrections
      themselves; CV F1 nearly unchanged (0.754 -> 0.757). Report both.
    - With 34 test positives, a few images move F1 by ~0.05; expected
      performance on new footage likely between CV (0.76) and test (0.88)

### Benchmark (Step 8, RTX 3050 Laptop 4 GB, plugged in, warm-up excluded)
- Throughput: 55.7 img/s (17.9 s for 999); latency batch-1: 49 ms
- Classifier: 21 ms for 999 images (~0.02 ms/img) -> negligible; CLIP is ~all compute
- Memory: GPU peak 0.77 GB, process RAM 1.2 GB
- Model: CLIP ViT-B-32, 151M params total (image encoder ~88M) + 513-param probe
- Cost: $0/image, fully local, no data leaves the machine
- Scale: 1M images ~5 h on this laptop; bottleneck is single-threaded image
  decoding (same speed with or without warm-up), not the GPU

### Step 9: end-to-end check
- filter.py: kept 100 of 999 (vs 103 labelled) -> kept.txt
  (sanity check only: includes training images, so not an accuracy measure)