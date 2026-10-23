"""List test-set mistakes and which labelling pool each image came from."""
import joblib, numpy as np, pandas as pd
from config import CLASSIFIER, EMBEDDINGS, FILENAMES

X = np.load(EMBEDDINGS)
idx = {n: i for i, n in enumerate(FILENAMES.read_text().split("\n"))}
b = joblib.load(CLASSIFIER)
test = pd.read_csv("split.csv").query("split == 'test'")
test["p"] = b["clf"].predict_proba(X[[idx[f] for f in test.filename]])[:, 1]
test["pred"] = (test.p >= b["threshold"]).astype(int)
print(test[test.pred != test.label].sort_values("p", ascending=False)
    [["filename", "label", "pred", "p", "pool"]].to_string(index=False))