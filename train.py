"""Steps 4-6: split, zero-shot baseline, linear probe.

Usage: python train.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     cross_val_predict)

from config import CLASSIFIER, EMBEDDINGS, FILENAMES

FRAMES_PER_GROUP = 20   # consecutive frames are near-duplicates: keep them together


def frame_groups(filenames):
    """'road (209).jpg' -> 209 // 20 = group 10."""
    return filenames.str.extract(r"(\d+)")[0].astype(int).to_numpy() // FRAMES_PER_GROUP


def best_threshold(y, scores):
    """Threshold that maximises F1 on the given (training) data."""
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / (p + r + 1e-9)
    return t[f1[:-1].argmax()]


def report(name, y_true, y_pred):
    print(f"\n{name}: test F1 = {f1_score(y_true, y_pred):.3f}")
    print(confusion_matrix(y_true, y_pred))


def main():
    X_all = np.load(EMBEDDINGS)
    idx = {n: i for i, n in enumerate(FILENAMES.read_text().split("\n"))}

    # Step 4: stratified GROUP split (~1/3 test), so near-duplicate frames
    # never sit on both sides
    labels = pd.read_csv("labels.csv")
    groups = frame_groups(labels.filename)
    outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    tr_i, te_i = next(outer.split(labels, labels.label, groups))
    train_df, test_df = labels.iloc[tr_i], labels.iloc[te_i]
    g_tr = groups[tr_i]
    pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]) \
      .to_csv("split.csv", index=False)
    X_tr, y_tr = X_all[[idx[f] for f in train_df.filename]], train_df.label.to_numpy()
    X_te, y_te = X_all[[idx[f] for f in test_df.filename]], test_df.label.to_numpy()
    print(f"train {len(y_tr)} ({y_tr.sum()} pos) | test {len(y_te)} ({y_te.sum()} pos)")

    # Step 5: zero-shot baseline, threshold chosen on train only
    zs = pd.read_csv("zeroshot.csv").set_index("filename").zs_score
    thr_zs = best_threshold(y_tr, zs[train_df.filename].to_numpy())
    report("zero-shot", y_te, (zs[test_df.filename].to_numpy() >= thr_zs).astype(int))

    # Step 6: linear probe, C and threshold chosen by CV on train only
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    search = GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=2000),
                          {"C": [1, 3, 10, 30, 100, 300]}, scoring="f1", cv=cv)
    search.fit(X_tr, y_tr, groups=g_tr)
    clf = search.best_estimator_
    oof = cross_val_predict(clf, X_tr, y_tr, groups=g_tr, cv=cv,
                            method="predict_proba")[:, 1]
    thr = best_threshold(y_tr, oof)
    print(f"\nbest C {search.best_params_['C']}, CV F1 {search.best_score_:.3f}, threshold {thr:.3f}")
    report("linear probe", y_te, (clf.predict_proba(X_te)[:, 1] >= thr).astype(int))

    joblib.dump({"clf": clf, "threshold": float(thr)}, CLASSIFIER)
    print(f"\nsaved {CLASSIFIER}")


if __name__ == "__main__":
    main()
