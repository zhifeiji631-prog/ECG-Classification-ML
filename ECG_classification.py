
"""PTB-XL multi-label ECG classification with outputs.
Run: python ECG_classification.py --root data --out outputs
Requires: numpy pandas matplotlib scikit-learn wfdb
"""
import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STATS = ["mean", "std", "rms", "ptp", "diff", "abs_p95"]
FEATURES = [f"{lead}_{stat}" for lead in LEADS for stat in STATS]
COLORS = ["#234B75", "#38A3A5", "#E7A55F"]

def load_labels(root):
    # Load ECG metadata and diagnostic code definitions
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)

    # Map diagnostic codes to broader diagnostic classes
    mapping = (scp.loc[scp["diagnostic"] == 1, "diagnostic_class"]
               .dropna().to_dict())

    # Initialize a binary label matrix for multi-label classification
    labels = np.zeros((len(db), len(CLASSES)), dtype=int)

    # Convert each ECG record's diagnostic codes into binary labels
    for i, raw in enumerate(db["scp_codes"]):
        codes = ast.literal_eval(raw) if isinstance(raw, str) else raw
        diagnoses = {mapping[c] for c in codes if c in mapping}
        labels[i] = [int(name in diagnoses) for name in CLASSES]

    return db, labels


def extract_features(signal):
    """Extract six basic waveform statistics from each of twelve leads."""
    if signal.shape[1] != 12:
        raise ValueError(f"Expected 12 ECG leads, got {signal.shape[1]}")
    signal = np.asarray(signal, dtype=float)
    values = np.stack([signal.mean(axis=0), signal.std(axis=0),
        np.sqrt(np.mean(signal ** 2, axis=0)),
        np.ptp(signal, axis=0),
        np.abs(np.diff(signal, axis=0)).mean(axis=0),
        np.percentile(np.abs(signal), 95, axis=0)], axis=1)
    return values.reshape(-1)

def build_features(db, root):
    rows = []
    print(f"Reading {len(db)} ECG waveforms...")
    for i, filename in enumerate(db["filename_lr"], 1):
        record = root / str(filename)
        if not record.with_suffix(".hea").is_file() or not record.with_suffix(".dat").is_file():
            raise FileNotFoundError(f"Missing ECG waveform: {record}.hea / .dat")
        signal, fields = wfdb.rdsamp(str(record))
        if fields["sig_name"] != LEADS:
            raise ValueError(f"Unexpected lead order in {record}: {fields['sig_name']}")
        rows.append(extract_features(signal))
        if i % 2000 == 0:
            print(f"  Processed {i}/{len(db)}")
    return np.asarray(rows)

def verify_splits(db):
    folds = db["strat_fold"].to_numpy()
    if not np.isin(folds, np.arange(1, 11)).all():
        raise ValueError("Unexpected strat_fold values")
    train, val, test = folds <= 8, folds == 9, folds == 10
    for a, b, mask_a, mask_b in [
        ("train", "val", train, val), ("train", "test", train, test),
        ("val", "test", val, test)
    ]:
        patients_a = set(db.loc[mask_a, "patient_id"].dropna())
        patients_b = set(db.loc[mask_b, "patient_id"].dropna())
        overlap = len(patients_a & patients_b)
        print(f"Patient overlap {a}/{b}: {overlap}")
        if overlap:
            raise ValueError("Patient leakage detected")
    if db["patient_id"].isna().any():
        raise ValueError("Missing patient IDs")
    return train, val, test


def scores(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    data = []
    for i, name in enumerate(CLASSES):
        actual, pred = y_true[:, i], predictions[:, i]
        auc = (roc_auc_score(actual, probabilities[:, i])
               if len(np.unique(actual)) == 2 else np.nan)
        data.append({"Class": name, "Precision": precision_score(actual, pred, zero_division=0),
                     "Recall": recall_score(actual, pred, zero_division=0),
                     "F1": f1_score(actual, pred, zero_division=0), "AUROC": auc})
    return pd.DataFrame(data).set_index("Class")


def make_figures(db, labels, masks, results, probabilities, best_model, models, out):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (name, mask) in enumerate(zip(["Train", "Validation", "Test"], masks)):
        ax.bar(x + (i - 1) * .24, labels[mask].mean(axis=0) * 100,
               width=.22, label=name, color=COLORS[i])
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Positive records (%)")
    ax.set_title("Diagnostic prevalence by data split", loc="left")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "01_label_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, (name, score) in enumerate(results.items()):
        ax.bar(x + (i - .5) * .36, score["AUROC"], width=.34,
           label=name, color=["#800000", "#A6A6A6"][i])

    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test AUROC")
    ax.set_title("Model performance by diagnosis", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "02_model_comparison.png", dpi=300)
    plt.close(fig) 

    y_test = labels[masks[2]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, name in enumerate(CLASSES):
        if len(np.unique(y_test[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test[:, i], probabilities[:, i])
        auc = roc_auc_score(y_test[:, i], probabilities[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"Test ROC curves: {best_model}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "03_roc_curves.png", dpi=300)
    plt.close(fig)

    forest = models["Random Forest"]
    importances = np.mean([m.feature_importances_ for m in forest.estimators_], axis=0)
    top = np.argsort(importances)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(np.arange(len(top)), importances[top][::-1], color=COLORS[0])
    ax.set_yticks(np.arange(len(top)), [FEATURES[j] for j in top[::-1]])
    ax.set_xlabel("Mean impurity-based importance across five classifiers")
    ax.set_title("Top 15 ECG features: Random Forest", loc="left")
    fig.tight_layout()
    fig.savefig(out / "04_feature_importance.png", dpi=300)
    plt.close(fig)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_dir / "data")
    parser.add_argument("--out", type=Path, default=project_dir / "outputs")
    args = parser.parse_args()
    root = args.root
    for file in ["ptbxl_database.csv", "scp_statements.csv"]:
        if not (root / file).is_file():
            raise FileNotFoundError(f"Required file missing: {root / file}")
    if not (root / "records100").is_dir():
        raise FileNotFoundError(f"Required waveform folder missing: {root / 'records100'}")
    args.out.mkdir(parents=True, exist_ok=True)

    db, y = load_labels(root)
    masks = verify_splits(db)
    X = build_features(db, root)
    train, val, test = masks
    pd.DataFrame(X, columns=FEATURES, index=db.index).to_csv(args.out / "ecg_features.csv")
    summary = pd.DataFrame({name: {"Records": int(mask.sum()),
                                         "Patients": db.loc[mask, "patient_id"].nunique(),
                                         "Multi_label": int((y[mask].sum(axis=1) > 1).sum())}
                            for name, mask in zip(["Train", "Validation", "Test"], masks)}).T
    summary.to_csv(args.out / "split_summary.csv")

    models = {"Logistic Regression": make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        "Random Forest": OneVsRestClassifier(RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)) }

    X = np.where(np.isfinite(X), X, np.nan)
    # Train-only medians: avoids leaking validation/test statistics.
    medians = np.nanmedian(X[train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)
    X = np.where(np.isnan(X), medians, X)
    best_model, best_auc = None, -np.inf
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X[train], y[train])
        fitted[name] = model
        val_prob = model.predict_proba(X[val])
        val_auc = np.nanmean(scores(y[val], val_prob)["AUROC"])
        print(f"  Validation macro AUROC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_model = val_auc, name

    results = {}
    for name, model in fitted.items():
        results[name] = scores(y[test], model.predict_proba(X[test]))
    table = pd.concat(results, names=["Model", "Class"])
    table.round(4).to_csv(args.out / "model_performance.csv")
    best_prob = fitted[best_model].predict_proba(X[test])
    make_figures(db, y, masks, results, best_prob, best_model, fitted, args.out)
    print("\nData split:\n", summary.to_string())
    print("\nTest performance:\n", table.round(3).to_string())
    print(f"\nBest model by validation AUROC: {best_model}")
    print(f"Saved CSV tables and 4 figures to: {args.out}")


if __name__ == "__main__":
    main()