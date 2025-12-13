import os
import json
import joblib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.20
    n_splits: int = 5

    # Column names expected in CSVs
    drop_cols: Tuple[str, ...] = ("No.",)
    label_col: str = "Label"

    categorical_cols: Tuple[str, ...] = ("Source", "Destination", "Protocol", "Info")
    numeric_cols: Tuple[str, ...] = ("Time", "Length")


# ----------------------------
# Data loading / cleaning
# ----------------------------
def fix_data_errors(df: pd.DataFrame) -> pd.DataFrame:
    # Robust numeric parsing
    for col in ["Time", "Length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df


def load_one_csv(path: str, label: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fix_data_errors(df)

    # Drop columns if present
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Ensure required columns exist (create if missing)
    for c in cfg.categorical_cols + cfg.numeric_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[cfg.label_col] = label
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def load_dataset(datasets: Dict[str, Union[str, List[str]]], cfg: Config) -> pd.DataFrame:
    frames = []
    for label, files in datasets.items():
        if isinstance(files, list):
            for f in files:
                frames.append(load_one_csv(f, label, cfg))
        else:
            frames.append(load_one_csv(files, label, cfg))
    data = pd.concat(frames, ignore_index=True)
    return data


# ----------------------------
# Pipeline building
# ----------------------------
def build_preprocessor(cfg: Config) -> ColumnTransformer:
    # We use OrdinalEncoder (NOT LabelEncoder) and fit ONCE on training data.
    cat_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    num_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, list(cfg.categorical_cols)),
            ("num", num_pipe, list(cfg.numeric_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def build_model_pipeline(cfg: Config, categorical_count: int) -> ImbPipeline:
    """
    Preprocess -> SMOTENC -> Scale -> LinearSVC
    SMOTENC needs categorical feature indices in the *preprocessed* array.
    With our ColumnTransformer order: [cats..., nums...]
    so categorical indices are range(0, categorical_count).
    """
    preprocessor = build_preprocessor(cfg)

    smote = SMOTENC(
        categorical_features=list(range(categorical_count)),
        random_state=cfg.random_state
    )

    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",   # helps with imbalance even before SMOTE
        dual="auto",
        random_state=cfg.random_state
    )

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", smote),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
    return pipe


# ----------------------------
# Evaluation helpers
# ----------------------------
def save_confusion_matrix(y_true, y_pred, labels, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return cm


def metrics_dict(y_true, y_pred, average="weighted", pos_label=None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if pos_label is not None:
        out["precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label))
        out["recall"] = float(recall_score(y_true, y_pred, pos_label=pos_label))
        out["f1"] = float(f1_score(y_true, y_pred, pos_label=pos_label))
    else:
        out["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    return out


# ----------------------------
# Main routine
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts", help="Where to save models/reports")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(args.outdir, exist_ok=True)

    # Your dataset mapping (same as your old code)
    datasets = {
        "Normal": "Normal.csv",
        "FlexiSpy_B": "FlexiSpy.csv",
        "MobileSpy_B": "MobileSpy.csv",
        "UMobix_B": "UMobix.csv",
        "TheWispy_B": "TheWispy.csv",
        "Mspy_B": ["Mspy1.csv", "Mspy2.csv"],
        "FlexiSpy_C": "FlexiSpy_Installation.csv",
        "MobileSpy_C": "MobileSpy_Installation.csv",
        "UMobix_C": "UMobix_Installation.csv",
        "TheWispy_C": "TheWispy_Installation.csv",
        "Mspy_C": "Mspy_Installation.csv"
    }

    data = load_dataset(datasets, cfg)

    # Split X/y
    X = data.drop(columns=[cfg.label_col])
    y = data[cfg.label_col].astype(str)

    # Common CV
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    # ============================
    # 1) Binary task: Normal vs Spyware
    # ============================
    y_bin = y.where(y == "Normal", other="Spyware")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin,
        test_size=cfg.test_size,
        stratify=y_bin,
        random_state=cfg.random_state
    )

    bin_pipe = build_model_pipeline(cfg, categorical_count=len(cfg.categorical_cols))

    # Proper CV (SMOTE happens inside each fold)
    bin_cv_scores = cross_val_score(bin_pipe, X_train, y_train, cv=skf, scoring="accuracy")
    print("\n[BINARY] CV scores:", bin_cv_scores)
    print("[BINARY] Mean CV accuracy: %.2f%%" % (100 * bin_cv_scores.mean()))

    # Fit and evaluate
    bin_pipe.fit(X_train, y_train)
    y_pred = bin_pipe.predict(X_test)

    bin_metrics = metrics_dict(y_test, y_pred, pos_label="Spyware")
    print("\n[BINARY] Test metrics:", bin_metrics)
    print("\n[BINARY] Classification report:\n", classification_report(y_test, y_pred, digits=4))

    bin_labels = ["Normal", "Spyware"]
    bin_cm = save_confusion_matrix(
        y_test, y_pred,
        labels=bin_labels,
        title="Confusion Matrix - Binary (Normal vs Spyware)",
        out_png=os.path.join(args.outdir, "cm_binary.png")
    )

    joblib.dump(bin_pipe, os.path.join(args.outdir, "model_binary.joblib"))

    # ============================
    # 2) Multiclass task: all labels
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state
    )

    multi_pipe = build_model_pipeline(cfg, categorical_count=len(cfg.categorical_cols))

    multi_cv_scores = cross_val_score(multi_pipe, X_train, y_train, cv=skf, scoring="accuracy")
    print("\n[MULTI] CV scores:", multi_cv_scores)
    print("[MULTI] Mean CV accuracy: %.2f%%" % (100 * multi_cv_scores.mean()))

    multi_pipe.fit(X_train, y_train)
    y_pred = multi_pipe.predict(X_test)

    multi_metrics = metrics_dict(y_test, y_pred, average="weighted")
    print("\n[MULTI] Test metrics:", multi_metrics)
    print("\n[MULTI] Classification report:\n", classification_report(y_test, y_pred, digits=4))

    labels_sorted = sorted(y.unique().tolist())
    multi_cm = save_confusion_matrix(
        y_test, y_pred,
        labels=labels_sorted,
        title="Confusion Matrix - Multiclass",
        out_png=os.path.join(args.outdir, "cm_multiclass.png")
    )

    joblib.dump(multi_pipe, os.path.join(args.outdir, "model_multiclass.joblib"))

    # Save JSON summary + confusion matrices table
    summary = {
        "binary": {
            "cv_scores": bin_cv_scores.tolist(),
            "cv_mean_accuracy": float(bin_cv_scores.mean()),
            "test_metrics": bin_metrics,
            "labels": bin_labels,
        },
        "multiclass": {
            "cv_scores": multi_cv_scores.tolist(),
            "cv_mean_accuracy": float(multi_cv_scores.mean()),
            "test_metrics": multi_metrics,
            "labels": labels_sorted,
        }
    }
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(bin_cm, index=bin_labels, columns=bin_labels).to_csv(os.path.join(args.outdir, "cm_binary.csv"))
    pd.DataFrame(multi_cm, index=labels_sorted, columns=labels_sorted).to_csv(os.path.join(args.outdir, "cm_multiclass.csv"))

    print(f"\nDone. Artifacts saved in: {args.outdir}/")
    print(" - model_binary.joblib, model_multiclass.joblib")
    print(" - cm_binary.png, cm_multiclass.png")
    print(" - metrics_summary.json, cm_binary.csv, cm_multiclass.csv")


if __name__ == "__main__":
    main()
