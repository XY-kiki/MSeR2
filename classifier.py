"""Fold-wise two-stage training and evaluation for MSeR2.

Stage 1: an auxiliary MLP supplies differentiable supervision to the graph
encoder using only the training pairs of the current outer fold.

Stage 2: the auxiliary MLP is removed, the encoder is frozen, and XGBoost is
trained on the concatenated final drug and disease representations.

The GIP matrices are loaded from the dataset directory as fixed precomputed
features.  This script does not recompute GIP kernels or bandwidths per fold.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, __version__ as XGBOOST_VERSION

from emb import (
    DEFAULT_ENCODER_CONFIG,
    build_heterograph,
    load_data,
    load_pairs,
    prepare_features,
    save_embedding_csv,
    set_seed,
    train_encoder_for_fold,
    validate_pairs,
)


BASE_DIR = Path(__file__).parent

EXPERIMENT_CONFIG = {
    "datasets": ["B-dataset", "C-dataset", "F-dataset"],
    "data_root": BASE_DIR / "dataset",
    "sample_root": BASE_DIR / "samples",
    "result_root": BASE_DIR / "results_mser2",
    "seed": 1234,
    "n_splits": 10,
    "save_fold_embeddings": True,
    "save_models": True,
    "encoder": deepcopy(DEFAULT_ENCODER_CONFIG),
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 15.0,
        "reg_lambda": 8.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    },
}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return torch.device("cuda:0" if requested == "cuda" else "cpu")


def set_experiment_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def build_pair_features(
    drug_embedding: np.ndarray,
    disease_embedding: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Construct x_ij = [z_dr_i || z_di_j] without extra feature operators."""
    drug_part = drug_embedding[pairs[:, 0]]
    disease_part = disease_embedding[pairs[:, 1]]
    return np.concatenate((drug_part, disease_part), axis=1).astype(np.float32)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def build_xgboost_model(
    config: Mapping[str, object],
    seed: int,
    use_cuda: bool,
) -> XGBClassifier:
    parameters = dict(config)
    parameters["random_state"] = seed
    major_version = int(XGBOOST_VERSION.split(".", maxsplit=1)[0])
    if major_version >= 2:
        # XGBoost >= 2.0 uses device together with tree_method="hist".
        parameters["device"] = "cuda" if use_cuda else "cpu"
    elif use_cuda:
        # Compatibility with XGBoost 1.x.
        parameters["tree_method"] = "gpu_hist"
    return XGBClassifier(**parameters)


def _serializable_config(config: Mapping[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, Mapping):
            serializable[key] = _serializable_config(value)
        else:
            serializable[key] = value
    return serializable


def _save_fold_membership(
    save_path: Path,
    pairs: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    membership = np.full(len(pairs), "train", dtype=object)
    membership[test_indices] = "test"
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(pairs), dtype=np.int64),
            "drug_id": pairs[:, 0],
            "disease_id": pairs[:, 1],
            "label": labels,
            "partition": membership,
        }
    )
    if len(train_indices) + len(test_indices) != len(pairs):
        raise RuntimeError("The saved fold does not cover every sample exactly once.")
    frame.to_csv(save_path, index=False)


def _save_predictions(
    save_path: Path,
    pairs: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    original_indices: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "sample_index": original_indices,
            "drug_id": pairs[:, 0],
            "disease_id": pairs[:, 1],
            "label": labels,
            "probability": probabilities,
            "prediction": predictions,
        }
    ).to_csv(save_path, index=False)


def run_single_dataset(
    dataset_name: str,
    config: Mapping[str, object],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["seed"])
    data_dir = Path(config["data_root"]) / dataset_name
    sample_dir = Path(config["sample_root"]) / dataset_name
    dataset_result_dir = Path(config["result_root"]) / dataset_name
    dataset_result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"Dataset: {dataset_name}")
    print(f"Torch device: {device}")
    print(
        "GIP protocol: DrugGIP.csv and DiseaseGIP.csv are reused as fixed "
        "precomputed inputs in every fold."
    )
    print("=" * 72)

    data = load_data(data_dir)
    pairs, labels = load_pairs(sample_dir)
    validate_pairs(pairs, int(data["num_drug"]), int(data["num_disease"]))

    graph = build_heterograph(data).to(device)
    features = prepare_features(data, device)

    splitter = StratifiedKFold(
        n_splits=int(config["n_splits"]),
        shuffle=True,
        random_state=seed,
    )
    fold_metrics = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(pairs, labels), start=1
    ):
        fold_seed = seed + fold
        set_experiment_seed(fold_seed)
        fold_dir = dataset_result_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFold {fold}/{config['n_splits']}")
        print(
            f"  encoder training pairs: {len(train_indices)} | "
            f"held-out test pairs: {len(test_indices)}"
        )

        encoder, embeddings, best_epoch, validation_auc = train_encoder_for_fold(
            graph=graph,
            features=features,
            train_pairs=pairs[train_indices],
            train_labels=labels[train_indices],
            config=config["encoder"],
            device=device,
            seed=fold_seed,
            verbose=True,
        )

        # The encoder is now frozen; XGBoost receives final embeddings only.
        train_features = build_pair_features(
            embeddings["drug"], embeddings["disease"], pairs[train_indices]
        )
        test_features = build_pair_features(
            embeddings["drug"], embeddings["disease"], pairs[test_indices]
        )
        classifier = build_xgboost_model(
            config=config["xgboost"],
            seed=fold_seed,
            use_cuda=device.type == "cuda",
        )
        classifier.fit(train_features, labels[train_indices])
        probabilities = classifier.predict_proba(test_features)[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        metrics = calculate_metrics(
            labels[test_indices], probabilities, predictions
        )
        metrics.update(
            {
                "fold": fold,
                "encoder_best_epoch": best_epoch,
                "encoder_validation_auc": validation_auc,
            }
        )
        fold_metrics.append(metrics)

        _save_fold_membership(
            fold_dir / "split.csv",
            pairs,
            labels,
            train_indices,
            test_indices,
        )
        _save_predictions(
            fold_dir / "test_predictions.csv",
            pairs[test_indices],
            labels[test_indices],
            probabilities,
            predictions,
            test_indices,
        )

        if bool(config["save_fold_embeddings"]):
            embedding_dir = fold_dir / "embeddings"
            save_embedding_csv(
                embeddings["drug"],
                embedding_dir / "drug_embedding.csv",
                "drug",
            )
            save_embedding_csv(
                embeddings["disease"],
                embedding_dir / "disease_embedding.csv",
                "disease",
            )
            save_embedding_csv(
                embeddings["protein"],
                embedding_dir / "protein_embedding.csv",
                "protein",
            )

        if bool(config["save_models"]):
            torch.save(encoder.state_dict(), fold_dir / "encoder_state.pt")
            classifier.save_model(str(fold_dir / "xgboost_model.json"))

        print(
            "  "
            f"Acc. {metrics['acc']:.4f} | F1 {metrics['f1']:.4f} | "
            f"AUC {metrics['auc']:.4f} | AUPR {metrics['aupr']:.4f}"
        )

        # Release fold-specific GPU memory before the next encoder is created.
        del encoder, classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_frame = pd.DataFrame(fold_metrics).sort_values("fold")
    fold_frame.to_csv(dataset_result_dir / "fold_metrics.csv", index=False)

    reported_metrics = ["acc", "f1", "auc", "aupr"]
    summary = pd.DataFrame(
        {
            "metric": reported_metrics,
            "mean": [fold_frame[name].mean() for name in reported_metrics],
            "std": [fold_frame[name].std(ddof=1) for name in reported_metrics],
        }
    )
    summary["std_percent"] = 100.0 * summary["std"]
    summary.to_csv(dataset_result_dir / "summary_metrics.csv", index=False)

    with (dataset_result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_serializable_config(config), handle, indent=2, ensure_ascii=False)

    print("\nSummary (mean ± standard deviation):")
    for row in summary.itertuples(index=False):
        print(f"  {row.metric:4s}: {row.mean:.4f} ± {row.std:.4f}")
    return fold_frame, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fold-wise MSeR2 encoder training and XGBoost evaluation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=EXPERIMENT_CONFIG["datasets"],
        help="Dataset directory names under --data-root and --sample-root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=EXPERIMENT_CONFIG["data_root"],
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=EXPERIMENT_CONFIG["sample_root"],
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=EXPERIMENT_CONFIG["result_root"],
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save fold-specific encoder and XGBoost parameters.",
    )
    parser.add_argument(
        "--no-save-embeddings",
        action="store_true",
        help="Do not save fold-specific node embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deepcopy(EXPERIMENT_CONFIG)
    config["datasets"] = args.datasets
    config["data_root"] = (
        args.data_root if args.data_root.is_absolute() else Path.cwd() / args.data_root
    )
    config["sample_root"] = (
        args.sample_root
        if args.sample_root.is_absolute()
        else Path.cwd() / args.sample_root
    )
    config["result_root"] = (
        args.result_root
        if args.result_root.is_absolute()
        else Path.cwd() / args.result_root
    )
    config["save_models"] = not args.no_save_models
    config["save_fold_embeddings"] = not args.no_save_embeddings

    device = choose_device(args.device)
    set_experiment_seed(int(config["seed"]))
    for dataset_name in config["datasets"]:
        run_single_dataset(dataset_name, config, device)
    print("\nAll MSeR2 experiments finished successfully.")


if __name__ == "__main__":
    main()
