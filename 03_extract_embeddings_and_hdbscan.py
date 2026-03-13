"""
Standalone embedding extraction and HDBSCAN clustering script.

Expected input files:
- outputs/{RUN_NAME}/preprocess/cycle_features.npy
- outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
- outputs/{RUN_NAME}/training/encoder.keras

Saved artifacts:
- outputs/{RUN_NAME}/clustering/embeddings.npy
- outputs/{RUN_NAME}/clustering/embedding_metadata.csv
- outputs/{RUN_NAME}/clustering/cluster_assignments.csv
- outputs/{RUN_NAME}/clustering/hdbscan_model.joblib
- outputs/{RUN_NAME}/clustering/clustering_summary.json
- outputs/{RUN_NAME}/clustering/pca_scatter.png
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import hdbscan
import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

from excel_export_utils import export_stage_workbook

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Editable configuration
# ============================================================================
class EmbeddingConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "test_dataset_260312_preprocess_v2"

    PREPROCESS_ROOT = OUTPUT_ROOT / RUN_NAME / "preprocess"
    TRAINING_ROOT = OUTPUT_ROOT / RUN_NAME / "training"
    CLUSTERING_ROOT = OUTPUT_ROOT / RUN_NAME / "clustering"
    REQUIRED_METADATA_COLUMNS = [
        "sample_id",
        "subject_id",
        "recording_id",
        "valid_flag",
    ]

    PREDICT_BATCH_SIZE = 256

    MIN_CLUSTER_SIZE = 10
    MIN_SAMPLES = 5
    METRIC = "euclidean"
    CLUSTER_SELECTION_METHOD = "eom"
    PREDICTION_DATA = True
    PCA_COMPONENTS = 2

    EXCEL_EXPORT_ENABLED = True
    EXCEL_FILENAME = "clustering_data_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40

    RANDOM_SEED = 42


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATHS = {
    "output_root": EmbeddingConfig.OUTPUT_ROOT,
}

RUN_NAME = EmbeddingConfig.RUN_NAME

DATA = {
    "preprocess_root": EmbeddingConfig.PREPROCESS_ROOT,
    "training_root": EmbeddingConfig.TRAINING_ROOT,
    "clustering_root": EmbeddingConfig.CLUSTERING_ROOT,
    "required_metadata_columns": EmbeddingConfig.REQUIRED_METADATA_COLUMNS,
}

PREPROCESS = {}

MODEL = {}

TRAINING = {}

EMBEDDING = {
    "predict_batch_size": EmbeddingConfig.PREDICT_BATCH_SIZE,
}

CLUSTERING = {
    "min_cluster_size": EmbeddingConfig.MIN_CLUSTER_SIZE,
    "min_samples": EmbeddingConfig.MIN_SAMPLES,
    "metric": EmbeddingConfig.METRIC,
    "cluster_selection_method": EmbeddingConfig.CLUSTER_SELECTION_METHOD,
    "prediction_data": EmbeddingConfig.PREDICTION_DATA,
    "pca_components": EmbeddingConfig.PCA_COMPONENTS,
}

EXCEL = {
    "export_enabled": EmbeddingConfig.EXCEL_EXPORT_ENABLED,
    "filename": EmbeddingConfig.EXCEL_FILENAME,
    "freeze_panes": EmbeddingConfig.EXCEL_FREEZE_PANES,
    "header_fill": EmbeddingConfig.EXCEL_HEADER_FILL,
    "header_font_color": EmbeddingConfig.EXCEL_HEADER_FONT_COLOR,
    "max_column_width": EmbeddingConfig.EXCEL_MAX_COLUMN_WIDTH,
}

RANDOM_SEED = EmbeddingConfig.RANDOM_SEED


# ============================================================================
# Dataset adapter section
# ============================================================================
def load_clustering_inputs(
    preprocess_root: Path,
    training_root: Path,
    required_metadata_columns: list[str],
) -> tuple[np.ndarray, pd.DataFrame, tf.keras.Model]:
    """
    Load aligned valid-cycle features, metadata, and the trained encoder.

    Expected input schema:
    - cycle_features.npy: shape (num_valid_cycles, num_features)
    - cycle_metadata.csv: valid_flag plus sample-alignment metadata
    - encoder.keras: trained encoder model from stage 02

    Output:
        features: float32 array with shape (num_valid_cycles, num_features)
        valid_metadata: DataFrame aligned 1:1 with features
        encoder: Keras model mapping (batch, num_features) -> (batch, latent_dim)
    """
    feature_path = preprocess_root / "cycle_features.npy"
    metadata_path = preprocess_root / "cycle_metadata.csv"
    encoder_path = training_root / "encoder.keras"

    for path in [feature_path, metadata_path, encoder_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")

    features = np.load(feature_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    encoder = tf.keras.models.load_model(encoder_path, compile=False)

    missing_columns = [
        column for column in required_metadata_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Invalid feature matrix shape: {features.shape}")
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Feature matrix contains NaN or Inf values.")

    valid_metadata = metadata.loc[metadata["valid_flag"] == True].copy()
    if len(valid_metadata) != features.shape[0]:
        raise ValueError(
            "Number of valid metadata rows does not match feature rows: "
            f"{len(valid_metadata)} vs {features.shape[0]}"
        )
    if valid_metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be unique across valid cycles.")

    if "feature_row_index" in valid_metadata.columns:
        feature_row_index = valid_metadata["feature_row_index"].dropna().to_numpy()
        if (
            len(feature_row_index) == len(valid_metadata)
            and len(np.unique(feature_row_index)) == len(valid_metadata)
            and np.array_equal(
                np.sort(feature_row_index.astype(int)),
                np.arange(len(valid_metadata), dtype=int),
            )
        ):
            valid_metadata["feature_row_index"] = valid_metadata["feature_row_index"].astype(int)
            valid_metadata = valid_metadata.sort_values(
                by="feature_row_index", kind="stable"
            ).reset_index(drop=True)
            return features, valid_metadata, encoder

    sort_columns = [
        column
        for column in ["recording_id", "cycle_index"]
        if column in valid_metadata.columns
    ]
    if sort_columns:
        valid_metadata = valid_metadata.sort_values(
            by=sort_columns, kind="stable"
        ).reset_index(drop=True)
    else:
        valid_metadata = valid_metadata.reset_index(drop=True)

    return features, valid_metadata, encoder


# ============================================================================
# Utility functions
# ============================================================================
def set_random_seed(seed: int) -> None:
    """Set reproducible seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_output_directories(output_root: Path, run_name: str) -> dict[str, Path]:
    """Create stable output directories for this run."""
    run_root = output_root / run_name
    preprocess_root = run_root / "preprocess"
    training_root = run_root / "training"
    clustering_root = run_root / "clustering"
    interpretation_root = run_root / "interpretation"

    for path in [
        run_root,
        preprocess_root,
        training_root,
        clustering_root,
        interpretation_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "preprocess_root": preprocess_root,
        "training_root": training_root,
        "clustering_root": clustering_root,
        "interpretation_root": interpretation_root,
    }


def extract_embeddings(
    encoder: tf.keras.Model,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Extract one latent vector per valid sample.

    Input:
        features: array with shape (num_valid_samples, input_dim)
    Output:
        embeddings: array with shape (num_valid_samples, latent_dim)
    """
    embeddings = encoder.predict(features, batch_size=batch_size, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != features.shape[0]:
        raise ValueError(
            "Encoder output shape is invalid: "
            f"features {features.shape} -> embeddings {embeddings.shape}"
        )
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains NaN or Inf values.")
    return embeddings


def run_hdbscan(embeddings: np.ndarray) -> hdbscan.HDBSCAN:
    """
    Fit HDBSCAN on latent embeddings.

    Input:
        embeddings: array with shape (num_valid_samples, latent_dim)
    Output:
        Fitted HDBSCAN clusterer.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CLUSTERING["min_cluster_size"],
        min_samples=CLUSTERING["min_samples"],
        metric=CLUSTERING["metric"],
        cluster_selection_method=CLUSTERING["cluster_selection_method"],
        prediction_data=CLUSTERING["prediction_data"],
    )
    clusterer.fit(embeddings)
    return clusterer


def cluster_assignment_table(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    outlier_scores: np.ndarray | None,
) -> pd.DataFrame:
    """
    Build a row-aligned cluster assignment table.

    Input:
        metadata: DataFrame with shape (num_valid_samples, num_columns)
        labels: array with shape (num_valid_samples,)
    Output:
        DataFrame with one clustering row per valid sample.
    """
    output = metadata[["sample_id", "subject_id", "recording_id"]].copy()
    output.insert(0, "embedding_row_index", np.arange(len(output), dtype=int))
    if "feature_row_index" in metadata.columns:
        output["feature_row_index"] = metadata["feature_row_index"].astype(int)
    if "waveform_row_index" in metadata.columns:
        output["waveform_row_index"] = metadata["waveform_row_index"].astype(int)
    output["cluster_label"] = labels.astype(int)
    if probabilities is not None:
        output["membership_probability"] = probabilities.astype(np.float32)
    if outlier_scores is not None:
        output["outlier_score"] = outlier_scores.astype(np.float32)
    return output


def embedding_metadata_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Save row-level metadata aligned to embeddings.npy.

    Output:
        DataFrame with one row per embedding vector.
    """
    preferred_columns = [
        "sample_id",
        "subject_id",
        "recording_id",
        "auscultation_site",
        "cycle_index",
        "feature_row_index",
        "waveform_row_index",
        "cycle_start_sample",
        "cycle_end_sample",
    ]
    kept_columns = [column for column in preferred_columns if column in metadata.columns]
    output = metadata[kept_columns].copy()
    output.insert(0, "embedding_row_index", np.arange(len(output), dtype=int))
    return output


def cluster_size_dict(labels: np.ndarray) -> dict[str, int]:
    """Convert cluster labels into a JSON-friendly size dictionary."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}


def save_pca_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    pca_components: int,
) -> list[float]:
    """Create a simple 2D PCA scatter plot colored by cluster label."""
    if embeddings.shape[1] < pca_components:
        raise ValueError(
            f"Latent dimension {embeddings.shape[1]} is smaller than PCA components {pca_components}."
        )

    pca = PCA(n_components=pca_components, random_state=RANDOM_SEED)
    projected = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    color_map = matplotlib.colormaps.get_cmap("tab20")

    cluster_counter = 0
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            plt.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=18,
                c="#9E9E9E",
                alpha=0.75,
                label="noise (-1)",
            )
        else:
            plt.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=18,
                color=color_map(cluster_counter % 20),
                alpha=0.8,
                label=f"cluster {int(label)}",
            )
            cluster_counter += 1

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA of Latent Embeddings")
    plt.legend(loc="best", fontsize=8, markerscale=1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return [float(value) for value in pca.explained_variance_ratio_]


def clustering_summary(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    pca_explained_variance_ratio: list[float],
) -> dict[str, Any]:
    """Create a compact JSON summary for the clustering stage."""
    unique_non_noise = sorted(label for label in np.unique(labels).tolist() if label != -1)
    noise_count = int(np.sum(labels == -1))
    summary: dict[str, Any] = {
        "run_name": RUN_NAME,
        "total_valid_samples": int(embeddings.shape[0]),
        "latent_dimension": int(embeddings.shape[1]),
        "number_of_clusters_excluding_noise": int(len(unique_non_noise)),
        "number_of_noise_samples": noise_count,
        "noise_ratio": float(noise_count / len(labels)),
        "cluster_sizes": cluster_size_dict(labels),
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
        "hdbscan_parameters": {
            "min_cluster_size": CLUSTERING["min_cluster_size"],
            "min_samples": CLUSTERING["min_samples"],
            "metric": CLUSTERING["metric"],
            "cluster_selection_method": CLUSTERING["cluster_selection_method"],
            "prediction_data": CLUSTERING["prediction_data"],
        },
    }
    if probabilities is not None:
        summary["membership_probability"] = {
            "mean": float(np.mean(probabilities)),
            "median": float(np.median(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
        }
    return summary


def export_clustering_excel(
    clustering_root: Path,
    summary: dict[str, Any],
    embedding_metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    embeddings: np.ndarray,
) -> Path:
    """Export clustering-stage tables to an Excel workbook."""
    overview_rows: list[dict[str, Any]] = []
    for key in [
        "run_name",
        "total_valid_samples",
        "latent_dimension",
        "number_of_clusters_excluding_noise",
        "number_of_noise_samples",
        "noise_ratio",
    ]:
        overview_rows.append({"section": "summary", "metric": key, "value": summary[key]})

    for key, value in summary["hdbscan_parameters"].items():
        overview_rows.append({"section": "hdbscan_parameters", "metric": key, "value": value})

    if "membership_probability" in summary:
        for key, value in summary["membership_probability"].items():
            overview_rows.append({"section": "membership_probability", "metric": key, "value": value})

    cluster_size_df = pd.DataFrame(
        [
            {"cluster_label": int(label), "sample_count": count}
            for label, count in summary["cluster_sizes"].items()
        ]
    )
    embedding_frame = pd.DataFrame(
        embeddings,
        columns=[f"embedding_dim_{index}" for index in range(embeddings.shape[1])],
    )
    embedding_view = pd.concat(
        [
            embedding_metadata.reset_index(drop=True),
            assignments.drop(
                columns=[
                    "sample_id",
                    "subject_id",
                    "recording_id",
                    "feature_row_index",
                    "waveform_row_index",
                ],
                errors="ignore",
            ).reset_index(drop=True),
            embedding_frame,
        ],
        axis=1,
    )

    workbook_path = clustering_root / EXCEL["filename"]
    return export_stage_workbook(
        workbook_path=workbook_path,
        sheets={
            "Overview": pd.DataFrame(overview_rows),
            "Cluster_Sizes": cluster_size_df,
            "Embedding_Metadata": embedding_metadata,
            "Cluster_Assignments": assignments,
            "Embedding_View": embedding_view,
        },
        freeze_panes=EXCEL["freeze_panes"],
        header_fill=EXCEL["header_fill"],
        header_font_color=EXCEL["header_font_color"],
        max_column_width=EXCEL["max_column_width"],
    )


def main() -> None:
    """Extract embeddings, run HDBSCAN, and save clustering artifacts."""
    set_random_seed(RANDOM_SEED)
    output_paths = ensure_output_directories(PATHS["output_root"], RUN_NAME)

    features, valid_metadata, encoder = load_clustering_inputs(
        preprocess_root=DATA["preprocess_root"],
        training_root=DATA["training_root"],
        required_metadata_columns=DATA["required_metadata_columns"],
    )

    embeddings = extract_embeddings(
        encoder=encoder,
        features=features,
        batch_size=EMBEDDING["predict_batch_size"],
    )
    clusterer = run_hdbscan(embeddings)

    labels = np.asarray(clusterer.labels_, dtype=int)
    probabilities = (
        np.asarray(clusterer.probabilities_, dtype=np.float32)
        if hasattr(clusterer, "probabilities_")
        else None
    )
    outlier_scores = (
        np.asarray(clusterer.outlier_scores_, dtype=np.float32)
        if hasattr(clusterer, "outlier_scores_")
        else None
    )

    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "Cluster label count does not match embedding rows: "
            f"{labels.shape[0]} vs {embeddings.shape[0]}"
        )

    clustering_root = output_paths["clustering_root"]
    np.save(clustering_root / "embeddings.npy", embeddings)

    embedding_metadata = embedding_metadata_table(valid_metadata)
    embedding_metadata.to_csv(clustering_root / "embedding_metadata.csv", index=False)

    assignments = cluster_assignment_table(
        metadata=valid_metadata,
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
    )
    assignments.to_csv(clustering_root / "cluster_assignments.csv", index=False)

    joblib.dump(clusterer, clustering_root / "hdbscan_model.joblib")

    pca_explained_variance_ratio = save_pca_scatter(
        embeddings=embeddings,
        labels=labels,
        output_path=clustering_root / "pca_scatter.png",
        pca_components=CLUSTERING["pca_components"],
    )

    summary = clustering_summary(
        embeddings=embeddings,
        labels=labels,
        probabilities=probabilities,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
    )
    with open(clustering_root / "clustering_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    excel_path = None
    if EXCEL["export_enabled"]:
        excel_path = export_clustering_excel(
            clustering_root=clustering_root,
            summary=summary,
            embedding_metadata=embedding_metadata,
            assignments=assignments,
            embeddings=embeddings,
        )

    logger.info("Saved clustering outputs to: %s", clustering_root)
    if excel_path is not None:
        logger.info("Saved clustering Excel export to: %s", excel_path)
    logger.info("Embedding shape: %s", embeddings.shape)
    logger.info(
        "Clusters excluding noise: %s",
        summary["number_of_clusters_excluding_noise"],
    )
    logger.info("Noise samples: %s", summary["number_of_noise_samples"])


if __name__ == "__main__":
    main()
