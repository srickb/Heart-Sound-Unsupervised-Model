"""
HDBSCAN clustering script for the revised Heart-Sound-Unsupervised-Model pipeline.

Expected input files:
- outputs/{PREPROCESS_RUN_NAME}/preprocess/beat_features_valid.csv
- outputs/{PREPROCESS_RUN_NAME}/preprocess/learning_input_columns.json
- outputs/{TRAINING_RUN_NAME}/training/latent_train.csv
- outputs/{TRAINING_RUN_NAME}/training/latent_val.csv
- outputs/{TRAINING_RUN_NAME}/training/latent_test.csv
- outputs/{TRAINING_RUN_NAME}/training/latent_all_valid.csv
- outputs/{TRAINING_RUN_NAME}/training/split_info.json

Saved artifacts:
- outputs/{RUN_NAME}/clustering/latent_train.csv
- outputs/{RUN_NAME}/clustering/latent_val.csv
- outputs/{RUN_NAME}/clustering/latent_test.csv
- outputs/{RUN_NAME}/clustering/hdbscan_labels_train.csv
- outputs/{RUN_NAME}/clustering/all_valid_with_latent.csv
- outputs/{RUN_NAME}/clustering/clustering_summary.json
- outputs/{RUN_NAME}/clustering/cluster_exemplars.csv
- outputs/{RUN_NAME}/clustering/cluster_stability_summary.csv
- outputs/{RUN_NAME}/clustering/record_distribution_summary.csv
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClusteringConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 2개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    PREPROCESS_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"
    TRAINING_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\training"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\clustering"

    RANDOM_SEED = 42
    MIN_CLUSTER_SIZE = 25
    MIN_SAMPLES = 10
    CLUSTER_SELECTION_EPSILON = 0.0
    CLUSTER_SELECTION_METHOD = "eom"
    PREDICTION_DATA_ENABLED = True
    EXEMPLARS_PER_CLUSTER = 5

    BEAT_FEATURES_FILENAME = "beat_features_valid.csv"
    LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"
    LATENT_TRAIN_FILENAME = "latent_train.csv"
    LATENT_VAL_FILENAME = "latent_val.csv"
    LATENT_TEST_FILENAME = "latent_test.csv"
    LATENT_ALL_VALID_FILENAME = "latent_all_valid.csv"
    SPLIT_INFO_FILENAME = "split_info.json"

    OUTPUT_LATENT_TRAIN_FILENAME = "latent_train.csv"
    OUTPUT_LATENT_VAL_FILENAME = "latent_val.csv"
    OUTPUT_LATENT_TEST_FILENAME = "latent_test.csv"
    HDBSCAN_LABELS_TRAIN_FILENAME = "hdbscan_labels_train.csv"
    ALL_VALID_WITH_LATENT_FILENAME = "all_valid_with_latent.csv"
    CLUSTERING_SUMMARY_FILENAME = "clustering_summary.json"
    CLUSTER_EXEMPLARS_FILENAME = "cluster_exemplars.csv"
    CLUSTER_STABILITY_SUMMARY_FILENAME = "cluster_stability_summary.csv"
    RECORD_DISTRIBUTION_SUMMARY_FILENAME = "record_distribution_summary.csv"


LATENT_METADATA_COLUMNS = [
    "record_id",
    "source_file",
    "beat_index",
    "cycle_index",
    "valid_flag",
]


@dataclass
class SplitLatentData:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    all_valid_frame: pd.DataFrame
    train_records: list[str]
    val_records: list[str]
    test_records: list[str]


# =================================================
# 1. Dependency Helpers
# =================================================


def load_hdbscan_modules() -> tuple[Any, Any]:
    try:
        import hdbscan  # type: ignore
        from hdbscan.prediction import approximate_predict  # type: ignore
    except Exception as error:
        raise ImportError(
            "hdbscan package is required to run 03_extract_embeddings_and_hdbscan.py"
        ) from error

    return hdbscan, approximate_predict


# =================================================
# 2. IO Helpers
# =================================================


def configured_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser()


def ensure_output_directories(stage_output_folder: Path) -> dict[str, Path]:
    clustering_root = stage_output_folder
    clustering_root.mkdir(parents=True, exist_ok=True)
    return {"clustering_root": clustering_root}


def load_json_list(file_path: Path) -> list[str]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected JSON string list: {file_path}")
    return values


def load_json_dict(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {file_path}")
    return value


def load_clustering_inputs(
    preprocess_root: Path,
    training_root: Path,
) -> tuple[pd.DataFrame, list[str], SplitLatentData]:
    beat_features_path = preprocess_root / ClusteringConfig.BEAT_FEATURES_FILENAME
    learning_input_columns_path = preprocess_root / ClusteringConfig.LEARNING_INPUT_COLUMNS_FILENAME
    latent_train_path = training_root / ClusteringConfig.LATENT_TRAIN_FILENAME
    latent_val_path = training_root / ClusteringConfig.LATENT_VAL_FILENAME
    latent_test_path = training_root / ClusteringConfig.LATENT_TEST_FILENAME
    latent_all_valid_path = training_root / ClusteringConfig.LATENT_ALL_VALID_FILENAME
    split_info_path = training_root / ClusteringConfig.SPLIT_INFO_FILENAME

    missing_paths = [
        path
        for path in [
            beat_features_path,
            learning_input_columns_path,
            latent_train_path,
            latent_val_path,
            latent_test_path,
            latent_all_valid_path,
            split_info_path,
        ]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing clustering inputs: {missing_paths}")

    beat_features = pd.read_csv(beat_features_path)
    learning_input_columns = load_json_list(learning_input_columns_path)
    split_info = load_json_dict(split_info_path)

    split_latent_data = SplitLatentData(
        train_frame=pd.read_csv(latent_train_path),
        val_frame=pd.read_csv(latent_val_path),
        test_frame=pd.read_csv(latent_test_path),
        all_valid_frame=pd.read_csv(latent_all_valid_path),
        train_records=[str(value) for value in split_info.get("train_records", [])],
        val_records=[str(value) for value in split_info.get("validation_records", [])],
        test_records=[str(value) for value in split_info.get("test_records", [])],
    )
    return beat_features, learning_input_columns, split_latent_data


# =================================================
# 3. Validation Helpers
# =================================================


def get_latent_columns(frame: pd.DataFrame) -> list[str]:
    latent_columns = [column for column in frame.columns if column.startswith("latent_")]
    if not latent_columns:
        raise ValueError("No latent columns found")
    return sorted(latent_columns)


def _validate_latent_frame(frame: pd.DataFrame, latent_columns: list[str], frame_name: str) -> None:
    required_columns = ["record_id", "beat_index", *latent_columns]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {frame_name}: {missing_columns}")

    matrix = frame.loc[:, latent_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite latent values found in {frame_name}")


def validate_inputs(
    beat_features: pd.DataFrame,
    learning_input_columns: list[str],
    split_latent_data: SplitLatentData,
) -> list[str]:
    if beat_features.empty:
        raise ValueError("beat_features_valid.csv is empty")

    missing_learning_columns = [column for column in learning_input_columns if column not in beat_features.columns]
    if missing_learning_columns:
        raise ValueError(f"Missing learning input columns in beat_features_valid.csv: {missing_learning_columns}")

    valid_flags = pd.to_numeric(beat_features["valid_flag"], errors="raise").astype(np.int64)
    if np.any(valid_flags != 1):
        raise ValueError("beat_features_valid.csv must contain only valid beats")

    latent_columns = get_latent_columns(split_latent_data.train_frame)
    for frame_name, frame in [
        ("latent_train.csv", split_latent_data.train_frame),
        ("latent_val.csv", split_latent_data.val_frame),
        ("latent_test.csv", split_latent_data.test_frame),
        ("latent_all_valid.csv", split_latent_data.all_valid_frame),
    ]:
        _validate_latent_frame(frame, latent_columns, frame_name)
        if get_latent_columns(frame) != latent_columns:
            raise ValueError(f"Latent column mismatch detected in {frame_name}")

    all_valid_pairs = split_latent_data.all_valid_frame.loc[:, ["record_id", "beat_index"]].copy()
    beat_pairs = beat_features.loc[:, ["record_id", "beat_index"]].copy()
    if len(all_valid_pairs) != len(beat_pairs):
        raise ValueError("beat_features_valid.csv and latent_all_valid.csv row counts do not match")
    if not all_valid_pairs.reset_index(drop=True).equals(beat_pairs.reset_index(drop=True)):
        raise ValueError("beat_features_valid.csv and latent_all_valid.csv metadata order does not match")

    if not split_latent_data.train_records or not split_latent_data.val_records or not split_latent_data.test_records:
        raise ValueError("split_info.json must contain non-empty train/validation/test record lists")

    return latent_columns


# =================================================
# 4. Clustering Helpers
# =================================================


def dataframe_to_latent_matrix(frame: pd.DataFrame, latent_columns: list[str]) -> np.ndarray:
    matrix = frame.loc[:, latent_columns].to_numpy(dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Latent matrix must be two-dimensional")
    return matrix


def fit_hdbscan(
    train_latent: np.ndarray,
    config: ClusteringConfig,
) -> Any:
    hdbscan_module, _approximate_predict = load_hdbscan_modules()
    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        min_samples=config.MIN_SAMPLES,
        cluster_selection_epsilon=config.CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=config.CLUSTER_SELECTION_METHOD,
        prediction_data=config.PREDICTION_DATA_ENABLED,
    )
    clusterer.fit(train_latent)
    logger.info("HDBSCAN fit 완료")
    return clusterer


def build_train_assignment_frame(
    train_frame: pd.DataFrame,
    train_latent: np.ndarray,
    clusterer: Any,
    latent_columns: list[str],
) -> pd.DataFrame:
    output = train_frame.copy()
    output["cluster_label"] = np.asarray(clusterer.labels_, dtype=np.int64)

    if hasattr(clusterer, "probabilities_"):
        output["membership_probability"] = np.asarray(clusterer.probabilities_, dtype=np.float32)
    else:
        output["membership_probability"] = np.nan

    if hasattr(clusterer, "outlier_scores_"):
        output["outlier_score"] = np.asarray(clusterer.outlier_scores_, dtype=np.float32)
    else:
        output["outlier_score"] = np.nan

    output["is_noise"] = (output["cluster_label"] == -1).astype(np.int64)
    return output


def build_cluster_reference_table(
    train_assignments: pd.DataFrame,
    latent_columns: list[str],
) -> pd.DataFrame:
    clustered_train = train_assignments.loc[train_assignments["cluster_label"] != -1].copy()
    if clustered_train.empty:
        return pd.DataFrame(columns=["cluster_label", *latent_columns])

    rows: list[dict[str, Any]] = []
    for cluster_label, cluster_frame in clustered_train.groupby("cluster_label", sort=True):
        row: dict[str, Any] = {"cluster_label": int(cluster_label)}
        for column in latent_columns:
            row[column] = float(cluster_frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_nearest_cluster_features(
    frame: pd.DataFrame,
    reference_table: pd.DataFrame,
    latent_columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    if reference_table.empty:
        output["nearest_train_cluster_label"] = -1
        output["nearest_train_cluster_distance"] = np.nan
        return output

    reference_labels = reference_table["cluster_label"].to_numpy(dtype=np.int64)
    reference_matrix = reference_table.loc[:, latent_columns].to_numpy(dtype=np.float32)
    latent_matrix = output.loc[:, latent_columns].to_numpy(dtype=np.float32)

    nearest_labels: list[int] = []
    nearest_distances: list[float] = []
    for row_vector in latent_matrix:
        distances = np.linalg.norm(reference_matrix - row_vector[None, :], axis=1)
        best_index = int(np.argmin(distances))
        nearest_labels.append(int(reference_labels[best_index]))
        nearest_distances.append(float(distances[best_index]))

    output["nearest_train_cluster_label"] = nearest_labels
    output["nearest_train_cluster_distance"] = nearest_distances
    return output


def approximate_predict_frame(
    clusterer: Any,
    frame: pd.DataFrame,
    latent_columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    output["predicted_cluster_label"] = np.nan
    output["predicted_membership_probability"] = np.nan

    if len(output) == 0:
        return output

    if not getattr(clusterer, "prediction_data_", None):
        return output

    _hdbscan_module, approximate_predict = load_hdbscan_modules()
    latent_matrix = output.loc[:, latent_columns].to_numpy(dtype=np.float32)
    labels, strengths = approximate_predict(clusterer, latent_matrix)
    output["predicted_cluster_label"] = np.asarray(labels, dtype=np.int64)
    output["predicted_membership_probability"] = np.asarray(strengths, dtype=np.float32)
    return output


def assign_split_name(
    frame: pd.DataFrame,
    split_latent_data: SplitLatentData,
) -> pd.DataFrame:
    output = frame.copy()
    output["split_name"] = "unknown"
    train_records = set(split_latent_data.train_records)
    val_records = set(split_latent_data.val_records)
    test_records = set(split_latent_data.test_records)

    output.loc[output["record_id"].astype(str).isin(train_records), "split_name"] = "train"
    output.loc[output["record_id"].astype(str).isin(val_records), "split_name"] = "val"
    output.loc[output["record_id"].astype(str).isin(test_records), "split_name"] = "test"
    return output


def compute_cluster_exemplars(
    train_assignments: pd.DataFrame,
    latent_columns: list[str],
    exemplars_per_cluster: int,
) -> pd.DataFrame:
    clustered_train = train_assignments.loc[train_assignments["cluster_label"] != -1].copy()
    if clustered_train.empty:
        return pd.DataFrame(
            columns=["cluster_label", "exemplar_rank", "record_id", "beat_index", "distance_to_cluster_mean"]
        )

    rows: list[dict[str, Any]] = []
    for cluster_label, cluster_frame in clustered_train.groupby("cluster_label", sort=True):
        cluster_matrix = cluster_frame.loc[:, latent_columns].to_numpy(dtype=np.float32)
        cluster_center = cluster_matrix.mean(axis=0, dtype=np.float32)
        distances = np.linalg.norm(cluster_matrix - cluster_center[None, :], axis=1)
        exemplar_order = np.argsort(distances)[:exemplars_per_cluster]
        for exemplar_rank, exemplar_index in enumerate(exemplar_order, start=1):
            exemplar_row = cluster_frame.iloc[int(exemplar_index)]
            rows.append(
                {
                    "cluster_label": int(cluster_label),
                    "exemplar_rank": int(exemplar_rank),
                    "record_id": exemplar_row["record_id"],
                    "beat_index": int(exemplar_row["beat_index"]),
                    "distance_to_cluster_mean": float(distances[int(exemplar_index)]),
                }
            )
    return pd.DataFrame(rows)


def compute_cluster_stability_summary(
    clusterer: Any,
    train_assignments: pd.DataFrame,
) -> pd.DataFrame:
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    unique_cluster_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    persistence = getattr(clusterer, "cluster_persistence_", np.array([], dtype=np.float32))

    rows: list[dict[str, Any]] = []
    for index, cluster_label in enumerate(unique_cluster_labels):
        cluster_frame = train_assignments.loc[train_assignments["cluster_label"] == cluster_label]
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "beat_count": int(len(cluster_frame)),
                "cluster_persistence": float(persistence[index]) if index < len(persistence) else float(np.nan),
                "membership_probability_mean": float(cluster_frame["membership_probability"].mean())
                if not cluster_frame.empty
                else float(np.nan),
                "outlier_score_mean": float(cluster_frame["outlier_score"].mean())
                if not cluster_frame.empty
                else float(np.nan),
            }
        )

    noise_frame = train_assignments.loc[train_assignments["cluster_label"] == -1]
    rows.append(
        {
            "cluster_label": -1,
            "beat_count": int(len(noise_frame)),
            "cluster_persistence": float(np.nan),
            "membership_probability_mean": float(noise_frame["membership_probability"].mean())
            if not noise_frame.empty
            else float(np.nan),
            "outlier_score_mean": float(noise_frame["outlier_score"].mean()) if not noise_frame.empty else float(np.nan),
        }
    )
    return pd.DataFrame(rows)


def compute_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return float(np.nan)
    probabilities = counts.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return float(np.nan)
    return float(-np.sum(probabilities * np.log(probabilities)))


def compute_record_distribution_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_label, cluster_frame in frame.groupby("cluster_label", sort=True):
        record_counts = cluster_frame["record_id"].astype(str).value_counts()
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "beat_count": int(len(cluster_frame)),
                "num_records": int(record_counts.shape[0]),
                "record_distribution_entropy": compute_entropy_from_counts(record_counts.to_numpy(dtype=np.int64)),
                "top_record_id": str(record_counts.index[0]) if not record_counts.empty else "",
                "top_record_fraction": float(record_counts.iloc[0] / max(1, len(cluster_frame))) if not record_counts.empty else float(np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_clustering_summary(
    clusterer: Any,
    train_assignments: pd.DataFrame,
    split_latent_data: SplitLatentData,
    latent_columns: list[str],
    config: ClusteringConfig,
) -> dict[str, Any]:
    train_labels = np.asarray(clusterer.labels_, dtype=np.int64)
    unique_clusters = sorted(int(label) for label in np.unique(train_labels) if label != -1)
    noise_ratio = float(np.mean(train_labels == -1)) if train_labels.size > 0 else float(np.nan)

    return {
        "latent_dim": int(len(latent_columns)),
        "num_train_records": int(len(split_latent_data.train_records)),
        "num_val_records": int(len(split_latent_data.val_records)),
        "num_test_records": int(len(split_latent_data.test_records)),
        "num_train_beats": int(len(split_latent_data.train_frame)),
        "num_val_beats": int(len(split_latent_data.val_frame)),
        "num_test_beats": int(len(split_latent_data.test_frame)),
        "num_all_valid_beats": int(len(split_latent_data.all_valid_frame)),
        "num_clusters_excluding_noise": int(len(unique_clusters)),
        "cluster_labels_excluding_noise": unique_clusters,
        "noise_ratio_train": noise_ratio,
        "config": {
            "min_cluster_size": int(config.MIN_CLUSTER_SIZE),
            "min_samples": int(config.MIN_SAMPLES),
            "cluster_selection_epsilon": float(config.CLUSTER_SELECTION_EPSILON),
            "cluster_selection_method": config.CLUSTER_SELECTION_METHOD,
            "prediction_data_enabled": bool(config.PREDICTION_DATA_ENABLED),
        },
        "train_membership_probability_mean": float(train_assignments["membership_probability"].mean()),
        "train_outlier_score_mean": float(train_assignments["outlier_score"].mean()),
    }


# =================================================
# 5. Main
# =================================================


def main() -> None:
    config = ClusteringConfig()
    output_paths = ensure_output_directories(stage_output_folder=configured_path(config.OUTPUT_FOLDER))
    clustering_root = output_paths["clustering_root"]
    preprocess_input_root = configured_path(config.PREPROCESS_INPUT_FOLDER)
    training_input_root = configured_path(config.TRAINING_INPUT_FOLDER)

    logger.info("전처리 입력 폴더: %s", preprocess_input_root)
    logger.info("표현학습 입력 폴더: %s", training_input_root)
    logger.info("클러스터링 출력 폴더: %s", clustering_root)

    beat_features, learning_input_columns, split_latent_data = load_clustering_inputs(
        preprocess_root=preprocess_input_root,
        training_root=training_input_root,
    )
    latent_columns = validate_inputs(
        beat_features=beat_features,
        learning_input_columns=learning_input_columns,
        split_latent_data=split_latent_data,
    )

    train_latent = dataframe_to_latent_matrix(split_latent_data.train_frame, latent_columns)
    clusterer = fit_hdbscan(train_latent=train_latent, config=config)

    train_assignments = build_train_assignment_frame(
        train_frame=split_latent_data.train_frame,
        train_latent=train_latent,
        clusterer=clusterer,
        latent_columns=latent_columns,
    )

    cluster_reference_table = build_cluster_reference_table(
        train_assignments=train_assignments,
        latent_columns=latent_columns,
    )

    val_with_predictions = compute_nearest_cluster_features(
        frame=split_latent_data.val_frame,
        reference_table=cluster_reference_table,
        latent_columns=latent_columns,
    )
    test_with_predictions = compute_nearest_cluster_features(
        frame=split_latent_data.test_frame,
        reference_table=cluster_reference_table,
        latent_columns=latent_columns,
    )
    all_valid_with_predictions = compute_nearest_cluster_features(
        frame=split_latent_data.all_valid_frame,
        reference_table=cluster_reference_table,
        latent_columns=latent_columns,
    )

    if config.PREDICTION_DATA_ENABLED:
        val_with_predictions = approximate_predict_frame(clusterer, val_with_predictions, latent_columns)
        test_with_predictions = approximate_predict_frame(clusterer, test_with_predictions, latent_columns)
        all_valid_with_predictions = approximate_predict_frame(clusterer, all_valid_with_predictions, latent_columns)
        train_assignments = approximate_predict_frame(clusterer, train_assignments, latent_columns)
    else:
        train_assignments["predicted_cluster_label"] = train_assignments["cluster_label"]
        train_assignments["predicted_membership_probability"] = train_assignments["membership_probability"]

    all_valid_with_predictions = assign_split_name(
        frame=all_valid_with_predictions,
        split_latent_data=split_latent_data,
    )

    beat_feature_metadata_columns = [
        column
        for column in [
            "record_id",
            "beat_index",
            "source_file",
            "cycle_index",
            "valid_flag",
            "s1_start",
            "s1_end",
            "s2_start",
            "s2_end",
            "next_s1_start",
        ]
        if column in beat_features.columns
    ]
    all_valid_with_latent = beat_features.loc[:, beat_feature_metadata_columns].merge(
        all_valid_with_predictions,
        on=["record_id", "beat_index"],
        how="inner",
        validate="one_to_one",
    )

    cluster_exemplars = compute_cluster_exemplars(
        train_assignments=train_assignments,
        latent_columns=latent_columns,
        exemplars_per_cluster=config.EXEMPLARS_PER_CLUSTER,
    )
    cluster_stability_summary = compute_cluster_stability_summary(
        clusterer=clusterer,
        train_assignments=train_assignments,
    )
    record_distribution_summary = compute_record_distribution_summary(
        frame=train_assignments.loc[:, ["record_id", "cluster_label"]],
    )
    clustering_summary = build_clustering_summary(
        clusterer=clusterer,
        train_assignments=train_assignments,
        split_latent_data=split_latent_data,
        latent_columns=latent_columns,
        config=config,
    )

    split_latent_data.train_frame.to_csv(clustering_root / config.OUTPUT_LATENT_TRAIN_FILENAME, index=False)
    split_latent_data.val_frame.to_csv(clustering_root / config.OUTPUT_LATENT_VAL_FILENAME, index=False)
    split_latent_data.test_frame.to_csv(clustering_root / config.OUTPUT_LATENT_TEST_FILENAME, index=False)
    train_assignments.to_csv(clustering_root / config.HDBSCAN_LABELS_TRAIN_FILENAME, index=False)
    all_valid_with_latent.to_csv(clustering_root / config.ALL_VALID_WITH_LATENT_FILENAME, index=False)
    cluster_exemplars.to_csv(clustering_root / config.CLUSTER_EXEMPLARS_FILENAME, index=False)
    cluster_stability_summary.to_csv(clustering_root / config.CLUSTER_STABILITY_SUMMARY_FILENAME, index=False)
    record_distribution_summary.to_csv(clustering_root / config.RECORD_DISTRIBUTION_SUMMARY_FILENAME, index=False)
    (clustering_root / config.CLUSTERING_SUMMARY_FILENAME).write_text(
        json.dumps(clustering_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "HDBSCAN 완료: train_beats=%s, clusters=%s, noise_ratio=%.4f",
        len(train_assignments),
        clustering_summary["num_clusters_excluding_noise"],
        clustering_summary["noise_ratio_train"],
    )


if __name__ == "__main__":
    main()
