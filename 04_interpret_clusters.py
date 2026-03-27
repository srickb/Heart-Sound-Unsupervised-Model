"""
Cluster interpretation script for the revised Heart-Sound-Unsupervised-Model pipeline.

Expected input files:
- outputs/{PREPROCESS_RUN_NAME}/preprocess/beat_features_valid.csv
- outputs/{PREPROCESS_RUN_NAME}/preprocess/learning_input_columns.json
- outputs/{PREPROCESS_RUN_NAME}/preprocess/feature_names.json
- outputs/{PREPROCESS_RUN_NAME}/preprocess/feature_groups.json
- outputs/{CLUSTERING_RUN_NAME}/clustering/hdbscan_labels_train.csv
- outputs/{CLUSTERING_RUN_NAME}/clustering/all_valid_with_latent.csv
- outputs/{CLUSTERING_RUN_NAME}/clustering/clustering_summary.json
- outputs/{CLUSTERING_RUN_NAME}/clustering/cluster_exemplars.csv
- outputs/{CLUSTERING_RUN_NAME}/clustering/cluster_stability_summary.csv
- outputs/{CLUSTERING_RUN_NAME}/clustering/record_distribution_summary.csv

Saved artifacts:
- outputs/{RUN_NAME}/interpretation/cluster_overview.csv
- outputs/{RUN_NAME}/interpretation/feature_summary_by_cluster.csv
- outputs/{RUN_NAME}/interpretation/top_features_per_cluster.csv
- outputs/{RUN_NAME}/interpretation/feature_group_summary_by_cluster.json
- outputs/{RUN_NAME}/interpretation/representative_beats.csv
- outputs/{RUN_NAME}/interpretation/record_cluster_distribution.csv
- outputs/{RUN_NAME}/interpretation/cluster_interpretation_report.xlsx
- outputs/{RUN_NAME}/interpretation/cluster_interpretation_summary.json
- outputs/{RUN_NAME}/interpretation/figures/*
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable

from excel_export_utils import export_stage_workbook


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InterpretationConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 3개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    PREPROCESS_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"
    CLUSTERING_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\clustering"
    TRAIN_DATA_FOLDER = r"C:\Users\LUI\Desktop\PCG\Data\Train 학습데이터(260109)"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\interpretation"

    TOP_FEATURES_PER_CLUSTER = 10
    REPRESENTATIVE_BEATS_PER_CLUSTER = 5
    ENVELOPE_SMOOTH_MS = 20.0
    SAMPLING_RATE = 4000.0
    EPS = 1e-8
    CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    SHOW_PROGRESS = True

    BEAT_FEATURES_VALID_FILENAME = "beat_features_valid.csv"
    LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"
    FEATURE_NAMES_FILENAME = "feature_names.json"
    FEATURE_GROUPS_FILENAME = "feature_groups.json"

    HDBSCAN_LABELS_TRAIN_FILENAME = "hdbscan_labels_train.csv"
    ALL_VALID_WITH_LATENT_FILENAME = "all_valid_with_latent.csv"
    CLUSTERING_SUMMARY_FILENAME = "clustering_summary.json"
    CLUSTER_EXEMPLARS_FILENAME = "cluster_exemplars.csv"
    CLUSTER_STABILITY_SUMMARY_FILENAME = "cluster_stability_summary.csv"
    RECORD_DISTRIBUTION_SUMMARY_FILENAME = "record_distribution_summary.csv"

    CLUSTER_OVERVIEW_FILENAME = "cluster_overview.csv"
    FEATURE_SUMMARY_FILENAME = "feature_summary_by_cluster.csv"
    TOP_FEATURES_FILENAME = "top_features_per_cluster.csv"
    FEATURE_GROUP_SUMMARY_FILENAME = "feature_group_summary_by_cluster.json"
    REPRESENTATIVE_BEATS_FILENAME = "representative_beats.csv"
    RECORD_CLUSTER_DISTRIBUTION_FILENAME = "record_cluster_distribution.csv"
    EXCEL_REPORT_FILENAME = "cluster_interpretation_report.xlsx"
    JSON_SUMMARY_FILENAME = "cluster_interpretation_summary.json"
    CLUSTERED_VALID_BEATS_FILENAME = "clustered_valid_beats.csv"

    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40


# =================================================
# 1. IO Helpers
# =================================================


def configured_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser()


def ensure_output_directories(stage_output_folder: Path) -> dict[str, Path]:
    interpretation_root = stage_output_folder
    figures_root = interpretation_root / "figures"
    interpretation_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    return {
        "interpretation_root": interpretation_root,
        "figures_root": figures_root,
    }


def load_json_list(file_path: Path) -> list[str]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected JSON string list: {file_path}")
    return values


def load_json_dict(file_path: Path) -> dict[str, Any]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Expected JSON object: {file_path}")
    return values


def load_optional_csv(file_path: Path) -> pd.DataFrame | None:
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_interpretation_inputs(
    preprocess_root: Path,
    clustering_root: Path,
) -> dict[str, Any]:
    required_paths = {
        "beat_features_valid": preprocess_root / InterpretationConfig.BEAT_FEATURES_VALID_FILENAME,
        "learning_input_columns": preprocess_root / InterpretationConfig.LEARNING_INPUT_COLUMNS_FILENAME,
        "feature_names": preprocess_root / InterpretationConfig.FEATURE_NAMES_FILENAME,
        "feature_groups": preprocess_root / InterpretationConfig.FEATURE_GROUPS_FILENAME,
        "hdbscan_labels_train": clustering_root / InterpretationConfig.HDBSCAN_LABELS_TRAIN_FILENAME,
        "all_valid_with_latent": clustering_root / InterpretationConfig.ALL_VALID_WITH_LATENT_FILENAME,
        "clustering_summary": clustering_root / InterpretationConfig.CLUSTERING_SUMMARY_FILENAME,
        "cluster_exemplars": clustering_root / InterpretationConfig.CLUSTER_EXEMPLARS_FILENAME,
        "cluster_stability_summary": clustering_root / InterpretationConfig.CLUSTER_STABILITY_SUMMARY_FILENAME,
        "record_distribution_summary": clustering_root / InterpretationConfig.RECORD_DISTRIBUTION_SUMMARY_FILENAME,
    }
    missing_paths = [path for path in required_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing interpretation inputs: {missing_paths}")

    return {
        "beat_features_valid": pd.read_csv(required_paths["beat_features_valid"]),
        "learning_input_columns": load_json_list(required_paths["learning_input_columns"]),
        "feature_names": load_json_list(required_paths["feature_names"]),
        "feature_groups": load_json_dict(required_paths["feature_groups"]),
        "hdbscan_labels_train": pd.read_csv(required_paths["hdbscan_labels_train"]),
        "all_valid_with_latent": pd.read_csv(required_paths["all_valid_with_latent"]),
        "clustering_summary": load_json_dict(required_paths["clustering_summary"]),
        "cluster_exemplars": pd.read_csv(required_paths["cluster_exemplars"]),
        "cluster_stability_summary": pd.read_csv(required_paths["cluster_stability_summary"]),
        "record_distribution_summary": pd.read_csv(required_paths["record_distribution_summary"]),
    }


# =================================================
# 2. Validation / Merge Helpers
# =================================================


def validate_inputs(inputs: dict[str, Any]) -> list[str]:
    beat_features_valid = inputs["beat_features_valid"]
    learning_input_columns = inputs["learning_input_columns"]
    feature_names = inputs["feature_names"]
    feature_groups = inputs["feature_groups"]
    train_assignments = inputs["hdbscan_labels_train"]
    all_valid_with_latent = inputs["all_valid_with_latent"]

    required_feature_columns = ["record_id", "beat_index", "valid_flag", "source_file"]
    missing_feature_columns = [column for column in required_feature_columns if column not in beat_features_valid.columns]
    if missing_feature_columns:
        raise ValueError(f"Missing required columns in beat_features_valid.csv: {missing_feature_columns}")

    missing_learning_columns = [column for column in learning_input_columns if column not in beat_features_valid.columns]
    if missing_learning_columns:
        raise ValueError(f"Missing learning input columns in beat_features_valid.csv: {missing_learning_columns}")

    unknown_learning_columns = [column for column in learning_input_columns if column not in feature_names]
    if unknown_learning_columns:
        raise ValueError(f"Unknown learning input columns: {unknown_learning_columns}")

    if not isinstance(feature_groups, dict) or not feature_groups:
        raise ValueError("feature_groups.json must contain a non-empty object")

    latent_columns = sorted(column for column in all_valid_with_latent.columns if column.startswith("latent_"))
    if not latent_columns:
        raise ValueError("No latent columns found in all_valid_with_latent.csv")

    required_assignment_columns = ["record_id", "beat_index", "cluster_label", "membership_probability", "outlier_score"]
    missing_assignment_columns = [column for column in required_assignment_columns if column not in train_assignments.columns]
    if missing_assignment_columns:
        raise ValueError(f"Missing required columns in hdbscan_labels_train.csv: {missing_assignment_columns}")

    return latent_columns


def build_clustered_valid_beats(
    beat_features_valid: pd.DataFrame,
    train_assignments: pd.DataFrame,
    all_valid_with_latent: pd.DataFrame,
) -> pd.DataFrame:
    merged = beat_features_valid.merge(
        all_valid_with_latent,
        on=["record_id", "beat_index", "source_file", "cycle_index", "valid_flag", "s1_start", "s1_end", "s2_start", "s2_end", "next_s1_start"],
        how="inner",
        validate="one_to_one",
    )

    train_assignment_columns = [
        column for column in train_assignments.columns if column not in {"source_file", "cycle_index", "valid_flag"}
    ]
    merged = merged.merge(
        train_assignments.loc[:, train_assignment_columns],
        on=["record_id", "beat_index"],
        how="left",
        validate="one_to_one",
    )

    for optional_column in [
        "cluster_label",
        "membership_probability",
        "outlier_score",
        "predicted_cluster_label",
        "predicted_membership_probability",
        "nearest_train_cluster_label",
        "nearest_train_cluster_distance",
        "split_name",
    ]:
        if optional_column not in merged.columns:
            merged[optional_column] = np.nan

    merged["analysis_cluster_label"] = merged["cluster_label"]
    predicted_mask = merged["analysis_cluster_label"].isnull() & merged["predicted_cluster_label"].notnull()
    merged.loc[predicted_mask, "analysis_cluster_label"] = merged.loc[predicted_mask, "predicted_cluster_label"]
    nearest_mask = merged["analysis_cluster_label"].isnull() & merged["nearest_train_cluster_label"].notnull()
    merged.loc[nearest_mask, "analysis_cluster_label"] = merged.loc[nearest_mask, "nearest_train_cluster_label"]
    merged["analysis_cluster_label"] = merged["analysis_cluster_label"].fillna(-1).astype(int)

    merged["analysis_label_source"] = "train_fit"
    merged.loc[predicted_mask, "analysis_label_source"] = "approximate_predict"
    merged.loc[nearest_mask, "analysis_label_source"] = "nearest_cluster"
    merged.loc[merged["analysis_cluster_label"] == -1, "analysis_label_source"] = merged.loc[
        merged["analysis_cluster_label"] == -1, "analysis_label_source"
    ].replace("", "noise")

    merged["analysis_membership_probability"] = merged["membership_probability"]
    replace_probability_mask = merged["analysis_membership_probability"].isnull() & merged["predicted_membership_probability"].notnull()
    merged.loc[replace_probability_mask, "analysis_membership_probability"] = merged.loc[
        replace_probability_mask, "predicted_membership_probability"
    ]

    return merged


def get_dynamic_cluster_labels(clustered_valid_beats: pd.DataFrame) -> list[int]:
    labels = sorted(int(label) for label in clustered_valid_beats["analysis_cluster_label"].dropna().unique())
    return labels


# =================================================
# 3. Summary Helpers
# =================================================


def summarize_feature_series(values: np.ndarray) -> dict[str, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "mean": float(np.nan),
            "std": float(np.nan),
            "median": float(np.nan),
            "q25": float(np.nan),
            "q75": float(np.nan),
        }
    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values, ddof=0)),
        "median": float(np.median(finite_values)),
        "q25": float(np.quantile(finite_values, 0.25)),
        "q75": float(np.quantile(finite_values, 0.75)),
    }


def compute_cluster_overview(clustered_valid_beats: pd.DataFrame) -> pd.DataFrame:
    total_beats = float(len(clustered_valid_beats))
    rows: list[dict[str, Any]] = []

    for cluster_label in get_dynamic_cluster_labels(clustered_valid_beats):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["analysis_cluster_label"] == cluster_label].copy()
        probability_values = cluster_frame["analysis_membership_probability"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "is_noise": int(cluster_label == -1),
                "beat_count": int(len(cluster_frame)),
                "beat_ratio": float(len(cluster_frame) / total_beats) if total_beats > 0 else float(np.nan),
                "num_records": int(cluster_frame["record_id"].nunique()),
                "train_count": int(np.sum(cluster_frame["split_name"] == "train")) if "split_name" in cluster_frame.columns else 0,
                "val_count": int(np.sum(cluster_frame["split_name"] == "val")) if "split_name" in cluster_frame.columns else 0,
                "test_count": int(np.sum(cluster_frame["split_name"] == "test")) if "split_name" in cluster_frame.columns else 0,
                "membership_probability_mean": float(np.nanmean(probability_values)) if len(cluster_frame) > 0 else float(np.nan),
                "membership_probability_std": float(np.nanstd(probability_values)) if len(cluster_frame) > 0 else float(np.nan),
            }
        )

    return pd.DataFrame(rows)


def compute_feature_summary(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_labels = get_dynamic_cluster_labels(clustered_valid_beats)

    for feature_name in tqdm(
        learning_input_columns,
        desc="Feature summary",
        disable=not InterpretationConfig.SHOW_PROGRESS,
    ):
        global_values = clustered_valid_beats[feature_name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "summary_level": "global",
                "cluster_label": -999,
                "feature_name": feature_name,
                **summarize_feature_series(global_values),
            }
        )

        for cluster_label in cluster_labels:
            cluster_values = clustered_valid_beats.loc[
                clustered_valid_beats["analysis_cluster_label"] == cluster_label,
                feature_name,
            ].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "summary_level": "cluster",
                    "cluster_label": int(cluster_label),
                    "feature_name": feature_name,
                    **summarize_feature_series(cluster_values),
                }
            )

    return pd.DataFrame(rows)


def compute_top_features_per_cluster(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
    top_k: int,
    eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_means = clustered_valid_beats.loc[:, learning_input_columns].mean(axis=0)
    global_stds = clustered_valid_beats.loc[:, learning_input_columns].std(axis=0, ddof=0)

    for cluster_label in tqdm(
        get_dynamic_cluster_labels(clustered_valid_beats),
        desc="Top features",
        disable=not InterpretationConfig.SHOW_PROGRESS,
    ):
        if cluster_label == -1:
            continue
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["analysis_cluster_label"] == cluster_label]
        if cluster_frame.empty:
            continue

        cluster_means = cluster_frame.loc[:, learning_input_columns].mean(axis=0)
        feature_rows: list[dict[str, Any]] = []
        for feature_name in learning_input_columns:
            cluster_mean = float(cluster_means[feature_name])
            global_mean = float(global_means[feature_name])
            global_std = float(global_stds[feature_name])
            delta_mean = cluster_mean - global_mean
            effect_z = float(delta_mean / (global_std + eps))
            feature_rows.append(
                {
                    "cluster_label": int(cluster_label),
                    "feature_name": feature_name,
                    "cluster_mean": cluster_mean,
                    "global_mean": global_mean,
                    "delta_mean": delta_mean,
                    "effect_z": effect_z,
                }
            )

        ranked_rows = sorted(feature_rows, key=lambda row: abs(float(row["effect_z"])), reverse=True)[:top_k]
        for rank_index, row in enumerate(ranked_rows, start=1):
            row["rank_within_cluster"] = rank_index
            rows.append(row)

    return pd.DataFrame(rows)


def compute_feature_group_summary(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
    feature_groups: dict[str, list[str]],
    eps: float,
) -> dict[str, dict[str, Any]]:
    global_means = clustered_valid_beats.loc[:, learning_input_columns].mean(axis=0)
    global_stds = clustered_valid_beats.loc[:, learning_input_columns].std(axis=0, ddof=0)
    summary: dict[str, dict[str, Any]] = {}

    for cluster_label in tqdm(
        get_dynamic_cluster_labels(clustered_valid_beats),
        desc="Feature groups",
        disable=not InterpretationConfig.SHOW_PROGRESS,
    ):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["analysis_cluster_label"] == cluster_label]
        cluster_summary: dict[str, Any] = {}
        if cluster_frame.empty:
            summary[f"cluster_{cluster_label}"] = cluster_summary
            continue

        cluster_means = cluster_frame.loc[:, learning_input_columns].mean(axis=0)
        for group_name, group_columns in feature_groups.items():
            usable_columns = [column for column in group_columns if column in learning_input_columns]
            if not usable_columns:
                cluster_summary[group_name] = {
                    "group_feature_count": 0,
                    "mean_abs_effect_z": float(np.nan),
                    "top_features": [],
                }
                continue

            effect_rows: list[dict[str, Any]] = []
            for feature_name in usable_columns:
                delta_mean = float(cluster_means[feature_name] - global_means[feature_name])
                effect_z = float(delta_mean / (float(global_stds[feature_name]) + eps))
                effect_rows.append(
                    {
                        "feature_name": feature_name,
                        "effect_z": effect_z,
                    }
                )

            ranked_rows = sorted(effect_rows, key=lambda row: abs(float(row["effect_z"])), reverse=True)
            cluster_summary[group_name] = {
                "group_feature_count": int(len(usable_columns)),
                "mean_abs_effect_z": float(np.mean([abs(float(row["effect_z"])) for row in effect_rows])),
                "top_features": [str(row["feature_name"]) for row in ranked_rows[:5]],
            }

        summary[f"cluster_{cluster_label}"] = cluster_summary

    return summary


def compute_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return float(np.nan)
    probabilities = counts.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return float(np.nan)
    return float(-np.sum(probabilities * np.log(probabilities)))


def compute_record_cluster_distribution(clustered_valid_beats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_label in get_dynamic_cluster_labels(clustered_valid_beats):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["analysis_cluster_label"] == cluster_label]
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


# =================================================
# 4. Heuristic Summary Helpers
# =================================================


def _heuristic_signal(
    cluster_mean: float,
    global_mean: float,
    global_std: float,
    eps: float,
) -> float:
    return float((cluster_mean - global_mean) / (global_std + eps))


def build_cluster_heuristic_summary(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
    eps: float,
) -> dict[int, list[str]]:
    global_means = clustered_valid_beats.loc[:, learning_input_columns].mean(axis=0)
    global_stds = clustered_valid_beats.loc[:, learning_input_columns].std(axis=0, ddof=0)
    heuristics: dict[int, list[str]] = {}

    for cluster_label in get_dynamic_cluster_labels(clustered_valid_beats):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["analysis_cluster_label"] == cluster_label]
        cluster_means = cluster_frame.loc[:, learning_input_columns].mean(axis=0)
        messages: list[str] = []

        if cluster_label == -1:
            messages.append("noise-like pattern 가능성: density clustering 기준으로 안정된 군집 밖 샘플 비율이 높음")
            heuristics[cluster_label] = messages
            continue

        systole_energy_signal = _heuristic_signal(
            float(cluster_means.get("seg_sys_energy", 0.0)),
            float(global_means.get("seg_sys_energy", 0.0)),
            float(global_stds.get("seg_sys_energy", 0.0)),
            eps,
        )
        systole_occupancy_signal = _heuristic_signal(
            float(cluster_means.get("seg_sys_env_occupancy", 0.0)),
            float(global_means.get("seg_sys_env_occupancy", 0.0)),
            float(global_stds.get("seg_sys_env_occupancy", 0.0)),
            eps,
        )
        ed_energy_signal = _heuristic_signal(
            float(cluster_means.get("zone_ed_energy", 0.0)),
            float(global_means.get("zone_ed_energy", 0.0)),
            float(global_stds.get("zone_ed_energy", 0.0)),
            eps,
        )
        ld_peak_signal = _heuristic_signal(
            float(cluster_means.get("zone_ld_peak_rel_to_s1", 0.0)),
            float(global_means.get("zone_ld_peak_rel_to_s1", 0.0)),
            float(global_stds.get("zone_ld_peak_rel_to_s1", 0.0)),
            eps,
        )
        hr_signal = _heuristic_signal(
            float(cluster_means.get("global_hr_bpm", 0.0)),
            float(global_means.get("global_hr_bpm", 0.0)),
            float(global_stds.get("global_hr_bpm", 0.0)),
            eps,
        )

        if systole_energy_signal > 0.8 and systole_occupancy_signal > 0.8:
            messages.append("systole energy and occupancy가 높아 murmur-like pattern 가능성")
        if ed_energy_signal > 0.8:
            messages.append("early diastole zone energy가 높아 S3-like pattern 가능성")
        if ld_peak_signal > 0.8:
            messages.append("late diastole zone peak가 높아 S4-like pattern 가능성")
        if hr_signal > 0.8:
            messages.append("cycle HR이 높아 tachycardia-like pattern 가능성")
        if not messages:
            messages.append("특정 임상적 heuristic이 두드러지지 않는 mixed pattern")

        heuristics[cluster_label] = messages

    return heuristics


# =================================================
# 5. Representative Beat / Figure Helpers
# =================================================


def normalize_column_name(column_name: Any) -> str:
    return str(column_name).replace("\ufeff", "").strip()


def read_tabular_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(file_path)

    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in InterpretationConfig.CSV_ENCODING_CANDIDATES:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error
                continue
        raise ValueError(f"Unsupported CSV encoding in {file_path.name}") from last_error

    raise ValueError(f"Unsupported input file extension: {file_path.suffix}")


def load_source_signal(file_path: Path) -> np.ndarray:
    dataframe = read_tabular_file(file_path)
    dataframe = dataframe.copy()
    dataframe.columns = [normalize_column_name(column_name) for column_name in dataframe.columns]
    if "Amplitude" not in dataframe.columns:
        raise ValueError(f"Amplitude column is required in {file_path}")
    return pd.to_numeric(dataframe["Amplitude"], errors="raise").to_numpy(dtype=np.float32)


def compute_smoothed_envelope(x: np.ndarray, fs: float, smooth_ms: float) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)
    abs_values = np.abs(x.astype(np.float32, copy=False))
    radius = max(1, int(round((smooth_ms / 1000.0) * fs / 2.0)))
    prefix = np.zeros(abs_values.size + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(abs_values, dtype=np.float64)
    smoothed = np.zeros(abs_values.size, dtype=np.float32)
    for index in range(abs_values.size):
        start = max(0, index - radius)
        end = min(abs_values.size - 1, index + radius)
        smoothed[index] = float((prefix[end + 1] - prefix[start]) / max(1, end - start + 1))
    return smoothed


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        raise ImportError("matplotlib is required to export representative figures") from error

    return plt


def plot_representative_beat(
    signal: np.ndarray,
    envelope: np.ndarray,
    representative_row: pd.Series,
    save_path: Path,
    fs: float,
) -> None:
    plt = _load_matplotlib_pyplot()

    start = int(representative_row["s1_start"])
    end = int(representative_row["next_s1_start"])
    beat_signal = signal[start:end]
    beat_envelope = envelope[start:end]
    time_ms = np.arange(len(beat_signal), dtype=np.float64) * (1000.0 / float(fs))

    boundary_offsets = {
        "S1 start": int(representative_row["s1_start"]) - start,
        "S1 end": int(representative_row["s1_end"]) - start,
        "S2 start": int(representative_row["s2_start"]) - start,
        "S2 end": int(representative_row["s2_end"]) - start,
    }

    plt.figure(figsize=(10, 4))
    plt.plot(time_ms, beat_signal, color="#1f4e78", linewidth=1.2, label="PCG")
    plt.plot(time_ms, beat_envelope, color="#d95f02", linewidth=1.3, alpha=0.9, label="Smoothed envelope")
    for label, offset in boundary_offsets.items():
        if 0 <= offset < len(time_ms):
            plt.axvline(time_ms[offset], linestyle="--", linewidth=1.0, label=label)
    plt.title(
        f"cluster_{int(representative_row['cluster_label'])} / rank_{int(representative_row['rank_in_cluster'])} "
        f"/ {representative_row['record_id']} beat {int(representative_row['beat_index'])}"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    handles, labels = plt.gca().get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    plt.legend(dedup.values(), dedup.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_representative_beats(
    clustered_valid_beats: pd.DataFrame,
    cluster_exemplars: pd.DataFrame,
    heuristics: dict[int, list[str]],
    figures_root: Path,
    data_root: Path,
) -> pd.DataFrame:
    if cluster_exemplars.empty:
        return pd.DataFrame()

    merged = cluster_exemplars.merge(
        clustered_valid_beats,
        on=["record_id", "beat_index"],
        how="left",
        validate="one_to_one",
    )
    if merged.empty:
        return merged

    keep_columns = [
        "cluster_label",
        "exemplar_rank",
        "record_id",
        "beat_index",
        "source_file",
        "distance_to_cluster_mean",
        "analysis_cluster_label",
        "analysis_label_source",
        "analysis_membership_probability",
        "s1_start",
        "s1_end",
        "s2_start",
        "s2_end",
        "next_s1_start",
        "seg_s1_peak_env",
        "seg_sys_energy",
        "seg_s2_peak_env",
        "zone_ed_energy",
        "zone_ld_peak_rel_to_s1",
        "global_hr_bpm",
    ]
    keep_columns = [column for column in keep_columns if column in merged.columns]
    representative_beats = merged.loc[:, keep_columns].copy()
    representative_beats = representative_beats.rename(columns={"exemplar_rank": "rank_in_cluster"})
    representative_beats["heuristic_summary"] = representative_beats["cluster_label"].map(
        lambda label: " | ".join(heuristics.get(int(label), []))
    )
    representative_beats["waveform_figure_path"] = ""

    try:
        cached_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        row_iterator = tqdm(
            representative_beats.iterrows(),
            total=len(representative_beats),
            desc="Representative figures",
            disable=not InterpretationConfig.SHOW_PROGRESS,
        )
        for row_index, row in row_iterator:
            source_file = str(row["source_file"])
            source_path = data_root / source_file
            if source_file not in cached_signals:
                signal = load_source_signal(source_path)
                envelope = compute_smoothed_envelope(signal, fs=InterpretationConfig.SAMPLING_RATE, smooth_ms=InterpretationConfig.ENVELOPE_SMOOTH_MS)
                cached_signals[source_file] = (signal, envelope)
            signal, envelope = cached_signals[source_file]
            cluster_dir = figures_root / f"cluster_{int(row['cluster_label'])}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            save_path = cluster_dir / f"representative_rank_{int(row['rank_in_cluster'])}.png"
            plot_representative_beat(signal=signal, envelope=envelope, representative_row=row, save_path=save_path, fs=InterpretationConfig.SAMPLING_RATE)
            representative_beats.loc[row_index, "waveform_figure_path"] = str(save_path)
    except Exception as error:
        logger.warning("대표 beat figure 생성 생략: %s", error)

    return representative_beats


# =================================================
# 6. JSON / Excel Helpers
# =================================================


def build_json_summary(
    cluster_overview: pd.DataFrame,
    top_features_per_cluster: pd.DataFrame,
    feature_group_summary: dict[str, dict[str, Any]],
    representative_beats: pd.DataFrame,
    heuristics: dict[int, list[str]],
    clustering_summary: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "clustering_summary": clustering_summary,
        "clusters": {},
    }
    for _, overview_row in cluster_overview.iterrows():
        cluster_label = int(overview_row["cluster_label"])
        cluster_key = f"cluster_{cluster_label}"
        cluster_top_features = top_features_per_cluster.loc[
            top_features_per_cluster["cluster_label"] == cluster_label
        ].copy()
        cluster_representatives = representative_beats.loc[
            representative_beats["cluster_label"] == cluster_label
        ].copy()
        summary["clusters"][cluster_key] = {
            "beat_count": int(overview_row["beat_count"]),
            "beat_ratio": float(overview_row["beat_ratio"]),
            "num_records": int(overview_row["num_records"]),
            "is_noise": bool(overview_row["is_noise"]),
            "heuristic_summary": heuristics.get(cluster_label, []),
            "top_features": cluster_top_features.to_dict(orient="records"),
            "feature_group_summary": feature_group_summary.get(cluster_key, {}),
            "representative_beats": cluster_representatives.to_dict(orient="records"),
        }
    return summary


# =================================================
# 7. Main
# =================================================


def main() -> None:
    output_paths = ensure_output_directories(stage_output_folder=configured_path(InterpretationConfig.OUTPUT_FOLDER))
    interpretation_root = output_paths["interpretation_root"]
    figures_root = output_paths["figures_root"]

    preprocess_root = configured_path(InterpretationConfig.PREPROCESS_INPUT_FOLDER)
    clustering_root = configured_path(InterpretationConfig.CLUSTERING_INPUT_FOLDER)
    data_root = configured_path(InterpretationConfig.TRAIN_DATA_FOLDER)

    logger.info("전처리 입력 폴더: %s", preprocess_root)
    logger.info("클러스터링 입력 폴더: %s", clustering_root)
    logger.info("해석 출력 폴더: %s", interpretation_root)

    inputs = load_interpretation_inputs(preprocess_root=preprocess_root, clustering_root=clustering_root)
    latent_columns = validate_inputs(inputs)

    clustered_valid_beats = build_clustered_valid_beats(
        beat_features_valid=inputs["beat_features_valid"],
        train_assignments=inputs["hdbscan_labels_train"],
        all_valid_with_latent=inputs["all_valid_with_latent"],
    )
    clustered_valid_beats.to_csv(interpretation_root / InterpretationConfig.CLUSTERED_VALID_BEATS_FILENAME, index=False)

    learning_input_columns = list(inputs["learning_input_columns"])
    feature_groups = dict(inputs["feature_groups"])

    cluster_overview = compute_cluster_overview(clustered_valid_beats)
    feature_summary = compute_feature_summary(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=learning_input_columns,
    )
    top_features_per_cluster = compute_top_features_per_cluster(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=learning_input_columns,
        top_k=InterpretationConfig.TOP_FEATURES_PER_CLUSTER,
        eps=InterpretationConfig.EPS,
    )
    feature_group_summary = compute_feature_group_summary(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=learning_input_columns,
        feature_groups=feature_groups,
        eps=InterpretationConfig.EPS,
    )
    record_cluster_distribution = compute_record_cluster_distribution(clustered_valid_beats)
    heuristics = build_cluster_heuristic_summary(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=learning_input_columns,
        eps=InterpretationConfig.EPS,
    )
    representative_beats = build_representative_beats(
        clustered_valid_beats=clustered_valid_beats,
        cluster_exemplars=inputs["cluster_exemplars"],
        heuristics=heuristics,
        figures_root=figures_root,
        data_root=data_root,
    )

    cluster_overview.to_csv(interpretation_root / InterpretationConfig.CLUSTER_OVERVIEW_FILENAME, index=False)
    feature_summary.to_csv(interpretation_root / InterpretationConfig.FEATURE_SUMMARY_FILENAME, index=False)
    top_features_per_cluster.to_csv(interpretation_root / InterpretationConfig.TOP_FEATURES_FILENAME, index=False)
    representative_beats.to_csv(interpretation_root / InterpretationConfig.REPRESENTATIVE_BEATS_FILENAME, index=False)
    record_cluster_distribution.to_csv(
        interpretation_root / InterpretationConfig.RECORD_CLUSTER_DISTRIBUTION_FILENAME,
        index=False,
    )
    (interpretation_root / InterpretationConfig.FEATURE_GROUP_SUMMARY_FILENAME).write_text(
        json.dumps(feature_group_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    json_summary = build_json_summary(
        cluster_overview=cluster_overview,
        top_features_per_cluster=top_features_per_cluster,
        feature_group_summary=feature_group_summary,
        representative_beats=representative_beats,
        heuristics=heuristics,
        clustering_summary=inputs["clustering_summary"],
    )
    (interpretation_root / InterpretationConfig.JSON_SUMMARY_FILENAME).write_text(
        json.dumps(json_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    export_stage_workbook(
        workbook_path=interpretation_root / InterpretationConfig.EXCEL_REPORT_FILENAME,
        sheets={
            "cluster_overview": cluster_overview,
            "feature_summary": feature_summary,
            "top_features": top_features_per_cluster,
            "representative_beats": representative_beats,
            "record_cluster_distribution": record_cluster_distribution,
            "cluster_stability_summary": inputs["cluster_stability_summary"],
            "record_distribution_from_clustering": inputs["record_distribution_summary"],
        },
        freeze_panes=InterpretationConfig.EXCEL_FREEZE_PANES,
        header_fill=InterpretationConfig.EXCEL_HEADER_FILL,
        header_font_color=InterpretationConfig.EXCEL_HEADER_FONT_COLOR,
        max_column_width=InterpretationConfig.EXCEL_MAX_COLUMN_WIDTH,
    )

    logger.info(
        "해석 완료: clusters=%s, representative_beats=%s",
        cluster_overview["cluster_label"].tolist(),
        len(representative_beats),
    )


if __name__ == "__main__":
    main()
