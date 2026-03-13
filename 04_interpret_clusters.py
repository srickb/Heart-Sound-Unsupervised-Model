"""
Standalone interpretation script for unsupervised heart sound clusters.

Expected input files:
- outputs/{RUN_NAME}/preprocess/cycle_features.npy
- outputs/{RUN_NAME}/preprocess/cycle_waveforms.npy
- outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
- outputs/{RUN_NAME}/preprocess/feature_names.json
- outputs/{RUN_NAME}/clustering/embeddings.npy
- outputs/{RUN_NAME}/clustering/cluster_assignments.csv
- outputs/{RUN_NAME}/clustering/clustering_summary.json

Saved artifacts:
- outputs/{RUN_NAME}/interpretation/cluster_summary.csv
- outputs/{RUN_NAME}/interpretation/cluster_feature_stats.csv
- outputs/{RUN_NAME}/interpretation/cluster_group_summary.csv
- outputs/{RUN_NAME}/interpretation/representative_samples.csv
- outputs/{RUN_NAME}/interpretation/noise_analysis.csv
- outputs/{RUN_NAME}/interpretation/cluster_waveform_panels.png
- outputs/{RUN_NAME}/interpretation/cluster_feature_boxplots.png
- outputs/{RUN_NAME}/interpretation/interpretation_report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from excel_export_utils import export_stage_workbook

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Editable configuration
# ============================================================================
class InterpretationConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "test_dataset_260312_preprocess_v2"

    PREPROCESS_ROOT = OUTPUT_ROOT / RUN_NAME / "preprocess"
    CLUSTERING_ROOT = OUTPUT_ROOT / RUN_NAME / "clustering"
    INTERPRETATION_ROOT = OUTPUT_ROOT / RUN_NAME / "interpretation"
    REQUIRED_METADATA_COLUMNS = [
        "sample_id",
        "recording_id",
        "subject_id",
        "valid_flag",
        "feature_row_index",
        "waveform_row_index",
    ]

    REPRESENTATIVES_PER_CLUSTER = 3
    WAVEFORM_ENVELOPE_LOW_Q = 0.10
    WAVEFORM_ENVELOPE_HIGH_Q = 0.90

    EXCEL_EXPORT_ENABLED = True
    EXCEL_FILENAME = "interpretation_data_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40

    RANDOM_SEED = 42


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATHS = {
    "output_root": InterpretationConfig.OUTPUT_ROOT,
}

RUN_NAME = InterpretationConfig.RUN_NAME

DATA = {
    "preprocess_root": InterpretationConfig.PREPROCESS_ROOT,
    "clustering_root": InterpretationConfig.CLUSTERING_ROOT,
    "interpretation_root": InterpretationConfig.INTERPRETATION_ROOT,
    "required_metadata_columns": InterpretationConfig.REQUIRED_METADATA_COLUMNS,
}

PREPROCESS = {}

MODEL = {}

TRAINING = {}

EMBEDDING = {}

CLUSTERING = {
    "representatives_per_cluster": InterpretationConfig.REPRESENTATIVES_PER_CLUSTER,
    "waveform_envelope_low_q": InterpretationConfig.WAVEFORM_ENVELOPE_LOW_Q,
    "waveform_envelope_high_q": InterpretationConfig.WAVEFORM_ENVELOPE_HIGH_Q,
}

EXCEL = {
    "export_enabled": InterpretationConfig.EXCEL_EXPORT_ENABLED,
    "filename": InterpretationConfig.EXCEL_FILENAME,
    "freeze_panes": InterpretationConfig.EXCEL_FREEZE_PANES,
    "header_fill": InterpretationConfig.EXCEL_HEADER_FILL,
    "header_font_color": InterpretationConfig.EXCEL_HEADER_FONT_COLOR,
    "max_column_width": InterpretationConfig.EXCEL_MAX_COLUMN_WIDTH,
}

RANDOM_SEED = InterpretationConfig.RANDOM_SEED


# ============================================================================
# Dataset adapter section
# ============================================================================
def load_interpretation_inputs() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Load and align preprocessing outputs with clustering results.

    Output:
        merged_frame: DataFrame with one row per valid sample
        feature_matrix: array with shape (num_valid_samples, num_features)
        waveform_matrix: array with shape (num_valid_samples, fixed_length)
        embeddings: array with shape (num_valid_samples, latent_dim)
        feature_names: list with length num_features
        feature_metadata: metadata rows aligned to feature_names when available
        clustering_summary: summary JSON from the clustering stage
    """
    preprocess_root = DATA["preprocess_root"]
    clustering_root = DATA["clustering_root"]

    feature_path = preprocess_root / "cycle_features.npy"
    waveform_path = preprocess_root / "cycle_waveforms.npy"
    metadata_path = preprocess_root / "cycle_metadata.csv"
    feature_names_path = preprocess_root / "feature_names.json"
    feature_metadata_path = preprocess_root / "feature_metadata.json"
    embeddings_path = clustering_root / "embeddings.npy"
    assignments_path = clustering_root / "cluster_assignments.csv"
    clustering_summary_path = clustering_root / "clustering_summary.json"

    for path in [
        feature_path,
        waveform_path,
        metadata_path,
        feature_names_path,
        embeddings_path,
        assignments_path,
        clustering_summary_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required interpretation input is missing: {path}")

    feature_matrix = np.load(feature_path).astype(np.float32)
    waveform_matrix = np.load(waveform_path).astype(np.float32)
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    assignments = pd.read_csv(assignments_path)
    with open(feature_names_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)
    if feature_metadata_path.exists():
        with open(feature_metadata_path, "r", encoding="utf-8") as file:
            feature_metadata = json.load(file)
    else:
        feature_metadata = []
    with open(clustering_summary_path, "r", encoding="utf-8") as file:
        clustering_summary = json.load(file)

    missing_columns = [
        column for column in DATA["required_metadata_columns"] if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    valid_metadata = metadata.loc[metadata["valid_flag"] == True].copy()
    if len(valid_metadata) != feature_matrix.shape[0]:
        raise ValueError(
            "Valid metadata row count does not match feature matrix rows: "
            f"{len(valid_metadata)} vs {feature_matrix.shape[0]}"
        )
    if waveform_matrix.shape[0] != feature_matrix.shape[0]:
        raise ValueError(
            "Waveform matrix row count does not match feature matrix rows: "
            f"{waveform_matrix.shape[0]} vs {feature_matrix.shape[0]}"
        )
    if embeddings.shape[0] != feature_matrix.shape[0]:
        raise ValueError(
            "Embedding row count does not match feature matrix rows: "
            f"{embeddings.shape[0]} vs {feature_matrix.shape[0]}"
        )
    if len(feature_names) != feature_matrix.shape[1]:
        raise ValueError(
            "Feature name count does not match feature matrix width: "
            f"{len(feature_names)} vs {feature_matrix.shape[1]}"
        )

    valid_metadata["feature_row_index"] = valid_metadata["feature_row_index"].astype(int)
    valid_metadata["waveform_row_index"] = valid_metadata["waveform_row_index"].astype(int)
    valid_metadata = valid_metadata.sort_values(
        by="feature_row_index", kind="stable"
    ).reset_index(drop=True)

    expected_index = np.arange(len(valid_metadata), dtype=int)
    if not np.array_equal(valid_metadata["feature_row_index"].to_numpy(), expected_index):
        raise ValueError("feature_row_index is not globally aligned for valid cycles.")
    if not np.array_equal(valid_metadata["waveform_row_index"].to_numpy(), expected_index):
        raise ValueError("waveform_row_index is not globally aligned for valid cycles.")
    if valid_metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be unique across valid cycles.")

    overlapping_feature_columns = [
        column for column in feature_names if column in valid_metadata.columns
    ]
    if overlapping_feature_columns:
        valid_metadata = valid_metadata.drop(columns=overlapping_feature_columns)

    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names)
    embedding_frame = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{index}" for index in range(embeddings.shape[1])],
    )
    feature_frame["feature_row_index"] = np.arange(len(feature_frame), dtype=int)
    embedding_frame["feature_row_index"] = np.arange(len(embedding_frame), dtype=int)

    assignments["feature_row_index"] = assignments["feature_row_index"].astype(int)
    assignments = assignments.sort_values(
        by="feature_row_index", kind="stable"
    ).reset_index(drop=True)

    merged = valid_metadata.merge(feature_frame, on="feature_row_index", how="inner")
    merged = merged.merge(embedding_frame, on="feature_row_index", how="inner")
    merged = merged.merge(
        assignments,
        on=["sample_id", "subject_id", "recording_id", "feature_row_index", "waveform_row_index"],
        how="inner",
    )
    if len(merged) != feature_matrix.shape[0]:
        raise ValueError(
            "Merged interpretation table is not aligned to valid sample count: "
            f"{len(merged)} vs {feature_matrix.shape[0]}"
        )

    return (
        merged,
        feature_matrix,
        waveform_matrix,
        embeddings,
        feature_names,
        feature_metadata,
        clustering_summary,
    )


# ============================================================================
# Utility functions
# ============================================================================
def ensure_output_directories(output_root: Path, run_name: str) -> dict[str, Path]:
    """Create stable output directories for this run."""
    run_root = output_root / run_name
    interpretation_root = run_root / "interpretation"
    interpretation_root.mkdir(parents=True, exist_ok=True)
    return {"run_root": run_root, "interpretation_root": interpretation_root}


def cluster_name(label: int) -> str:
    """Return a neutral display name for a cluster label."""
    return "noise" if label == -1 else f"cluster_{label}"


def feature_group_from_name(feature_name: str) -> str:
    """Infer feature group from the configured prefixes."""
    prefix_map = {
        "rs_": "rs",
        "time_": "time",
        "amp_raw_": "amp_raw",
        "amp_norm_": "amp_norm",
        "ratio_": "ratio",
        "qc_": "qc",
    }
    for prefix, group_name in prefix_map.items():
        if feature_name.startswith(prefix):
            return group_name
    return "other"


def build_feature_group_map(
    feature_columns: list[str],
    feature_metadata: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Group feature columns by prefix so interpretation is prefix-driven."""
    groups = {
        "rs": [],
        "time": [],
        "amp_raw": [],
        "amp_norm": [],
        "ratio": [],
        "qc": [],
        "other": [],
    }

    if feature_metadata:
        metadata_by_name = {
            str(row["feature_name"]): str(row.get("feature_group", "other"))
            for row in feature_metadata
        }
        for column in feature_columns:
            groups.setdefault(metadata_by_name.get(column, "other"), []).append(column)
        return groups

    for column in feature_columns:
        groups.setdefault(feature_group_from_name(column), []).append(column)
    return groups


def preferred_report_features(
    feature_group_map: dict[str, list[str]],
    max_total: int = 6,
) -> list[str]:
    """Choose a compact report feature set from the available prefixed groups."""
    preferred_patterns = [
        "cycle_duration_sec",
        "s1_duration_sec",
        "systole_duration_sec",
        "s2_duration_sec",
        "diastole_duration_sec",
        "cycle_rms",
        "cycle_peak_to_peak",
        "cycle_energy",
        "cycle_max_abs",
        "s1_rms",
        "s2_rms",
        "duration_over",
    ]
    candidate_pool = (
        feature_group_map.get("time", [])
        + feature_group_map.get("amp_raw", [])
        + feature_group_map.get("amp_norm", [])
        + feature_group_map.get("ratio", [])
    )

    chosen: list[str] = []
    for pattern in preferred_patterns:
        match = next((name for name in candidate_pool if pattern in name and name not in chosen), None)
        if match is not None:
            chosen.append(match)
        if len(chosen) >= max_total:
            return chosen

    for column in candidate_pool:
        if column not in chosen:
            chosen.append(column)
        if len(chosen) >= max_total:
            break
    return chosen


def standardized_effect(cluster_values: pd.Series, baseline_values: pd.Series) -> float:
    """Compute a stable standardized mean difference."""
    baseline_std = float(baseline_values.std(ddof=0))
    if baseline_std == 0.0:
        return 0.0
    return float((cluster_values.mean() - baseline_values.mean()) / baseline_std)


def build_cluster_summary(
    merged: pd.DataFrame,
    feature_group_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Build one summary row per cluster with prefix-group level tendencies."""
    rows: list[dict[str, Any]] = []
    total_count = len(merged)
    baseline = merged.copy()

    for label in sorted(merged["cluster_label"].unique()):
        cluster_frame = merged.loc[merged["cluster_label"] == label].copy()
        row: dict[str, Any] = {
            "cluster_label": int(label),
            "cluster_name": cluster_name(int(label)),
            "sample_count": int(len(cluster_frame)),
            "proportion": float(len(cluster_frame) / total_count),
            "noise_proportion": float((merged["cluster_label"] == -1).mean()),
        }

        for group_name in ["rs", "time", "amp_raw", "amp_norm", "ratio", "qc"]:
            group_features = feature_group_map.get(group_name, [])
            if not group_features:
                row[f"{group_name}_feature_count"] = 0
                row[f"{group_name}_top_feature"] = ""
                row[f"{group_name}_top_effect"] = 0.0
                continue

            effect_rows: list[tuple[str, float]] = []
            for feature_name in group_features:
                effect_rows.append(
                    (
                        feature_name,
                        standardized_effect(cluster_frame[feature_name], baseline[feature_name]),
                    )
                )

            effect_rows.sort(key=lambda item: abs(item[1]), reverse=True)
            top_feature, top_effect = effect_rows[0]
            row[f"{group_name}_feature_count"] = int(len(group_features))
            row[f"{group_name}_top_feature"] = top_feature
            row[f"{group_name}_top_effect"] = float(top_effect)
        rows.append(row)

    return pd.DataFrame(rows)


def build_cluster_feature_stats(
    merged: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Create a long-format feature statistics table by cluster."""
    rows: list[dict[str, Any]] = []
    for label in sorted(merged["cluster_label"].unique()):
        cluster_frame = merged.loc[merged["cluster_label"] == label]
        for feature_name in selected_features:
            values = cluster_frame[feature_name].to_numpy(dtype=np.float32)
            rows.append(
                {
                    "cluster_label": int(label),
                    "cluster_name": cluster_name(int(label)),
                    "feature_name": feature_name,
                    "feature_group": feature_group_from_name(feature_name),
                    "sample_count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def build_cluster_group_summary(
    merged: pd.DataFrame,
    feature_group_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize each cluster by feature-group level standardized effects."""
    rows: list[dict[str, Any]] = []
    baseline = merged.copy()

    for label in sorted(merged["cluster_label"].unique()):
        cluster_frame = merged.loc[merged["cluster_label"] == label].copy()
        for group_name, feature_names in feature_group_map.items():
            if group_name == "other" or not feature_names:
                continue

            effect_rows = [
                (
                    feature_name,
                    standardized_effect(cluster_frame[feature_name], baseline[feature_name]),
                )
                for feature_name in feature_names
            ]
            effect_rows.sort(key=lambda item: abs(item[1]), reverse=True)
            top_feature, top_effect = effect_rows[0]
            rows.append(
                {
                    "cluster_label": int(label),
                    "cluster_name": cluster_name(int(label)),
                    "feature_group": group_name,
                    "feature_count": int(len(feature_names)),
                    "top_feature": top_feature,
                    "top_effect": float(top_effect),
                    "mean_abs_effect": float(np.mean(np.abs([effect for _, effect in effect_rows]))),
                }
            )

    return pd.DataFrame(rows)


def representative_samples(
    merged: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Choose representative samples per cluster using probability and centroid distance."""
    rows: list[dict[str, Any]] = []

    for label in sorted(merged["cluster_label"].unique()):
        cluster_frame = merged.loc[merged["cluster_label"] == label].copy()
        embedding_rows = cluster_frame["feature_row_index"].to_numpy(dtype=int)
        cluster_embeddings = embeddings[embedding_rows]

        if label == -1:
            ranking_score = cluster_frame["outlier_score"].to_numpy(dtype=np.float32)
            order = np.argsort(ranking_score)
            selection_method = "lowest_outlier_score"
            distances = np.full(len(cluster_frame), np.nan, dtype=np.float32)
        else:
            centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            probabilities = cluster_frame["membership_probability"].to_numpy(dtype=np.float32)
            order = np.lexsort((distances, -probabilities))
            selection_method = "highest_probability_then_centroid_distance"

        chosen = cluster_frame.iloc[order[: CLUSTERING["representatives_per_cluster"]]].copy()
        for rank, (_, sample_row) in enumerate(chosen.iterrows(), start=1):
            rows.append(
                {
                    "cluster_label": int(label),
                    "cluster_name": cluster_name(int(label)),
                    "representative_rank": rank,
                    "selection_method": selection_method,
                    "sample_id": sample_row["sample_id"],
                    "subject_id": sample_row["subject_id"],
                    "recording_id": sample_row["recording_id"],
                    "feature_row_index": int(sample_row["feature_row_index"]),
                    "waveform_row_index": int(sample_row["waveform_row_index"]),
                    "membership_probability": float(sample_row.get("membership_probability", np.nan)),
                    "outlier_score": float(sample_row.get("outlier_score", np.nan)),
                }
            )

    return pd.DataFrame(rows)


def noise_analysis_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Analyze noise samples for duration/amplitude extremes and structural inconsistency."""
    noise_frame = merged.loc[merged["cluster_label"] == -1].copy()
    if noise_frame.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "subject_id",
                "recording_id",
                "feature_row_index",
                "waveform_row_index",
                "duration_zscore",
                "energy_zscore",
                "peak_to_peak_zscore",
                "candidate_mismatch_score",
                "duration_outlier_flag",
                "amplitude_outlier_flag",
                "candidate_mismatch_flag",
                "likely_artifact_like_flag",
            ]
        )

    reference_frame = merged.loc[merged["cluster_label"] != -1].copy()
    if reference_frame.empty:
        reference_frame = merged.copy()

    def zscore(values: pd.Series, reference: pd.Series) -> np.ndarray:
        ref_std = float(reference.std(ddof=0))
        if ref_std == 0:
            return np.zeros(len(values), dtype=np.float32)
        return ((values - float(reference.mean())) / ref_std).to_numpy(dtype=np.float32)

    duration_column = (
        "time_cycle_duration_sec"
        if "time_cycle_duration_sec" in merged.columns
        else "cycle_duration_sec"
    )
    energy_column = (
        "amp_raw_cycle_energy"
        if "amp_raw_cycle_energy" in merged.columns
        else "cycle_energy"
    )
    peak_column = (
        "amp_raw_cycle_peak_to_peak"
        if "amp_raw_cycle_peak_to_peak" in merged.columns
        else "cycle_peak_to_peak"
    )
    s1_count_column = (
        "qc_num_s1_end_candidates"
        if "qc_num_s1_end_candidates" in merged.columns
        else "num_s1_end_candidates"
    )
    s2_start_count_column = (
        "qc_num_s2_start_candidates"
        if "qc_num_s2_start_candidates" in merged.columns
        else "num_s2_start_candidates"
    )
    s2_end_count_column = (
        "qc_num_s2_end_candidates"
        if "qc_num_s2_end_candidates" in merged.columns
        else "num_s2_end_candidates"
    )

    duration_z = zscore(noise_frame[duration_column], reference_frame[duration_column])
    energy_z = zscore(noise_frame[energy_column], reference_frame[energy_column])
    peak_z = zscore(noise_frame[peak_column], reference_frame[peak_column])

    candidate_mismatch_score = (
        np.abs(noise_frame[s1_count_column] - 1)
        + np.abs(noise_frame[s2_start_count_column] - 1)
        + np.abs(noise_frame[s2_end_count_column] - 1)
    ).to_numpy(dtype=np.int32)

    output = noise_frame[
        ["sample_id", "subject_id", "recording_id", "feature_row_index", "waveform_row_index"]
    ].copy()
    output["duration_zscore"] = duration_z
    output["energy_zscore"] = energy_z
    output["peak_to_peak_zscore"] = peak_z
    output["candidate_mismatch_score"] = candidate_mismatch_score
    output["duration_outlier_flag"] = np.abs(duration_z) >= 2.0
    output["amplitude_outlier_flag"] = np.maximum(np.abs(energy_z), np.abs(peak_z)) >= 2.0
    output["candidate_mismatch_flag"] = candidate_mismatch_score > 0
    output["likely_artifact_like_flag"] = (
        output["duration_outlier_flag"]
        | output["amplitude_outlier_flag"]
        | output["candidate_mismatch_flag"]
    )
    return output


def save_waveform_panels(
    merged: pd.DataFrame,
    waveforms: np.ndarray,
    representatives: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot cluster-level waveform envelopes and representative examples."""
    labels = sorted(merged["cluster_label"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 3.5 * len(labels)), squeeze=False)
    x_axis = np.arange(waveforms.shape[1])

    for axis, label in zip(axes.ravel(), labels):
        cluster_frame = merged.loc[merged["cluster_label"] == label]
        waveform_rows = cluster_frame["waveform_row_index"].to_numpy(dtype=int)
        cluster_waveforms = waveforms[waveform_rows]

        median_waveform = np.median(cluster_waveforms, axis=0)
        low_band = np.quantile(cluster_waveforms, CLUSTERING["waveform_envelope_low_q"], axis=0)
        high_band = np.quantile(cluster_waveforms, CLUSTERING["waveform_envelope_high_q"], axis=0)

        axis.fill_between(x_axis, low_band, high_band, color="#9ecae1", alpha=0.35)
        axis.plot(x_axis, median_waveform, color="#08519c", linewidth=2.0, label="median")

        reps = representatives.loc[representatives["cluster_label"] == label]
        for _, rep_row in reps.iterrows():
            rep_waveform = waveforms[int(rep_row["waveform_row_index"])]
            axis.plot(x_axis, rep_waveform, linewidth=0.9, alpha=0.7)

        axis.set_title(
            f"{cluster_name(int(label))} | n={len(cluster_frame)} | "
            f"proportion={len(cluster_frame) / len(merged):.3f}"
        )
        axis.set_xlabel("Sample")
        axis.set_ylabel("Normalized amplitude")
        axis.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_boxplots(
    merged: pd.DataFrame,
    selected_features: list[str],
    output_path: Path,
) -> None:
    """Plot simple feature boxplots across clusters."""
    labels = sorted(merged["cluster_label"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    if not selected_features:
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    for axis, feature_name in zip(axes, selected_features):
        data = [
            merged.loc[merged["cluster_label"] == label, feature_name].to_numpy(dtype=np.float32)
            for label in labels
        ]
        axis.boxplot(
            data,
            tick_labels=[cluster_name(int(label)) for label in labels],
            showfliers=False,
        )
        axis.set_title(feature_name)
        axis.tick_params(axis="x", rotation=20)

    for axis in axes[len(selected_features) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def feature_difference_text(
    merged: pd.DataFrame,
    label: int,
    report_features: list[str],
) -> str:
    """Summarize top feature differences for one cluster in neutral language."""
    cluster_frame = merged.loc[merged["cluster_label"] == label]
    baseline = merged.copy()
    rows: list[tuple[str, float, float]] = []

    for feature_name in report_features:
        baseline_std = float(baseline[feature_name].std(ddof=0))
        if baseline_std == 0:
            continue
        effect = float((cluster_frame[feature_name].mean() - baseline[feature_name].mean()) / baseline_std)
        rows.append((feature_name, effect, float(cluster_frame[feature_name].mean())))

    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    top_rows = rows[:3]
    if not top_rows:
        return "No strong feature shifts were available from the selected summary features."

    descriptions = []
    for feature_name, effect, _ in top_rows:
        direction = "higher" if effect > 0 else "lower"
        readable_name = feature_name.replace("_", " ")
        descriptions.append(f"{direction} {readable_name}")
    return ", ".join(descriptions)


def write_interpretation_report(
    merged: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    noise_analysis: pd.DataFrame,
    clustering_summary: dict[str, Any],
    report_features: list[str],
    output_path: Path,
) -> None:
    """Write a concise markdown report for manual scientific inspection."""
    non_noise_labels = [label for label in sorted(merged["cluster_label"].unique()) if label != -1]
    lines = [
        "# Cluster Interpretation Report",
        "",
        "## Overview",
        f"- Clusters excluding noise: {clustering_summary['number_of_clusters_excluding_noise']}",
        f"- Noise ratio: {clustering_summary['noise_ratio']:.3f}",
        f"- Total valid samples: {clustering_summary['total_valid_samples']}",
        "",
        "## Cluster Sizes",
    ]

    for _, row in cluster_summary.iterrows():
        lines.append(
            f"- {row['cluster_name']}: n={int(row['sample_count'])}, proportion={row['proportion']:.3f}"
        )

    lines.extend(["", "## Cluster Tendencies"])
    for label in non_noise_labels:
        rep_ids = representatives.loc[
            representatives["cluster_label"] == label, "sample_id"
        ].tolist()
        lines.append(f"### {cluster_name(int(label))}")
        lines.append(
            f"- Representative samples: {', '.join(rep_ids) if rep_ids else 'none'}"
        )
        lines.append(
            f"- Main feature shifts: {feature_difference_text(merged, int(label), report_features)}."
        )
        lines.append(
            "- Interpretation note: waveform differences should be treated as structural tendencies only, not clinical labels."
        )

    lines.extend(["", "## Noise Group"])
    if noise_analysis.empty:
        lines.append("- No noise samples were present.")
    else:
        artifact_ratio = float(noise_analysis["likely_artifact_like_flag"].mean())
        lines.append(f"- Noise samples: {len(noise_analysis)}")
        lines.append(f"- Artifact-like flag ratio within noise: {artifact_ratio:.3f}")
        lines.append(
            "- Noise appears to include cycles with extreme duration/amplitude values or less consistent structure, but this is only a heuristic inspection."
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- Clusters are unsupervised and should be interpreted cautiously.",
            "- The PCA visualization is a convenience view, not evidence of separability on its own.",
            "- No clinical labels were used, so cluster meaning remains descriptive rather than diagnostic.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_interpretation_excel(
    interpretation_root: Path,
    clustering_summary: dict[str, Any],
    merged: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_feature_stats: pd.DataFrame,
    cluster_group_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    noise_analysis: pd.DataFrame,
) -> Path:
    """Export interpretation-stage tables to an Excel workbook."""
    overview_rows = [
        {"section": "summary", "metric": "run_name", "value": clustering_summary["run_name"]},
        {"section": "summary", "metric": "total_valid_samples", "value": clustering_summary["total_valid_samples"]},
        {
            "section": "summary",
            "metric": "number_of_clusters_excluding_noise",
            "value": clustering_summary["number_of_clusters_excluding_noise"],
        },
        {"section": "summary", "metric": "number_of_noise_samples", "value": clustering_summary["number_of_noise_samples"]},
        {"section": "summary", "metric": "noise_ratio", "value": clustering_summary["noise_ratio"]},
    ]
    workbook_path = interpretation_root / EXCEL["filename"]
    return export_stage_workbook(
        workbook_path=workbook_path,
        sheets={
            "Overview": pd.DataFrame(overview_rows),
            "Cluster_Summary": cluster_summary,
            "Cluster_Feature_Stats": cluster_feature_stats,
            "Cluster_Group_Summary": cluster_group_summary,
            "Representative_Samples": representatives,
            "Noise_Analysis": noise_analysis,
            "Merged_View": merged,
        },
        freeze_panes=EXCEL["freeze_panes"],
        header_fill=EXCEL["header_fill"],
        header_font_color=EXCEL["header_font_color"],
        max_column_width=EXCEL["max_column_width"],
    )


def main() -> None:
    """Run the cluster interpretation pipeline and save readable outputs."""
    output_paths = ensure_output_directories(PATHS["output_root"], RUN_NAME)
    (
        merged,
        feature_matrix,
        waveform_matrix,
        embeddings,
        feature_names,
        feature_metadata,
        clustering_summary,
    ) = load_interpretation_inputs()

    feature_columns = feature_names

    feature_group_map = build_feature_group_map(feature_columns, feature_metadata)
    report_features = preferred_report_features(feature_group_map)
    selected_boxplot_features = report_features[:6]

    cluster_summary = build_cluster_summary(merged, feature_group_map)
    cluster_feature_stats = build_cluster_feature_stats(merged, selected_features=report_features)
    cluster_group_summary = build_cluster_group_summary(merged, feature_group_map)
    representatives = representative_samples(merged, embeddings)
    noise_analysis = noise_analysis_table(merged)

    interpretation_root = output_paths["interpretation_root"]
    cluster_summary.to_csv(interpretation_root / "cluster_summary.csv", index=False)
    cluster_feature_stats.to_csv(interpretation_root / "cluster_feature_stats.csv", index=False)
    cluster_group_summary.to_csv(interpretation_root / "cluster_group_summary.csv", index=False)
    representatives.to_csv(interpretation_root / "representative_samples.csv", index=False)
    noise_analysis.to_csv(interpretation_root / "noise_analysis.csv", index=False)

    save_waveform_panels(
        merged=merged,
        waveforms=waveform_matrix,
        representatives=representatives,
        output_path=interpretation_root / "cluster_waveform_panels.png",
    )
    save_feature_boxplots(
        merged=merged,
        selected_features=selected_boxplot_features,
        output_path=interpretation_root / "cluster_feature_boxplots.png",
    )
    write_interpretation_report(
        merged=merged,
        cluster_summary=cluster_summary,
        representatives=representatives,
        noise_analysis=noise_analysis,
        clustering_summary=clustering_summary,
        report_features=report_features,
        output_path=interpretation_root / "interpretation_report.md",
    )

    excel_path = None
    if EXCEL["export_enabled"]:
        excel_path = export_interpretation_excel(
            interpretation_root=interpretation_root,
            clustering_summary=clustering_summary,
            merged=merged,
            cluster_summary=cluster_summary,
            cluster_feature_stats=cluster_feature_stats,
            cluster_group_summary=cluster_group_summary,
            representatives=representatives,
            noise_analysis=noise_analysis,
        )

    logger.info("Saved interpretation outputs to: %s", interpretation_root)
    if excel_path is not None:
        logger.info("Saved interpretation Excel export to: %s", excel_path)
    logger.info("Cluster summary rows: %s", len(cluster_summary))
    logger.info("Representative sample rows: %s", len(representatives))
    logger.info("Noise analysis rows: %s", len(noise_analysis))


if __name__ == "__main__":
    main()
