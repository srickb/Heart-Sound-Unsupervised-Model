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
- outputs/{RUN_NAME}/interpretation/representative_samples.csv
- outputs/{RUN_NAME}/interpretation/noise_analysis.csv
- outputs/{RUN_NAME}/interpretation/cluster_waveform_panels.png
- outputs/{RUN_NAME}/interpretation/cluster_feature_boxplots.png
- outputs/{RUN_NAME}/interpretation/interpretation_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Editable configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PATHS = {
    "output_root": PROJECT_ROOT / "outputs",
}

RUN_NAME = "test_dataset_260312_preprocess_v1"

DATA = {
    "preprocess_root": PATHS["output_root"] / RUN_NAME / "preprocess",
    "clustering_root": PATHS["output_root"] / RUN_NAME / "clustering",
    "interpretation_root": PATHS["output_root"] / RUN_NAME / "interpretation",
    "required_metadata_columns": [
        "sample_id",
        "recording_id",
        "subject_id",
        "valid_flag",
        "feature_row_index",
        "waveform_row_index",
    ],
}

PREPROCESS = {}

MODEL = {}

TRAINING = {}

EMBEDDING = {}

CLUSTERING = {
    "representatives_per_cluster": 3,
    "waveform_envelope_low_q": 0.10,
    "waveform_envelope_high_q": 0.90,
}

RANDOM_SEED = 42


# ============================================================================
# Dataset adapter section
# ============================================================================
def load_interpretation_inputs() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]
]:
    """
    Load and align preprocessing outputs with clustering results.

    Output:
        merged_frame: DataFrame with one row per valid sample
        feature_matrix: array with shape (num_valid_samples, num_features)
        waveform_matrix: array with shape (num_valid_samples, fixed_length)
        embeddings: array with shape (num_valid_samples, latent_dim)
        feature_names: list with length num_features
        clustering_summary: summary JSON from the clustering stage
    """
    preprocess_root = DATA["preprocess_root"]
    clustering_root = DATA["clustering_root"]

    feature_path = preprocess_root / "cycle_features.npy"
    waveform_path = preprocess_root / "cycle_waveforms.npy"
    metadata_path = preprocess_root / "cycle_metadata.csv"
    feature_names_path = preprocess_root / "feature_names.json"
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

    return merged, feature_matrix, waveform_matrix, embeddings, feature_names, clustering_summary


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


def important_feature_groups(feature_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Choose timing, amplitude, and report features from the available feature set."""
    timing_candidates = [
        "cycle_duration_sec",
        "s1_to_s2_start_sec",
        "s2_end_to_next_s1_sec",
        "systole_duration_sec",
        "diastole_duration_sec",
    ]
    amplitude_candidates = [
        "cycle_rms",
        "cycle_peak_to_peak",
        "cycle_energy",
        "cycle_abs_area",
        "cycle_max_abs",
    ]
    report_candidates = [
        "cycle_duration_sec",
        "s1_to_s2_start_sec",
        "s2_end_to_next_s1_sec",
        "cycle_rms",
        "cycle_peak_to_peak",
        "cycle_energy",
    ]
    timing_features = [name for name in timing_candidates if name in feature_columns]
    amplitude_features = [name for name in amplitude_candidates if name in feature_columns]
    report_features = [name for name in report_candidates if name in feature_columns]
    return timing_features, amplitude_features, report_features


def build_cluster_summary(
    merged: pd.DataFrame,
    timing_features: list[str],
    amplitude_features: list[str],
) -> pd.DataFrame:
    """Build one summary row per cluster with count and key feature statistics."""
    rows: list[dict[str, Any]] = []
    total_count = len(merged)

    for label in sorted(merged["cluster_label"].unique()):
        cluster_frame = merged.loc[merged["cluster_label"] == label].copy()
        row: dict[str, Any] = {
            "cluster_label": int(label),
            "cluster_name": cluster_name(int(label)),
            "sample_count": int(len(cluster_frame)),
            "proportion": float(len(cluster_frame) / total_count),
            "noise_proportion": float((merged["cluster_label"] == -1).mean()),
        }
        for feature_name in timing_features:
            row[f"{feature_name}_mean"] = float(cluster_frame[feature_name].mean())
            row[f"{feature_name}_median"] = float(cluster_frame[feature_name].median())
        for feature_name in amplitude_features:
            row[f"{feature_name}_mean"] = float(cluster_frame[feature_name].mean())
            row[f"{feature_name}_std"] = float(cluster_frame[feature_name].std(ddof=0))
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
                    "sample_count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
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

    duration_z = zscore(noise_frame["cycle_duration_sec"], reference_frame["cycle_duration_sec"])
    energy_z = zscore(noise_frame["cycle_energy"], reference_frame["cycle_energy"])
    peak_z = zscore(noise_frame["cycle_peak_to_peak"], reference_frame["cycle_peak_to_peak"])

    candidate_mismatch_score = (
        np.abs(noise_frame["num_s1_end_candidates"] - 1)
        + np.abs(noise_frame["num_s2_start_candidates"] - 1)
        + np.abs(noise_frame["num_s2_end_candidates"] - 1)
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


def main() -> None:
    """Run the cluster interpretation pipeline and save readable outputs."""
    output_paths = ensure_output_directories(PATHS["output_root"], RUN_NAME)
    merged, feature_matrix, waveform_matrix, embeddings, feature_names, clustering_summary = (
        load_interpretation_inputs()
    )

    feature_columns = feature_names

    timing_features, amplitude_features, report_features = important_feature_groups(feature_columns)
    selected_boxplot_features = report_features[:6]

    cluster_summary = build_cluster_summary(merged, timing_features, amplitude_features)
    cluster_feature_stats = build_cluster_feature_stats(
        merged, selected_features=timing_features + amplitude_features
    )
    representatives = representative_samples(merged, embeddings)
    noise_analysis = noise_analysis_table(merged)

    interpretation_root = output_paths["interpretation_root"]
    cluster_summary.to_csv(interpretation_root / "cluster_summary.csv", index=False)
    cluster_feature_stats.to_csv(interpretation_root / "cluster_feature_stats.csv", index=False)
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

    print(f"Saved interpretation outputs to: {interpretation_root}")
    print(f"Cluster summary rows: {len(cluster_summary)}")
    print(f"Representative sample rows: {len(representatives)}")
    print(f"Noise analysis rows: {len(noise_analysis)}")


if __name__ == "__main__":
    main()
