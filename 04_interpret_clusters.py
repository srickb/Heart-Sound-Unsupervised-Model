"""
Cluster interpretation script fixed to the current unsupervised design.

Expected input files:
- outputs/beat_level_preprocess_fixed/preprocess/beat_features_valid.csv
- outputs/beat_level_preprocess_fixed/preprocess/learning_input_columns.json
- outputs/beat_level_preprocess_fixed/preprocess/feature_names.json
- outputs/cluster_training_fixed/clustering/cluster_assignments_valid.csv
- outputs/cluster_training_fixed/clustering/cluster_centers.npy

Optional input files:
- outputs/beat_level_preprocess_fixed/preprocess/beat_features_all.csv
- outputs/beat_level_preprocess_fixed/preprocess/record_summary.csv

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

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from excel_export_utils import export_stage_workbook

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class InterpretationConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    INPUT_ROOT = PROJECT_ROOT / "outputs" / "cluster_training_fixed" / "clustering"
    PREPROCESS_INPUT_ROOT = PROJECT_ROOT / "outputs" / "beat_level_preprocess_fixed" / "preprocess"
    DATA_ROOT = PROJECT_ROOT / "Data" / "Test_Dataset_260312"
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "cluster_interpretation_fixed"
    RANDOM_SEED = 42
    NUM_CLUSTERS = 4
    TOP_FEATURES_PER_CLUSTER = 10
    REPRESENTATIVE_BEATS_PER_CLUSTER = 5
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1

    BEAT_FEATURES_VALID_FILENAME = "beat_features_valid.csv"
    LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"
    FEATURE_NAMES_FILENAME = "feature_names.json"
    BEAT_FEATURES_ALL_FILENAME = "beat_features_all.csv"
    RECORD_SUMMARY_FILENAME = "record_summary.csv"
    ASSIGNMENTS_FILENAME = "cluster_assignments_valid.csv"
    CLUSTER_CENTERS_FILENAME = "cluster_centers.npy"

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

    SAMPLING_RATE = 4000
    ENVELOPE_SMOOTH_MS = 10.0
    CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    EPS = 1e-12


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


FEATURE_GROUP_PREFIXES = ["time_", "amp_", "shape_", "stat_", "stab_"]


def ensure_output_directories(output_root: Path, run_name: str) -> dict[str, Path]:
    run_root = output_root / run_name
    preprocess_root = run_root / "preprocess"
    training_root = run_root / "training"
    clustering_root = run_root / "clustering"
    interpretation_root = run_root / "interpretation"
    figures_root = interpretation_root / "figures"

    for path in [preprocess_root, training_root, clustering_root, interpretation_root, figures_root]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "preprocess_root": preprocess_root,
        "training_root": training_root,
        "clustering_root": clustering_root,
        "interpretation_root": interpretation_root,
        "figures_root": figures_root,
    }


def load_json_list(file_path: Path) -> list[str]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected JSON string list: {file_path}")
    return values


def load_optional_csv(file_path: Path) -> pd.DataFrame | None:
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_interpretation_inputs(
    preprocess_root: Path,
    clustering_root: Path,
) -> dict[str, Any]:
    beat_features_valid_path = preprocess_root / InterpretationConfig.BEAT_FEATURES_VALID_FILENAME
    learning_input_columns_path = preprocess_root / InterpretationConfig.LEARNING_INPUT_COLUMNS_FILENAME
    feature_names_path = preprocess_root / InterpretationConfig.FEATURE_NAMES_FILENAME
    assignments_path = clustering_root / InterpretationConfig.ASSIGNMENTS_FILENAME
    cluster_centers_path = clustering_root / InterpretationConfig.CLUSTER_CENTERS_FILENAME

    missing_paths = [
        path
        for path in [
            beat_features_valid_path,
            learning_input_columns_path,
            feature_names_path,
            assignments_path,
            cluster_centers_path,
        ]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing interpretation inputs: {missing_paths}")

    inputs = {
        "beat_features_valid": pd.read_csv(beat_features_valid_path),
        "learning_input_columns": load_json_list(learning_input_columns_path),
        "feature_names": load_json_list(feature_names_path),
        "cluster_assignments_valid": pd.read_csv(assignments_path),
        "cluster_centers": np.load(cluster_centers_path),
        "beat_features_all": load_optional_csv(preprocess_root / InterpretationConfig.BEAT_FEATURES_ALL_FILENAME),
        "record_summary": load_optional_csv(preprocess_root / InterpretationConfig.RECORD_SUMMARY_FILENAME),
    }
    logger.info("input file load 완료")
    return inputs


def validate_inputs(inputs: dict[str, Any]) -> None:
    beat_features_valid = inputs["beat_features_valid"]
    learning_input_columns = inputs["learning_input_columns"]
    feature_names = inputs["feature_names"]
    cluster_assignments_valid = inputs["cluster_assignments_valid"]
    cluster_centers = inputs["cluster_centers"]

    required_feature_columns = ["record_id", "beat_index", "valid_flag", "source_file", "s1_on", "s1_off", "s2_on", "s2_off", "s1_on_next"]
    missing_feature_columns = [column for column in required_feature_columns if column not in beat_features_valid.columns]
    if missing_feature_columns:
        raise ValueError(f"Missing required columns in beat_features_valid.csv: {missing_feature_columns}")

    missing_learning_columns = [column for column in learning_input_columns if column not in beat_features_valid.columns]
    if missing_learning_columns:
        raise ValueError(f"Missing learning input columns in beat_features_valid.csv: {missing_learning_columns}")

    unknown_learning_columns = [column for column in learning_input_columns if column not in feature_names]
    if unknown_learning_columns:
        raise ValueError(f"Unknown learning input columns: {unknown_learning_columns}")

    required_assignment_columns = ["record_id", "beat_index", "cluster_label", "cluster_confidence"]
    required_assignment_columns += [f"q_cluster_{index}" for index in range(InterpretationConfig.NUM_CLUSTERS)]
    required_assignment_columns += [f"latent_{index:02d}" for index in range(cluster_centers.shape[1])]
    missing_assignment_columns = [column for column in required_assignment_columns if column not in cluster_assignments_valid.columns]
    if missing_assignment_columns:
        raise ValueError(f"Missing required columns in cluster_assignments_valid.csv: {missing_assignment_columns}")

    if cluster_centers.shape != (InterpretationConfig.NUM_CLUSTERS, 12):
        raise ValueError(
            f"cluster_centers.npy must have shape {(InterpretationConfig.NUM_CLUSTERS, 12)}, got {cluster_centers.shape}"
        )


def build_clustered_valid_beats(
    beat_features_valid: pd.DataFrame,
    cluster_assignments_valid: pd.DataFrame,
) -> pd.DataFrame:
    clustered_valid_beats = beat_features_valid.merge(
        cluster_assignments_valid,
        on=["record_id", "beat_index"],
        how="inner",
        validate="one_to_one",
    )
    logger.info("merge 완료")
    return clustered_valid_beats


def compute_cluster_overview(clustered_valid_beats: pd.DataFrame, num_clusters: int) -> pd.DataFrame:
    total_beats = float(len(clustered_valid_beats))
    rows: list[dict[str, float | int]] = []

    for cluster_label in range(num_clusters):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["cluster_label"] == cluster_label]
        confidence_values = cluster_frame["cluster_confidence"].to_numpy(dtype=np.float64) if not cluster_frame.empty else np.array([], dtype=np.float64)
        rows.append(
            {
                "cluster_label": cluster_label,
                "beat_count": int(len(cluster_frame)),
                "beat_ratio": float(len(cluster_frame) / total_beats) if total_beats > 0 else float(np.nan),
                "confidence_mean": float(np.mean(confidence_values)) if confidence_values.size > 0 else float(np.nan),
                "confidence_std": float(np.std(confidence_values, ddof=0)) if confidence_values.size > 0 else float(np.nan),
            }
        )

    overview = pd.DataFrame(rows)
    logger.info("cluster별 beat 수: %s", overview.loc[:, ["cluster_label", "beat_count"]].to_dict(orient="records"))
    return overview


def summarize_feature_series(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def compute_feature_summary(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
    num_clusters: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for feature_name in learning_input_columns:
        global_values = clustered_valid_beats[feature_name].to_numpy(dtype=np.float64)
        global_summary = summarize_feature_series(global_values)
        rows.append(
            {
                "summary_level": "global",
                "cluster_label": -1,
                "feature_name": feature_name,
                **global_summary,
            }
        )

        for cluster_label in range(num_clusters):
            cluster_values = clustered_valid_beats.loc[
                clustered_valid_beats["cluster_label"] == cluster_label,
                feature_name,
            ].to_numpy(dtype=np.float64)
            if cluster_values.size == 0:
                rows.append(
                    {
                        "summary_level": "cluster",
                        "cluster_label": cluster_label,
                        "feature_name": feature_name,
                        "mean": float(np.nan),
                        "std": float(np.nan),
                        "median": float(np.nan),
                        "q25": float(np.nan),
                        "q75": float(np.nan),
                    }
                )
                continue

            cluster_summary = summarize_feature_series(cluster_values)
            rows.append(
                {
                    "summary_level": "cluster",
                    "cluster_label": cluster_label,
                    "feature_name": feature_name,
                    **cluster_summary,
                }
            )

    feature_summary = pd.DataFrame(rows)
    logger.info("feature summary 생성 완료")
    return feature_summary


def compute_top_features_per_cluster(
    clustered_valid_beats: pd.DataFrame,
    learning_input_columns: list[str],
    num_clusters: int,
    top_k: int,
    eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    global_means = clustered_valid_beats.loc[:, learning_input_columns].mean(axis=0)
    global_stds = clustered_valid_beats.loc[:, learning_input_columns].std(axis=0, ddof=0)

    for cluster_label in range(num_clusters):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["cluster_label"] == cluster_label]
        if cluster_frame.empty:
            continue

        feature_rows: list[dict[str, float | int | str]] = []
        cluster_means = cluster_frame.loc[:, learning_input_columns].mean(axis=0)
        for feature_name in learning_input_columns:
            cluster_mean = float(cluster_means[feature_name])
            global_mean = float(global_means[feature_name])
            global_std = float(global_stds[feature_name])
            delta_mean = cluster_mean - global_mean
            effect_z = delta_mean / (global_std + eps)
            feature_rows.append(
                {
                    "cluster_label": cluster_label,
                    "feature_name": feature_name,
                    "cluster_mean": cluster_mean,
                    "global_mean": global_mean,
                    "delta_mean": delta_mean,
                    "effect_z": effect_z,
                }
            )

        ranked_rows = sorted(feature_rows, key=lambda row: abs(float(row["effect_z"])), reverse=True)[:top_k]
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank_within_cluster"] = rank
            rows.append(row)

    top_features = pd.DataFrame(rows)
    logger.info("top feature 추출 완료")
    return top_features


def compute_feature_group_summary(
    top_features_per_cluster: pd.DataFrame,
    learning_input_columns: list[str],
    clustered_valid_beats: pd.DataFrame,
    num_clusters: int,
    eps: float,
) -> dict[str, dict[str, Any]]:
    global_means = clustered_valid_beats.loc[:, learning_input_columns].mean(axis=0)
    global_stds = clustered_valid_beats.loc[:, learning_input_columns].std(axis=0, ddof=0)
    summary: dict[str, dict[str, Any]] = {}

    for cluster_label in range(num_clusters):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["cluster_label"] == cluster_label]
        cluster_summary: dict[str, Any] = {}
        if cluster_frame.empty:
            summary[f"cluster_{cluster_label}"] = cluster_summary
            continue

        cluster_means = cluster_frame.loc[:, learning_input_columns].mean(axis=0)
        effect_rows = []
        for feature_name in learning_input_columns:
            delta_mean = float(cluster_means[feature_name] - global_means[feature_name])
            effect_z = delta_mean / (float(global_stds[feature_name]) + eps)
            effect_rows.append(
                {
                    "feature_name": feature_name,
                    "effect_z": effect_z,
                }
            )

        for prefix in FEATURE_GROUP_PREFIXES:
            group_rows = [row for row in effect_rows if str(row["feature_name"]).startswith(prefix)]
            ranked_group_rows = sorted(group_rows, key=lambda row: abs(float(row["effect_z"])), reverse=True)
            cluster_summary[prefix] = {
                "group_feature_count": int(len(group_rows)),
                "mean_abs_effect_z": float(np.mean([abs(float(row["effect_z"])) for row in group_rows])) if group_rows else float(np.nan),
                "top_features": [str(row["feature_name"]) for row in ranked_group_rows[:5]],
            }

        summary[f"cluster_{cluster_label}"] = cluster_summary

    return summary


def compute_representative_beats(
    clustered_valid_beats: pd.DataFrame,
    cluster_centers: np.ndarray,
    num_clusters: int,
    representative_count: int,
) -> pd.DataFrame:
    latent_columns = [f"latent_{index:02d}" for index in range(cluster_centers.shape[1])]
    rows: list[dict[str, float | int | str]] = []

    for cluster_label in range(num_clusters):
        cluster_frame = clustered_valid_beats.loc[clustered_valid_beats["cluster_label"] == cluster_label].copy()
        if cluster_frame.empty:
            continue

        cluster_latent = cluster_frame.loc[:, latent_columns].to_numpy(dtype=np.float64)
        center = cluster_centers[cluster_label].astype(np.float64)
        distances = np.linalg.norm(cluster_latent - center[None, :], axis=1)
        cluster_frame["distance_to_center"] = distances.astype(np.float64)
        cluster_frame = cluster_frame.sort_values(["distance_to_center", "record_id", "beat_index"], kind="stable").reset_index(drop=True)

        selected_frame = cluster_frame.head(representative_count).copy()
        for rank, (_, selected_row) in enumerate(selected_frame.iterrows(), start=1):
            rows.append(
                {
                    "cluster_label": cluster_label,
                    "record_id": selected_row["record_id"],
                    "beat_index": int(selected_row["beat_index"]),
                    "rank_in_cluster": rank,
                    "distance_to_center": float(selected_row["distance_to_center"]),
                    "cluster_confidence": float(selected_row["cluster_confidence"]),
                    "source_file": selected_row["source_file"],
                    "s1_on": int(selected_row["s1_on"]),
                    "s1_off": int(selected_row["s1_off"]),
                    "s2_on": int(selected_row["s2_on"]),
                    "s2_off": int(selected_row["s2_off"]),
                    "s1_on_next": int(selected_row["s1_on_next"]),
                }
            )

    representative_beats = pd.DataFrame(rows)
    logger.info("representative beat 선택 완료")
    return representative_beats


def compute_record_cluster_distribution(clustered_valid_beats: pd.DataFrame, num_clusters: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for record_id, record_frame in clustered_valid_beats.groupby("record_id", sort=True):
        total_valid_beats = int(len(record_frame))
        row: dict[str, float | int | str] = {
            "record_id": record_id,
            "total_valid_beat_count": total_valid_beats,
        }
        for cluster_label in range(num_clusters):
            cluster_count = int(np.sum(record_frame["cluster_label"].to_numpy(dtype=np.int64) == cluster_label))
            row[f"cluster_{cluster_label}_count"] = cluster_count
            row[f"cluster_{cluster_label}_ratio"] = float(cluster_count / total_valid_beats) if total_valid_beats > 0 else float(np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


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
    amplitude = pd.to_numeric(dataframe["Amplitude"], errors="raise").to_numpy(dtype=np.float32)
    return amplitude


def compute_smoothed_envelope(x: np.ndarray, fs: int, smooth_ms: float) -> np.ndarray:
    envelope_raw = np.abs(hilbert(x))
    window_size = max(3, int(round((smooth_ms / 1000.0) * fs)))
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(envelope_raw.astype(np.float32), kernel, mode="same").astype(np.float32)


def plot_representative_beat(
    signal: np.ndarray,
    envelope: np.ndarray,
    representative_row: pd.Series,
    save_path: Path,
    fs: int,
) -> None:
    start = int(representative_row["s1_on"])
    end = int(representative_row["s1_on_next"])
    beat_signal = signal[start:end]
    beat_envelope = envelope[start:end]
    time_ms = np.arange(len(beat_signal), dtype=np.float64) * (1000.0 / float(fs))

    boundary_offsets = {
        "S1 on": 0,
        "S1 off": int(representative_row["s1_off"]) - start,
        "S2 on": int(representative_row["s2_on"]) - start,
        "S2 off": int(representative_row["s2_off"]) - start,
    }
    max_envelope = float(np.max(beat_envelope)) if beat_envelope.size > 0 else 1.0
    normalized_envelope = beat_envelope / (max_envelope + InterpretationConfig.EPS)
    envelope_overlay = normalized_envelope * (np.max(np.abs(beat_signal)) + InterpretationConfig.EPS)

    plt.figure(figsize=(10, 4))
    plt.plot(time_ms, beat_signal, color="#1f4e78", linewidth=1.2, label="PCG")
    plt.plot(time_ms, envelope_overlay, color="#d95f02", linewidth=1.5, alpha=0.9, label="Envelope")
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


def export_representative_figures(
    representative_beats: pd.DataFrame,
    data_root: Path,
    figures_root: Path,
    fs: int,
    smooth_ms: float,
) -> None:
    cached_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if representative_beats.empty:
        return

    for _, row in representative_beats.iterrows():
        source_file = str(row["source_file"])
        source_path = data_root / source_file
        if source_file not in cached_signals:
            signal = load_source_signal(source_path)
            envelope = compute_smoothed_envelope(signal, fs=fs, smooth_ms=smooth_ms)
            cached_signals[source_file] = (signal, envelope)
        signal, envelope = cached_signals[source_file]

        cluster_dir = figures_root / f"cluster_{int(row['cluster_label'])}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        save_path = cluster_dir / f"representative_rank_{int(row['rank_in_cluster'])}.png"
        plot_representative_beat(signal=signal, envelope=envelope, representative_row=row, save_path=save_path, fs=fs)
    logger.info("representative waveform figure 저장 완료")


def export_umap_figure(
    clustered_valid_beats: pd.DataFrame,
    figures_root: Path,
    random_seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    try:
        import umap.umap_ as umap
    except ImportError as error:
        raise ImportError("UMAP is required for interpretation visualization") from error

    latent_columns = [column for column in clustered_valid_beats.columns if column.startswith("latent_")]
    latent_matrix = clustered_valid_beats.loc[:, latent_columns].to_numpy(dtype=np.float32)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_seed,
    )
    embedding = reducer.fit_transform(latent_matrix)
    plot_frame = clustered_valid_beats.loc[:, ["cluster_label"]].copy()
    plot_frame["umap_0"] = embedding[:, 0]
    plot_frame["umap_1"] = embedding[:, 1]

    plt.figure(figsize=(8, 6))
    for cluster_label in range(InterpretationConfig.NUM_CLUSTERS):
        cluster_frame = plot_frame.loc[plot_frame["cluster_label"] == cluster_label]
        plt.scatter(
            cluster_frame["umap_0"],
            cluster_frame["umap_1"],
            s=18,
            alpha=0.7,
            label=f"cluster_{cluster_label}",
        )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Latent UMAP by Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "latent_umap_by_cluster.png", dpi=150)
    plt.close()
    logger.info("UMAP figure 저장 완료")


def build_json_summary(
    cluster_overview: pd.DataFrame,
    top_features_per_cluster: pd.DataFrame,
    feature_group_summary: dict[str, dict[str, Any]],
    representative_beats: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"clusters": {}}
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
            "confidence_mean": float(overview_row["confidence_mean"]) if pd.notna(overview_row["confidence_mean"]) else float(np.nan),
            "top_features": cluster_top_features.to_dict(orient="records"),
            "feature_group_summary": feature_group_summary.get(cluster_key, {}),
            "representative_beats": cluster_representatives.loc[
                :,
                ["record_id", "beat_index", "rank_in_cluster", "distance_to_center", "cluster_confidence"],
            ].to_dict(orient="records"),
        }
    return summary


def save_outputs(
    interpretation_root: Path,
    cluster_overview: pd.DataFrame,
    feature_summary: pd.DataFrame,
    top_features_per_cluster: pd.DataFrame,
    feature_group_summary: dict[str, dict[str, Any]],
    representative_beats: pd.DataFrame,
    record_cluster_distribution: pd.DataFrame,
    clustered_valid_beats: pd.DataFrame,
    json_summary: dict[str, Any],
) -> None:
    cluster_overview.to_csv(interpretation_root / InterpretationConfig.CLUSTER_OVERVIEW_FILENAME, index=False)
    feature_summary.to_csv(interpretation_root / InterpretationConfig.FEATURE_SUMMARY_FILENAME, index=False)
    top_features_per_cluster.to_csv(interpretation_root / InterpretationConfig.TOP_FEATURES_FILENAME, index=False)
    representative_beats.to_csv(interpretation_root / InterpretationConfig.REPRESENTATIVE_BEATS_FILENAME, index=False)
    record_cluster_distribution.to_csv(
        interpretation_root / InterpretationConfig.RECORD_CLUSTER_DISTRIBUTION_FILENAME,
        index=False,
    )
    clustered_valid_beats.to_csv(interpretation_root / InterpretationConfig.CLUSTERED_VALID_BEATS_FILENAME, index=False)

    (interpretation_root / InterpretationConfig.FEATURE_GROUP_SUMMARY_FILENAME).write_text(
        json.dumps(feature_group_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (interpretation_root / InterpretationConfig.JSON_SUMMARY_FILENAME).write_text(
        json.dumps(json_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    export_stage_workbook(
        workbook_path=interpretation_root / InterpretationConfig.EXCEL_REPORT_FILENAME,
        sheets={
            "cluster_overview": cluster_overview,
            "top_features": top_features_per_cluster,
            "feature_summary": feature_summary,
            "representative_beats": representative_beats,
            "record_cluster_distribution": record_cluster_distribution,
        },
        freeze_panes=InterpretationConfig.EXCEL_FREEZE_PANES,
        header_fill=InterpretationConfig.EXCEL_HEADER_FILL,
        header_font_color=InterpretationConfig.EXCEL_HEADER_FONT_COLOR,
        max_column_width=InterpretationConfig.EXCEL_MAX_COLUMN_WIDTH,
    )
    logger.info("Excel / JSON report 저장 완료")


def main() -> None:
    config = InterpretationConfig()
    output_paths = ensure_output_directories(output_root=config.OUTPUT_ROOT, run_name=config.RUN_NAME)
    interpretation_root = output_paths["interpretation_root"]
    figures_root = output_paths["figures_root"]

    inputs = load_interpretation_inputs(
        preprocess_root=config.PREPROCESS_INPUT_ROOT,
        clustering_root=config.INPUT_ROOT,
    )
    validate_inputs(inputs)

    clustered_valid_beats = build_clustered_valid_beats(
        beat_features_valid=inputs["beat_features_valid"],
        cluster_assignments_valid=inputs["cluster_assignments_valid"],
    )
    cluster_overview = compute_cluster_overview(
        clustered_valid_beats=clustered_valid_beats,
        num_clusters=config.NUM_CLUSTERS,
    )
    feature_summary = compute_feature_summary(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=inputs["learning_input_columns"],
        num_clusters=config.NUM_CLUSTERS,
    )
    top_features_per_cluster = compute_top_features_per_cluster(
        clustered_valid_beats=clustered_valid_beats,
        learning_input_columns=inputs["learning_input_columns"],
        num_clusters=config.NUM_CLUSTERS,
        top_k=config.TOP_FEATURES_PER_CLUSTER,
        eps=config.EPS,
    )
    feature_group_summary = compute_feature_group_summary(
        top_features_per_cluster=top_features_per_cluster,
        learning_input_columns=inputs["learning_input_columns"],
        clustered_valid_beats=clustered_valid_beats,
        num_clusters=config.NUM_CLUSTERS,
        eps=config.EPS,
    )
    representative_beats = compute_representative_beats(
        clustered_valid_beats=clustered_valid_beats,
        cluster_centers=inputs["cluster_centers"],
        num_clusters=config.NUM_CLUSTERS,
        representative_count=config.REPRESENTATIVE_BEATS_PER_CLUSTER,
    )
    export_representative_figures(
        representative_beats=representative_beats,
        data_root=config.DATA_ROOT,
        figures_root=figures_root,
        fs=config.SAMPLING_RATE,
        smooth_ms=config.ENVELOPE_SMOOTH_MS,
    )
    record_cluster_distribution = compute_record_cluster_distribution(
        clustered_valid_beats=clustered_valid_beats,
        num_clusters=config.NUM_CLUSTERS,
    )
    export_umap_figure(
        clustered_valid_beats=clustered_valid_beats,
        figures_root=figures_root,
        random_seed=config.RANDOM_SEED,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
    )

    json_summary = build_json_summary(
        cluster_overview=cluster_overview,
        top_features_per_cluster=top_features_per_cluster,
        feature_group_summary=feature_group_summary,
        representative_beats=representative_beats,
    )
    save_outputs(
        interpretation_root=interpretation_root,
        cluster_overview=cluster_overview,
        feature_summary=feature_summary,
        top_features_per_cluster=top_features_per_cluster,
        feature_group_summary=feature_group_summary,
        representative_beats=representative_beats,
        record_cluster_distribution=record_cluster_distribution,
        clustered_valid_beats=clustered_valid_beats,
        json_summary=json_summary,
    )


if __name__ == "__main__":
    main()
