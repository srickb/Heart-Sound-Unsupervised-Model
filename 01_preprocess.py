"""
Standalone preprocessing script for unsupervised heart sound cycle analysis.

Expected input schema per Excel file:
- Time_Index
- Amplitude
- S1-Start_RS_Score
- S1-End_RS_Score
- S2-Start_RS_Score
- S2-End_RS_Score

Saved artifacts:
- outputs/{RUN_NAME}/preprocess/cycle_features.npy
  shape: (num_valid_cycles, num_features)
- outputs/{RUN_NAME}/preprocess/cycle_waveforms.npy
  shape: (num_valid_cycles, fixed_length)
- outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
- outputs/{RUN_NAME}/preprocess/feature_names.json
- outputs/{RUN_NAME}/preprocess/preprocess_summary.json
- outputs/{RUN_NAME}/preprocess/qc/*
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.signal import butter, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Editable configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PATHS = {
    "data_root": PROJECT_ROOT / "Data" / "Test_Dataset_260312",
    "output_root": PROJECT_ROOT / "outputs",
}

RUN_NAME = "test_dataset_260312_preprocess_v1"

DATA = {
    "file_glob": "*.xlsx",
    "sampling_rate": 4000,
    "expected_columns": [
        "Time_Index",
        "Amplitude",
        "S1-Start_RS_Score",
        "S1-End_RS_Score",
        "S2-Start_RS_Score",
        "S2-End_RS_Score",
    ],
}

PREPROCESS = {
    "enable_bandpass": False,
    "bandpass_low_hz": 20.0,
    "bandpass_high_hz": 800.0,
    "bandpass_order": 4,
    "fixed_length": 4000,
    "min_cycle_seconds": 0.25,
    "max_cycle_seconds": 1.50,
}

FEATURE = {
    "normalize_waveform_by": "max_abs",
    "subtract_cycle_mean_before_normalization": True,
    "rs_score_columns": [
        "S1-Start_RS_Score",
        "S1-End_RS_Score",
        "S2-Start_RS_Score",
        "S2-End_RS_Score",
    ],
}

QC = {
    "num_example_waveforms": 8,
    "waveform_alpha": 0.7,
}

RANDOM_SEED = 42


# ============================================================================
# Dataset adapter section
# ============================================================================
def parse_recording_identity(file_path: Path) -> dict[str, str]:
    """
    Parse subject/site identity from a dataset filename.

    Input:
        file_path: Path to one .xlsx recording file.
    Output:
        Dictionary with recording_id, subject_id, and auscultation_site.
    """
    stem_parts = file_path.stem.split("_")
    if len(stem_parts) < 4:
        raise ValueError(
            f"Unexpected filename format for recording identity: {file_path.name}"
        )

    subject_id = stem_parts[0]
    auscultation_site = stem_parts[1]
    recording_id = file_path.stem
    return {
        "recording_id": recording_id,
        "subject_id": subject_id,
        "auscultation_site": auscultation_site,
    }


def load_recording_table(
    file_path: Path,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Load one recording workbook and enforce the expected internal schema.

    Expected input columns:
    - Time_Index
    - Amplitude
    - S1-Start_RS_Score
    - S1-End_RS_Score
    - S2-Start_RS_Score
    - S2-End_RS_Score

    Input:
        file_path: Excel workbook path.
        expected_columns: Required schema columns in the adapter.
    Output:
        DataFrame with shape (num_samples, 6).
    """
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    rows = worksheet.iter_rows(values_only=True)

    try:
        header = [str(value) if value is not None else "" for value in next(rows)]
    except StopIteration as exc:
        workbook.close()
        raise ValueError(f"Workbook is empty: {file_path}") from exc

    missing_columns = [column for column in expected_columns if column not in header]
    if missing_columns:
        workbook.close()
        raise ValueError(
            f"Missing required columns in {file_path.name}: {missing_columns}"
        )

    column_indices = [header.index(column) for column in expected_columns]
    data = {column: [] for column in expected_columns}

    for row in rows:
        if row is None:
            continue
        values = [row[index] if index < len(row) else None for index in column_indices]
        if all(value is None for value in values):
            continue
        for column, value in zip(expected_columns, values):
            data[column].append(value)

    workbook.close()

    dataframe = pd.DataFrame(data)
    if dataframe.empty:
        raise ValueError(f"No data rows found in workbook: {file_path}")

    dataframe = dataframe.replace({None: np.nan})
    if dataframe[expected_columns].isnull().any().any():
        null_counts = dataframe[expected_columns].isnull().sum()
        failing = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values detected in {file_path.name}: {failing}")

    dataframe["Time_Index"] = dataframe["Time_Index"].astype(np.int64)
    dataframe["Amplitude"] = dataframe["Amplitude"].astype(np.float32)
    for column in expected_columns[2:]:
        dataframe[column] = dataframe[column].astype(np.float32)

    if len(dataframe) < 2:
        raise ValueError(f"Recording is too short to segment cycles: {file_path.name}")

    time_index = dataframe["Time_Index"].to_numpy()
    time_diffs = np.diff(time_index)
    if np.any(time_diffs <= 0):
        raise ValueError(f"Non-monotonic Time_Index detected in {file_path.name}")
    if np.any(time_diffs != 1):
        raise ValueError(
            f"Time_Index must increment by 1 sample in {file_path.name}, "
            f"but observed diffs {np.unique(time_diffs)}"
        )

    return dataframe


# ============================================================================
# Utility functions
# ============================================================================
def set_random_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_directories(output_root: Path, run_name: str) -> dict[str, Path]:
    """Create stable output directories for this run."""
    run_root = output_root / run_name
    preprocess_root = run_root / "preprocess"
    qc_root = preprocess_root / "qc"
    training_root = run_root / "training"
    clustering_root = run_root / "clustering"
    interpretation_root = run_root / "interpretation"

    for path in [
        preprocess_root,
        qc_root,
        training_root,
        clustering_root,
        interpretation_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "preprocess_root": preprocess_root,
        "qc_root": qc_root,
        "training_root": training_root,
        "clustering_root": clustering_root,
        "interpretation_root": interpretation_root,
    }


def maybe_filter_signal(
    waveform: np.ndarray,
    sampling_rate: int,
    preprocess_config: dict[str, Any],
) -> np.ndarray:
    """
    Apply optional conservative bandpass filtering.

    Input:
        waveform: Array with shape (num_samples,).
        sampling_rate: Sampling rate in Hz.
        preprocess_config: Filtering configuration dictionary.
    Output:
        Array with shape (num_samples,).
    """
    waveform = waveform.astype(np.float32, copy=True)
    if not preprocess_config["enable_bandpass"]:
        return waveform

    sos = butter(
        preprocess_config["bandpass_order"],
        [
            preprocess_config["bandpass_low_hz"],
            preprocess_config["bandpass_high_hz"],
        ],
        btype="bandpass",
        output="sos",
        fs=sampling_rate,
    )
    return sosfiltfilt(sos, waveform).astype(np.float32)


def find_positive_runs(score_array: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous positive-score runs.

    Input:
        score_array: Array with shape (num_samples,).
    Output:
        List of (start_row, end_row) inclusive tuples.
    """
    positive_rows = np.flatnonzero(score_array > 0)
    if positive_rows.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    run_start = int(positive_rows[0])
    previous_row = int(positive_rows[0])

    for row_index in positive_rows[1:]:
        row_index = int(row_index)
        if row_index == previous_row + 1:
            previous_row = row_index
            continue
        runs.append((run_start, previous_row))
        run_start = row_index
        previous_row = row_index

    runs.append((run_start, previous_row))
    return runs


def representative_peak_row(score_array: np.ndarray, start_row: int, end_row: int) -> int:
    """
    Pick one representative row from a positive RS-score run.

    Rule:
    - find the maximum score inside the run
    - if the maximum spans multiple rows, pick the plateau center

    Input:
        score_array: Array with shape (num_samples,).
        start_row: Inclusive run start.
        end_row: Inclusive run end.
    Output:
        Representative row index as int.
    """
    run_scores = score_array[start_row : end_row + 1]
    max_score = np.max(run_scores)
    plateau_rows = np.flatnonzero(run_scores == max_score)
    plateau_center = int(plateau_rows[len(plateau_rows) // 2])
    return start_row + plateau_center


def extract_event_candidates(
    time_index: np.ndarray,
    score_array: np.ndarray,
    event_name: str,
) -> list[dict[str, Any]]:
    """
    Convert RS-score runs into event candidates.

    Input:
        time_index: Sample indices with shape (num_samples,).
        score_array: RS scores with shape (num_samples,).
        event_name: Human-readable event label.
    Output:
        List of dictionaries, one per positive-score run.
    """
    candidates: list[dict[str, Any]] = []
    for run_start_row, run_end_row in find_positive_runs(score_array):
        peak_row = representative_peak_row(score_array, run_start_row, run_end_row)
        candidates.append(
            {
                "event_name": event_name,
                "run_start_row": run_start_row,
                "run_end_row": run_end_row,
                "peak_row": peak_row,
                "run_start_sample": int(time_index[run_start_row]),
                "run_end_sample": int(time_index[run_end_row]),
                "sample_index": int(time_index[peak_row]),
                "peak_score": float(score_array[peak_row]),
            }
        )
    return candidates


def events_between(
    events: list[dict[str, Any]],
    left_sample: int,
    right_sample: int,
) -> list[dict[str, Any]]:
    """
    Return events strictly inside one cycle boundary.

    Input:
        events: Event candidate list.
        left_sample: Exclusive lower boundary.
        right_sample: Exclusive upper boundary.
    Output:
        Candidate list inside the interval.
    """
    return [
        event
        for event in events
        if left_sample < event["sample_index"] < right_sample
    ]


def resolve_cycle_event(
    candidates: list[dict[str, Any]],
    missing_reason: str,
    multiple_reason: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Resolve one cycle-local event candidate list to a unique event.

    Input:
        candidates: Candidates inside one S1-start to next-S1-start interval.
    Output:
        (event or None, exclusion_reason or None)
    """
    if len(candidates) == 0:
        return None, missing_reason
    if len(candidates) > 1:
        return None, multiple_reason
    return candidates[0], None


def normalize_cycle_waveform(
    waveform: np.ndarray,
    subtract_mean: bool,
) -> np.ndarray:
    """
    Normalize one cycle waveform for feature extraction and visualization.

    Input:
        waveform: Cycle waveform with shape (cycle_length,).
    Output:
        Normalized waveform with shape (cycle_length,).
    """
    normalized = waveform.astype(np.float32, copy=True)
    if subtract_mean:
        normalized = normalized - np.mean(normalized)

    max_abs = float(np.max(np.abs(normalized)))
    if max_abs > 0.0:
        normalized = normalized / max_abs
    return normalized.astype(np.float32)


def fixed_length_representation(
    waveform: np.ndarray,
    fixed_length: int,
) -> np.ndarray:
    """
    Convert one variable-length cycle waveform into a fixed-length array.

    Input:
        waveform: Normalized cycle waveform with shape (cycle_length,).
        fixed_length: Target number of samples.
    Output:
        Array with shape (fixed_length,).
    """
    output = np.zeros(fixed_length, dtype=np.float32)
    copy_length = min(len(waveform), fixed_length)
    output[:copy_length] = waveform[:copy_length]
    return output


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when undefined."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def waveform_statistics(
    waveform: np.ndarray,
    prefix: str,
    include_peak_to_peak: bool,
) -> dict[str, float]:
    """
    Compute simple interpretable statistics from one waveform segment.

    Input:
        waveform: Array with shape (segment_length,).
        prefix: Feature name prefix.
    Output:
        Ordered feature dictionary for this segment.
    """
    stats: dict[str, float] = {
        f"{prefix}_mean": float(np.mean(waveform)),
        f"{prefix}_std": float(np.std(waveform)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(waveform)))),
        f"{prefix}_max_abs": float(np.max(np.abs(waveform))),
        f"{prefix}_energy": float(np.sum(np.square(waveform))),
        f"{prefix}_abs_area": float(np.sum(np.abs(waveform))),
    }
    if include_peak_to_peak:
        stats[f"{prefix}_peak_to_peak"] = float(np.ptp(waveform))
    return stats


def custom_rs_feature_hook(
    recording_table: pd.DataFrame,
    start_row: int,
    end_row_exclusive: int,
    rs_columns: list[str],
) -> dict[str, float]:
    """
    Placeholder hook for user-defined RS-based features.

    Input:
        recording_table: Recording table with shape (num_samples, num_columns).
        start_row: Inclusive cycle start row.
        end_row_exclusive: Exclusive cycle end row.
        rs_columns: RS score column names.
    Output:
        Feature dictionary to be merged into the cycle feature row.

    TODO:
    - Add user-approved RS feature definitions here once the intended
      interpretation of RS scores is finalized.
    - Keep any future RS-derived features isolated in this hook so the
      remainder of the preprocessing pipeline stays stable.
    """
    _ = recording_table
    _ = start_row
    _ = end_row_exclusive
    _ = rs_columns
    return {}


def build_feature_row(
    cycle_waveform: np.ndarray,
    s1_segment: np.ndarray,
    systole_segment: np.ndarray,
    s2_segment: np.ndarray,
    diastole_segment: np.ndarray,
    cycle_length_samples: int,
    s1_to_s2_samples: int,
    s2_end_to_next_s1_samples: int,
    s1_duration_samples: int,
    systole_duration_samples: int,
    s2_duration_samples: int,
    diastole_duration_samples: int,
    sampling_rate: int,
    recording_table: pd.DataFrame,
    cycle_start_row: int,
    cycle_end_row_exclusive: int,
) -> dict[str, float]:
    """
    Build one feature row for one valid cycle.

    Input waveforms:
        cycle_waveform: shape (cycle_length,)
        s1_segment: shape (s1_length,)
        systole_segment: shape (systole_length,)
        s2_segment: shape (s2_length,)
        diastole_segment: shape (diastole_length,)
    Output:
        Feature dictionary with stable key order.
    """
    cycle_length_sec = cycle_length_samples / sampling_rate
    s1_to_s2_sec = s1_to_s2_samples / sampling_rate
    s2_end_to_next_s1_sec = s2_end_to_next_s1_samples / sampling_rate
    s1_duration_sec = s1_duration_samples / sampling_rate
    systole_duration_sec = systole_duration_samples / sampling_rate
    s2_duration_sec = s2_duration_samples / sampling_rate
    diastole_duration_sec = diastole_duration_samples / sampling_rate

    features: dict[str, float] = {
        "cycle_duration_sec": float(cycle_length_sec),
        "s1_to_s2_start_sec": float(s1_to_s2_sec),
        "s2_end_to_next_s1_sec": float(s2_end_to_next_s1_sec),
        "s1_duration_sec": float(s1_duration_sec),
        "systole_duration_sec": float(systole_duration_sec),
        "s2_duration_sec": float(s2_duration_sec),
        "diastole_duration_sec": float(diastole_duration_sec),
        "s1_ratio": safe_ratio(s1_duration_sec, cycle_length_sec),
        "systole_ratio": safe_ratio(systole_duration_sec, cycle_length_sec),
        "s2_ratio": safe_ratio(s2_duration_sec, cycle_length_sec),
        "diastole_ratio": safe_ratio(diastole_duration_sec, cycle_length_sec),
        "s1_to_s2_ratio": safe_ratio(s1_to_s2_sec, cycle_length_sec),
        "s2_end_to_next_s1_ratio": safe_ratio(
            s2_end_to_next_s1_sec, cycle_length_sec
        ),
    }

    features.update(
        waveform_statistics(
            cycle_waveform, prefix="cycle", include_peak_to_peak=True
        )
    )
    features.update(
        waveform_statistics(s1_segment, prefix="s1_segment", include_peak_to_peak=False)
    )
    features.update(
        waveform_statistics(
            systole_segment, prefix="systole_segment", include_peak_to_peak=False
        )
    )
    features.update(
        waveform_statistics(s2_segment, prefix="s2_segment", include_peak_to_peak=False)
    )
    features.update(
        waveform_statistics(
            diastole_segment, prefix="diastole_segment", include_peak_to_peak=False
        )
    )
    features.update(
        custom_rs_feature_hook(
            recording_table=recording_table,
            start_row=cycle_start_row,
            end_row_exclusive=cycle_end_row_exclusive,
            rs_columns=FEATURE["rs_score_columns"],
        )
    )
    return features


def summarize_array(values: np.ndarray) -> dict[str, float] | dict[str, None]:
    """Return min/median/mean/max summary for a 1D numeric array."""
    if values.size == 0:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def save_qc_plots(
    metadata_frame: pd.DataFrame,
    waveform_matrix: np.ndarray,
    qc_root: Path,
    fixed_length: int,
    random_seed: int,
) -> None:
    """Generate simple QC figures for the preprocessing output."""
    valid_metadata = metadata_frame[metadata_frame["valid_flag"] == True].copy()
    excluded_metadata = metadata_frame[metadata_frame["valid_flag"] == False].copy()

    plt.figure(figsize=(8, 4.5))
    if not valid_metadata.empty:
        plt.hist(valid_metadata["cycle_duration_sec"], bins=30, color="#4C78A8")
    plt.xlabel("Cycle duration (sec)")
    plt.ylabel("Count")
    plt.title("Valid Cycle Duration Histogram")
    plt.tight_layout()
    plt.savefig(qc_root / "cycle_duration_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["valid", "excluded"],
        [len(valid_metadata), len(excluded_metadata)],
        color=["#59A14F", "#E15759"],
    )
    plt.ylabel("Count")
    plt.title("Valid vs Excluded Cycles")
    plt.tight_layout()
    plt.savefig(qc_root / "valid_vs_excluded_cycles.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if waveform_matrix.size > 0:
        rng = np.random.default_rng(random_seed)
        num_examples = min(QC["num_example_waveforms"], waveform_matrix.shape[0])
        chosen_rows = np.sort(
            rng.choice(waveform_matrix.shape[0], size=num_examples, replace=False)
        )
        x_axis = np.arange(fixed_length)
        for row_index in chosen_rows:
            plt.plot(
                x_axis,
                waveform_matrix[row_index],
                alpha=QC["waveform_alpha"],
                linewidth=1.0,
            )
    plt.xlabel("Sample")
    plt.ylabel("Normalized amplitude")
    plt.title("Example Normalized Cycle Waveforms")
    plt.tight_layout()
    plt.savefig(qc_root / "example_normalized_cycle_waveforms.png", dpi=150)
    plt.close()


# ============================================================================
# Main preprocessing pipeline
# ============================================================================
def process_recording(
    file_path: Path,
    sampling_rate: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[np.ndarray], dict[str, Any]]:
    """
    Process one recording into cycle-level metadata, features, and waveforms.

    Output:
        metadata_rows: one row per candidate cycle, valid or excluded
        feature_rows: one row per valid cycle
        waveform_rows: one fixed-length waveform per valid cycle
        recording_summary: per-recording counts
    """
    identity = parse_recording_identity(file_path)
    recording_table = load_recording_table(file_path, DATA["expected_columns"])
    filtered_amplitude = maybe_filter_signal(
        recording_table["Amplitude"].to_numpy(dtype=np.float32),
        sampling_rate=sampling_rate,
        preprocess_config=PREPROCESS,
    )
    time_index = recording_table["Time_Index"].to_numpy(dtype=np.int64)

    event_candidates = {
        "s1_start": extract_event_candidates(
            time_index,
            recording_table["S1-Start_RS_Score"].to_numpy(dtype=np.float32),
            event_name="S1_START",
        ),
        "s1_end": extract_event_candidates(
            time_index,
            recording_table["S1-End_RS_Score"].to_numpy(dtype=np.float32),
            event_name="S1_END",
        ),
        "s2_start": extract_event_candidates(
            time_index,
            recording_table["S2-Start_RS_Score"].to_numpy(dtype=np.float32),
            event_name="S2_START",
        ),
        "s2_end": extract_event_candidates(
            time_index,
            recording_table["S2-End_RS_Score"].to_numpy(dtype=np.float32),
            event_name="S2_END",
        ),
    }

    if len(event_candidates["s1_start"]) < 2:
        raise NotImplementedError(
            f"Recording {file_path.name} does not contain enough S1-start annotations "
            f"to define cycles."
        )

    min_cycle_samples = int(PREPROCESS["min_cycle_seconds"] * sampling_rate)
    max_cycle_samples = int(PREPROCESS["max_cycle_seconds"] * sampling_rate)

    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    waveform_rows: list[np.ndarray] = []
    recording_exclusion_counter: Counter[str] = Counter()

    s1_starts = event_candidates["s1_start"]
    for cycle_index in range(len(s1_starts) - 1):
        s1_start_event = s1_starts[cycle_index]
        next_s1_start_event = s1_starts[cycle_index + 1]
        cycle_start_sample = int(s1_start_event["sample_index"])
        next_s1_start_sample = int(next_s1_start_event["sample_index"])
        cycle_start_row = int(s1_start_event["peak_row"])
        cycle_end_row_exclusive = int(next_s1_start_event["peak_row"])
        cycle_length_samples = next_s1_start_sample - cycle_start_sample

        sample_id = f"{identity['recording_id']}_cycle_{cycle_index:04d}"
        cycle_metadata: dict[str, Any] = {
            "sample_id": sample_id,
            "subject_id": identity["subject_id"],
            "recording_id": identity["recording_id"],
            "auscultation_site": identity["auscultation_site"],
            "source_file": file_path.name,
            "cycle_index": cycle_index,
            "cycle_start_sample": cycle_start_sample,
            "cycle_end_sample": next_s1_start_sample,
            "s1_start_sample": cycle_start_sample,
            "next_s1_start_sample": next_s1_start_sample,
            "original_cycle_length": int(cycle_end_row_exclusive - cycle_start_row),
            "fixed_length": PREPROCESS["fixed_length"],
            "sampling_rate": sampling_rate,
            "valid_flag": False,
            "exclusion_reason": "",
            "feature_row_index": np.nan,
            "waveform_row_index": np.nan,
        }

        if cycle_length_samples < min_cycle_samples:
            cycle_metadata["exclusion_reason"] = "cycle_too_short"
            metadata_rows.append(cycle_metadata)
            recording_exclusion_counter["cycle_too_short"] += 1
            continue
        if cycle_length_samples > max_cycle_samples:
            cycle_metadata["exclusion_reason"] = "cycle_too_long"
            metadata_rows.append(cycle_metadata)
            recording_exclusion_counter["cycle_too_long"] += 1
            continue

        s1_end_candidates = events_between(
            event_candidates["s1_end"], cycle_start_sample, next_s1_start_sample
        )
        s2_start_candidates = events_between(
            event_candidates["s2_start"], cycle_start_sample, next_s1_start_sample
        )
        s2_end_candidates = events_between(
            event_candidates["s2_end"], cycle_start_sample, next_s1_start_sample
        )

        s1_end_event, exclusion_reason = resolve_cycle_event(
            s1_end_candidates,
            missing_reason="missing_s1_end",
            multiple_reason="multiple_s1_end",
        )
        if exclusion_reason is None:
            s2_start_event, exclusion_reason = resolve_cycle_event(
                s2_start_candidates,
                missing_reason="missing_s2_start",
                multiple_reason="multiple_s2_start",
            )
        else:
            s2_start_event = None
        if exclusion_reason is None:
            s2_end_event, exclusion_reason = resolve_cycle_event(
                s2_end_candidates,
                missing_reason="missing_s2_end",
                multiple_reason="multiple_s2_end",
            )
        else:
            s2_end_event = None

        cycle_metadata["num_s1_end_candidates"] = len(s1_end_candidates)
        cycle_metadata["num_s2_start_candidates"] = len(s2_start_candidates)
        cycle_metadata["num_s2_end_candidates"] = len(s2_end_candidates)

        cycle_metadata["s1_end_sample"] = (
            int(s1_end_event["sample_index"]) if s1_end_event is not None else np.nan
        )
        cycle_metadata["s2_start_sample"] = (
            int(s2_start_event["sample_index"])
            if s2_start_event is not None
            else np.nan
        )
        cycle_metadata["s2_end_sample"] = (
            int(s2_end_event["sample_index"]) if s2_end_event is not None else np.nan
        )

        if exclusion_reason is not None:
            cycle_metadata["exclusion_reason"] = exclusion_reason
            metadata_rows.append(cycle_metadata)
            recording_exclusion_counter[exclusion_reason] += 1
            continue

        assert s1_end_event is not None
        assert s2_start_event is not None
        assert s2_end_event is not None

        ordering = [
            cycle_start_sample,
            int(s1_end_event["sample_index"]),
            int(s2_start_event["sample_index"]),
            int(s2_end_event["sample_index"]),
            next_s1_start_sample,
        ]
        if ordering != sorted(ordering) or len(set(ordering)) != len(ordering):
            cycle_metadata["exclusion_reason"] = "invalid_event_order"
            metadata_rows.append(cycle_metadata)
            recording_exclusion_counter["invalid_event_order"] += 1
            continue

        s1_end_row = int(s1_end_event["peak_row"])
        s2_start_row = int(s2_start_event["peak_row"])
        s2_end_row = int(s2_end_event["peak_row"])

        cycle_waveform = filtered_amplitude[cycle_start_row:cycle_end_row_exclusive]
        normalized_cycle = normalize_cycle_waveform(
            cycle_waveform,
            subtract_mean=FEATURE["subtract_cycle_mean_before_normalization"],
        )

        s1_segment = normalized_cycle[: s1_end_row - cycle_start_row]
        systole_segment = normalized_cycle[
            s1_end_row - cycle_start_row : s2_start_row - cycle_start_row
        ]
        s2_segment = normalized_cycle[
            s2_start_row - cycle_start_row : s2_end_row - cycle_start_row
        ]
        diastole_segment = normalized_cycle[
            s2_end_row - cycle_start_row : cycle_end_row_exclusive - cycle_start_row
        ]

        if any(
            len(segment) == 0
            for segment in [s1_segment, systole_segment, s2_segment, diastole_segment]
        ):
            cycle_metadata["exclusion_reason"] = "empty_segment_after_ordering"
            metadata_rows.append(cycle_metadata)
            recording_exclusion_counter["empty_segment_after_ordering"] += 1
            continue

        s1_to_s2_samples = int(s2_start_event["sample_index"] - cycle_start_sample)
        s2_end_to_next_s1_samples = int(
            next_s1_start_sample - s2_end_event["sample_index"]
        )
        s1_duration_samples = int(s1_end_event["sample_index"] - cycle_start_sample)
        systole_duration_samples = int(
            s2_start_event["sample_index"] - s1_end_event["sample_index"]
        )
        s2_duration_samples = int(
            s2_end_event["sample_index"] - s2_start_event["sample_index"]
        )
        diastole_duration_samples = int(
            next_s1_start_sample - s2_end_event["sample_index"]
        )

        feature_row = build_feature_row(
            cycle_waveform=normalized_cycle,
            s1_segment=s1_segment,
            systole_segment=systole_segment,
            s2_segment=s2_segment,
            diastole_segment=diastole_segment,
            cycle_length_samples=cycle_length_samples,
            s1_to_s2_samples=s1_to_s2_samples,
            s2_end_to_next_s1_samples=s2_end_to_next_s1_samples,
            s1_duration_samples=s1_duration_samples,
            systole_duration_samples=systole_duration_samples,
            s2_duration_samples=s2_duration_samples,
            diastole_duration_samples=diastole_duration_samples,
            sampling_rate=sampling_rate,
            recording_table=recording_table,
            cycle_start_row=cycle_start_row,
            cycle_end_row_exclusive=cycle_end_row_exclusive,
        )

        fixed_waveform = fixed_length_representation(
            normalized_cycle,
            fixed_length=PREPROCESS["fixed_length"],
        )

        cycle_metadata.update(
            {
                "s1_end_sample": int(s1_end_event["sample_index"]),
                "s2_start_sample": int(s2_start_event["sample_index"]),
                "s2_end_sample": int(s2_end_event["sample_index"]),
                "cycle_duration_sec": float(cycle_length_samples / sampling_rate),
                "s1_to_s2_start_sec": float(s1_to_s2_samples / sampling_rate),
                "s2_end_to_next_s1_sec": float(
                    s2_end_to_next_s1_samples / sampling_rate
                ),
                "s1_duration_sec": float(s1_duration_samples / sampling_rate),
                "systole_duration_sec": float(
                    systole_duration_samples / sampling_rate
                ),
                "s2_duration_sec": float(s2_duration_samples / sampling_rate),
                "diastole_duration_sec": float(
                    diastole_duration_samples / sampling_rate
                ),
                "valid_flag": True,
                "exclusion_reason": "",
            }
        )

        metadata_rows.append(cycle_metadata)
        feature_rows.append(feature_row)
        waveform_rows.append(fixed_waveform)

    recording_summary = {
        "recording_id": identity["recording_id"],
        "subject_id": identity["subject_id"],
        "auscultation_site": identity["auscultation_site"],
        "source_file": file_path.name,
        "num_samples": int(len(recording_table)),
        "num_s1_start_events": int(len(event_candidates["s1_start"])),
        "num_candidate_cycles": int(max(len(event_candidates["s1_start"]) - 1, 0)),
        "num_valid_cycles": int(sum(row["valid_flag"] for row in metadata_rows)),
        "num_excluded_cycles": int(sum(not row["valid_flag"] for row in metadata_rows)),
        "exclusion_reasons": dict(recording_exclusion_counter),
    }

    return metadata_rows, feature_rows, waveform_rows, recording_summary


def main() -> None:
    """Run the standalone preprocessing pipeline from raw Excel files."""
    set_random_seed(RANDOM_SEED)
    output_paths = ensure_output_directories(PATHS["output_root"], RUN_NAME)

    data_root = PATHS["data_root"]
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    file_paths = sorted(data_root.glob(DATA["file_glob"]))
    file_paths = [
        file_path for file_path in file_paths if not file_path.name.startswith("~$")
    ]
    if not file_paths:
        raise FileNotFoundError(f"No input files found under: {data_root}")

    all_metadata_rows: list[dict[str, Any]] = []
    all_feature_rows: list[dict[str, float]] = []
    all_waveform_rows: list[np.ndarray] = []
    recording_summaries: list[dict[str, Any]] = []

    for file_path in file_paths:
        metadata_rows, feature_rows, waveform_rows, recording_summary = process_recording(
            file_path=file_path,
            sampling_rate=DATA["sampling_rate"],
        )
        all_metadata_rows.extend(metadata_rows)
        all_feature_rows.extend(feature_rows)
        all_waveform_rows.extend(waveform_rows)
        recording_summaries.append(recording_summary)

    metadata_frame = pd.DataFrame(all_metadata_rows)
    metadata_frame = metadata_frame.sort_values(
        by=["recording_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)

    valid_mask = metadata_frame["valid_flag"] == True
    metadata_frame.loc[valid_mask, "feature_row_index"] = np.arange(
        int(valid_mask.sum()), dtype=np.float32
    )
    metadata_frame.loc[valid_mask, "waveform_row_index"] = np.arange(
        int(valid_mask.sum()), dtype=np.float32
    )

    if all_feature_rows:
        feature_frame = pd.DataFrame(all_feature_rows)
        feature_names = feature_frame.columns.tolist()
        feature_matrix = feature_frame.to_numpy(dtype=np.float32)
        waveform_matrix = np.stack(all_waveform_rows).astype(np.float32)
    else:
        feature_names = []
        feature_matrix = np.empty((0, 0), dtype=np.float32)
        waveform_matrix = np.empty((0, PREPROCESS["fixed_length"]), dtype=np.float32)

    preprocess_root = output_paths["preprocess_root"]
    np.save(preprocess_root / "cycle_features.npy", feature_matrix)
    np.save(preprocess_root / "cycle_waveforms.npy", waveform_matrix)
    metadata_frame.to_csv(preprocess_root / "cycle_metadata.csv", index=False)

    with open(preprocess_root / "feature_names.json", "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    valid_metadata = metadata_frame[metadata_frame["valid_flag"] == True].copy()
    excluded_metadata = metadata_frame[metadata_frame["valid_flag"] == False].copy()
    exclusion_counts = excluded_metadata["exclusion_reason"].value_counts().to_dict()

    summary = {
        "run_name": RUN_NAME,
        "data_root": str(data_root),
        "num_input_files": len(file_paths),
        "sampling_rate": DATA["sampling_rate"],
        "fixed_length": PREPROCESS["fixed_length"],
        "feature_shape": list(feature_matrix.shape),
        "waveform_shape": list(waveform_matrix.shape),
        "num_candidate_cycles": int(len(metadata_frame)),
        "num_valid_cycles": int(len(valid_metadata)),
        "num_excluded_cycles": int(len(excluded_metadata)),
        "valid_cycle_duration_sec": summarize_array(
            valid_metadata["cycle_duration_sec"].to_numpy(dtype=np.float32)
            if not valid_metadata.empty
            else np.array([], dtype=np.float32)
        ),
        "exclusion_reasons": exclusion_counts,
        "recordings": recording_summaries,
    }

    with open(
        preprocess_root / "preprocess_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    save_qc_plots(
        metadata_frame=metadata_frame,
        waveform_matrix=waveform_matrix,
        qc_root=output_paths["qc_root"],
        fixed_length=PREPROCESS["fixed_length"],
        random_seed=RANDOM_SEED,
    )

    print(f"Saved preprocess outputs to: {preprocess_root}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Waveform matrix shape: {waveform_matrix.shape}")
    print(f"Metadata rows: {len(metadata_frame)}")
    print(f"Valid cycles: {len(valid_metadata)}")
    print(f"Excluded cycles: {len(excluded_metadata)}")


if __name__ == "__main__":
    main()
