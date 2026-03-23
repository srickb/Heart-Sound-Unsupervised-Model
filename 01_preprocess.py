"""
Beat-level preprocessing script fixed to the current unsupervised design.

Expected input schema per tabular file (.xlsx or .csv):
- Amplitude
- S1-Start_RS_Score
- S1-End_RS_Score
- S2-Start_RS_Score
- S2-End_RS_Score

Optional column:
- Time_Index (ignored during preprocessing)

Saved artifacts:
- outputs/{RUN_NAME}/preprocess/beat_features_all.csv
- outputs/{RUN_NAME}/preprocess/beat_features_valid.csv
- outputs/{RUN_NAME}/preprocess/record_summary.csv
- outputs/{RUN_NAME}/preprocess/feature_names.json
- outputs/{RUN_NAME}/preprocess/learning_input_columns.json
- outputs/{RUN_NAME}/preprocess/preprocess_export.xlsx
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


class PreprocessConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 2개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    TRAIN_DATA_FOLDER = r"C:\Users\LUI\Desktop\PCG\Data\Train 학습데이터(260109)"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"

    FILE_GLOB = "*"
    SUPPORTED_INPUT_SUFFIXES = (".xlsx", ".csv")
    CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    SAMPLING_RATE = 4000
    MIN_CYCLE_MS = 250.0
    MAX_CYCLE_MS = 1500.0
    ENVELOPE_SMOOTH_MS = 10.0
    TEMPLATE_RESAMPLE_LENGTH = 128
    EPS = 1e-12

    EXPECTED_COLUMNS = [
        "Amplitude",
        "S1-Start_RS_Score",
        "S1-End_RS_Score",
        "S2-Start_RS_Score",
        "S2-End_RS_Score",
    ]

    EXPORT_EXCEL = True
    EXPORT_CSV = True
    EXPORT_FEATURE_NAMES_JSON = True
    SAVE_INVALID_ROWS = True
    LEARNING_INPUT_EXCLUDE_COLUMNS = [
        "time_s1_center_time_ms",
        "time_s2_center_time_ms",
    ]

    EXCEL_FILENAME = "preprocess_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TIME_FEATURE_COLUMNS = [
    "time_s1_duration_ms",
    "time_s2_duration_ms",
    "time_s1_center_time_ms",
    "time_s2_center_time_ms",
    "time_s1_on_to_s2_on_ms",
    "time_s1_off_to_s2_on_ms",
    "time_s2_on_to_next_s1_on_ms",
    "time_s2_off_to_next_s1_on_ms",
    "time_cycle_length_ms",
    "time_cycle_length_s2_anchor_ms",
    "time_heart_rate_bpm",
    "time_systolic_fraction",
    "time_diastolic_fraction",
    "time_s1_fraction",
    "time_s2_fraction",
    "time_s1_s2_duration_ratio",
    "time_sys_dia_ratio",
    "time_center_to_center_interval_ms",
]

AMPLITUDE_FEATURE_COLUMNS = [
    "amp_s1_peak_abs",
    "amp_s2_peak_abs",
    "amp_s1_ptp",
    "amp_s2_ptp",
    "amp_s1_mean_abs",
    "amp_s2_mean_abs",
    "amp_s1_rms",
    "amp_s2_rms",
    "amp_s1_area_abs",
    "amp_s2_area_abs",
    "amp_s1_energy",
    "amp_s2_energy",
    "amp_s1_log_energy",
    "amp_s2_log_energy",
    "amp_s1_energy_per_ms",
    "amp_s2_energy_per_ms",
    "amp_s1_s2_peak_ratio",
    "amp_s1_s2_energy_ratio",
]

SHAPE_FEATURE_COLUMNS = [
    "shape_s1_attack_time_ms",
    "shape_s2_attack_time_ms",
    "shape_s1_decay_time_ms",
    "shape_s2_decay_time_ms",
    "shape_s1_attack_decay_ratio",
    "shape_s2_attack_decay_ratio",
    "shape_s1_max_rise_slope",
    "shape_s2_max_rise_slope",
    "shape_s1_max_fall_slope",
    "shape_s2_max_fall_slope",
    "shape_s1_temporal_centroid_ms",
    "shape_s2_temporal_centroid_ms",
]

STATISTICS_FEATURE_COLUMNS = [
    "stat_s1_skewness",
    "stat_s2_skewness",
    "stat_s1_kurtosis",
    "stat_s2_kurtosis",
    "stat_s1_zero_crossing_rate",
    "stat_s2_zero_crossing_rate",
    "stat_s1_abs_sum_first_diff",
    "stat_s2_abs_sum_first_diff",
]

STABILITY_FEATURE_COLUMNS = [
    "stab_s1_template_corr",
    "stab_s2_template_corr",
]

ALL_FEATURE_COLUMNS = (
    TIME_FEATURE_COLUMNS
    + AMPLITUDE_FEATURE_COLUMNS
    + SHAPE_FEATURE_COLUMNS
    + STATISTICS_FEATURE_COLUMNS
    + STABILITY_FEATURE_COLUMNS
)

SUMMARY_SOURCE_COLUMNS = [
    "time_s1_duration_ms",
    "time_s2_duration_ms",
    "time_s1_off_to_s2_on_ms",
    "time_s2_off_to_next_s1_on_ms",
    "time_cycle_length_ms",
    "time_heart_rate_bpm",
    "time_sys_dia_ratio",
]

METADATA_COLUMNS = [
    "record_id",
    "source_file",
    "beat_index",
    "valid_flag",
    "invalid_reason",
    "s1_on",
    "s1_off",
    "s2_on",
    "s2_off",
    "s1_on_next",
    "s2_on_next",
]

EVENT_COLUMN_MAP = {
    "s1_on": "S1-Start_RS_Score",
    "s1_off": "S1-End_RS_Score",
    "s2_on": "S2-Start_RS_Score",
    "s2_off": "S2-End_RS_Score",
}


def parse_record_id(file_path: Path) -> str:
    return file_path.stem


def configured_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser()


def ensure_output_directories(stage_output_folder: Path) -> dict[str, Path]:
    preprocess_root = stage_output_folder
    preprocess_root.mkdir(parents=True, exist_ok=True)
    return {"preprocess_root": preprocess_root}


def normalize_column_name(column_name: Any) -> str:
    return str(column_name).replace("\ufeff", "").strip()


def read_tabular_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(file_path)

    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in PreprocessConfig.CSV_ENCODING_CANDIDATES:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error
                continue
        raise ValueError(f"Unsupported CSV encoding in {file_path.name}") from last_error

    raise ValueError(f"Unsupported input file extension: {file_path.suffix}")


def load_recording_table(file_path: Path, expected_columns: list[str]) -> pd.DataFrame:
    logger.info("파일 로드 시작: %s", file_path.name)
    dataframe = read_tabular_file(file_path)
    if dataframe.empty:
        raise ValueError(f"No data rows found in tabular file: {file_path}")

    dataframe = dataframe.copy()
    dataframe.columns = [normalize_column_name(column_name) for column_name in dataframe.columns]

    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {file_path.name}: {missing_columns}")

    dataframe = dataframe.loc[:, expected_columns].copy()
    dataframe = dataframe.replace({None: np.nan})
    dataframe = dataframe.replace(r"^\s*$", np.nan, regex=True)
    if dataframe[expected_columns].isnull().any().any():
        null_counts = dataframe[expected_columns].isnull().sum()
        failing = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values detected in {file_path.name}: {failing}")

    for column in expected_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise").astype(np.float32)

    if len(dataframe) < 2:
        raise ValueError(f"Recording is too short to define beat boundaries: {file_path.name}")

    logger.info("파일 로드 완료: %s, num_rows=%s", file_path.name, len(dataframe))
    return dataframe


def find_positive_runs(score_array: np.ndarray) -> list[tuple[int, int]]:
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
    run_scores = score_array[start_row : end_row + 1]
    max_score = np.max(run_scores)
    plateau_rows = np.flatnonzero(run_scores == max_score)
    plateau_center = int(plateau_rows[len(plateau_rows) // 2])
    return start_row + plateau_center


def extract_event_candidates(score_array: np.ndarray, event_name: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_start_row, run_end_row in find_positive_runs(score_array):
        peak_row = representative_peak_row(score_array, run_start_row, run_end_row)
        candidates.append(
            {
                "event_name": event_name,
                "run_start_row": int(run_start_row),
                "run_end_row": int(run_end_row),
                "row_index": int(peak_row),
                "sample_index": int(peak_row),
                "peak_score": float(score_array[peak_row]),
            }
        )
    return candidates


def events_between(
    events: list[dict[str, Any]],
    left_sample: int,
    right_sample: int,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if left_sample < int(event["sample_index"]) < right_sample
    ]


def choose_ordered_boundaries(
    s1_on: dict[str, Any],
    s1_on_next: dict[str, Any],
    s1_off_candidates: list[dict[str, Any]],
    s2_on_candidates: list[dict[str, Any]],
    s2_off_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    cycle_start = int(s1_on["sample_index"])
    cycle_end = int(s1_on_next["sample_index"])

    if len(s1_off_candidates) == 0:
        return None, "missing_s1_off"
    if len(s2_on_candidates) == 0:
        return None, "missing_s2_on"
    if len(s2_off_candidates) == 0:
        return None, "missing_s2_off"

    valid_combinations: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for s1_off in s1_off_candidates:
        s1_off_sample = int(s1_off["sample_index"])
        if not cycle_start < s1_off_sample < cycle_end:
            continue
        for s2_on in s2_on_candidates:
            s2_on_sample = int(s2_on["sample_index"])
            if not s1_off_sample <= s2_on_sample < cycle_end:
                continue
            for s2_off in s2_off_candidates:
                s2_off_sample = int(s2_off["sample_index"])
                if not s2_on_sample < s2_off_sample < cycle_end:
                    continue

                total_peak_score = (
                    float(s1_off["peak_score"])
                    + float(s2_on["peak_score"])
                    + float(s2_off["peak_score"])
                )
                total_span = (
                    (int(s1_off["run_end_row"]) - int(s1_off["run_start_row"]))
                    + (int(s2_on["run_end_row"]) - int(s2_on["run_start_row"]))
                    + (int(s2_off["run_end_row"]) - int(s2_off["run_start_row"]))
                )
                sort_key = (total_peak_score, float(total_span), float(s2_off_sample))
                valid_combinations.append(
                    (
                        sort_key,
                        {
                            "s1_on": s1_on,
                            "s1_off": s1_off,
                            "s2_on": s2_on,
                            "s2_off": s2_off,
                            "s1_on_next": s1_on_next,
                        },
                    )
                )

    if not valid_combinations:
        return None, "invalid_boundary_order"

    valid_combinations.sort(key=lambda item: item[0], reverse=True)
    return valid_combinations[0][1], ""


def load_boundaries_from_recording(recording_table: pd.DataFrame) -> list[dict[str, Any]]:
    event_candidates = {
        event_name: extract_event_candidates(
            recording_table[column_name].to_numpy(dtype=np.float32),
            event_name=event_name,
        )
        for event_name, column_name in EVENT_COLUMN_MAP.items()
    }

    s1_on_events = event_candidates["s1_on"]
    if len(s1_on_events) < 2:
        raise ValueError("At least two S1 onset events are required to define beats.")

    beats: list[dict[str, Any]] = []
    for beat_index in range(len(s1_on_events) - 1):
        s1_on = s1_on_events[beat_index]
        s1_on_next = s1_on_events[beat_index + 1]
        left_sample = int(s1_on["sample_index"])
        right_sample = int(s1_on_next["sample_index"])

        selected, invalid_reason = choose_ordered_boundaries(
            s1_on=s1_on,
            s1_on_next=s1_on_next,
            s1_off_candidates=events_between(event_candidates["s1_off"], left_sample, right_sample),
            s2_on_candidates=events_between(event_candidates["s2_on"], left_sample, right_sample),
            s2_off_candidates=events_between(event_candidates["s2_off"], left_sample, right_sample),
        )

        beat_row: dict[str, Any] = {
            "beat_index": int(beat_index),
            "invalid_reason": invalid_reason,
            "s1_on": int(s1_on["sample_index"]),
            "s1_off": np.nan,
            "s2_on": np.nan,
            "s2_off": np.nan,
            "s1_on_next": int(s1_on_next["sample_index"]),
            "s2_on_next": np.nan,
            "s1_on_row": int(s1_on["row_index"]),
            "s1_off_row": np.nan,
            "s2_on_row": np.nan,
            "s2_off_row": np.nan,
            "s1_on_next_row": int(s1_on_next["row_index"]),
        }

        if selected is not None:
            beat_row.update(
                {
                    "s1_off": int(selected["s1_off"]["sample_index"]),
                    "s2_on": int(selected["s2_on"]["sample_index"]),
                    "s2_off": int(selected["s2_off"]["sample_index"]),
                    "s1_off_row": int(selected["s1_off"]["row_index"]),
                    "s2_on_row": int(selected["s2_on"]["row_index"]),
                    "s2_off_row": int(selected["s2_off"]["row_index"]),
                }
            )
        beats.append(beat_row)

    for beat_index in range(len(beats) - 1):
        next_s2_on = beats[beat_index + 1]["s2_on"]
        if pd.notna(next_s2_on):
            beats[beat_index]["s2_on_next"] = int(next_s2_on)

    return beats


def samples_to_ms(num_samples: float, fs: int) -> float:
    return float(1000.0 * float(num_samples) / float(fs))


def safe_nan_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0:
        return float(np.nan)
    return float(numerator / denominator)


def safe_eps_ratio(numerator: float, denominator: float, eps: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if np.isnan(numerator) or np.isnan(denominator):
        return float(np.nan)
    return float(numerator / (denominator + eps))


def nan_feature_map() -> dict[str, float]:
    return {column: float(np.nan) for column in ALL_FEATURE_COLUMNS}


def validate_beat_boundaries(
    beat: dict[str, Any],
    signal_length: int,
    fs: int,
    min_cycle_ms: float,
    max_cycle_ms: float,
) -> tuple[bool, str]:
    required_columns = ["s1_on", "s1_off", "s2_on", "s2_off", "s1_on_next"]
    if any(pd.isna(beat[column]) for column in required_columns):
        return False, beat["invalid_reason"] or "missing_boundary"

    s1_on = int(beat["s1_on"])
    s1_off = int(beat["s1_off"])
    s2_on = int(beat["s2_on"])
    s2_off = int(beat["s2_off"])
    s1_on_next = int(beat["s1_on_next"])

    if not (0 <= s1_on < s1_off <= s2_on < s2_off < s1_on_next <= signal_length):
        return False, "invalid_boundary_order"

    cycle_length_ms = samples_to_ms(s1_on_next - s1_on, fs)
    if cycle_length_ms <= 0:
        return False, "non_positive_cycle_length"
    if cycle_length_ms < min_cycle_ms or cycle_length_ms > max_cycle_ms:
        return False, "cycle_length_out_of_range"

    return True, ""


def compute_smoothed_envelope(x: np.ndarray, fs: int, smooth_ms: float) -> np.ndarray:
    envelope_raw = np.abs(hilbert(x))
    window_size = max(3, int(round((smooth_ms / 1000.0) * fs)))
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(envelope_raw.astype(np.float32), kernel, mode="same").astype(np.float32)


def segment_from_rows(signal: np.ndarray, row_start: int, row_end: int) -> np.ndarray:
    if row_start < 0 or row_end > len(signal) or row_start >= row_end:
        return np.empty(0, dtype=np.float32)
    return signal[row_start:row_end].astype(np.float32, copy=False)


def zscore_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean_value) / std_value).astype(np.float32)


def resample_linear(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.full(target_length, np.nan, dtype=np.float32)
    if values.size == 1:
        return np.full(target_length, float(values[0]), dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32)


def compute_time_features(beat: dict[str, Any], fs: int) -> dict[str, float]:
    s1_on = int(beat["s1_on"])
    s1_off = int(beat["s1_off"])
    s2_on = int(beat["s2_on"])
    s2_off = int(beat["s2_off"])
    s1_on_next = int(beat["s1_on_next"])
    cycle_length_samples = s1_on_next - s1_on
    cycle_length_ms = samples_to_ms(cycle_length_samples, fs)

    s1_center = 0.5 * (s1_on + s1_off)
    s2_center = 0.5 * (s2_on + s2_off)

    s2_on_next = float(beat["s2_on_next"]) if pd.notna(beat["s2_on_next"]) else float(np.nan)
    if np.isnan(s2_on_next):
        cycle_length_s2_anchor_ms = float(np.nan)
    else:
        cycle_length_s2_anchor_ms = samples_to_ms(s2_on_next - s2_on, fs)

    return {
        "time_s1_duration_ms": samples_to_ms(s1_off - s1_on, fs),
        "time_s2_duration_ms": samples_to_ms(s2_off - s2_on, fs),
        "time_s1_center_time_ms": samples_to_ms(s1_center, fs),
        "time_s2_center_time_ms": samples_to_ms(s2_center, fs),
        "time_s1_on_to_s2_on_ms": samples_to_ms(s2_on - s1_on, fs),
        "time_s1_off_to_s2_on_ms": samples_to_ms(s2_on - s1_off, fs),
        "time_s2_on_to_next_s1_on_ms": samples_to_ms(s1_on_next - s2_on, fs),
        "time_s2_off_to_next_s1_on_ms": samples_to_ms(s1_on_next - s2_off, fs),
        "time_cycle_length_ms": cycle_length_ms,
        "time_cycle_length_s2_anchor_ms": cycle_length_s2_anchor_ms,
        "time_heart_rate_bpm": float(60000.0 / cycle_length_ms) if cycle_length_ms > 0 else float(np.nan),
        "time_systolic_fraction": safe_nan_ratio(s2_on - s1_on, cycle_length_samples),
        "time_diastolic_fraction": safe_nan_ratio(s1_on_next - s2_on, cycle_length_samples),
        "time_s1_fraction": safe_nan_ratio(s1_off - s1_on, cycle_length_samples),
        "time_s2_fraction": safe_nan_ratio(s2_off - s2_on, cycle_length_samples),
        "time_s1_s2_duration_ratio": safe_nan_ratio(s1_off - s1_on, s2_off - s2_on),
        "time_sys_dia_ratio": safe_nan_ratio(s2_on - s1_on, s1_on_next - s2_on),
        "time_center_to_center_interval_ms": samples_to_ms(s2_center - s1_center, fs),
    }


def compute_segment_amplitude_features(
    segment: np.ndarray,
    duration_ms: float,
    prefix: str,
    eps: float,
) -> dict[str, float]:
    if segment.size == 0:
        return {
            f"amp_{prefix}_peak_abs": float(np.nan),
            f"amp_{prefix}_ptp": float(np.nan),
            f"amp_{prefix}_mean_abs": float(np.nan),
            f"amp_{prefix}_rms": float(np.nan),
            f"amp_{prefix}_area_abs": float(np.nan),
            f"amp_{prefix}_energy": float(np.nan),
            f"amp_{prefix}_log_energy": float(np.nan),
            f"amp_{prefix}_energy_per_ms": float(np.nan),
        }

    energy = float(np.sum(np.square(segment)))
    return {
        f"amp_{prefix}_peak_abs": float(np.max(np.abs(segment))),
        f"amp_{prefix}_ptp": float(np.ptp(segment)),
        f"amp_{prefix}_mean_abs": float(np.mean(np.abs(segment))),
        f"amp_{prefix}_rms": float(np.sqrt(np.mean(np.square(segment)))),
        f"amp_{prefix}_area_abs": float(np.sum(np.abs(segment))),
        f"amp_{prefix}_energy": energy,
        f"amp_{prefix}_log_energy": float(np.log(energy + eps)),
        f"amp_{prefix}_energy_per_ms": float(energy / (duration_ms + eps)),
    }


def compute_amplitude_energy_features(
    x: np.ndarray,
    envelope: np.ndarray,
    beat: dict[str, Any],
    fs: int,
    eps: float,
) -> dict[str, float]:
    del envelope
    s1 = segment_from_rows(x, int(beat["s1_on_row"]), int(beat["s1_off_row"]))
    s2 = segment_from_rows(x, int(beat["s2_on_row"]), int(beat["s2_off_row"]))

    s1_duration_ms = samples_to_ms(int(beat["s1_off"]) - int(beat["s1_on"]), fs)
    s2_duration_ms = samples_to_ms(int(beat["s2_off"]) - int(beat["s2_on"]), fs)

    features = {}
    features.update(compute_segment_amplitude_features(s1, s1_duration_ms, "s1", eps))
    features.update(compute_segment_amplitude_features(s2, s2_duration_ms, "s2", eps))
    features["amp_s1_s2_peak_ratio"] = safe_eps_ratio(
        features["amp_s1_peak_abs"],
        features["amp_s2_peak_abs"],
        eps,
    )
    features["amp_s1_s2_energy_ratio"] = safe_eps_ratio(
        features["amp_s1_energy"],
        features["amp_s2_energy"],
        eps,
    )
    return features


def compute_segment_shape_features(
    env_segment: np.ndarray,
    prefix: str,
    fs: int,
    eps: float,
) -> dict[str, float]:
    if env_segment.size == 0:
        return {
            f"shape_{prefix}_attack_time_ms": float(np.nan),
            f"shape_{prefix}_decay_time_ms": float(np.nan),
            f"shape_{prefix}_attack_decay_ratio": float(np.nan),
            f"shape_{prefix}_max_rise_slope": float(np.nan),
            f"shape_{prefix}_max_fall_slope": float(np.nan),
            f"shape_{prefix}_temporal_centroid_ms": float(np.nan),
        }

    peak_idx_rel = int(np.argmax(env_segment))
    attack_time_ms = samples_to_ms(peak_idx_rel, fs)
    decay_time_ms = samples_to_ms(len(env_segment) - peak_idx_rel, fs)

    if len(env_segment) < 2:
        max_rise_slope = float(np.nan)
        max_fall_slope = float(np.nan)
    else:
        env_diff = np.diff(env_segment)
        max_rise_slope = float(np.max(env_diff) * fs)
        max_fall_slope = float(np.min(env_diff) * fs)

    t_rel = np.arange(len(env_segment), dtype=np.float32) / float(fs)
    temporal_centroid_ms = float(
        1000.0 * np.sum(t_rel * env_segment) / (float(np.sum(env_segment)) + eps)
    )

    return {
        f"shape_{prefix}_attack_time_ms": attack_time_ms,
        f"shape_{prefix}_decay_time_ms": decay_time_ms,
        f"shape_{prefix}_attack_decay_ratio": safe_eps_ratio(attack_time_ms, decay_time_ms, eps),
        f"shape_{prefix}_max_rise_slope": max_rise_slope,
        f"shape_{prefix}_max_fall_slope": max_fall_slope,
        f"shape_{prefix}_temporal_centroid_ms": temporal_centroid_ms,
    }


def compute_shape_features(
    envelope: np.ndarray,
    beat: dict[str, Any],
    fs: int,
    eps: float,
) -> dict[str, float]:
    env_s1 = segment_from_rows(envelope, int(beat["s1_on_row"]), int(beat["s1_off_row"]))
    env_s2 = segment_from_rows(envelope, int(beat["s2_on_row"]), int(beat["s2_off_row"]))

    features = {}
    features.update(compute_segment_shape_features(env_s1, "s1", fs, eps))
    features.update(compute_segment_shape_features(env_s2, "s2", fs, eps))
    return features


def compute_segment_statistics_features(segment: np.ndarray, prefix: str, eps: float) -> dict[str, float]:
    if segment.size == 0:
        return {
            f"stat_{prefix}_skewness": float(np.nan),
            f"stat_{prefix}_kurtosis": float(np.nan),
            f"stat_{prefix}_zero_crossing_rate": float(np.nan),
            f"stat_{prefix}_abs_sum_first_diff": float(np.nan),
        }

    mean_value = float(np.mean(segment))
    std_value = float(np.std(segment))
    normalized = (segment - mean_value) / (std_value + eps)
    skewness = float(np.mean(np.power(normalized, 3)))
    kurtosis = float(np.mean(np.power(normalized, 4)))

    if len(segment) < 2:
        zcr = float(np.nan)
        abs_sum_first_diff = float(np.nan)
    else:
        sign_changes = np.count_nonzero(segment[:-1] * segment[1:] < 0)
        zcr = float(sign_changes / (len(segment) - 1))
        abs_sum_first_diff = float(np.sum(np.abs(np.diff(segment))))

    return {
        f"stat_{prefix}_skewness": skewness,
        f"stat_{prefix}_kurtosis": kurtosis,
        f"stat_{prefix}_zero_crossing_rate": zcr,
        f"stat_{prefix}_abs_sum_first_diff": abs_sum_first_diff,
    }


def compute_statistics_complexity_features(
    x: np.ndarray,
    beat: dict[str, Any],
    eps: float,
) -> dict[str, float]:
    s1 = segment_from_rows(x, int(beat["s1_on_row"]), int(beat["s1_off_row"]))
    s2 = segment_from_rows(x, int(beat["s2_on_row"]), int(beat["s2_off_row"]))

    features = {}
    features.update(compute_segment_statistics_features(s1, "s1", eps))
    features.update(compute_segment_statistics_features(s2, "s2", eps))
    return features


def build_template(
    segments: list[np.ndarray],
    target_length: int,
    eps: float,
) -> np.ndarray:
    if not segments:
        return np.full(target_length, np.nan, dtype=np.float32)

    normalized_rows: list[np.ndarray] = []
    for segment in segments:
        resampled = resample_linear(segment, target_length)
        normalized_rows.append(zscore_normalize(resampled, eps))
    stacked = np.stack(normalized_rows).astype(np.float32)
    return np.median(stacked, axis=0).astype(np.float32)


def build_record_templates(
    x: np.ndarray,
    beat_rows: list[dict[str, Any]],
    template_resample_length: int,
    eps: float,
) -> dict[str, np.ndarray]:
    s1_segments: list[np.ndarray] = []
    s2_segments: list[np.ndarray] = []
    for beat in beat_rows:
        if int(beat["valid_flag"]) != 1:
            continue
        s1 = segment_from_rows(x, int(beat["s1_on_row"]), int(beat["s1_off_row"]))
        s2 = segment_from_rows(x, int(beat["s2_on_row"]), int(beat["s2_off_row"]))
        if s1.size > 0:
            s1_segments.append(s1)
        if s2.size > 0:
            s2_segments.append(s2)

    return {
        "s1": build_template(s1_segments, template_resample_length, eps),
        "s2": build_template(s2_segments, template_resample_length, eps),
    }


def pearson_corr(left: np.ndarray, right: np.ndarray, eps: float) -> float:
    if left.size == 0 or right.size == 0:
        return float(np.nan)
    if np.isnan(left).any() or np.isnan(right).any():
        return float(np.nan)

    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= eps or right_std <= eps:
        return float(np.nan)

    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    return float(np.mean(left_centered * right_centered) / (left_std * right_std))


def compute_stability_features(
    x: np.ndarray,
    beat: dict[str, Any],
    templates: dict[str, np.ndarray],
    template_resample_length: int,
    eps: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, row_start_key, row_end_key in [
        ("s1", "s1_on_row", "s1_off_row"),
        ("s2", "s2_on_row", "s2_off_row"),
    ]:
        segment = segment_from_rows(x, int(beat[row_start_key]), int(beat[row_end_key]))
        if segment.size == 0:
            features[f"stab_{prefix}_template_corr"] = float(np.nan)
            continue
        normalized_segment = zscore_normalize(
            resample_linear(segment, template_resample_length),
            eps,
        )
        template = templates[prefix]
        features[f"stab_{prefix}_template_corr"] = pearson_corr(
            normalized_segment,
            template,
            eps,
        )
    return features


def build_feature_row(
    record_id: str,
    source_file: str,
    beat: dict[str, Any],
    x: np.ndarray,
    envelope: np.ndarray,
    fs: int,
    eps: float,
    templates: dict[str, np.ndarray],
    template_resample_length: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id,
        "source_file": source_file,
        "beat_index": int(beat["beat_index"]),
        "valid_flag": int(beat["valid_flag"]),
        "invalid_reason": beat.get("invalid_reason", ""),
        "s1_on": beat["s1_on"],
        "s1_off": beat["s1_off"],
        "s2_on": beat["s2_on"],
        "s2_off": beat["s2_off"],
        "s1_on_next": beat["s1_on_next"],
        "s2_on_next": beat["s2_on_next"],
    }

    if int(beat["valid_flag"]) != 1:
        row.update(nan_feature_map())
        return row

    features: dict[str, float] = {}
    features.update(compute_time_features(beat, fs))
    features.update(compute_amplitude_energy_features(x, envelope, beat, fs, eps))
    features.update(compute_shape_features(envelope, beat, fs, eps))
    features.update(compute_statistics_complexity_features(x, beat, eps))
    features.update(
        compute_stability_features(
            x=x,
            beat=beat,
            templates=templates,
            template_resample_length=template_resample_length,
            eps=eps,
        )
    )
    row.update(features)
    return row


def build_feature_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    ordered_columns = METADATA_COLUMNS + ALL_FEATURE_COLUMNS
    missing_columns = [column for column in ordered_columns if column not in dataframe.columns]
    for column in missing_columns:
        dataframe[column] = np.nan
    return dataframe[ordered_columns].copy()


def summary_stat(values: np.ndarray, eps: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "mean": float(np.nan),
            "std": float(np.nan),
            "cv": float(np.nan),
            "rmssd": float(np.nan),
        }

    std_value = float(np.std(values, ddof=0))
    mean_value = float(np.mean(values))
    if values.size < 2:
        rmssd = float(np.nan)
    else:
        rmssd = float(np.sqrt(np.mean(np.square(np.diff(values)))))
    return {
        "mean": mean_value,
        "std": std_value,
        "cv": float(std_value / (mean_value + eps)),
        "rmssd": rmssd,
    }


def build_record_summary(
    feature_frame: pd.DataFrame,
    eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_id, record_frame in feature_frame.groupby("record_id", sort=True):
        valid_frame = record_frame.loc[record_frame["valid_flag"] == 1].copy()
        row: dict[str, Any] = {
            "record_id": record_id,
            "total_beats": int(len(record_frame)),
            "valid_beats": int(len(valid_frame)),
            "invalid_beats": int(len(record_frame) - len(valid_frame)),
        }
        for column in SUMMARY_SOURCE_COLUMNS:
            stats = summary_stat(valid_frame[column].to_numpy(dtype=np.float64), eps)
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_std"] = stats["std"]
            row[f"{column}_cv"] = stats["cv"]
            row[f"{column}_rmssd"] = stats["rmssd"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_learning_input_columns(feature_columns: list[str], excluded_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if column not in set(excluded_columns)]


def export_feature_outputs(
    preprocess_root: Path,
    beat_features_all: pd.DataFrame,
    beat_features_valid: pd.DataFrame,
    record_summary: pd.DataFrame,
    feature_names: list[str],
    learning_input_columns: list[str],
) -> None:
    if PreprocessConfig.EXPORT_CSV:
        beat_features_all.to_csv(preprocess_root / "beat_features_all.csv", index=False)
        beat_features_valid.to_csv(preprocess_root / "beat_features_valid.csv", index=False)
        record_summary.to_csv(preprocess_root / "record_summary.csv", index=False)

    if PreprocessConfig.EXPORT_FEATURE_NAMES_JSON:
        (preprocess_root / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (preprocess_root / "learning_input_columns.json").write_text(
            json.dumps(learning_input_columns, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if PreprocessConfig.EXPORT_EXCEL:
        export_stage_workbook(
            workbook_path=preprocess_root / PreprocessConfig.EXCEL_FILENAME,
            sheets={
                "beat_features_all": beat_features_all,
                "beat_features_valid": beat_features_valid,
                "record_summary": record_summary,
            },
            freeze_panes=PreprocessConfig.EXCEL_FREEZE_PANES,
            header_fill=PreprocessConfig.EXCEL_HEADER_FILL,
            header_font_color=PreprocessConfig.EXCEL_HEADER_FONT_COLOR,
            max_column_width=PreprocessConfig.EXCEL_MAX_COLUMN_WIDTH,
        )

    logger.info("export 경로: %s", preprocess_root)


def process_recording(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_id = parse_record_id(file_path)
    recording_table = load_recording_table(file_path, PreprocessConfig.EXPECTED_COLUMNS)
    x = recording_table["Amplitude"].to_numpy(dtype=np.float32)
    envelope = compute_smoothed_envelope(
        x=x,
        fs=PreprocessConfig.SAMPLING_RATE,
        smooth_ms=PreprocessConfig.ENVELOPE_SMOOTH_MS,
    )

    beats = load_boundaries_from_recording(recording_table)
    logger.info("beat 수: record_id=%s, total_beats=%s", record_id, len(beats))

    prepared_beats: list[dict[str, Any]] = []
    for beat in beats:
        is_valid, invalid_reason = validate_beat_boundaries(
            beat=beat,
            signal_length=len(x),
            fs=PreprocessConfig.SAMPLING_RATE,
            min_cycle_ms=PreprocessConfig.MIN_CYCLE_MS,
            max_cycle_ms=PreprocessConfig.MAX_CYCLE_MS,
        )
        beat_copy = beat.copy()
        beat_copy["valid_flag"] = int(is_valid)
        beat_copy["invalid_reason"] = invalid_reason
        prepared_beats.append(beat_copy)

    valid_count = sum(int(beat["valid_flag"]) for beat in prepared_beats)
    invalid_count = len(prepared_beats) - valid_count
    logger.info(
        "valid / invalid beat 수: record_id=%s, valid=%s, invalid=%s",
        record_id,
        valid_count,
        invalid_count,
    )

    templates = build_record_templates(
        x=x,
        beat_rows=prepared_beats,
        template_resample_length=PreprocessConfig.TEMPLATE_RESAMPLE_LENGTH,
        eps=PreprocessConfig.EPS,
    )

    rows = [
        build_feature_row(
            record_id=record_id,
            source_file=file_path.name,
            beat=beat,
            x=x,
            envelope=envelope,
            fs=PreprocessConfig.SAMPLING_RATE,
            eps=PreprocessConfig.EPS,
            templates=templates,
            template_resample_length=PreprocessConfig.TEMPLATE_RESAMPLE_LENGTH,
        )
        for beat in prepared_beats
    ]
    feature_frame = build_feature_dataframe(rows)
    logger.info("feature 추출 완료: record_id=%s", record_id)

    valid_frame = feature_frame.loc[feature_frame["valid_flag"] == 1].copy()
    return feature_frame, valid_frame


def collect_input_files(data_root: Path, file_glob: str, supported_suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in data_root.glob(file_glob)
        if path.is_file()
        and not path.name.startswith("~$")
        and path.suffix.lower() in supported_suffixes
    )


def main() -> None:
    output_paths = ensure_output_directories(
        stage_output_folder=configured_path(PreprocessConfig.OUTPUT_FOLDER),
    )

    data_root = configured_path(PreprocessConfig.TRAIN_DATA_FOLDER)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    logger.info("입력 데이터 폴더: %s", data_root)
    logger.info("전처리 출력 폴더: %s", output_paths["preprocess_root"])

    file_paths = collect_input_files(
        data_root=data_root,
        file_glob=PreprocessConfig.FILE_GLOB,
        supported_suffixes=PreprocessConfig.SUPPORTED_INPUT_SUFFIXES,
    )
    if not file_paths:
        raise FileNotFoundError(f"No input files found under: {data_root}")

    all_feature_frames: list[pd.DataFrame] = []
    all_valid_frames: list[pd.DataFrame] = []
    for file_path in file_paths:
        feature_frame, valid_frame = process_recording(file_path)
        all_feature_frames.append(feature_frame)
        all_valid_frames.append(valid_frame)

    beat_features_all = pd.concat(all_feature_frames, axis=0, ignore_index=True)
    if PreprocessConfig.SAVE_INVALID_ROWS:
        beat_features_all = beat_features_all.copy()
    else:
        beat_features_all = beat_features_all.loc[beat_features_all["valid_flag"] == 1].copy()

    beat_features_valid = pd.concat(all_valid_frames, axis=0, ignore_index=True)
    feature_names = list(ALL_FEATURE_COLUMNS)
    learning_input_columns = build_learning_input_columns(
        feature_columns=feature_names,
        excluded_columns=PreprocessConfig.LEARNING_INPUT_EXCLUDE_COLUMNS,
    )
    record_summary = build_record_summary(beat_features_all, eps=PreprocessConfig.EPS)

    logger.info("학습 입력 feature 수: %s", len(learning_input_columns))
    logger.info("제외된 column 목록: %s", PreprocessConfig.LEARNING_INPUT_EXCLUDE_COLUMNS)

    export_feature_outputs(
        preprocess_root=output_paths["preprocess_root"],
        beat_features_all=beat_features_all,
        beat_features_valid=beat_features_valid,
        record_summary=record_summary,
        feature_names=feature_names,
        learning_input_columns=learning_input_columns,
    )


if __name__ == "__main__":
    main()
