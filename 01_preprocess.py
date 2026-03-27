"""
Cycle-level preprocessing script for the revised Heart-Sound-Unsupervised-Model pipeline.

Expected input schema per tabular file (.xlsx or .csv):
- Amplitude
- S1-Start_RS_Score
- S1-End_RS_Score
- S2-Start_RS_Score
- S2-End_RS_Score

Saved artifacts:
- outputs/{RUN_NAME}/preprocess/beat_features_all.csv
- outputs/{RUN_NAME}/preprocess/beat_features_valid.csv
- outputs/{RUN_NAME}/preprocess/record_summary.csv
- outputs/{RUN_NAME}/preprocess/feature_names.json
- outputs/{RUN_NAME}/preprocess/learning_input_columns.json
- outputs/{RUN_NAME}/preprocess/feature_groups.json
- outputs/{RUN_NAME}/preprocess/preprocess_summary.json
- outputs/{RUN_NAME}/preprocess/preprocess_export.xlsx
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from excel_export_utils import export_stage_workbook


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreprocessConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 2개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    TRAIN_DATA_FOLDER = r"C:\Users\LUI\Desktop\PCG\Data\Train 학습데이터(260109)"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"

    FILE_GLOB = "*"
    SUPPORTED_INPUT_SUFFIXES = (".xlsx", ".csv")
    CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

    SAMPLING_RATE = 4000.0
    SAMPLE_MS = 1000.0 / SAMPLING_RATE
    REGION_THRESHOLD = 15.0
    DEFAULT_CYCLE_SPACING = 4000
    MAX_REGION_WIDTH_RATIO = 0.45
    EPS = 1e-8

    ENVELOPE_SMOOTH_MS = 20.0
    ENVELOPE_OCCUPANCY_THRESHOLD_RATIO = 0.30
    TEMPLATE_RESAMPLE_POINTS = 64
    TEMPLATE_MIN_VALID_SEGMENTS = 3

    EARLY_DIASTOLE_RATIO = (0.00, 0.33)
    MID_DIASTOLE_RATIO = (0.33, 0.66)
    LATE_DIASTOLE_RATIO = (0.66, 1.00)

    EXPECTED_COLUMNS = [
        "Amplitude",
        "S1-Start_RS_Score",
        "S1-End_RS_Score",
        "S2-Start_RS_Score",
        "S2-End_RS_Score",
    ]

    KEY_RECORD_SUMMARY_COLUMNS = [
        "global_cycle_length_ms",
        "global_hr_bpm",
        "seg_sys_energy",
        "zone_ed_peak_rel_to_s2",
        "zone_ld_peak_rel_to_s1",
    ]

    EXPORT_EXCEL = True
    EXPORT_CSV = True
    EXPORT_FEATURE_NAMES_JSON = True
    SAVE_INVALID_ROWS = True
    EXCEL_FILENAME = "preprocess_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40


# =================================================
# 1. Feature Schema
# =================================================


GLOBAL_FEATURE_COLUMNS = [
    "global_cycle_length_ms",
    "global_hr_bpm",
    "global_s1_width_ratio",
    "global_s2_width_ratio",
    "global_systole_ratio",
    "global_diastole_ratio",
    "global_total_energy",
    "global_env_mean",
    "global_env_peak",
    "global_env_rms",
]

SEGMENT_TEMPLATE_SUFFIXES = [
    "duration_ms",
    "mean_env",
    "peak_env",
    "rms",
    "energy",
    "energy_ratio_to_cycle",
    "energy_centroid",
    "energy_spread",
    "env_occupancy",
]

ZONE_TEMPLATE_SUFFIXES = [
    "duration_ms",
    "mean_env",
    "peak_env",
    "rms",
    "energy",
    "energy_ratio_to_diastole",
    "peak_timing_relative",
    "energy_centroid",
    "energy_spread",
    "env_occupancy",
]

SEGMENT_S1_COLUMNS = [f"seg_s1_{suffix}" for suffix in SEGMENT_TEMPLATE_SUFFIXES]
SEGMENT_SYSTOLE_COLUMNS = [f"seg_sys_{suffix}" for suffix in SEGMENT_TEMPLATE_SUFFIXES]
SEGMENT_S2_COLUMNS = [f"seg_s2_{suffix}" for suffix in SEGMENT_TEMPLATE_SUFFIXES]
SEGMENT_DIASTOLE_COLUMNS = [f"seg_dia_{suffix}" for suffix in SEGMENT_TEMPLATE_SUFFIXES]

ZONE_EARLY_COLUMNS = [f"zone_ed_{suffix}" for suffix in ZONE_TEMPLATE_SUFFIXES]
ZONE_MID_COLUMNS = [f"zone_md_{suffix}" for suffix in ZONE_TEMPLATE_SUFFIXES]
ZONE_LATE_COLUMNS = [f"zone_ld_{suffix}" for suffix in ZONE_TEMPLATE_SUFFIXES]

S3S4_RELATIVE_COLUMNS = [
    "zone_ed_peak_rel_to_s2",
    "zone_ed_mean_rel_to_s2",
    "zone_ld_peak_rel_to_s1",
    "zone_ld_mean_rel_to_s1",
    "zone_md_peak_rel_to_s1s2_mean",
]

SHAPE_FEATURE_COLUMNS = [
    "shape_s1_attack_ratio",
    "shape_s1_decay_ratio",
    "shape_s1_temporal_centroid_rel",
    "shape_s2_attack_ratio",
    "shape_s2_decay_ratio",
    "shape_s2_temporal_centroid_rel",
]

STAT_FEATURE_COLUMNS = [
    "stat_cycle_zero_crossing_rate",
    "stat_cycle_diff_mean_abs",
    "stat_s1_skewness",
    "stat_s1_kurtosis",
    "stat_s2_skewness",
    "stat_s2_kurtosis",
]

STABILITY_FEATURE_COLUMNS = [
    "stab_s1_template_corr",
    "stab_s2_template_corr",
]

PRIMARY_FEATURE_GROUPS = {
    "global": GLOBAL_FEATURE_COLUMNS,
    "segment_s1": SEGMENT_S1_COLUMNS,
    "segment_systole": SEGMENT_SYSTOLE_COLUMNS,
    "segment_s2": SEGMENT_S2_COLUMNS,
    "segment_diastole": SEGMENT_DIASTOLE_COLUMNS,
    "zone_early_diastole": ZONE_EARLY_COLUMNS,
    "zone_mid_diastole": ZONE_MID_COLUMNS,
    "zone_late_diastole": ZONE_LATE_COLUMNS,
    "s3s4_relative": S3S4_RELATIVE_COLUMNS,
    "shape": SHAPE_FEATURE_COLUMNS,
    "stat": STAT_FEATURE_COLUMNS,
    "stability": STABILITY_FEATURE_COLUMNS,
}

SECONDARY_FEATURE_GROUPS = {
    "murmur_related": [
        "seg_sys_energy",
        "seg_sys_energy_ratio_to_cycle",
        "seg_sys_energy_centroid",
        "seg_sys_energy_spread",
        "seg_sys_env_occupancy",
    ],
}

FEATURE_GROUPS = {**PRIMARY_FEATURE_GROUPS, **SECONDARY_FEATURE_GROUPS}

ALL_FEATURE_COLUMNS: list[str] = []
for group_columns in PRIMARY_FEATURE_GROUPS.values():
    for column in group_columns:
        if column not in ALL_FEATURE_COLUMNS:
            ALL_FEATURE_COLUMNS.append(column)

METADATA_COLUMNS = [
    "record_id",
    "source_file",
    "beat_index",
    "cycle_index",
    "valid_flag",
    "invalid_reason",
    "s1_start",
    "s1_end",
    "s2_start",
    "s2_end",
    "next_s1_start",
    "next_s1_end",
    "cycle_start",
    "cycle_end",
    "s1_on",
    "s1_off",
    "s2_on",
    "s2_off",
    "s1_on_next",
]

EVENT_COLUMN_MAP = {
    "S1_start": "S1-Start_RS_Score",
    "S1_end": "S1-End_RS_Score",
    "S2_start": "S2-Start_RS_Score",
    "S2_end": "S2-End_RS_Score",
}


# =================================================
# 2. File IO Helpers
# =================================================


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
        raise ValueError(f"Recording is too short to define cycles: {file_path.name}")

    logger.info("파일 로드 완료: %s, num_rows=%s", file_path.name, len(dataframe))
    return dataframe


# =================================================
# 3. Cycle Detection Helpers
# =================================================


def _extract_threshold_peaks(values: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    best_index: int | None = None
    best_value: float | None = None

    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if not np.isfinite(value) or value < threshold:
            if best_index is not None and best_value is not None:
                peaks.append((best_index, best_value))
            best_index = None
            best_value = None
            continue

        if best_index is None or best_value is None or value > best_value:
            best_index = index
            best_value = value

    if best_index is not None and best_value is not None:
        peaks.append((best_index, best_value))

    return peaks


def _estimate_peak_spacing(peaks: list[tuple[int, float]]) -> int | None:
    if len(peaks) < 2:
        return None

    spacings = sorted(
        peaks[index][0] - peaks[index - 1][0]
        for index in range(1, len(peaks))
        if peaks[index][0] - peaks[index - 1][0] > 0
    )
    if not spacings:
        return None

    return int(spacings[len(spacings) // 2])


def _build_region_overlays(label: str, start_values: np.ndarray, end_values: np.ndarray) -> list[dict[str, Any]]:
    start_peaks = _extract_threshold_peaks(start_values, PreprocessConfig.REGION_THRESHOLD)
    end_peaks = _extract_threshold_peaks(end_values, PreprocessConfig.REGION_THRESHOLD)
    if not start_peaks or not end_peaks:
        return []

    cycle_spacing = (
        _estimate_peak_spacing(start_peaks)
        or _estimate_peak_spacing(end_peaks)
        or PreprocessConfig.DEFAULT_CYCLE_SPACING
    )
    max_region_width = max(1, int(cycle_spacing * PreprocessConfig.MAX_REGION_WIDTH_RATIO))

    overlays: list[dict[str, Any]] = []
    end_peak_index = 0

    for index, start_peak in enumerate(start_peaks):
        next_start_index = start_peaks[index + 1][0] if index + 1 < len(start_peaks) else None

        while end_peak_index < len(end_peaks) and end_peaks[end_peak_index][0] <= start_peak[0]:
            end_peak_index += 1

        if end_peak_index >= len(end_peaks):
            break

        end_peak = end_peaks[end_peak_index]
        if next_start_index is not None and end_peak[0] >= next_start_index:
            continue

        region_width = end_peak[0] - start_peak[0]
        if region_width <= 0 or region_width > max_region_width:
            continue

        overlays.append(
            {
                "label": label,
                "startPeak": start_peak,
                "endPeak": end_peak,
                "areaStart": start_peak[0],
                "areaEnd": end_peak[0],
            }
        )
        end_peak_index += 1

    return overlays


def _get_region_score(overlay: dict[str, Any]) -> float:
    start_peak = overlay.get("startPeak") or (0, 0.0)
    end_peak = overlay.get("endPeak") or (0, 0.0)
    return max(float(start_peak[1]), float(end_peak[1]))


def _regions_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["areaStart"]) <= int(right["areaEnd"]) and int(right["areaStart"]) <= int(left["areaEnd"])


def _resolve_region_overlaps(
    first_overlays: list[dict[str, Any]],
    second_overlays: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        [*first_overlays, *second_overlays],
        key=lambda overlay: (int(overlay["areaStart"]), -_get_region_score(overlay)),
    )
    resolved: list[dict[str, Any]] = []

    for overlay in ordered:
        previous = resolved[-1] if resolved else None
        if previous is None or previous["label"] == overlay["label"] or not _regions_overlap(previous, overlay):
            resolved.append(overlay)
            continue

        if _get_region_score(overlay) > _get_region_score(previous):
            resolved[-1] = overlay

    resolved_first = [overlay for overlay in resolved if overlay["label"] == "S1"]
    resolved_second = [overlay for overlay in resolved if overlay["label"] == "S2"]
    return resolved_first, resolved_second


def _is_valid_cycle_order(
    s1_start: int | float | None,
    s1_end: int | float | None,
    s2_start: int | float | None,
    s2_end: int | float | None,
    next_s1_start: int | float | None,
) -> bool:
    try:
        values = [s1_start, s1_end, s2_start, s2_end, next_s1_start]
        if any(value is None or not np.isfinite(float(value)) for value in values):
            return False
        safe_s1_start = int(float(s1_start))
        safe_s1_end = int(float(s1_end))
        safe_s2_start = int(float(s2_start))
        safe_s2_end = int(float(s2_end))
        safe_next_s1_start = int(float(next_s1_start))
    except Exception:
        return False

    return safe_s1_start < safe_s1_end < safe_s2_start < safe_s2_end < safe_next_s1_start


def _validate_cycle_boundaries(cycle_row: dict[str, Any], signal_length: int) -> tuple[bool, str]:
    required_columns = ["s1_start", "s1_end", "s2_start", "s2_end", "next_s1_start"]
    if any(pd.isna(cycle_row[column]) for column in required_columns):
        return False, "missing_boundary"

    if not _is_valid_cycle_order(
        cycle_row["s1_start"],
        cycle_row["s1_end"],
        cycle_row["s2_start"],
        cycle_row["s2_end"],
        cycle_row["next_s1_start"],
    ):
        return False, "invalid_cycle_order"

    s1_start = int(cycle_row["s1_start"])
    s1_end = int(cycle_row["s1_end"])
    s2_start = int(cycle_row["s2_start"])
    s2_end = int(cycle_row["s2_end"])
    next_s1_start = int(cycle_row["next_s1_start"])
    if not (0 <= s1_start < s1_end < s2_start < s2_end < next_s1_start <= signal_length):
        return False, "out_of_bounds"

    return True, ""


def _build_cycles_from_recording(recording_table: pd.DataFrame) -> list[dict[str, Any]]:
    s1_start_values = recording_table[EVENT_COLUMN_MAP["S1_start"]].to_numpy(dtype=np.float32)
    s1_end_values = recording_table[EVENT_COLUMN_MAP["S1_end"]].to_numpy(dtype=np.float32)
    s2_start_values = recording_table[EVENT_COLUMN_MAP["S2_start"]].to_numpy(dtype=np.float32)
    s2_end_values = recording_table[EVENT_COLUMN_MAP["S2_end"]].to_numpy(dtype=np.float32)

    s1_overlays = _build_region_overlays("S1", s1_start_values, s1_end_values)
    s2_overlays = _build_region_overlays("S2", s2_start_values, s2_end_values)
    resolved_s1, _ = _resolve_region_overlaps(s1_overlays, [])
    _, resolved_s2 = _resolve_region_overlaps([], s2_overlays)
    sorted_s1 = sorted(resolved_s1, key=lambda overlay: int(overlay["areaStart"]))
    sorted_s2 = sorted(resolved_s2, key=lambda overlay: int(overlay["areaStart"]))

    if not sorted_s1:
        raise ValueError("No valid S1 overlays were detected from RS score channels.")

    rows: list[dict[str, Any]] = []
    s2_index = 0
    for cycle_index, s1_overlay in enumerate(sorted_s1, start=1):
        s1_start = int(s1_overlay["areaStart"])
        s1_end = int(s1_overlay["areaEnd"])
        next_s1_overlay = sorted_s1[cycle_index] if cycle_index < len(sorted_s1) else None
        next_s1_start = int(next_s1_overlay["areaStart"]) if next_s1_overlay is not None else np.nan
        next_s1_end = int(next_s1_overlay["areaEnd"]) if next_s1_overlay is not None else np.nan

        while s2_index < len(sorted_s2) and int(sorted_s2[s2_index]["areaEnd"]) <= s1_start:
            s2_index += 1

        matched_s2: dict[str, Any] | None = None
        probe_index = s2_index
        while probe_index < len(sorted_s2):
            candidate = sorted_s2[probe_index]
            candidate_start = int(candidate["areaStart"])
            candidate_end = int(candidate["areaEnd"])
            if candidate_end <= s1_start:
                probe_index += 1
                continue
            if np.isfinite(next_s1_start) and candidate_start >= int(next_s1_start):
                break
            if s1_start < s1_end < candidate_start < candidate_end and (
                not np.isfinite(next_s1_start) or candidate_end < int(next_s1_start)
            ):
                matched_s2 = candidate
                s2_index = probe_index + 1
                break
            probe_index += 1

        s2_start = int(matched_s2["areaStart"]) if matched_s2 is not None else np.nan
        s2_end = int(matched_s2["areaEnd"]) if matched_s2 is not None else np.nan

        rows.append(
            {
                "cycle_index": cycle_index,
                "beat_index": cycle_index - 1,
                "s1_start": s1_start,
                "s1_end": s1_end,
                "s2_start": s2_start,
                "s2_end": s2_end,
                "next_s1_start": next_s1_start,
                "next_s1_end": next_s1_end,
                "cycle_start": s1_start,
                "cycle_end": next_s1_start,
                "s1_on": s1_start,
                "s1_off": s1_end,
                "s2_on": s2_start,
                "s2_off": s2_end,
                "s1_on_next": next_s1_start,
            }
        )

    return rows


# =================================================
# 4. Numeric Helpers
# =================================================


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= PreprocessConfig.EPS:
        return float(default)
    return float(numerator / denominator)


def _nan_feature_map() -> dict[str, float]:
    return {column: float(np.nan) for column in ALL_FEATURE_COLUMNS}


def _nan_row(columns: list[str]) -> dict[str, float]:
    return {column: float(np.nan) for column in columns}


def _ms_to_sample_count(duration_ms: float) -> int:
    return max(1, int(round(duration_ms / PreprocessConfig.SAMPLE_MS)))


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float32)
    safe_radius = max(1, int(radius))
    prefix = np.zeros(values.size + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(values, dtype=np.float64)
    smoothed = np.zeros(values.size, dtype=np.float32)
    for index in range(values.size):
        start = max(0, index - safe_radius)
        end = min(values.size - 1, index + safe_radius)
        total = prefix[end + 1] - prefix[start]
        smoothed[index] = float(total / max(1, end - start + 1))
    return smoothed


def _build_smoothed_envelope(amplitude: np.ndarray) -> np.ndarray:
    abs_amplitude = np.abs(np.nan_to_num(amplitude.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    radius = _ms_to_sample_count(PreprocessConfig.ENVELOPE_SMOOTH_MS) // 2
    return _moving_average(abs_amplitude, radius=max(1, radius))


def _slice_signal(signal: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    safe_start = max(0, int(start_index))
    safe_end = min(len(signal), int(end_index))
    if safe_start >= safe_end:
        return np.array([], dtype=np.float32)
    return signal[safe_start:safe_end].astype(np.float32, copy=False)


def _resample_vector(values: np.ndarray, target_points: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(target_points, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values.astype(np.float32), target_points)
    source_positions = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_points, dtype=np.float64)
    return np.interp(target_positions, source_positions, values.astype(np.float64)).astype(np.float32)


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean_value = float(np.mean(values)) if values.size > 0 else 0.0
    std_value = float(np.std(values, ddof=0)) if values.size > 0 else 0.0
    if std_value <= PreprocessConfig.EPS:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean_value) / std_value).astype(np.float32)


def _safe_weighted_centroid(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.5
    total_weight = float(np.sum(weights))
    if total_weight <= PreprocessConfig.EPS:
        return 0.5
    positions = np.linspace(0.0, 1.0, weights.size, dtype=np.float64)
    return float(np.sum(positions * weights) / total_weight)


def _safe_weighted_spread(weights: np.ndarray, centroid: float) -> float:
    if weights.size == 0:
        return 0.0
    total_weight = float(np.sum(weights))
    if total_weight <= PreprocessConfig.EPS:
        return 0.0
    positions = np.linspace(0.0, 1.0, weights.size, dtype=np.float64)
    variance = float(np.sum(((positions - centroid) ** 2) * weights) / total_weight)
    return float(np.sqrt(max(variance, 0.0)))


def _safe_peak_timing_relative(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.5
    peak_index = int(np.argmax(values))
    return float(peak_index / max(1, values.size - 1))


def _safe_skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    if std_value <= PreprocessConfig.EPS:
        return 0.0
    z_values = (values - mean_value) / std_value
    return float(np.mean(z_values ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    if std_value <= PreprocessConfig.EPS:
        return 0.0
    z_values = (values - mean_value) / std_value
    return float(np.mean(z_values ** 4) - 3.0)


def _safe_zero_crossing_rate(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    signs = np.sign(values)
    transitions = np.sum(signs[1:] * signs[:-1] < 0)
    return float(transitions / max(1, values.size - 1))


def _safe_diff_mean_abs(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


# =================================================
# 5. Template Helpers
# =================================================


def _build_segment_template(
    envelope: np.ndarray,
    cycle_rows: list[dict[str, Any]],
    start_key: str,
    end_key: str,
) -> np.ndarray | None:
    template_vectors: list[np.ndarray] = []
    for cycle_row in cycle_rows:
        start_index = int(cycle_row[start_key])
        end_index = int(cycle_row[end_key])
        segment = _slice_signal(envelope, start_index, end_index)
        if segment.size < 2:
            continue
        resampled = _resample_vector(segment, PreprocessConfig.TEMPLATE_RESAMPLE_POINTS)
        template_vectors.append(_zscore_vector(resampled))

    if len(template_vectors) < PreprocessConfig.TEMPLATE_MIN_VALID_SEGMENTS:
        return None

    stacked = np.stack(template_vectors, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _compute_template_correlation(
    envelope: np.ndarray,
    start_index: int,
    end_index: int,
    template: np.ndarray | None,
) -> float:
    if template is None:
        return 0.0
    segment = _slice_signal(envelope, start_index, end_index)
    if segment.size < 2:
        return 0.0
    resampled = _zscore_vector(_resample_vector(segment, len(template)))
    template_z = _zscore_vector(template)
    correlation = float(np.mean(resampled * template_z))
    if not np.isfinite(correlation):
        return 0.0
    return correlation


# =================================================
# 6. Feature Computation Helpers
# =================================================


def _compute_global_features(
    cycle_raw: np.ndarray,
    cycle_env: np.ndarray,
    s1_duration_samples: int,
    s2_duration_samples: int,
    systole_duration_samples: int,
    diastole_duration_samples: int,
) -> dict[str, float]:
    cycle_length_samples = int(cycle_raw.size)
    cycle_length_ms = float(cycle_length_samples * PreprocessConfig.SAMPLE_MS)
    cycle_energy = float(np.sum(cycle_env.astype(np.float64) ** 2))
    cycle_hr = _safe_divide(60000.0, cycle_length_ms, default=0.0)

    return {
        "global_cycle_length_ms": cycle_length_ms,
        "global_hr_bpm": cycle_hr,
        "global_s1_width_ratio": _safe_divide(float(s1_duration_samples), float(cycle_length_samples)),
        "global_s2_width_ratio": _safe_divide(float(s2_duration_samples), float(cycle_length_samples)),
        "global_systole_ratio": _safe_divide(float(systole_duration_samples), float(cycle_length_samples)),
        "global_diastole_ratio": _safe_divide(float(diastole_duration_samples), float(cycle_length_samples)),
        "global_total_energy": cycle_energy,
        "global_env_mean": float(np.mean(cycle_env)) if cycle_env.size > 0 else 0.0,
        "global_env_peak": float(np.max(cycle_env)) if cycle_env.size > 0 else 0.0,
        "global_env_rms": float(np.sqrt(np.mean(cycle_env.astype(np.float64) ** 2))) if cycle_env.size > 0 else 0.0,
    }


def _compute_segment_block(
    prefix: str,
    raw_signal: np.ndarray,
    env_signal: np.ndarray,
    cycle_energy: float,
    *,
    ratio_key: str,
) -> dict[str, float]:
    if raw_signal.size == 0 or env_signal.size == 0:
        return {
            f"{prefix}_duration_ms": 0.0,
            f"{prefix}_mean_env": 0.0,
            f"{prefix}_peak_env": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_energy": 0.0,
            f"{prefix}_{ratio_key}": 0.0,
            f"{prefix}_energy_centroid": 0.5,
            f"{prefix}_energy_spread": 0.0,
            f"{prefix}_env_occupancy": 0.0,
        }

    env_energy_weights = env_signal.astype(np.float64) ** 2
    peak_env = float(np.max(env_signal))
    occupancy_threshold = peak_env * PreprocessConfig.ENVELOPE_OCCUPANCY_THRESHOLD_RATIO
    occupancy = float(np.mean(env_signal >= occupancy_threshold)) if peak_env > PreprocessConfig.EPS else 0.0
    centroid = _safe_weighted_centroid(env_energy_weights)
    spread = _safe_weighted_spread(env_energy_weights, centroid)
    energy_value = float(np.sum(env_energy_weights))

    return {
        f"{prefix}_duration_ms": float(raw_signal.size * PreprocessConfig.SAMPLE_MS),
        f"{prefix}_mean_env": float(np.mean(env_signal)),
        f"{prefix}_peak_env": peak_env,
        f"{prefix}_rms": float(np.sqrt(np.mean(raw_signal.astype(np.float64) ** 2))),
        f"{prefix}_energy": energy_value,
        f"{prefix}_{ratio_key}": _safe_divide(energy_value, cycle_energy if ratio_key == "energy_ratio_to_cycle" else cycle_energy),
        f"{prefix}_energy_centroid": centroid,
        f"{prefix}_energy_spread": spread,
        f"{prefix}_env_occupancy": occupancy,
    }


def _compute_zone_block(
    prefix: str,
    raw_signal: np.ndarray,
    env_signal: np.ndarray,
    diastole_energy: float,
) -> dict[str, float]:
    if raw_signal.size == 0 or env_signal.size == 0:
        return {
            f"{prefix}_duration_ms": 0.0,
            f"{prefix}_mean_env": 0.0,
            f"{prefix}_peak_env": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_energy": 0.0,
            f"{prefix}_energy_ratio_to_diastole": 0.0,
            f"{prefix}_peak_timing_relative": 0.5,
            f"{prefix}_energy_centroid": 0.5,
            f"{prefix}_energy_spread": 0.0,
            f"{prefix}_env_occupancy": 0.0,
        }

    env_energy_weights = env_signal.astype(np.float64) ** 2
    peak_env = float(np.max(env_signal))
    occupancy_threshold = peak_env * PreprocessConfig.ENVELOPE_OCCUPANCY_THRESHOLD_RATIO
    occupancy = float(np.mean(env_signal >= occupancy_threshold)) if peak_env > PreprocessConfig.EPS else 0.0
    centroid = _safe_weighted_centroid(env_energy_weights)
    spread = _safe_weighted_spread(env_energy_weights, centroid)
    energy_value = float(np.sum(env_energy_weights))

    return {
        f"{prefix}_duration_ms": float(raw_signal.size * PreprocessConfig.SAMPLE_MS),
        f"{prefix}_mean_env": float(np.mean(env_signal)),
        f"{prefix}_peak_env": peak_env,
        f"{prefix}_rms": float(np.sqrt(np.mean(raw_signal.astype(np.float64) ** 2))),
        f"{prefix}_energy": energy_value,
        f"{prefix}_energy_ratio_to_diastole": _safe_divide(energy_value, diastole_energy),
        f"{prefix}_peak_timing_relative": _safe_peak_timing_relative(env_signal),
        f"{prefix}_energy_centroid": centroid,
        f"{prefix}_energy_spread": spread,
        f"{prefix}_env_occupancy": occupancy,
    }


def _compute_shape_features(s1_env: np.ndarray, s2_env: np.ndarray) -> dict[str, float]:
    s1_attack = _safe_peak_timing_relative(s1_env)
    s2_attack = _safe_peak_timing_relative(s2_env)

    return {
        "shape_s1_attack_ratio": s1_attack,
        "shape_s1_decay_ratio": float(max(0.0, 1.0 - s1_attack)),
        "shape_s1_temporal_centroid_rel": _safe_weighted_centroid(s1_env.astype(np.float64) ** 2),
        "shape_s2_attack_ratio": s2_attack,
        "shape_s2_decay_ratio": float(max(0.0, 1.0 - s2_attack)),
        "shape_s2_temporal_centroid_rel": _safe_weighted_centroid(s2_env.astype(np.float64) ** 2),
    }


def _compute_stat_features(cycle_raw: np.ndarray, s1_raw: np.ndarray, s2_raw: np.ndarray) -> dict[str, float]:
    return {
        "stat_cycle_zero_crossing_rate": _safe_zero_crossing_rate(cycle_raw),
        "stat_cycle_diff_mean_abs": _safe_diff_mean_abs(cycle_raw),
        "stat_s1_skewness": _safe_skewness(s1_raw.astype(np.float64)),
        "stat_s1_kurtosis": _safe_kurtosis(s1_raw.astype(np.float64)),
        "stat_s2_skewness": _safe_skewness(s2_raw.astype(np.float64)),
        "stat_s2_kurtosis": _safe_kurtosis(s2_raw.astype(np.float64)),
    }


def _resolve_relative_zone_bounds(start_index: int, end_index: int, ratio_range: tuple[float, float]) -> tuple[int, int]:
    safe_start = int(start_index)
    safe_end = int(end_index)
    if safe_end <= safe_start:
        return safe_start, safe_start

    length = safe_end - safe_start
    left_ratio, right_ratio = ratio_range
    zone_start = safe_start + int(np.floor(length * left_ratio))
    zone_end = safe_start + int(np.floor(length * right_ratio))
    zone_start = max(safe_start, min(zone_start, safe_end))
    zone_end = max(zone_start, min(zone_end, safe_end))
    return zone_start, zone_end


def _compute_relative_features(
    zone_ed_block: dict[str, float],
    zone_md_block: dict[str, float],
    zone_ld_block: dict[str, float],
    seg_s1_block: dict[str, float],
    seg_s2_block: dict[str, float],
) -> dict[str, float]:
    s1_peak_env = float(seg_s1_block["seg_s1_peak_env"])
    s1_mean_env = float(seg_s1_block["seg_s1_mean_env"])
    s2_peak_env = float(seg_s2_block["seg_s2_peak_env"])
    s2_mean_env = float(seg_s2_block["seg_s2_mean_env"])
    s1s2_mean_peak = float(np.mean([s1_peak_env, s2_peak_env]))

    return {
        "zone_ed_peak_rel_to_s2": _safe_divide(float(zone_ed_block["zone_ed_peak_env"]), s2_peak_env),
        "zone_ed_mean_rel_to_s2": _safe_divide(float(zone_ed_block["zone_ed_mean_env"]), s2_mean_env),
        "zone_ld_peak_rel_to_s1": _safe_divide(float(zone_ld_block["zone_ld_peak_env"]), s1_peak_env),
        "zone_ld_mean_rel_to_s1": _safe_divide(float(zone_ld_block["zone_ld_mean_env"]), s1_mean_env),
        "zone_md_peak_rel_to_s1s2_mean": _safe_divide(float(zone_md_block["zone_md_peak_env"]), s1s2_mean_peak),
    }


def _compute_stability_features(
    envelope: np.ndarray,
    cycle_row: dict[str, Any],
    s1_template: np.ndarray | None,
    s2_template: np.ndarray | None,
) -> dict[str, float]:
    return {
        "stab_s1_template_corr": _compute_template_correlation(
            envelope,
            int(cycle_row["s1_start"]),
            int(cycle_row["s1_end"]),
            s1_template,
        ),
        "stab_s2_template_corr": _compute_template_correlation(
            envelope,
            int(cycle_row["s2_start"]),
            int(cycle_row["s2_end"]),
            s2_template,
        ),
    }


def _build_feature_row(
    record_id: str,
    source_file: str,
    cycle_row: dict[str, Any],
    amplitude: np.ndarray,
    envelope: np.ndarray,
    valid_flag: int,
    invalid_reason: str,
    s1_template: np.ndarray | None,
    s2_template: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id,
        "source_file": source_file,
        "beat_index": int(cycle_row["beat_index"]),
        "cycle_index": int(cycle_row["cycle_index"]),
        "valid_flag": int(valid_flag),
        "invalid_reason": invalid_reason,
        "s1_start": cycle_row["s1_start"],
        "s1_end": cycle_row["s1_end"],
        "s2_start": cycle_row["s2_start"],
        "s2_end": cycle_row["s2_end"],
        "next_s1_start": cycle_row["next_s1_start"],
        "next_s1_end": cycle_row["next_s1_end"],
        "cycle_start": cycle_row["cycle_start"],
        "cycle_end": cycle_row["cycle_end"],
        "s1_on": cycle_row["s1_on"],
        "s1_off": cycle_row["s1_off"],
        "s2_on": cycle_row["s2_on"],
        "s2_off": cycle_row["s2_off"],
        "s1_on_next": cycle_row["s1_on_next"],
    }

    if valid_flag != 1:
        row.update(_nan_feature_map())
        return row

    s1_start = int(cycle_row["s1_start"])
    s1_end = int(cycle_row["s1_end"])
    s2_start = int(cycle_row["s2_start"])
    s2_end = int(cycle_row["s2_end"])
    next_s1_start = int(cycle_row["next_s1_start"])

    cycle_raw = _slice_signal(amplitude, s1_start, next_s1_start)
    cycle_env = _slice_signal(envelope, s1_start, next_s1_start)
    s1_raw = _slice_signal(amplitude, s1_start, s1_end)
    s1_env = _slice_signal(envelope, s1_start, s1_end)
    systole_raw = _slice_signal(amplitude, s1_end, s2_start)
    systole_env = _slice_signal(envelope, s1_end, s2_start)
    s2_raw = _slice_signal(amplitude, s2_start, s2_end)
    s2_env = _slice_signal(envelope, s2_start, s2_end)
    diastole_raw = _slice_signal(amplitude, s2_end, next_s1_start)
    diastole_env = _slice_signal(envelope, s2_end, next_s1_start)

    s1_duration_samples = max(0, s1_end - s1_start)
    s2_duration_samples = max(0, s2_end - s2_start)
    systole_duration_samples = max(0, s2_start - s1_end)
    diastole_duration_samples = max(0, next_s1_start - s2_end)

    global_block = _compute_global_features(
        cycle_raw=cycle_raw,
        cycle_env=cycle_env,
        s1_duration_samples=s1_duration_samples,
        s2_duration_samples=s2_duration_samples,
        systole_duration_samples=systole_duration_samples,
        diastole_duration_samples=diastole_duration_samples,
    )

    cycle_energy = float(global_block["global_total_energy"])
    seg_s1_block = _compute_segment_block(
        "seg_s1",
        s1_raw,
        s1_env,
        cycle_energy,
        ratio_key="energy_ratio_to_cycle",
    )
    seg_sys_block = _compute_segment_block(
        "seg_sys",
        systole_raw,
        systole_env,
        cycle_energy,
        ratio_key="energy_ratio_to_cycle",
    )
    seg_s2_block = _compute_segment_block(
        "seg_s2",
        s2_raw,
        s2_env,
        cycle_energy,
        ratio_key="energy_ratio_to_cycle",
    )
    seg_dia_block = _compute_segment_block(
        "seg_dia",
        diastole_raw,
        diastole_env,
        cycle_energy,
        ratio_key="energy_ratio_to_cycle",
    )

    diastole_energy = float(seg_dia_block["seg_dia_energy"])
    ed_start, ed_end = _resolve_relative_zone_bounds(s2_end, next_s1_start, PreprocessConfig.EARLY_DIASTOLE_RATIO)
    md_start, md_end = _resolve_relative_zone_bounds(s2_end, next_s1_start, PreprocessConfig.MID_DIASTOLE_RATIO)
    ld_start, ld_end = _resolve_relative_zone_bounds(s2_end, next_s1_start, PreprocessConfig.LATE_DIASTOLE_RATIO)

    zone_ed_block = _compute_zone_block(
        "zone_ed",
        _slice_signal(amplitude, ed_start, ed_end),
        _slice_signal(envelope, ed_start, ed_end),
        diastole_energy,
    )
    zone_md_block = _compute_zone_block(
        "zone_md",
        _slice_signal(amplitude, md_start, md_end),
        _slice_signal(envelope, md_start, md_end),
        diastole_energy,
    )
    zone_ld_block = _compute_zone_block(
        "zone_ld",
        _slice_signal(amplitude, ld_start, ld_end),
        _slice_signal(envelope, ld_start, ld_end),
        diastole_energy,
    )

    relative_block = _compute_relative_features(
        zone_ed_block=zone_ed_block,
        zone_md_block=zone_md_block,
        zone_ld_block=zone_ld_block,
        seg_s1_block=seg_s1_block,
        seg_s2_block=seg_s2_block,
    )
    shape_block = _compute_shape_features(s1_env=s1_env, s2_env=s2_env)
    stat_block = _compute_stat_features(cycle_raw=cycle_raw, s1_raw=s1_raw, s2_raw=s2_raw)
    stability_block = _compute_stability_features(
        envelope=envelope,
        cycle_row=cycle_row,
        s1_template=s1_template,
        s2_template=s2_template,
    )

    feature_row: dict[str, float] = {}
    feature_row.update(global_block)
    feature_row.update(seg_s1_block)
    feature_row.update(seg_sys_block)
    feature_row.update(seg_s2_block)
    feature_row.update(seg_dia_block)
    feature_row.update(zone_ed_block)
    feature_row.update(zone_md_block)
    feature_row.update(zone_ld_block)
    feature_row.update(relative_block)
    feature_row.update(shape_block)
    feature_row.update(stat_block)
    feature_row.update(stability_block)

    row.update(feature_row)
    return row


# =================================================
# 7. Export Helpers
# =================================================


def build_feature_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    ordered_columns = METADATA_COLUMNS + list(ALL_FEATURE_COLUMNS)
    missing_columns = [column for column in ordered_columns if column not in dataframe.columns]
    for column in missing_columns:
        dataframe[column] = np.nan
    return dataframe.loc[:, ordered_columns].copy()


def summary_stat(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float(np.nan),
            "std": float(np.nan),
            "cv": float(np.nan),
            "rmssd": float(np.nan),
        }

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(values))))) if values.size >= 2 else float(np.nan)
    return {
        "mean": mean_value,
        "std": std_value,
        "cv": _safe_divide(std_value, mean_value, default=float(np.nan)),
        "rmssd": rmssd,
    }


def build_record_summary(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_id, record_frame in feature_frame.groupby("record_id", sort=True):
        valid_frame = record_frame.loc[record_frame["valid_flag"] == 1].copy()
        row: dict[str, Any] = {
            "record_id": record_id,
            "total_cycles": int(len(record_frame)),
            "valid_cycles": int(len(valid_frame)),
            "invalid_cycles": int(len(record_frame) - len(valid_frame)),
            "valid_ratio": _safe_divide(float(len(valid_frame)), float(max(1, len(record_frame)))),
            "mean_feature_missing_fraction": float(
                valid_frame.loc[:, ALL_FEATURE_COLUMNS].isnull().mean().mean()
            )
            if not valid_frame.empty
            else float(np.nan),
        }
        for column in PreprocessConfig.KEY_RECORD_SUMMARY_COLUMNS:
            stats = summary_stat(valid_frame[column].to_numpy(dtype=np.float64))
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_std"] = stats["std"]
            row[f"{column}_cv"] = stats["cv"]
            row[f"{column}_rmssd"] = stats["rmssd"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_learning_input_columns(feature_columns: list[str]) -> list[str]:
    return list(feature_columns)


def build_preprocess_summary(
    beat_features_all: pd.DataFrame,
    beat_features_valid: pd.DataFrame,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, Any]:
    invalid_reason_counts = (
        beat_features_all.loc[beat_features_all["valid_flag"] == 0, "invalid_reason"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )
    missingness_by_feature = beat_features_valid.loc[:, feature_names].isnull().mean().sort_values(ascending=False)
    return {
        "num_records": int(beat_features_all["record_id"].nunique()),
        "total_cycles": int(len(beat_features_all)),
        "valid_cycles": int(len(beat_features_valid)),
        "invalid_cycles": int(len(beat_features_all) - len(beat_features_valid)),
        "feature_dimension": int(len(feature_names)),
        "group_feature_counts": {group_name: len(group_columns) for group_name, group_columns in feature_groups.items()},
        "invalid_reason_counts": {str(key): int(value) for key, value in invalid_reason_counts.items()},
        "mean_missing_fraction_valid": float(missingness_by_feature.mean()) if not missingness_by_feature.empty else 0.0,
        "top_missing_features_valid": {
            str(key): float(value)
            for key, value in missingness_by_feature.head(10).to_dict().items()
        },
    }


def _feature_groups_to_frame(feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group_columns in feature_groups.items():
        for order_index, feature_name in enumerate(group_columns):
            rows.append(
                {
                    "feature_group": group_name,
                    "feature_order": order_index,
                    "feature_name": feature_name,
                }
            )
    return pd.DataFrame(rows)


def _summary_to_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for key, value in summary.items():
        rows.append({"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)})
    return pd.DataFrame(rows)


def export_feature_outputs(
    preprocess_root: Path,
    beat_features_all: pd.DataFrame,
    beat_features_valid: pd.DataFrame,
    record_summary: pd.DataFrame,
    feature_names: list[str],
    learning_input_columns: list[str],
    feature_groups: dict[str, list[str]],
    preprocess_summary: dict[str, Any],
) -> None:
    preprocess_root.mkdir(parents=True, exist_ok=True)

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
        (preprocess_root / "feature_groups.json").write_text(
            json.dumps(feature_groups, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (preprocess_root / "preprocess_summary.json").write_text(
            json.dumps(preprocess_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if PreprocessConfig.EXPORT_EXCEL:
        export_stage_workbook(
            workbook_path=preprocess_root / PreprocessConfig.EXCEL_FILENAME,
            sheets={
                "beat_features_all": beat_features_all,
                "beat_features_valid": beat_features_valid,
                "record_summary": record_summary,
                "feature_groups": _feature_groups_to_frame(feature_groups),
                "preprocess_summary": _summary_to_frame(preprocess_summary),
            },
            freeze_panes=PreprocessConfig.EXCEL_FREEZE_PANES,
            header_fill=PreprocessConfig.EXCEL_HEADER_FILL,
            header_font_color=PreprocessConfig.EXCEL_HEADER_FONT_COLOR,
            max_column_width=PreprocessConfig.EXCEL_MAX_COLUMN_WIDTH,
        )

    logger.info("export 경로: %s", preprocess_root)


# =================================================
# 8. Record Processing
# =================================================


def process_recording(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_id = parse_record_id(file_path)
    recording_table = load_recording_table(file_path, PreprocessConfig.EXPECTED_COLUMNS)
    amplitude = recording_table["Amplitude"].to_numpy(dtype=np.float32)
    envelope = _build_smoothed_envelope(amplitude)

    cycles = _build_cycles_from_recording(recording_table)
    logger.info("cycle 수: record_id=%s, total_cycles=%s", record_id, len(cycles))

    valid_cycles_for_templates: list[dict[str, Any]] = []
    for cycle_row in cycles:
        is_valid, _ = _validate_cycle_boundaries(cycle_row, signal_length=len(amplitude))
        if is_valid:
            valid_cycles_for_templates.append(cycle_row)

    s1_template = _build_segment_template(envelope, valid_cycles_for_templates, "s1_start", "s1_end")
    s2_template = _build_segment_template(envelope, valid_cycles_for_templates, "s2_start", "s2_end")

    rows: list[dict[str, Any]] = []
    valid_count = 0
    for cycle_row in cycles:
        is_valid, invalid_reason = _validate_cycle_boundaries(cycle_row, signal_length=len(amplitude))
        valid_flag = int(is_valid)
        if valid_flag == 1:
            valid_count += 1
        rows.append(
            _build_feature_row(
                record_id=record_id,
                source_file=file_path.name,
                cycle_row=cycle_row,
                amplitude=amplitude,
                envelope=envelope,
                valid_flag=valid_flag,
                invalid_reason=invalid_reason,
                s1_template=s1_template,
                s2_template=s2_template,
            )
        )

    invalid_count = len(cycles) - valid_count
    logger.info(
        "valid / invalid cycle 수: record_id=%s, valid=%s, invalid=%s",
        record_id,
        valid_count,
        invalid_count,
    )

    feature_frame = build_feature_dataframe(rows)
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


# =================================================
# 9. Main
# =================================================


def main() -> None:
    output_paths = ensure_output_directories(stage_output_folder=configured_path(PreprocessConfig.OUTPUT_FOLDER))

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
    if not PreprocessConfig.SAVE_INVALID_ROWS:
        beat_features_all = beat_features_all.loc[beat_features_all["valid_flag"] == 1].copy()

    beat_features_valid = pd.concat(all_valid_frames, axis=0, ignore_index=True)
    if beat_features_valid.empty:
        raise ValueError("No valid cycles were extracted from the provided recordings.")

    feature_names = list(ALL_FEATURE_COLUMNS)
    learning_input_columns = build_learning_input_columns(feature_names)
    record_summary = build_record_summary(beat_features_all)
    preprocess_summary = build_preprocess_summary(
        beat_features_all=beat_features_all,
        beat_features_valid=beat_features_valid,
        feature_names=feature_names,
        feature_groups=FEATURE_GROUPS,
    )

    export_feature_outputs(
        preprocess_root=output_paths["preprocess_root"],
        beat_features_all=beat_features_all,
        beat_features_valid=beat_features_valid,
        record_summary=record_summary,
        feature_names=feature_names,
        learning_input_columns=learning_input_columns,
        feature_groups=FEATURE_GROUPS,
        preprocess_summary=preprocess_summary,
    )

    logger.info(
        "전처리 완료: total_cycles=%s, valid_cycles=%s, feature_dim=%s",
        len(beat_features_all),
        len(beat_features_valid),
        len(feature_names),
    )


if __name__ == "__main__":
    main()
