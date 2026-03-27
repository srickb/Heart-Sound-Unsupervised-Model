"""
Cycle-level preprocessing script aligned to the current HeartSound Tool parameter design.

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
from typing import Any, Callable

import numpy as np
import pandas as pd

from excel_export_utils import export_stage_workbook


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
    EPS = 1e-12

    DIASTOLIC_WINDOW_MIN_DURATION_MS = 18.0
    CANDIDATE_MIN_WINDOW_MULTIPLIER = 2
    S3_WINDOW_CONFIG = {
        "offset_start_ms": 120.0,
        "offset_end_ms": 200.0,
        "fallback_start_ratio": 0.18,
        "fallback_end_ratio": 0.34,
    }
    S4_WINDOW_CONFIG = {
        "offset_start_ms": 200.0,
        "offset_end_ms": 80.0,
        "fallback_start_ratio": 0.72,
        "fallback_end_ratio": 0.88,
    }

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
    EXCEL_FILENAME = "preprocess_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


S1_PARAMETER_COLUMNS = [
    "S1_Width_ms",
    "S1_Peak_mV",
    "S1_mean_mV",
    "S1_RMS_mV",
    "S1_Area_mVms",
    "S1_Sumation",
]

S2_PARAMETER_COLUMNS = [
    "S2_Width_ms",
    "S2_Peak_mV",
    "S2_mean_mV",
    "S2_RMS_mV",
    "S2_Area_mVms",
    "S2_Sumation",
]

SYSTOLE_PARAMETER_COLUMNS = [
    "Systole_Duration_ms",
    "Systole_Peak_mV",
    "Systole_mean_mV",
]

DIASTOLE_PARAMETER_COLUMNS = [
    "Diastole_Duration_ms",
    "Diastole_Peak_mV",
    "Diastole_mean_mV",
    "Diastole_S3_Expected_Delta_mV",
    "Diastole_NaN_Gap_Delta_mV",
    "Diastole_S4_Expected_Delta_mV",
]

RS_PEAK_PARAMETER_COLUMNS = [
    "S1S_RS_Peak",
    "S1E_RS_Peak",
    "S2S_RS_Peak",
    "S2E_RS_Peak",
]

RS_WIDTH_PARAMETER_COLUMNS = [
    "S1S_RS_Width_ms",
    "S1E_RS_Width_ms",
    "S2S_RS_Width_ms",
    "S2E_RS_Width_ms",
]

RS_SUMATION_PARAMETER_COLUMNS = [
    "S1S_RS_Sumation",
    "S1E_RS_Sumation",
    "S2S_RS_Sumation",
    "S2E_RS_Sumation",
]

HEART_RATE_COLUMNS = [
    "HeartRate_bpm",
]

ALL_FEATURE_COLUMNS = (
    S1_PARAMETER_COLUMNS
    + S2_PARAMETER_COLUMNS
    + SYSTOLE_PARAMETER_COLUMNS
    + DIASTOLE_PARAMETER_COLUMNS
    + RS_PEAK_PARAMETER_COLUMNS
    + RS_WIDTH_PARAMETER_COLUMNS
    + RS_SUMATION_PARAMETER_COLUMNS
    + HEART_RATE_COLUMNS
)

SUMMARY_SOURCE_COLUMNS = [
    "S1_Width_ms",
    "S2_Width_ms",
    "Systole_Duration_ms",
    "Diastole_Duration_ms",
    "HeartRate_bpm",
    "S1_Peak_mV",
    "S2_Peak_mV",
]

METADATA_COLUMNS = [
    "record_id",
    "source_file",
    "beat_index",
    "cycle_index",
    "valid_flag",
    "invalid_reason",
    "S1_start",
    "S1_end",
    "S2_start",
    "S2_end",
    "next_S1_start",
    "next_S1_end",
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


def _nan_feature_map() -> dict[str, float]:
    return {column: float(np.nan) for column in ALL_FEATURE_COLUMNS}


def _nan_row(columns: list[str]) -> dict[str, float]:
    return {column: float(np.nan) for column in columns}


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
    required_columns = ["S1_start", "S1_end", "S2_start", "S2_end", "next_S1_start"]
    if any(pd.isna(cycle_row[column]) for column in required_columns):
        return False, "missing_boundary"

    if not _is_valid_cycle_order(
        cycle_row["S1_start"],
        cycle_row["S1_end"],
        cycle_row["S2_start"],
        cycle_row["S2_end"],
        cycle_row["next_S1_start"],
    ):
        return False, "invalid_cycle_order"

    s1_start = int(cycle_row["S1_start"])
    s1_end = int(cycle_row["S1_end"])
    s2_start = int(cycle_row["S2_start"])
    s2_end = int(cycle_row["S2_end"])
    next_s1_start = int(cycle_row["next_S1_start"])
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
                "S1_start": s1_start,
                "S1_end": s1_end,
                "S2_start": s2_start,
                "S2_end": s2_end,
                "next_S1_start": next_s1_start,
                "next_S1_end": next_s1_end,
                "s1_on": s1_start,
                "s1_off": s1_end,
                "s2_on": s2_start,
                "s2_off": s2_end,
                "s1_on_next": next_s1_start,
            }
        )

    return rows


def _compute_sound_parameter_row(
    amplitude: np.ndarray,
    start_index: int,
    end_index: int,
    *,
    width_key: str,
    peak_key: str,
    mean_key: str,
    rms_key: str,
    area_key: str,
    sumation_key: str,
    nan_factory: Callable[[], dict[str, float]],
) -> dict[str, float]:
    safe_start = max(0, int(start_index))
    safe_end = min(len(amplitude), int(end_index))
    if safe_start >= safe_end or amplitude.size == 0:
        return nan_factory()

    segment = amplitude[safe_start:safe_end].astype(float, copy=False)
    if segment.size == 0:
        return nan_factory()

    absolute_segment = np.abs(segment)
    total_absolute = float(np.sum(absolute_segment))
    peak_value = float(np.max(absolute_segment))
    sumation_value = float(peak_value / total_absolute) if total_absolute > 0.0 else float(np.nan)

    return {
        width_key: float((safe_end - safe_start) * PreprocessConfig.SAMPLE_MS),
        peak_key: peak_value,
        mean_key: float(np.mean(absolute_segment)),
        rms_key: float(np.sqrt(np.mean(segment ** 2))),
        area_key: float(np.sum(absolute_segment) * PreprocessConfig.SAMPLE_MS),
        sumation_key: sumation_value,
    }


def _compute_s1_parameter_row(amplitude: np.ndarray, start_index: int, end_index: int) -> dict[str, float]:
    return _compute_sound_parameter_row(
        amplitude,
        start_index,
        end_index,
        width_key="S1_Width_ms",
        peak_key="S1_Peak_mV",
        mean_key="S1_mean_mV",
        rms_key="S1_RMS_mV",
        area_key="S1_Area_mVms",
        sumation_key="S1_Sumation",
        nan_factory=lambda: _nan_row(S1_PARAMETER_COLUMNS),
    )


def _compute_s2_parameter_row(amplitude: np.ndarray, start_index: int, end_index: int) -> dict[str, float]:
    return _compute_sound_parameter_row(
        amplitude,
        start_index,
        end_index,
        width_key="S2_Width_ms",
        peak_key="S2_Peak_mV",
        mean_key="S2_mean_mV",
        rms_key="S2_RMS_mV",
        area_key="S2_Area_mVms",
        sumation_key="S2_Sumation",
        nan_factory=lambda: _nan_row(S2_PARAMETER_COLUMNS),
    )


def _compute_gap_parameter_row(
    amplitude: np.ndarray,
    *,
    start_index: int,
    end_index: int,
    duration_key: str,
    peak_key: str,
    mean_key: str,
    nan_factory: Callable[[], dict[str, float]],
) -> dict[str, float]:
    safe_start = max(0, int(start_index))
    safe_end = min(len(amplitude), int(end_index))
    if amplitude.size == 0 or safe_start >= safe_end:
        return nan_factory()

    segment = amplitude[safe_start:safe_end].astype(float, copy=False)
    if segment.size == 0:
        return nan_factory()

    absolute_segment = np.abs(segment)
    return {
        duration_key: float((safe_end - safe_start) * PreprocessConfig.SAMPLE_MS),
        peak_key: float(np.max(absolute_segment)),
        mean_key: float(np.mean(absolute_segment)),
    }


def _compute_systole_parameter_row(
    amplitude: np.ndarray,
    s1_end_index: int,
    s2_start_index: int,
) -> dict[str, float]:
    return _compute_gap_parameter_row(
        amplitude,
        start_index=s1_end_index,
        end_index=s2_start_index,
        duration_key="Systole_Duration_ms",
        peak_key="Systole_Peak_mV",
        mean_key="Systole_mean_mV",
        nan_factory=lambda: _nan_row(SYSTOLE_PARAMETER_COLUMNS),
    )


def _ms_to_sample_count(duration_ms: float) -> int:
    return max(1, int(round(duration_ms / PreprocessConfig.SAMPLE_MS)))


def _resolve_diastolic_expected_window(
    window_kind: str,
    diastole_start: int,
    diastole_end: int,
    current_s2_end: int,
    next_s1_start: int,
) -> tuple[int, int] | None:
    if diastole_end <= diastole_start:
        return None

    diastolic_length = diastole_end - diastole_start
    minimum_window_length = _ms_to_sample_count(
        PreprocessConfig.DIASTOLIC_WINDOW_MIN_DURATION_MS * PreprocessConfig.CANDIDATE_MIN_WINDOW_MULTIPLIER
    )
    config = PreprocessConfig.S3_WINDOW_CONFIG if window_kind == "S3" else PreprocessConfig.S4_WINDOW_CONFIG

    if window_kind == "S3":
        default_start = current_s2_end + _ms_to_sample_count(float(config["offset_start_ms"]))
        default_end = current_s2_end + _ms_to_sample_count(float(config["offset_end_ms"]))
    else:
        default_start = next_s1_start - _ms_to_sample_count(float(config["offset_start_ms"]))
        default_end = next_s1_start - _ms_to_sample_count(float(config["offset_end_ms"]))

    clipped_default_start = max(diastole_start, min(default_start, diastole_end))
    clipped_default_end = max(clipped_default_start, min(default_end, diastole_end))
    if clipped_default_end - clipped_default_start >= minimum_window_length:
        return clipped_default_start, clipped_default_end

    fallback_start = diastole_start + int(diastolic_length * float(config["fallback_start_ratio"]))
    fallback_end = diastole_start + int(diastolic_length * float(config["fallback_end_ratio"]))
    clipped_fallback_start = max(diastole_start, min(fallback_start, diastole_end))
    clipped_fallback_end = max(clipped_fallback_start, min(fallback_end, diastole_end))
    if clipped_fallback_end - clipped_fallback_start < minimum_window_length:
        return None

    return clipped_fallback_start, clipped_fallback_end


def _compute_segment_delta_value(amplitude: np.ndarray, start_index: int, end_index: int) -> float:
    safe_start = max(0, int(start_index))
    safe_end = min(len(amplitude), int(end_index))
    if amplitude.size == 0 or safe_end - safe_start < 2:
        return float(np.nan)

    segment = np.nan_to_num(amplitude[safe_start:safe_end].astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if segment.size < 2:
        return float(np.nan)

    return float(np.mean(np.abs(np.diff(segment))))


def _compute_diastolic_expected_delta_row(
    amplitude: np.ndarray,
    s2_end_index: int,
    next_s1_start_index: int,
) -> dict[str, float]:
    diastole_start = int(s2_end_index) + 1
    diastole_end = int(next_s1_start_index) - 1
    if diastole_end <= diastole_start:
        return {
            "Diastole_S3_Expected_Delta_mV": float(np.nan),
            "Diastole_NaN_Gap_Delta_mV": float(np.nan),
            "Diastole_S4_Expected_Delta_mV": float(np.nan),
        }

    s3_window = _resolve_diastolic_expected_window(
        "S3",
        diastole_start,
        diastole_end,
        int(s2_end_index),
        int(next_s1_start_index),
    )
    s4_window = _resolve_diastolic_expected_window(
        "S4",
        diastole_start,
        diastole_end,
        int(s2_end_index),
        int(next_s1_start_index),
    )

    nan_gap_delta = float(np.nan)
    if s3_window is not None and s4_window is not None:
        gap_start = int(s3_window[1])
        gap_end = int(s4_window[0])
        if gap_end - gap_start >= 2:
            nan_gap_delta = _compute_segment_delta_value(amplitude, gap_start, gap_end)

    return {
        "Diastole_S3_Expected_Delta_mV": _compute_segment_delta_value(amplitude, *s3_window)
        if s3_window is not None
        else float(np.nan),
        "Diastole_NaN_Gap_Delta_mV": nan_gap_delta,
        "Diastole_S4_Expected_Delta_mV": _compute_segment_delta_value(amplitude, *s4_window)
        if s4_window is not None
        else float(np.nan),
    }


def _compute_diastole_parameter_row(
    amplitude: np.ndarray,
    s2_end_index: int,
    next_s1_start_index: int,
) -> dict[str, float]:
    row = _compute_gap_parameter_row(
        amplitude,
        start_index=s2_end_index,
        end_index=next_s1_start_index,
        duration_key="Diastole_Duration_ms",
        peak_key="Diastole_Peak_mV",
        mean_key="Diastole_mean_mV",
        nan_factory=lambda: _nan_row(DIASTOLE_PARAMETER_COLUMNS),
    )
    row.update(_compute_diastolic_expected_delta_row(amplitude, s2_end_index, next_s1_start_index))
    return row


def _get_rs_peak_value(signal: np.ndarray, event_index: int | None) -> float:
    if event_index is None:
        return float(np.nan)

    safe_index = int(event_index)
    if safe_index < 0 or safe_index >= len(signal):
        return float(np.nan)

    value = float(signal[safe_index])
    if not np.isfinite(value):
        return float(np.nan)
    return float(int(round(value)))


def _get_rs_width_bounds(signal: np.ndarray, event_index: int | None) -> tuple[int, int] | None:
    if event_index is None:
        return None

    safe_index = int(event_index)
    if safe_index < 0 or safe_index >= len(signal):
        return None

    peak_value = float(signal[safe_index])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        return None

    threshold = 0.5 * peak_value
    left_index = safe_index
    while left_index - 1 >= 0:
        next_value = float(signal[left_index - 1])
        if not np.isfinite(next_value) or next_value < threshold:
            break
        left_index -= 1

    right_index = safe_index
    while right_index + 1 < len(signal):
        next_value = float(signal[right_index + 1])
        if not np.isfinite(next_value) or next_value < threshold:
            break
        right_index += 1

    if left_index > right_index:
        return None

    return left_index, right_index


def _get_rs_width_value(signal: np.ndarray, event_index: int | None) -> float:
    bounds = _get_rs_width_bounds(signal, event_index)
    if bounds is None:
        return float(np.nan)

    left_index, right_index = bounds
    return float((right_index - left_index) * PreprocessConfig.SAMPLE_MS)


def _get_rs_sumation_value(signal: np.ndarray, event_index: int | None) -> float:
    bounds = _get_rs_width_bounds(signal, event_index)
    if bounds is None:
        return float(np.nan)

    safe_index = int(event_index) if event_index is not None else -1
    if safe_index < 0 or safe_index >= len(signal):
        return float(np.nan)

    left_index, right_index = bounds
    segment = np.nan_to_num(signal[left_index : right_index + 1].astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(np.abs(segment)))
    peak_value = float(signal[safe_index])
    if total <= 0.0 or not np.isfinite(peak_value):
        return float(np.nan)
    return float(abs(peak_value) / total)


def _compute_rs_peak_parameter_row(
    s1_start_signal: np.ndarray,
    s1_end_signal: np.ndarray,
    s2_start_signal: np.ndarray,
    s2_end_signal: np.ndarray,
    cycle_row: dict[str, Any],
) -> dict[str, float]:
    return {
        "S1S_RS_Peak": _get_rs_peak_value(s1_start_signal, int(cycle_row["S1_start"])),
        "S1E_RS_Peak": _get_rs_peak_value(s1_end_signal, int(cycle_row["S1_end"])),
        "S2S_RS_Peak": _get_rs_peak_value(s2_start_signal, int(cycle_row["S2_start"])),
        "S2E_RS_Peak": _get_rs_peak_value(s2_end_signal, int(cycle_row["S2_end"])),
    }


def _compute_rs_width_parameter_row(
    s1_start_signal: np.ndarray,
    s1_end_signal: np.ndarray,
    s2_start_signal: np.ndarray,
    s2_end_signal: np.ndarray,
    cycle_row: dict[str, Any],
) -> dict[str, float]:
    return {
        "S1S_RS_Width_ms": _get_rs_width_value(s1_start_signal, int(cycle_row["S1_start"])),
        "S1E_RS_Width_ms": _get_rs_width_value(s1_end_signal, int(cycle_row["S1_end"])),
        "S2S_RS_Width_ms": _get_rs_width_value(s2_start_signal, int(cycle_row["S2_start"])),
        "S2E_RS_Width_ms": _get_rs_width_value(s2_end_signal, int(cycle_row["S2_end"])),
    }


def _compute_rs_sumation_parameter_row(
    s1_start_signal: np.ndarray,
    s1_end_signal: np.ndarray,
    s2_start_signal: np.ndarray,
    s2_end_signal: np.ndarray,
    cycle_row: dict[str, Any],
) -> dict[str, float]:
    return {
        "S1S_RS_Sumation": _get_rs_sumation_value(s1_start_signal, int(cycle_row["S1_start"])),
        "S1E_RS_Sumation": _get_rs_sumation_value(s1_end_signal, int(cycle_row["S1_end"])),
        "S2S_RS_Sumation": _get_rs_sumation_value(s2_start_signal, int(cycle_row["S2_start"])),
        "S2E_RS_Sumation": _get_rs_sumation_value(s2_end_signal, int(cycle_row["S2_end"])),
    }


def _compute_heart_rate_row(s1_start_index: int, next_s1_start_index: int) -> dict[str, float]:
    cycle_duration_ms = float((int(next_s1_start_index) - int(s1_start_index)) * PreprocessConfig.SAMPLE_MS)
    if cycle_duration_ms <= 0.0:
        return _nan_row(HEART_RATE_COLUMNS)

    return {
        "HeartRate_bpm": float(60000.0 / cycle_duration_ms),
    }


def _build_feature_row(
    record_id: str,
    source_file: str,
    cycle_row: dict[str, Any],
    amplitude: np.ndarray,
    s1_start_signal: np.ndarray,
    s1_end_signal: np.ndarray,
    s2_start_signal: np.ndarray,
    s2_end_signal: np.ndarray,
    valid_flag: int,
    invalid_reason: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id,
        "source_file": source_file,
        "beat_index": int(cycle_row["beat_index"]),
        "cycle_index": int(cycle_row["cycle_index"]),
        "valid_flag": int(valid_flag),
        "invalid_reason": invalid_reason,
        "S1_start": cycle_row["S1_start"],
        "S1_end": cycle_row["S1_end"],
        "S2_start": cycle_row["S2_start"],
        "S2_end": cycle_row["S2_end"],
        "next_S1_start": cycle_row["next_S1_start"],
        "next_S1_end": cycle_row["next_S1_end"],
        "s1_on": cycle_row["s1_on"],
        "s1_off": cycle_row["s1_off"],
        "s2_on": cycle_row["s2_on"],
        "s2_off": cycle_row["s2_off"],
        "s1_on_next": cycle_row["s1_on_next"],
    }

    if valid_flag != 1:
        row.update(_nan_feature_map())
        return row

    features: dict[str, float] = {}
    features.update(_compute_s1_parameter_row(amplitude, int(cycle_row["S1_start"]), int(cycle_row["S1_end"])))
    features.update(_compute_s2_parameter_row(amplitude, int(cycle_row["S2_start"]), int(cycle_row["S2_end"])))
    features.update(_compute_systole_parameter_row(amplitude, int(cycle_row["S1_end"]), int(cycle_row["S2_start"])))
    features.update(_compute_diastole_parameter_row(amplitude, int(cycle_row["S2_end"]), int(cycle_row["next_S1_start"])))
    features.update(
        _compute_rs_peak_parameter_row(
            s1_start_signal,
            s1_end_signal,
            s2_start_signal,
            s2_end_signal,
            cycle_row,
        )
    )
    features.update(
        _compute_rs_width_parameter_row(
            s1_start_signal,
            s1_end_signal,
            s2_start_signal,
            s2_end_signal,
            cycle_row,
        )
    )
    features.update(
        _compute_rs_sumation_parameter_row(
            s1_start_signal,
            s1_end_signal,
            s2_start_signal,
            s2_end_signal,
            cycle_row,
        )
    )
    features.update(_compute_heart_rate_row(int(cycle_row["S1_start"]), int(cycle_row["next_S1_start"])))
    row.update(features)
    return row


def build_feature_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    ordered_columns = METADATA_COLUMNS + list(ALL_FEATURE_COLUMNS)
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
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(values))))) if values.size >= 2 else float(np.nan)
    return {
        "mean": mean_value,
        "std": std_value,
        "cv": float(std_value / (mean_value + eps)),
        "rmssd": rmssd,
    }


def build_record_summary(feature_frame: pd.DataFrame, eps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_id, record_frame in feature_frame.groupby("record_id", sort=True):
        valid_frame = record_frame.loc[record_frame["valid_flag"] == 1].copy()
        row: dict[str, Any] = {
            "record_id": record_id,
            "total_cycles": int(len(record_frame)),
            "valid_cycles": int(len(valid_frame)),
            "invalid_cycles": int(len(record_frame) - len(valid_frame)),
        }
        for column in SUMMARY_SOURCE_COLUMNS:
            stats = summary_stat(valid_frame[column].to_numpy(dtype=np.float64), eps)
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_std"] = stats["std"]
            row[f"{column}_cv"] = stats["cv"]
            row[f"{column}_rmssd"] = stats["rmssd"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_learning_input_columns(feature_columns: list[str]) -> list[str]:
    return list(feature_columns)


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
    amplitude = recording_table["Amplitude"].to_numpy(dtype=np.float32)
    s1_start_signal = recording_table["S1-Start_RS_Score"].to_numpy(dtype=np.float32)
    s1_end_signal = recording_table["S1-End_RS_Score"].to_numpy(dtype=np.float32)
    s2_start_signal = recording_table["S2-Start_RS_Score"].to_numpy(dtype=np.float32)
    s2_end_signal = recording_table["S2-End_RS_Score"].to_numpy(dtype=np.float32)

    cycles = _build_cycles_from_recording(recording_table)
    logger.info("cycle 수: record_id=%s, total_cycles=%s", record_id, len(cycles))

    rows = []
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
                s1_start_signal=s1_start_signal,
                s1_end_signal=s1_end_signal,
                s2_start_signal=s2_start_signal,
                s2_end_signal=s2_end_signal,
                valid_flag=valid_flag,
                invalid_reason=invalid_reason,
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
    record_summary = build_record_summary(beat_features_all, eps=PreprocessConfig.EPS)

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
