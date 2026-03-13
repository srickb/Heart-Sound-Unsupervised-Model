"""
Standalone export script for cycle-level final cluster assignments.

Expected inputs:
- outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
- outputs/{RUN_NAME}/clustering/cluster_assignments.csv

Saved artifact:
- outputs/{RUN_NAME}/Total_Result/cycle_cluster_assignments.xlsx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from excel_export_utils import export_stage_workbook


# ============================================================================
# Editable configuration
# ============================================================================
class ExportConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "test_dataset_260312_preprocess_v2"

    PREPROCESS_ROOT = OUTPUT_ROOT / RUN_NAME / "preprocess"
    CLUSTERING_ROOT = OUTPUT_ROOT / RUN_NAME / "clustering"
    INTERPRETATION_ROOT = OUTPUT_ROOT / RUN_NAME / "interpretation"

    EXPORT_ROOT = OUTPUT_ROOT / RUN_NAME / "Total_Result"
    EXPORT_FILENAME = "cycle_cluster_assignments.xlsx"
    PER_RECORDING_FILENAME_SUFFIX = "_cycle_cluster_assignments.xlsx"

    FREEZE_PANES = "A2"
    HEADER_FILL = "1F4E78"
    HEADER_FONT_COLOR = "FFFFFF"
    MAX_COLUMN_WIDTH = 30

    NOISE_LABEL_VALUE = -1
    NOISE_DISPLAY_NAME = "Noise"
    CLUSTER_PREFIX = "Cluster "
    EXCLUDED_DISPLAY_NAME = "Excluded"
    UNMATCHED_DISPLAY_NAME = "Unmatched"

    REQUIRED_METADATA_COLUMNS = [
        "sample_id",
        "recording_id",
        "source_file",
        "cycle_index",
        "cycle_start_sample",
        "cycle_end_sample",
        "valid_flag",
        "feature_row_index",
        "waveform_row_index",
    ]
    REQUIRED_ASSIGNMENT_COLUMNS = ["cluster_label"]
    ASSIGNMENT_FILENAME_CANDIDATES = [
        "cluster_assignments.csv",
        "final_cluster_assignments.csv",
        "cluster_labels.csv",
    ]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_export_root(export_root: Path) -> None:
    """Create the target export directory."""
    export_root.mkdir(parents=True, exist_ok=True)


def find_cluster_assignment_path(clustering_root: Path) -> Path:
    """Locate the most likely cluster assignment CSV in the clustering folder."""
    for filename in ExportConfig.ASSIGNMENT_FILENAME_CANDIDATES:
        candidate = clustering_root / filename
        if candidate.exists():
            return candidate

    csv_candidates = sorted(clustering_root.glob("*.csv"))
    scored_candidates: list[tuple[int, Path]] = []
    for path in csv_candidates:
        stem = path.stem.lower()
        score = 0
        if "cluster" in stem:
            score += 2
        if "assignment" in stem:
            score += 2
        if "label" in stem:
            score += 1
        if score > 0:
            scored_candidates.append((score, path))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (-item[0], item[1].name))
        return scored_candidates[0][1]

    raise FileNotFoundError(
        f"No cluster assignment CSV found under: {clustering_root}"
    )


def load_cycle_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load cycle metadata and validate the minimum required schema."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    missing_columns = [
        column
        for column in ExportConfig.REQUIRED_METADATA_COLUMNS
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["recording_id"] = metadata["recording_id"].astype(str)
    metadata["source_file"] = metadata["source_file"].astype(str)
    metadata["cycle_index"] = metadata["cycle_index"].astype(int)
    metadata["valid_flag"] = metadata["valid_flag"].astype(bool)
    return metadata


def load_cluster_assignments(assignments_path: Path) -> pd.DataFrame:
    """Load cluster assignments and validate a usable key exists."""
    assignments = pd.read_csv(assignments_path)

    missing_columns = [
        column
        for column in ExportConfig.REQUIRED_ASSIGNMENT_COLUMNS
        if column not in assignments.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required assignment columns: {missing_columns}")

    usable_keys = [
        key
        for key in ["sample_id", "feature_row_index", "waveform_row_index"]
        if key in assignments.columns
    ]
    if not usable_keys:
        raise ValueError(
            "Cluster assignments must contain at least one join key among "
            "sample_id, feature_row_index, waveform_row_index."
        )

    assignments = assignments.copy()
    if "sample_id" in assignments.columns:
        assignments["sample_id"] = assignments["sample_id"].astype(str)
    for key in ["feature_row_index", "waveform_row_index"]:
        if key in assignments.columns:
            assignments[key] = pd.to_numeric(assignments[key], errors="coerce")
    assignments["cluster_label"] = pd.to_numeric(
        assignments["cluster_label"], errors="coerce"
    )
    return assignments


def choose_match_key(metadata: pd.DataFrame, assignments: pd.DataFrame) -> str:
    """Choose the most stable key for joining cycles to cluster assignments."""
    key_priority = ["sample_id", "feature_row_index", "waveform_row_index"]
    valid_metadata = metadata.loc[metadata["valid_flag"]].copy()

    for key in key_priority:
        if key not in assignments.columns or key not in valid_metadata.columns:
            continue

        left_values = valid_metadata[key].dropna()
        right_values = assignments[key].dropna()
        if left_values.empty or right_values.empty:
            continue
        if left_values.duplicated().any():
            continue
        if right_values.duplicated().any():
            continue

        overlap_count = int(pd.Index(left_values).intersection(pd.Index(right_values)).size)
        if overlap_count > 0:
            logger.info("Using '%s' as cycle-cluster match key (overlap=%s).", key, overlap_count)
            return key

    raise ValueError(
        "Unable to find a stable match key between cycle metadata and cluster assignments."
    )


def cluster_display_name(cluster_label: Any) -> str:
    """Convert the raw cluster label into a human-friendly display value."""
    if pd.isna(cluster_label):
        return ExportConfig.UNMATCHED_DISPLAY_NAME

    cluster_label_int = int(cluster_label)
    if cluster_label_int == ExportConfig.NOISE_LABEL_VALUE:
        return ExportConfig.NOISE_DISPLAY_NAME
    return f"{ExportConfig.CLUSTER_PREFIX}{cluster_label_int}"


def build_cycle_assignment_table(
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    match_key: str,
) -> pd.DataFrame:
    """Merge preprocess cycle metadata with final cluster assignments."""
    assignment_columns = [match_key, "cluster_label"]
    extra_assignment_columns = [
        column
        for column in ["membership_probability", "outlier_score"]
        if column in assignments.columns
    ]
    assignment_columns.extend(extra_assignment_columns)

    assignments_for_merge = assignments[assignment_columns].copy()

    valid_metadata = metadata.loc[metadata["valid_flag"] == True].copy()
    excluded_metadata = metadata.loc[metadata["valid_flag"] == False].copy()

    merged_valid = valid_metadata.merge(
        assignments_for_merge,
        on=match_key,
        how="left",
        validate="one_to_one",
    )
    for column in assignments_for_merge.columns:
        if column == match_key:
            continue
        if column not in excluded_metadata.columns:
            excluded_metadata[column] = np.nan

    merged = pd.concat([merged_valid, excluded_metadata], axis=0, ignore_index=True)

    merged["cluster_label_raw"] = merged["cluster_label"]
    merged["cluster_display"] = merged["cluster_label_raw"].apply(cluster_display_name)
    merged.loc[merged["valid_flag"] == False, "cluster_display"] = ExportConfig.EXCLUDED_DISPLAY_NAME
    merged.loc[
        (merged["valid_flag"] == True) & (merged["cluster_label_raw"].isna()),
        "cluster_display",
    ] = ExportConfig.UNMATCHED_DISPLAY_NAME

    merged["cycle_num_display"] = merged["cycle_index"].apply(
        lambda cycle_index: f"Cycle{int(cycle_index) + 1}"
    )
    merged["cluster_label_raw"] = pd.to_numeric(
        merged["cluster_label_raw"], errors="coerce"
    )
    merged = merged.sort_values(
        by=["source_file", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    return merged


def sanitize_sheet_name(sheet_name: str) -> str:
    """Make a source-file-derived sheet name Excel-safe."""
    invalid_chars = set(r'[]:*?/\\')
    clean_name = "".join("_" if char in invalid_chars else char for char in sheet_name)
    clean_name = clean_name.replace(".xlsx", "").strip()
    return clean_name or "Sheet"


def make_unique_sheet_name(base_name: str, used_names: set[str]) -> str:
    """Enforce Excel's sheet-name uniqueness and 31-character limit."""
    base_name = sanitize_sheet_name(base_name)
    if len(base_name) > 31:
        base_name = base_name[:31]

    candidate = base_name
    suffix_index = 1
    while candidate in used_names:
        suffix = f"_{suffix_index}"
        candidate = base_name[: max(0, 31 - len(suffix))] + suffix
        suffix_index += 1

    used_names.add(candidate)
    return candidate


def build_overview_dataframe(
    merged: pd.DataFrame,
    match_key: str,
) -> pd.DataFrame:
    """Build a compact overview sheet for export integrity checks."""
    rows: list[dict[str, object]] = [
        {"section": "run", "metric": "run_name", "value": ExportConfig.RUN_NAME},
        {"section": "counts", "metric": "total_recordings", "value": int(merged["source_file"].nunique())},
        {"section": "counts", "metric": "total_cycles", "value": int(len(merged))},
        {"section": "counts", "metric": "valid_cycles", "value": int(merged["valid_flag"].sum())},
        {"section": "counts", "metric": "excluded_cycles", "value": int((~merged["valid_flag"]).sum())},
        {
            "section": "counts",
            "metric": "matched_cluster_rows",
            "value": int(((merged["valid_flag"] == True) & merged["cluster_label_raw"].notna()).sum()),
        },
        {
            "section": "counts",
            "metric": "unmatched_rows",
            "value": int(((merged["valid_flag"] == True) & merged["cluster_label_raw"].isna()).sum()),
        },
        {
            "section": "counts",
            "metric": "noise_count",
            "value": int((merged["cluster_display"] == ExportConfig.NOISE_DISPLAY_NAME).sum()),
        },
        {"section": "matching", "metric": "match_key", "value": match_key},
    ]

    cluster_counts = (
        merged.loc[merged["valid_flag"] == True, "cluster_display"]
        .value_counts(dropna=False)
        .sort_index()
    )
    for cluster_name, count in cluster_counts.items():
        rows.append(
            {
                "section": "cluster_counts",
                "metric": str(cluster_name),
                "value": int(count),
            }
        )
    return pd.DataFrame(rows)


def build_all_cycles_dataframe(merged: pd.DataFrame) -> pd.DataFrame:
    """Build the unified cycle-level export sheet."""
    columns = [
        "source_file",
        "recording_id",
        "sample_id",
        "cycle_index",
        "cycle_num_display",
        "cycle_start_sample",
        "cycle_end_sample",
        "valid_flag",
        "cluster_label_raw",
        "cluster_display",
    ]
    optional_columns = [
        column
        for column in ["feature_row_index", "waveform_row_index", "membership_probability", "outlier_score"]
        if column in merged.columns
    ]
    return merged[columns + optional_columns].copy()


def build_recording_sheet_dataframe(recording_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the minimal A-D recording sheet requested by the user."""
    return pd.DataFrame(
        {
            "Cycle Num": recording_frame["cycle_num_display"].tolist(),
            "Cycle Start": recording_frame["cycle_start_sample"].astype("Int64").tolist(),
            "Cycle End": recording_frame["cycle_end_sample"].astype("Int64").tolist(),
            "Cluster": recording_frame["cluster_display"].tolist(),
        }
    )


def export_per_recording_workbooks(merged: pd.DataFrame) -> list[Path]:
    """Export one workbook per source recording file."""
    workbook_paths: list[Path] = []
    used_filenames: set[str] = set()

    for source_file, recording_frame in merged.groupby("source_file", sort=True):
        base_stem = sanitize_sheet_name(source_file)
        filename = f"{base_stem}{ExportConfig.PER_RECORDING_FILENAME_SUFFIX}"
        if len(filename) > 180:
            filename = filename[:180]

        candidate_name = filename
        suffix_index = 1
        while candidate_name.lower() in used_filenames:
            stem = base_stem
            suffix = f"_{suffix_index}{ExportConfig.PER_RECORDING_FILENAME_SUFFIX}"
            candidate_name = f"{stem[: max(1, 180 - len(suffix))]}{suffix}"
            suffix_index += 1
        used_filenames.add(candidate_name.lower())

        workbook_path = ExportConfig.EXPORT_ROOT / candidate_name
        sheet_name = make_unique_sheet_name(base_stem, set())
        recording_sheet = build_recording_sheet_dataframe(
            recording_frame.sort_values("cycle_index", kind="stable")
        )
        export_stage_workbook(
            workbook_path=workbook_path,
            sheets={sheet_name: recording_sheet},
            freeze_panes=ExportConfig.FREEZE_PANES,
            header_fill=ExportConfig.HEADER_FILL,
            header_font_color=ExportConfig.HEADER_FONT_COLOR,
            max_column_width=ExportConfig.MAX_COLUMN_WIDTH,
        )
        workbook_paths.append(workbook_path)

    return workbook_paths


def build_export_sheets(merged: pd.DataFrame, match_key: str) -> dict[str, pd.DataFrame]:
    """Assemble the workbook sheets in a predictable order."""
    sheets: dict[str, pd.DataFrame] = {
        "Overview": build_overview_dataframe(merged, match_key),
        "All_Cycles": build_all_cycles_dataframe(merged),
    }

    used_sheet_names = set(sheets.keys())
    for source_file, recording_frame in merged.groupby("source_file", sort=True):
        sheet_name = make_unique_sheet_name(source_file, used_sheet_names)
        sheets[sheet_name] = build_recording_sheet_dataframe(
            recording_frame.sort_values("cycle_index", kind="stable")
        )
    return sheets


def export_cycle_cluster_assignments() -> Path:
    """Load artifacts, merge cluster results, and export the final workbook."""
    ensure_export_root(ExportConfig.EXPORT_ROOT)

    metadata_path = ExportConfig.PREPROCESS_ROOT / "cycle_metadata.csv"
    assignments_path = find_cluster_assignment_path(ExportConfig.CLUSTERING_ROOT)

    metadata = load_cycle_metadata(metadata_path)
    assignments = load_cluster_assignments(assignments_path)
    match_key = choose_match_key(metadata, assignments)
    merged = build_cycle_assignment_table(metadata, assignments, match_key)

    workbook_path = ExportConfig.EXPORT_ROOT / ExportConfig.EXPORT_FILENAME
    sheets = build_export_sheets(merged, match_key)
    export_stage_workbook(
        workbook_path=workbook_path,
        sheets=sheets,
        freeze_panes=ExportConfig.FREEZE_PANES,
        header_fill=ExportConfig.HEADER_FILL,
        header_font_color=ExportConfig.HEADER_FONT_COLOR,
        max_column_width=ExportConfig.MAX_COLUMN_WIDTH,
    )
    per_recording_paths = export_per_recording_workbooks(merged)

    matched_count = int(
        ((merged["valid_flag"] == True) & merged["cluster_label_raw"].notna()).sum()
    )
    unmatched_count = int(
        ((merged["valid_flag"] == True) & merged["cluster_label_raw"].isna()).sum()
    )
    logger.info("Read metadata rows: %s", len(metadata))
    logger.info("Valid cycles: %s", int(metadata["valid_flag"].sum()))
    logger.info("Matched cluster rows: %s", matched_count)
    logger.info("Unmatched rows: %s", unmatched_count)
    logger.info("Recording sheet count: %s", int(metadata["source_file"].nunique()))
    logger.info("Per-recording workbook count: %s", len(per_recording_paths))
    logger.info("Saved workbook to: %s", workbook_path)
    logger.info("Cluster assignment source: %s", assignments_path)
    logger.info("Match key used: %s", match_key)
    return workbook_path


def main() -> None:
    workbook_path = export_cycle_cluster_assignments()
    print(f"Saved cycle cluster assignment workbook to: {workbook_path}")


if __name__ == "__main__":
    main()
