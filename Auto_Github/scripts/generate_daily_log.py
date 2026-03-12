#!/usr/bin/env python3
"""Generate a daily markdown research log from same-day Idea_DataBase specs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


AUTOMATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTOMATION_ROOT.parent
IDEA_DIR = REPO_ROOT / "Idea_DataBase"
NOTE_DIR = AUTOMATION_ROOT / "notes" / "daily_raw"
PROJECT_CONTEXT_PATH = AUTOMATION_ROOT / "project_context.json"
KST = ZoneInfo("Asia/Seoul") if ZoneInfo else timezone(timedelta(hours=9))

IDEA_FILE_ORDER = [
    "Code_Guide_v1.py",
    "Processed_Idea.py",
    "Train_Idea.py",
    "Embedded_Idea.py",
    "Result_Idea.py",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
LABEL_ONLY_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _/\-]{0,40}:$")
DATE_LOG_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.md$")


@dataclass
class IdeaEvidence:
    path: str
    modified_at: datetime
    purpose: str
    key_points: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        dest="target_date",
        help="Optional target date in YYYY-MM-DD. Defaults to today in Asia/Seoul.",
    )
    return parser.parse_args()


def resolve_target_date(date_text: str | None) -> tuple[str, datetime, datetime]:
    parsed_date = date.fromisoformat(date_text) if date_text else datetime.now(KST).date()
    start = datetime.combine(parsed_date, time.min, tzinfo=KST)
    end = start + timedelta(days=1)
    return parsed_date.isoformat(), start, end


def load_project_context() -> dict[str, str]:
    if not PROJECT_CONTEXT_PATH.exists():
        return {
            "project": "Heart-Sound-Unsupervised-Model",
            "research_direction": "heart sound unsupervised learning / PCG-related research",
        }
    return json.loads(PROJECT_CONTEXT_PATH.read_text(encoding="utf-8"))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_code_block = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped in {'"""', "'''"}:
            continue
        cleaned = re.sub(r"^#{1,6}\s*", "", stripped)
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", cleaned)
        cleaned = normalize_whitespace(cleaned)
        if not cleaned:
            continue
        if LABEL_ONLY_RE.fullmatch(cleaned):
            continue
        if cleaned.startswith("Create a standalone script named"):
            continue
        if cleaned.startswith("Output files to save"):
            continue
        if cleaned.startswith("Done when"):
            continue
        lines.append(cleaned)

    return lines


def split_sentences(text: str) -> list[str]:
    sentences = [normalize_whitespace(part) for part in SENTENCE_SPLIT_RE.split(text)]
    return [sentence for sentence in sentences if len(sentence) >= 20]


def choose_purpose(text: str, fallback_name: str) -> str:
    lines = extract_clean_lines(text)

    for line in lines:
        if line.lower().startswith("goal"):
            continue
        for sentence in split_sentences(line):
            lowered = sentence.lower()
            if "goal" in lowered and len(sentence) < 40:
                continue
            return sentence

    return f"{fallback_name} related work was updated."


def choose_key_points(text: str, purpose: str, limit: int = 3) -> list[str]:
    keywords = (
        "must",
        "should",
        "required",
        "preserve",
        "save",
        "load",
        "train",
        "extract",
        "apply",
        "interpret",
        "validate",
        "output",
        "cluster",
        "feature",
        "metadata",
        "waveform",
        "embedding",
        "preprocess",
    )

    key_points: list[str] = []
    for line in extract_clean_lines(text):
        if line == purpose:
            continue
        lowered = line.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        if len(line) > 160:
            continue
        if line not in key_points:
            key_points.append(line)
        if len(key_points) >= limit:
            break

    return key_points


def idea_sort_key(path: Path) -> tuple[int, str]:
    try:
        return IDEA_FILE_ORDER.index(path.name), path.name
    except ValueError:
        return len(IDEA_FILE_ORDER), path.name


def collect_same_day_idea_files(start: datetime, end: datetime) -> list[IdeaEvidence]:
    if not IDEA_DIR.exists():
        return []

    evidences: list[IdeaEvidence] = []
    for path in sorted(IDEA_DIR.glob("*.py"), key=idea_sort_key):
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=KST)
        if not (start <= modified_at < end):
            continue
        text = path.read_text(encoding="utf-8")
        purpose = choose_purpose(text, path.name)
        key_points = choose_key_points(text, purpose)
        evidences.append(
            IdeaEvidence(
                path=path.relative_to(REPO_ROOT).as_posix(),
                modified_at=modified_at,
                purpose=purpose,
                key_points=key_points,
            )
        )
    return evidences


def read_same_day_note(target_date: str) -> list[str]:
    note_path = NOTE_DIR / f"{target_date}.md"
    if not note_path.exists():
        return []
    text = note_path.read_text(encoding="utf-8")
    lines = extract_clean_lines(text)
    return [line for line in lines if not DATE_LOG_RE.fullmatch(line)][:6]


def make_statement(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"[.!?]+$", "", cleaned)
    if not cleaned:
        return ""
    if cleaned.endswith("입니다"):
        return f"{cleaned}."
    if cleaned.endswith("입니다."):
        return cleaned
    return f"{cleaned}입니다."


def build_summary(idea_files: list[IdeaEvidence], note_lines: list[str]) -> str:
    if note_lines:
        return make_statement(note_lines[0])

    stage_names = [Path(item.path).stem.replace("_Idea", "") for item in idea_files]
    if stage_names:
        joined = ", ".join(stage_names)
        return make_statement(f"오늘은 {joined} 관련 설계 문서를 바탕으로 비지도 심장음 파이프라인 작업을 정리한 날")

    return make_statement("오늘 연구 활동을 정리할 근거가 부족함")


def build_markdown(
    target_date: str,
    project_context: dict[str, str],
    idea_files: list[IdeaEvidence],
    note_lines: list[str],
) -> str:
    summary = build_summary(idea_files, note_lines)

    lines: list[str] = [
        f"# Daily Research Log - {target_date}",
        "",
        "## Summary",
        summary,
        "",
        "## Project Context",
        f"- 프로젝트: `{project_context.get('project', '')}`",
        f"- 연구 방향: {make_statement(project_context.get('research_direction', ''))}",
        "",
        "## Updated Idea Files",
    ]

    if idea_files:
        for item in idea_files:
            timestamp = item.modified_at.strftime("%Y-%m-%d %H:%M:%S %Z")
            lines.append(f"- `{item.path}` ({timestamp}): {make_statement(item.purpose)}")
    else:
        lines.append("- 당일 수정된 Idea_DataBase `.py` 파일이 없습니다.")

    lines.extend(["", "## Key Points"])
    key_points_written = False
    for item in idea_files:
        if not item.key_points:
            continue
        key_points_written = True
        lines.append(f"- `{item.path}`")
        for point in item.key_points:
            lines.append(f"  - {make_statement(point)}")
    if not key_points_written:
        lines.append("- 핵심 포인트로 정리할 항목이 충분하지 않습니다.")

    lines.extend(["", "## Manual Notes"])
    if note_lines:
        for note in note_lines:
            lines.append(f"- {make_statement(note)}")
    else:
        lines.append("- 별도 수동 메모는 없습니다.")

    lines.extend(["", "## Evidence"])
    if idea_files:
        lines.append(f"- 당일 근거로 사용한 Idea 파일 수는 {len(idea_files)}개입니다.")
        for item in idea_files:
            lines.append(f"- `{item.path}`")
    else:
        lines.append("- 당일 Idea 파일 근거는 없습니다.")

    return "\n".join(lines) + "\n"


def main() -> int:
    target_date, start, end = resolve_target_date(parse_args().target_date)
    project_context = load_project_context()
    idea_files = collect_same_day_idea_files(start, end)
    note_lines = read_same_day_note(target_date)

    if not idea_files and not note_lines:
        print(f"[skip] No same-day Idea_DataBase updates for {target_date}.")
        return 0

    output_path = IDEA_DIR / f"{target_date}.md"
    markdown = build_markdown(target_date, project_context, idea_files, note_lines)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[ok] Wrote {output_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
