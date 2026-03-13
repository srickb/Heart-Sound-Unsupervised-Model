#!/usr/bin/env python3
"""Generate a detailed daily markdown log from same-day Idea_DataBase specs."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
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

DOCSTRING_FENCE_RE = re.compile(r'^\s*("""|\'\'\')\s*$')
SECTION_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /()_\-]{1,60}):\s*$")
DATE_LOG_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.md$")
GENERIC_NOTE_LINES = {"오늘 작업", "메모", "notes", "note"}

SECTION_PRIORITY = [
    "Goal",
    "Important context",
    "Critical constraints",
    "Required behavior",
    "Segmentation rules",
    "Signal preprocessing",
    "Feature extraction",
    "Feature groups",
    "Model requirements",
    "Training requirements",
    "HDBSCAN requirements",
    "Required analyses",
    "Important constraints",
    "Inputs to load",
    "Output files to save",
    "Outputs to save",
    "The metadata file must include at least",
    "The cluster assignment file must include at least",
    "The clustering summary must include at least",
    "The training summary should include",
    "The markdown report should summarize",
    "Implementation guidance",
    "Done when",
]


@dataclass
class IdeaEvidence:
    path: str
    modified_at: datetime
    raw_text: str
    purpose: str
    sections: OrderedDict[str, list[str]]


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


def strip_docstring_fences(text: str) -> list[str]:
    lines = text.splitlines()
    return [line.rstrip() for line in lines if not DOCSTRING_FENCE_RE.fullmatch(line)]


def extract_clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in strip_docstring_fences(text):
        stripped = raw_line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^#{1,6}\s*", "", stripped)
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", cleaned)
        cleaned = normalize_whitespace(cleaned)
        if not cleaned:
            continue
        if SECTION_HEADER_RE.fullmatch(cleaned):
            continue
        if cleaned.startswith("Create a standalone script named"):
            continue
        lines.append(cleaned)
    return lines


def parse_sections(text: str) -> OrderedDict[str, list[str]]:
    sections: OrderedDict[str, list[str]] = OrderedDict()
    current_name = "Overview"
    current_lines: list[str] = []

    for raw_line in strip_docstring_fences(text):
        stripped = raw_line.strip()
        header_match = SECTION_HEADER_RE.fullmatch(stripped)
        if header_match:
            if current_lines:
                sections[current_name] = clean_section_lines(current_lines)
            current_name = header_match.group(1)
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_lines:
        sections[current_name] = clean_section_lines(current_lines)

    return OrderedDict((name, lines) for name, lines in sections.items() if lines)


def clean_section_lines(lines: list[str]) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", stripped)
        cleaned = normalize_whitespace(cleaned)
        if not cleaned:
            continue
        cleaned_lines.append(cleaned)
    return cleaned_lines


def choose_purpose(sections: OrderedDict[str, list[str]], fallback_name: str) -> str:
    if "Goal" in sections and sections["Goal"]:
        return sections["Goal"][0]

    overview_lines = sections.get("Overview", [])
    for line in overview_lines:
        if len(line) >= 20:
            return line

    return f"{fallback_name} related work was updated."


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
        raw_text = path.read_text(encoding="utf-8")
        sections = parse_sections(raw_text)
        purpose = choose_purpose(sections, path.name)
        evidences.append(
            IdeaEvidence(
                path=path.relative_to(REPO_ROOT).as_posix(),
                modified_at=modified_at,
                raw_text=raw_text.rstrip() + "\n",
                purpose=purpose,
                sections=sections,
            )
        )
    return evidences


def read_same_day_note(target_date: str) -> list[str]:
    note_path = NOTE_DIR / f"{target_date}.md"
    if not note_path.exists():
        return []
    lines = extract_clean_lines(note_path.read_text(encoding="utf-8"))
    filtered: list[str] = []
    for line in lines:
        normalized = normalize_whitespace(line).lower()
        if DATE_LOG_RE.fullmatch(line):
            continue
        if normalized in GENERIC_NOTE_LINES:
            continue
        filtered.append(line)
    return filtered[:8]


def build_summary(idea_files: list[IdeaEvidence], note_lines: list[str]) -> str:
    if note_lines:
        return make_statement(note_lines[0])

    if not idea_files:
        return make_statement("오늘 정리할 Idea 파일 근거가 충분하지 않음")

    stage_names = [Path(item.path).stem.replace("_Idea", "") for item in idea_files]
    joined = ", ".join(stage_names)
    return make_statement(
        f"오늘은 {joined} 관련 Idea 문서를 바탕으로 비지도 심장음 파이프라인 설계와 구현 방향을 정리함"
    )


def format_section_points(section_name: str, section_lines: list[str], limit: int = 5) -> list[str]:
    if not section_lines:
        return []
    trimmed = section_lines[:limit]
    return [f"{section_name}: {line}" for line in trimmed]


def iter_priority_sections(sections: OrderedDict[str, list[str]]) -> list[tuple[str, list[str]]]:
    ordered: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    for name in SECTION_PRIORITY:
        if name in sections:
            ordered.append((name, sections[name]))
            seen.add(name)

    for name, lines in sections.items():
        if name not in seen:
            ordered.append((name, lines))

    return ordered


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
            lines.append(f"- `{item.path}` ({timestamp})")
            lines.append(f"  - 목적 요약: {make_statement(item.purpose)}")
    else:
        lines.append("- 당일 반영된 Idea_DataBase `.py` 파일이 없습니다.")

    lines.extend(["", "## Detailed Summary"])
    if idea_files:
        for item in idea_files:
            lines.append(f"### {item.path}")
            lines.append(f"- 핵심 목적: {make_statement(item.purpose)}")
            section_items_written = False
            for section_name, section_lines in iter_priority_sections(item.sections):
                if section_name == "Overview":
                    continue
                bullet_lines = format_section_points(section_name, section_lines)
                if not bullet_lines:
                    continue
                section_items_written = True
                for bullet in bullet_lines:
                    lines.append(f"- {make_statement(bullet)}")
            if not section_items_written:
                lines.append("- 세부 섹션으로 정리할 내용이 충분하지 않습니다.")
            lines.append("")
    else:
        lines.append("- 세부 요약을 생성할 파일이 없습니다.")
        lines.append("")

    lines.append("## Manual Notes")
    if note_lines:
        for note in note_lines:
            lines.append(f"- {make_statement(note)}")
    else:
        lines.append("- 별도 수동 메모는 없습니다.")

    lines.extend(["", "## Raw Added Content"])
    if idea_files:
        for item in idea_files:
            lines.append(f"### {item.path}")
            lines.append("```python")
            lines.extend(item.raw_text.rstrip("\n").splitlines())
            lines.append("```")
            lines.append("")
    else:
        lines.append("당일 반영된 원문 파일이 없습니다.")
        lines.append("")

    lines.append("## Evidence")
    if idea_files:
        lines.append(f"- 당일 근거로 사용한 Idea 파일 수는 {len(idea_files)}개입니다.")
        for item in idea_files:
            lines.append(f"- `{item.path}`")
    else:
        lines.append("- 당일 Idea 파일 근거는 없습니다.")

    return "\n".join(lines).rstrip() + "\n"


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
