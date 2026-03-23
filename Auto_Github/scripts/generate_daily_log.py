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

SECTION_NAME_KO = {
    "Goal": "목표",
    "Important context": "중요 맥락",
    "Critical constraints": "핵심 제약",
    "Required behavior": "필수 동작",
    "Segmentation rules": "분할 규칙",
    "Signal preprocessing": "신호 전처리",
    "Feature extraction": "특징 추출",
    "Feature groups": "특징 그룹",
    "Model requirements": "모델 요구사항",
    "Training requirements": "학습 요구사항",
    "HDBSCAN requirements": "HDBSCAN 요구사항",
    "Required analyses": "필수 분석",
    "Important constraints": "중요 제약",
    "Inputs to load": "입력 파일",
    "Output files to save": "저장할 출력 파일",
    "Outputs to save": "저장할 출력물",
    "The metadata file must include at least": "메타데이터 필수 항목",
    "The cluster assignment file must include at least": "군집 할당 파일 필수 항목",
    "The clustering summary must include at least": "클러스터링 요약 필수 항목",
    "The training summary should include": "학습 요약 포함 항목",
    "The markdown report should summarize": "마크다운 보고서 요약 항목",
    "Implementation guidance": "구현 가이드",
    "Done when": "완료 조건",
}

EXACT_TEXT_TRANSLATIONS = {
    "how many clusters were found": "발견된 군집 개수",
    "noise ratio": "노이즈 비율",
    "major feature differences between clusters": "군집 간 주요 특징 차이",
    "major differences between clusters": "군집 간 주요 차이",
    "which feature groups seem to characterize each cluster": "각 군집을 특징짓는 feature 그룹",
    "whether the noise group appears artifact-like or structurally inconsistent": "노이즈 그룹이 artifact 성향인지 또는 구조적으로 불안정한지 여부",
    "whether the noise group appears artifact-like": "노이즈 그룹이 artifact 성향으로 보이는지 여부",
    "waveform tendencies observed in each cluster": "각 군집에서 관찰된 파형 경향",
    "This project is unsupervised": "이 프로젝트는 비지도학습 기반임",
    "Do not retrain the model here": "이 단계에서 모델을 다시 학습하지 않음",
    "Do not use labels": "라벨을 사용하지 않음",
    "Do not manually set the number of clusters": "군집 개수를 수동으로 고정하지 않음",
    "Load only valid cycles from preprocessing outputs": "전처리 결과에서 valid cycle만 불러옴",
    "Preserve sample ordering and sample_id": "sample 순서와 sample_id 정렬을 유지함",
    "Preserve sample_id alignment": "sample_id 정렬을 유지함",
    "Load all valid cycles in the same sample order as preprocessing": "전처리 단계와 동일한 sample 순서로 모든 valid cycle을 불러옴",
    "Extract one latent embedding vector per sample": "sample마다 잠재 임베딩 벡터 1개를 추출함",
    "Run HDBSCAN on the latent vectors": "잠재 벡터에 HDBSCAN을 적용함",
    "cluster-level tables are saved": "군집 단위 테이블이 저장됨",
    "feature-based comparison plots are saved": "feature 기반 비교 플롯이 저장됨",
    "representative sample IDs are saved": "대표 sample ID가 저장됨",
    "noise analysis is saved": "노이즈 분석 결과가 저장됨",
    "a concise markdown interpretation report is generated": "간단한 마크다운 해석 보고서가 생성됨",
}

PHRASE_TRANSLATIONS = [
    ("The purpose is to learn a compact latent representation of each heart sound cycle in feature space", "각 심음 주기를 feature 공간에서 압축된 잠재 표현으로 학습하는 것"),
    ("The purpose is to convert cluster labels into understandable heart sound cycle patterns in feature space", "군집 라벨을 feature 공간에서 이해 가능한 심음 주기 패턴으로 해석하는 것"),
    ("The purpose is to convert cluster labels into understandable heart sound pattern descriptions without overclaiming clinical meaning", "군집 라벨을 임상적 과장을 피하면서 이해 가능한 심음 패턴 설명으로 바꾸는 것"),
    ("Build the preprocessing stage for unsupervised heart sound cycle analysis", "비지도 심음 주기 분석용 전처리 단계를 구축"),
    ("Train an unsupervised autoencoder on the cycle-level numeric feature matrix produced by 01_preprocess.py", "01_preprocess.py에서 생성한 주기 단위 수치형 feature matrix로 비지도 autoencoder를 학습"),
    ("Load the trained encoder and the preprocessed cycle feature matrix, extract latent embeddings for all valid cycles, apply HDBSCAN in latent space, and save clustering outputs for later interpretation", "학습된 encoder와 전처리된 cycle feature matrix를 불러와 모든 valid cycle의 잠재 임베딩을 추출하고 latent space에서 HDBSCAN을 수행한 뒤 이후 해석에 필요한 결과를 저장"),
    ("Load the trained encoder and the preprocessed cycle features, extract latent embeddings for all valid cycles, apply HDBSCAN in latent space, and save clustering outputs for later interpretation", "학습된 encoder와 전처리된 cycle feature를 불러와 모든 valid cycle의 잠재 임베딩을 추출하고 latent space에서 HDBSCAN을 수행한 뒤 이후 해석에 필요한 결과를 저장"),
    ("Interpret the HDBSCAN clustering results by comparing cluster-wise numeric feature distributions and identifying the dominant structural characteristics of each cluster", "HDBSCAN 군집 결과를 군집별 수치형 feature 분포 비교와 주요 구조 특성 파악을 통해 해석"),
    ("This project is NOT primarily using raw waveform sequences as model input", "이 프로젝트는 원시 파형 시퀀스를 주된 모델 입력으로 사용하지 않음"),
    ("The main learning input must be a numeric feature vector extracted from each cycle", "주된 학습 입력은 각 cycle에서 추출한 수치형 feature vector임"),
    ("One heart cycle should be defined primarily as S1(i) onset to S1(i+1) onset, with S2(i) occurring in between", "하나의 심장 주기는 기본적으로 S1(i) 시작부터 다음 S1(i+1) 시작 전까지로 정의하며 그 사이에 S2(i)가 포함됨"),
    ("Sampling rate is fixed at 4000 Hz", "샘플링 주파수는 4000 Hz로 고정함"),
    ("Avoid resampling by default", "기본적으로 리샘플링은 피함"),
    ("This is a tabular feature-learning task, not a raw time-series reconstruction task", "이 단계는 원시 시계열 복원이 아니라 tabular feature 학습 작업임"),
    ("The model input is a fixed-length numeric feature vector for each cycle", "모델 입력은 각 cycle의 고정 길이 수치형 feature vector임"),
    ("Do not use raw waveform tensors as model input", "원시 파형 텐서를 모델 입력으로 사용하지 않음"),
    ("Do not build sequence models such as 1D CNN, RNN, LSTM, or Transformer for this stage", "이 단계에서는 1D CNN, RNN, LSTM, Transformer 같은 시퀀스 모델을 사용하지 않음"),
    ("Use a simple dense autoencoder for tabular numeric data", "tabular 수치형 데이터에 맞는 단순 dense autoencoder를 사용함"),
    ("Split train/validation at the subject level if subject_id exists; otherwise use recording_id level splitting", "subject_id가 있으면 subject 단위로, 없으면 recording_id 단위로 train/validation을 분리함"),
    ("Do not leak normalization statistics from validation into training", "validation의 정규화 통계가 training에 섞이지 않도록 함"),
    ("This stage operates on tabular feature embeddings", "이 단계는 tabular feature 임베딩을 대상으로 동작함"),
    ("Each sample corresponds to one heart sound cycle represented by one numeric feature vector", "각 sample은 하나의 수치형 feature vector로 표현된 심음 cycle 1개에 대응함"),
    ("HDBSCAN should discover stable density-based groups in latent feature space", "HDBSCAN은 잠재 feature 공간에서 안정적인 밀도 기반 군집을 찾아야 함"),
    ("Interpretation should be based primarily on the extracted numeric feature vectors", "해석은 추출된 수치형 feature vector를 중심으로 이루어져야 함"),
    ("Do not rely primarily on raw waveform visualization", "원시 파형 시각화에만 의존하지 않음"),
    ("Cluster interpretation is more important than reporting a single numeric score", "단일 수치 점수보다 군집 해석이 더 중요함"),
    ("Use neutral cluster names by default, such as cluster_0, cluster_1, and noise", "기본적으로 cluster_0, cluster_1, noise 같은 중립적인 군집 이름을 사용함"),
    ("Do not automatically name clusters as “normal”, “murmur”, or other clinical labels unless there is explicit evidence and the code only presents them as tentative hypotheses", "명확한 근거가 없는 한 군집을 normal, murmur 같은 임상 라벨로 자동 명명하지 않으며, 필요 시에도 가설 수준으로만 제시함"),
    ("Use neutral language in outputs", "출력에서는 중립적인 표현을 사용함"),
    ("Keep the interpretation code reproducible and easy to inspect", "해석 코드는 재현 가능하고 검토하기 쉽게 유지함"),
    ("Focus on feature-based cluster interpretability rather than signal reconstruction", "신호 복원보다 feature 기반 군집 해석 가능성에 초점을 둠"),
    ("You are helping build research code for unsupervised heart sound cycle analysis", "비지도 심음 주기 분석을 위한 연구 코드를 작성하는 작업임"),
    ("heart sound unsupervised learning / PCG-related research", "심음 비지도학습 및 PCG 관련 연구"),
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


def translate_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""

    if cleaned in EXACT_TEXT_TRANSLATIONS:
        return EXACT_TEXT_TRANSLATIONS[cleaned]

    translated = cleaned
    for source, target in sorted(
        EXACT_TEXT_TRANSLATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        translated = translated.replace(source, target)

    for source, target in PHRASE_TRANSLATIONS:
        translated = translated.replace(source, target)

    for source, target in SECTION_NAME_KO.items():
        if translated.startswith(f"{source}:"):
            translated = f"{target}:{translated[len(source) + 1:]}"
            break

    translated = translated.replace("Output files to save:", "저장할 출력 파일:")
    translated = translated.replace("Outputs to save:", "저장할 출력물:")
    translated = translated.replace("Inputs to load:", "입력 파일:")
    translated = translated.replace("Required behavior:", "필수 동작:")
    translated = translated.replace("Important context:", "중요 맥락:")
    translated = translated.replace("Implementation guidance:", "구현 가이드:")
    translated = translated.replace("Done when:", "완료 조건:")
    translated = translated.replace("Training requirements:", "학습 요구사항:")
    translated = translated.replace("Model requirements:", "모델 요구사항:")
    translated = translated.replace("HDBSCAN requirements:", "HDBSCAN 요구사항:")
    translated = translated.replace("Required analyses:", "필수 분석:")
    translated = translated.replace("Important constraints:", "중요 제약:")
    translated = translated.replace("Critical constraints:", "핵심 제약:")
    translated = translated.replace("Segmentation rules:", "분할 규칙:")
    translated = translated.replace("Feature extraction:", "특징 추출:")
    translated = translated.replace("Feature groups:", "특징 그룹:")
    translated = translated.replace("The metadata file must include at least:", "메타데이터 필수 항목:")
    translated = translated.replace("The cluster assignment file must include at least:", "군집 할당 파일 필수 항목:")
    translated = translated.replace("The clustering summary must include at least:", "클러스터링 요약 필수 항목:")
    translated = translated.replace("The training summary should include:", "학습 요약 포함 항목:")
    translated = translated.replace("The markdown report should summarize:", "마크다운 보고서 요약 항목:")

    return translated


def make_statement(text: str) -> str:
    cleaned = translate_text(text)
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
    display_name = SECTION_NAME_KO.get(section_name, section_name)
    return [f"{display_name}: {line}" for line in trimmed]


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
