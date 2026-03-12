# Daily Log Usage

## Purpose

This document explains how the daily markdown generator works after the automation was simplified around `Idea_DataBase`.

## Evidence Scope

The generator reads only the target date in Asia/Seoul timezone.

- Same-day `Idea_DataBase/*.py` files
- Optional `Auto_Github/notes/daily_raw/YYYY-MM-DD.md`
- `Auto_Github/project_context.json`

The generator does not require commits, experiment JSON files, or diff parsing.

## Meaningful Day Rule

The generator creates a markdown log when at least one of the following is true:

- There is at least one same-day `Idea_DataBase/*.py` file.
- There is a non-empty manual note for that date.

If neither exists, the script exits with success and creates nothing.

## Generated File

When the day is meaningful, the script creates:

- `Idea_DataBase/YYYY-MM-DD.md`

## Markdown Log

The markdown file is the human-readable daily record.

- It is written in Korean `입니다.` style.
- It summarizes same-day idea files and optional manual notes.
- It is meant to capture research intent and work scope at a practical level.

## Recommended Workflow

1. Update the idea spec files in `Idea_DataBase`.
2. Optionally write a short note in `Auto_Github/notes/daily_raw/YYYY-MM-DD.md`.
3. Run `python Auto_Github/scripts/generate_daily_log.py` locally when needed.
4. Let GitHub Actions regenerate and push the markdown file at 23:00 Asia/Seoul time.
