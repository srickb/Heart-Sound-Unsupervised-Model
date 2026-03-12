# Auto_Github

## Daily Research Log Automation

This folder contains the daily markdown logging system for `Heart-Sound-Unsupervised-Model`.

- `Auto_Github/scripts/generate_daily_log.py` reads same-day `Idea_DataBase/*.py` files.
- Optional manual notes can be added in `Auto_Github/notes/daily_raw/YYYY-MM-DD.md`.
- The script writes one file:
  - `Idea_DataBase/YYYY-MM-DD.md`
- If there are no same-day idea-file updates and no manual note, the script exits successfully and creates nothing.
- The script uses only Python standard library modules.
- `.github/workflows/daily_research_log.yml` runs the generator on `push` and every day at 23:00 Asia/Seoul time (`14:00 UTC` in GitHub Actions cron).

## Project Context

- Project: `Heart-Sound-Unsupervised-Model`
- Research direction: heart sound unsupervised learning / PCG-related research
- Local root: `C:\Users\LUI\Desktop\Unsupervised Model`
- Automation root: `C:\Users\LUI\Desktop\Unsupervised Model\Auto_Github`
- Markdown output root: `C:\Users\LUI\Desktop\Unsupervised Model\Idea_DataBase`

## Manual Inputs

- Optional note: `Auto_Github/notes/daily_raw/YYYY-MM-DD.md`
- Stable context: `Auto_Github/project_context.json`

## Local Run

```bash
python Auto_Github/scripts/generate_daily_log.py
```

Optional date override for inspection:

```bash
python Auto_Github/scripts/generate_daily_log.py --date 2026-03-12
```

## Output Policy

- Logs are written in Korean `입니다.` style.
- The generator does not copy raw diffs.
- The generator does not invent results beyond the available evidence.
- The generator summarizes what was added or updated in `Idea_DataBase`.
