# AGENT.md

## Purpose

This repository should follow the coding style, pipeline structure, and implementation habits used in the PCG/260108 code unless the user explicitly requests a different direction.

The default goal is not to redesign the project.  
The default goal is to extend the existing codebase consistently.

## Core Principle

Prefer the existing script-based pipeline style over large abstractions.

Keep the code easy to trace from top to bottom.  
Do not introduce unnecessary framework layers, package restructuring, or architecture changes unless explicitly requested.

Default preference:

- separate executable scripts
- explicit helper functions
- top-level config class
- direct file-based pipeline
- minimal abstraction
- readable procedural flow

## Required File Structure

For Python scripts, preserve this order whenever possible:

- imports
- logging setup
- config class
- section divider comments
- helper functions
- `main()`
- `if __name__ == "__main__": main()`

Use section dividers in this style:

```python
# =================================================
# 1. Section Title
# =================================================
```

Do not replace this with an overly abstract structure.

## Pipeline Separation Rule

Keep responsibilities split by stage.

Preferred separation:

- preprocessing script
- training script
- inference / test script
- evaluation script
- optional analysis / visualization script

Do not mix all stages into one large file unless explicitly requested.

## Config Rules

Every executable file should define a config class near the top.

Examples:

- `PreprocessConfig`
- `TrainConfig`
- `TestConfig`
- `EvalConfig`
- `GlobalConfig`

Rules:

- config fields should use uppercase names
- paths should live in the config class
- tunable constants should live in the config class
- shared constants should not be scattered across function bodies
- if a value already exists in config, logic must reference config instead of hardcoding

Always prefer:

- `config.SAMPLING_RATE`
- `config.WINDOW_SIZE`
- `config.OUTPUT`
- `config.NULL_CLASS`
- `config.EVENT_NAMES`

instead of raw numeric literals.

## Hardcoding Rule

Do not leave stale magic numbers in logic.

If the code uses configurable values like:

- `OUTPUT`
- `NULL_CLASS`
- `WINDOW_SIZE`
- `FRAME_SIZE`
- `FRAME_OVERLAP`

then downstream logic must reference the config values, not old literals.

Examples of what to avoid:

- using `39`, `79`, `80`, `395`, `711` directly in logic when they are derived from config
- keeping an old class index threshold after the output dimension changed
- changing one stage but forgetting related dependent dimensions

If a dimension changes, update every dependent part consistently.

## Expected Domain Constants

Unless explicitly changed by the user, preserve the current 260108-style defaults:

```python
SAMPLING_RATE = 4000
PEAK_FREQUENCIES = [30, 60, 90, 120, 150]
Q_FACTOR = 10
ENVELOPE_WINDOW_SIZE = 200
FRAME_SIZE = 100
FRAME_OVERLAP = 50
WINDOW_SIZE = 79
EVENT_NAMES = ['S1-Start', 'S1-End', 'S2-Start', 'S2-End']
```

Do not silently change these in one file only.  
If one is changed, check preprocessing, training, inference, and evaluation together.

## Preprocessing Style

When writing preprocessing code, stay close to the existing flow.

Preferred pattern:

- load raw signal
- normalize signal if needed
- construct filter bank
- apply filtering
- compute envelopes
- convert to multi-channel per-sample features
- convert sample features into frame-level features
- use standard deviation based frame representation
- apply `np.log1p`
- create sliding windows
- create labels per event
- save processed arrays

Keep preprocessing explicit and readable.

Do not replace the existing feature flow with a very different representation unless explicitly requested.

## Training Style

When writing training code in this repo style:

- preserve the stage-based structure
- preserve the per-event loop structure
- keep training logic script-readable
- prefer helper functions for model creation and feature fusion
- save intermediate artifacts clearly

For the 260108 2mDNN-style code, the default assumption is:

- Stage 1: train one model per event
- generate RS-like signals from Stage 1
- Stage 2: combine original features and RS features
- train one final model per event

Loop over `EVENT_NAMES` instead of duplicating code manually.

Preferred behavior:

- load processed arrays once when possible
- reuse normalization stats
- skip already existing model files when intended
- save logs per event
- save models per event
- write a final summary report if the surrounding code already does this

## Inference Style

Inference code should remain separate from training code.

Preferred inference responsibilities:

- load normalization statistics
- load trained models
- load raw test files
- resample if needed
- preprocess using the same feature logic as training
- generate stage-1 outputs
- build stage-2 input
- generate final outputs
- save per-file results

Do not mix training callbacks or training-only logic into inference scripts.

Inference outputs should remain easy to inspect in tabular form.

## Evaluation Style

Evaluation should remain a separate analysis layer.

Preferred evaluation responsibilities:

- load prediction files
- load GT files
- align lengths safely
- detect peaks from RS score
- compute event-wise metrics
- compute timing errors
- compute overlap-based metrics where needed
- compute simple clinical summary values if relevant
- save final report as Excel

Keep metric functions explicit.

Prefer separate helper functions for:

- Dice
- F1-related logic
- MAE / residual extraction
- clinical metric extraction

If column order matters in the output report, define it explicitly before saving.

## Naming Conventions

Keep naming close to the current repo style.

### Classes

Use descriptive config-style names:

- `PreprocessConfig`
- `TrainConfig`
- `TestConfig`
- `EvalConfig`
- `GlobalConfig`

### Functions

Prefer readable helper names.  
Do not rename working legacy functions only for stylistic reasons.

Examples of acceptable style:

- `Filter`
- `Filter_diff`
- `RS_Score`
- `combine_features_rs`
- `calculate_dice`
- `extract_clinical_metrics`

For newly added functions, readable snake_case is preferred.  
However, consistency with the surrounding file is more important than forced renaming.

### Variables

Use explicit names that reflect the current pipeline, for example:

- `X_full`
- `Y_full_dict`
- `rs_signals`
- `X_stage2`
- `result_data`
- `pred_files`
- `det_peaks`

Avoid generic one-letter naming except for short loop indices.

## File and Artifact Naming Rules

Preserve compatibility with the current file-based workflow.

Preferred artifact style:

### Processed data

- `X_train_unnormalized.npy`
- `normalization_stats.npy`
- `Y_train_{event}.npy`

### Saved models

- `stage1_{event}.h5`
- `final_model_{event}.h5`

### Training logs

- `log_stage1_{event}.csv`
- `log_final_{event}.csv`

### Inference outputs

- `{basename}_RS_Score.xlsx`

### Summary reports

- Excel report files with explicit names derived from folder or experiment name

Do not rename output files casually if downstream scripts already depend on them.

## Logging Rules

Use the current repo-style logging approach:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

Use `logger.info`, `logger.warning`, and `logger.error` where appropriate.

For loops over many files, epochs, batches, representative export tasks, or workbook sheet generation,
use `tqdm` by default unless the loop is truly trivial.

Progress visibility is part of the repository's default UX.
If a script may take noticeable time, the user should be able to see that it is moving forward.

Preferred pattern:

- define a config flag such as `SHOW_PROGRESS = True`
- wrap long-running loops with `tqdm(..., disable=not config.SHOW_PROGRESS)`
- keep progress labels explicit, for example:
  - `Processing files`
  - `Training epochs`
  - `Generating RS scores`
  - `Saving representative figures`
  - `Writing Excel sheets`

Do not silently remove visible progress reporting from existing script-style pipelines unless explicitly requested.

For per-file batch processing, it is acceptable to use `try/except` so one broken file does not stop the entire run.

## Numeric Stability Rules

Protect common failure points.

Always handle safely:

- zero standard deviation
- empty arrays
- empty windows
- no matching GT events
- divide-by-zero in metrics
- invalid file path cases

Use small stabilizers like `1e-8` when numerically appropriate.

Do not let one empty case crash the pipeline.

## Data Compatibility Rules

Whenever one stage changes, check downstream compatibility.

At minimum verify:

- preprocessing output shape still matches training loader assumptions
- window size still matches flatten dimension
- input dimension still matches model input shape
- event names still match across scripts
- saved filenames still match downstream readers
- evaluation column names still match inference outputs

Examples of consistency checks:

- if `WINDOW_SIZE` changes, re-check flattened input dimensions
- if event channels change, re-check `EVENT_NAMES`, saved `Y_train_*`, and output columns
- if output dimension changes, re-check null class index and any class threshold logic
- if RS feature count changes, re-check stage-2 input dimension

## Output Dimension Rules

If the code uses flattened window input, derived dimensions must be kept consistent.

Examples:

- stage-1 input dimension should reflect `WINDOW_SIZE × number_of_base_features`
- stage-2 input dimension should reflect `WINDOW_SIZE × (base_features + RS_features)`

Do not keep outdated dimension numbers after changing upstream feature construction.

If dimensions are written explicitly for clarity, verify they match the actual constructed arrays.

## Comment Style

Keep comments practical and local.

Allowed style:

- short Korean explanations
- English identifiers
- brief mixed technical wording when helpful

Do not over-document trivial lines.  
Do not write long theoretical essays inside production scripts unless explicitly requested.

Prefer comments that explain:

- why a step exists
- why a parameter is chosen
- why a safety condition is needed
- why an output is saved in a certain format

## Editing Philosophy

When modifying this repository:

- make the smallest compatible change first
- preserve the surrounding file style
- preserve downstream compatibility
- avoid unnecessary refactoring
- avoid framework-driven rewrites
- prioritize consistency over generic “best practices”

Default behavior should be:

extend the existing pipeline cleanly, do not reinvent it.

## Things To Avoid By Default

Do not introduce these unless explicitly requested:

- large package refactor
- deep object-oriented redesign
- CLI frameworks
- external config frameworks
- dataclass-heavy redesign
- hidden dependency injection
- replacing Excel outputs with a totally new format
- renaming stable output files casually
- changing signal-processing assumptions in only one script
- changing event labels without checking the full pipeline

## Required Pre-Submit Check

Before finalizing any code change, verify all of the following:

- config constants and actual logic are consistent
- no stale magic numbers remain
- file naming still matches downstream readers
- input dimensions still match built features
- `EVENT_NAMES` are identical across related files
- preprocessing, training, inference, and evaluation still agree on schema
- long-running loops still expose visible progress when appropriate

If there is any mismatch, fix the mismatch before finishing.

## Default Agent Behavior

When asked to add or modify code in this repository, first infer which layer the task belongs to:

- preprocessing
- training
- inference
- evaluation
- analysis / visualization

Then follow the conventions of that layer.

If the requested change would break compatibility across layers, update all affected layers or explicitly note the required follow-up files.

The default assumption is:

maintain the repo’s existing script-first PCG pipeline style unless the user clearly requests a new architecture.

When updating long-running pipeline scripts, default to preserving or adding:

- repo-style logging
- section divider comments
- config-controlled progress bars
- readable top-to-bottom execution flow
