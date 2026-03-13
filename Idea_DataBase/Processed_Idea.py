"""

Create a standalone script named 01_preprocess.py.

Goal:
Build the preprocessing stage for unsupervised heart sound cycle analysis. The script must load raw heart sound recordings and segmentation annotations, split recordings into heart cycles, extract a fixed-length numeric feature vector from each valid cycle, and save a clean feature table for later unsupervised learning.

Important context:
- This project is NOT primarily using raw waveform sequences as model input.
- The main learning input must be a numeric feature vector extracted from each cycle.
- One heart cycle should be defined primarily as S1(i) onset to S1(i+1) onset, with S2(i) occurring in between.
- Sampling rate is fixed at 4000 Hz.
- Avoid resampling by default.
- The final output of this stage is a tabular feature representation: one row per cycle, one fixed-length feature vector per row.

Critical constraints:
- Do not invent the dataset schema.
- Create a clearly marked adapter section where the user can map their actual files/columns into the expected internal schema.
- If S1/S2 annotations are missing, raise a clear NotImplementedError instead of guessing.
- If annotation format differs, isolate that logic in one adapter function.
- Keep the code readable and heavily commented.

Expected internal schema:
For each recording, expect at minimum:
- recording_id
- subject_id (if unavailable, fall back to recording_id)
- waveform or waveform file path
- sampling_rate
- S1/S2 timing information or sample indices

Segmentation rules:
- Validate cycle ordering.
- Reject cycles with invalid order or clearly abnormal duration using configurable thresholds.
- Save the reason for exclusion.
- Preserve sample_id alignment for all valid cycles.

Signal preprocessing before feature extraction:
- Use conservative filtering only if needed.
- Make filter parameters configurable.
- Avoid aggressive processing that may distort morphology.
- Normalize amplitude at the cycle level if needed for stable feature extraction.
- Do not prepare fixed-length waveform tensors for model input in this script.
- The main output is a feature matrix, not a waveform tensor dataset.

Feature extraction:
Implement a fixed-length numeric feature vector for each valid cycle.

The feature groups should include:
1. Time features
   - cycle duration
   - S1-to-S2 interval
   - S2-to-next-S1 interval
   - relative interval ratios if valid

2. Amplitude/statistical features
   - mean
   - std
   - rms
   - max
   - min
   - max absolute amplitude
   - peak-to-peak amplitude

3. Energy/area features
   - total signal energy
   - absolute area
   - segment-specific energy/area if segment boundaries are unambiguous

4. Peak-based features
   - S1 peak amplitude if definable
   - S2 peak amplitude if definable
   - other peak-related summary values only when clearly supported by the available annotations

5. User-defined RS feature block
   - create a clearly marked placeholder function for RS-based parameters
   - do not invent a proprietary RS definition
   - allow the user to extend this block later

Outputs to save:
outputs/{RUN_NAME}/preprocess/cycle_features.npy
outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
outputs/{RUN_NAME}/preprocess/feature_names.json
outputs/{RUN_NAME}/preprocess/preprocess_summary.json
outputs/{RUN_NAME}/preprocess/feature_table.csv

The metadata file must include at least:
- sample_id
- subject_id
- recording_id
- cycle_start_sample
- cycle_end_sample
- s1_sample
- s2_sample if available
- original_cycle_length
- sampling_rate
- valid_flag
- exclusion_reason

Also save a few simple QC outputs such as:
- cycle duration histogram
- valid vs excluded count
- feature missingness summary
- feature distribution summary

Implementation guidance:
- Use numpy, pandas, scipy.
- Keep the pipeline deterministic and configurable.
- Write helper functions with docstrings and comments.
- Do not use labels.
- Focus on creating a reliable tabular feature dataset for later unsupervised learning.

Done when:
- the script runs as a standalone preprocessing script
- valid cycles are segmented
- one fixed-length feature vector is created per valid cycle
- feature matrix and metadata are saved
- exclusion reasons are tracked
- basic QC summaries are generated

"""