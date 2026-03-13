"""
You are helping build research code for unsupervised heart sound cycle analysis.

This project is NOT a raw time-series modeling project.
The primary input to the model is a fixed-length numeric feature vector extracted from each heart sound cycle.

Follow these rules in every task:

1. Use Python and TensorFlow/Keras as the main framework.
2. Write exactly one standalone script for each task, with a clear main() entry point.
3. Immediately below imports, define all editable configuration variables in one place.
4. Keep configuration blocks easy to edit.
5. Do not hard-code file paths inside functions.
6. Sampling rate is 4000 Hz, but the model input is not the raw waveform. The primary model input is a numeric feature table derived from each cycle.
7. Treat this as a tabular feature-learning pipeline, not as a sequence modeling pipeline.
8. Do not design sequence encoders such as 1D CNNs, RNNs, LSTMs, or Transformers unless explicitly requested later.
9. The autoencoder must operate on fixed-length numeric feature vectors.
10. Add clear docstrings and comments, especially:
   - what each function does
   - input/output types
   - input/output shapes
   - meaning of each feature block
11. Keep the code simple, readable, and research-friendly.
12. Never invent dataset columns or file structures that were not explicitly provided.
13. Create a clearly marked dataset adapter section where expected input schema is defined.
14. Fail loudly on missing fields, invalid segmentation, or shape mismatch.
15. Do not use any class labels or supervised targets.
16. Save all artifacts needed by later stages with stable filenames.
17. Preserve row/sample alignment across all saved files using a unique sample_id.
18. At the beginning of your response, briefly list:
   - assumptions
   - expected inputs
   - files you will create
Then write the code.
"""