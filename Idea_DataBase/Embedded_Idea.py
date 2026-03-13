"""

Create a standalone script named 03_extract_embeddings_and_hdbscan.py.

Goal:
Load the trained encoder and the preprocessed cycle feature matrix, extract latent embeddings for all valid cycles, apply HDBSCAN in latent space, and save clustering outputs for later interpretation.

Important context:
- This stage operates on tabular feature embeddings.
- Do not retrain the model here.
- Each sample corresponds to one heart sound cycle represented by one numeric feature vector.
- HDBSCAN should discover stable density-based groups in latent feature space.

Inputs to load:
outputs/{RUN_NAME}/preprocess/cycle_features.npy
outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
outputs/{RUN_NAME}/training/encoder.keras

Required behavior:
- Load all valid cycles in the same sample order as preprocessing.
- Extract one latent embedding vector per sample.
- Preserve sample_id alignment.
- Run HDBSCAN on the latent vectors.
- Do not manually set the number of clusters.

HDBSCAN requirements:
- Make key parameters configurable at the top:
  - min_cluster_size
  - min_samples
  - metric
  - cluster_selection_method
  - prediction_data
- Default to prediction_data=True.
- Keep noise label -1 explicit.
- Save soft cluster strength/probability when available.

Outputs to save:
outputs/{RUN_NAME}/clustering/embeddings.npy
outputs/{RUN_NAME}/clustering/embedding_metadata.csv
outputs/{RUN_NAME}/clustering/cluster_assignments.csv
outputs/{RUN_NAME}/clustering/hdbscan_model.joblib
outputs/{RUN_NAME}/clustering/clustering_summary.json
outputs/{RUN_NAME}/clustering/pca_scatter.png

The cluster assignment file must include at least:
- sample_id
- subject_id
- recording_id
- cluster_label
- membership_probability if available

The clustering summary must include at least:
- total number of valid samples
- latent dimension
- number of clusters excluding noise
- number of noise samples
- noise ratio
- cluster sizes
- HDBSCAN parameters used

Visualization:
- Create a simple 2D PCA scatter plot colored by cluster label.
- Use this only as a compact visualization of latent feature space.
- Do not interpret cluster meaning from the 2D scatter alone.

Implementation guidance:
- Keep the code straightforward and reproducible.
- Save enough information for the interpretation stage to join cluster assignments back to feature statistics and metadata.
- Do not use labels or supervised evaluation.
- Do not silently skip failures.

Done when:
- embeddings are extracted for all valid samples
- HDBSCAN labels are assigned
- noise is explicitly reported
- outputs are saved with sample_id alignment
- basic visualization and summaries are generated

"""