"""

Create a standalone script named 04_interpret_clusters.py.

Goal:
Interpret the HDBSCAN clustering results by comparing cluster-wise numeric feature distributions and identifying the dominant structural characteristics of each cluster. The purpose is to convert cluster labels into understandable heart sound cycle patterns in feature space.

Important context:
- This project is unsupervised.
- Interpretation should be based primarily on the extracted numeric feature vectors.
- Do not rely primarily on raw waveform visualization.
- Cluster interpretation is more important than reporting a single numeric score.
- Use neutral cluster names by default, such as cluster_0, cluster_1, and noise.

Inputs to load:
outputs/{RUN_NAME}/preprocess/cycle_features.npy
outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
outputs/{RUN_NAME}/preprocess/feature_names.json
outputs/{RUN_NAME}/clustering/embeddings.npy
outputs/{RUN_NAME}/clustering/cluster_assignments.csv
outputs/{RUN_NAME}/clustering/clustering_summary.json

Required analyses:
1. Cluster summary table
   - number of samples per cluster
   - proportion per cluster
   - noise proportion

2. Cluster-wise feature statistics
   - mean, std, median, IQR for major features
   - compare timing-related features across clusters
   - compare amplitude/peak/energy/area features across clusters

3. Representative sample selection
   - select representative cycles for each cluster
   - prefer samples near the cluster center in latent space or samples with high membership probability
   - save representative sample_ids

4. Feature importance for interpretation
   - identify features showing the clearest differences across clusters
   - rank features using simple unsupervised group-difference summaries
   - keep the method transparent and easy to inspect

5. Noise analysis
   - analyze the feature profile of noise-assigned samples
   - check whether they tend to show abnormal duration, extreme amplitudes, outlier energy, or suspicious segmentation-related statistics
   - do not overstate conclusions

Important constraints:
- Do not automatically name clusters as “normal”, “murmur”, or other clinical labels unless there is explicit evidence and the code only presents them as tentative hypotheses.
- Use neutral language in outputs.
- Keep the interpretation code reproducible and easy to inspect.
- Preserve sample_id alignment in all merged tables.

Outputs to save:
outputs/{RUN_NAME}/interpretation/cluster_summary.csv
outputs/{RUN_NAME}/interpretation/cluster_feature_stats.csv
outputs/{RUN_NAME}/interpretation/representative_samples.csv
outputs/{RUN_NAME}/interpretation/noise_analysis.csv
outputs/{RUN_NAME}/interpretation/feature_difference_ranking.csv
outputs/{RUN_NAME}/interpretation/cluster_feature_boxplots.png
outputs/{RUN_NAME}/interpretation/cluster_feature_heatmap.png
outputs/{RUN_NAME}/interpretation/interpretation_report.md

The markdown report should summarize:
- how many clusters were found
- noise ratio
- major feature differences between clusters
- which feature groups seem to characterize each cluster
- whether the noise group appears artifact-like or structurally inconsistent
- cautious interpretation notes and limitations

Implementation guidance:
- Use pandas, numpy, matplotlib.
- Add comments explaining what each analysis block is testing.
- Keep the code practical and readable.
- Focus on feature-based cluster interpretability rather than signal reconstruction.

Done when:
- cluster-level tables are saved
- feature-based comparison plots are saved
- representative sample IDs are saved
- noise analysis is saved
- a concise markdown interpretation report is generated

"""