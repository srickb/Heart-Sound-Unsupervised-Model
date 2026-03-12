"""
Create a standalone script named 04_interpret_clusters.py.

Goal:
Interpret the HDBSCAN clustering results by linking cluster assignments back to cycle waveforms, timing structure, and feature statistics. The purpose is to convert cluster labels into understandable heart sound pattern descriptions without overclaiming clinical meaning.

Important context:
- This project is unsupervised.
- Interpretation is more important than reporting a single numeric score.
- The script should help inspect whether clusters reflect meaningful cycle structures or mainly segmentation/noise artifacts.
- Use neutral cluster names by default, such as cluster_0, cluster_1, and noise.

Inputs to load:
outputs/{RUN_NAME}/preprocess/cycle_features.npy
outputs/{RUN_NAME}/preprocess/cycle_waveforms.npy
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
   - mean and median of available timing features
   - mean and std of important amplitude/energy/area features

2. Representative cycle selection
   - choose representative samples for each cluster
   - prefer medoid-like samples in embedding space or highest membership-probability samples
   - save representative sample_ids

3. Waveform interpretation
   - plot representative waveforms for each cluster
   - plot average or median normalized waveform per cluster
   - include a simple variability band or percentile envelope when possible

4. Structural comparison
   - compare available interval features across clusters
   - compare peak/energy/area distributions across clusters
   - highlight clear differences without making unsupported clinical claims

5. Noise analysis
   - analyze samples assigned to noise
   - report whether they tend to show abnormal duration, extreme amplitude, inconsistent structure, or likely segmentation artifacts based on available metadata/features
   - do not overstate conclusions

Important constraints:
- Do not automatically name clusters as “normal”, “murmur”, or other clinical labels unless there is explicit evidence and the code is only presenting them as tentative hypotheses.
- Use neutral language in outputs.
- Keep the interpretation code reproducible and easy to inspect.
- Preserve sample_id alignment in all merged tables.

Outputs to save:
outputs/{RUN_NAME}/interpretation/cluster_summary.csv
outputs/{RUN_NAME}/interpretation/cluster_feature_stats.csv
outputs/{RUN_NAME}/interpretation/representative_samples.csv
outputs/{RUN_NAME}/interpretation/noise_analysis.csv
outputs/{RUN_NAME}/interpretation/cluster_waveform_panels.png
outputs/{RUN_NAME}/interpretation/cluster_feature_boxplots.png
outputs/{RUN_NAME}/interpretation/interpretation_report.md

The markdown report should summarize:
- how many clusters were found
- noise ratio
- major differences between clusters
- waveform tendencies observed in each cluster
- whether the noise group appears artifact-like
- cautious interpretation notes and limitations

Implementation guidance:
- Use pandas/matplotlib for summaries and plots.
- Add comments explaining what each analysis block is testing.
- Keep the code practical and readable.
- Focus on cluster interpretability rather than fancy visualization.

Done when:
- cluster-level tables are saved
- representative waveforms are saved
- noise analysis is saved
- a concise markdown interpretation report is generated
- outputs clearly support manual scientific inspection
"""