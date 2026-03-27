# Schema Contract 0327

## 목적

이 문서는 [Pipeline Revision 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Pipeline%20Revision%200327.md)와 [Recommend Sequence 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Recommend%20Sequence%200327.md)를 실제 코드로 옮기기 전에, 이후 모든 stage가 공통으로 참조할 **고정 계약(contract)** 을 정리한 문서다.

이 문서의 목적은 다음 두 가지다.

1. `01_preprocess.py`에서 생성할 최종 feature schema를 먼저 고정한다.
2. `02`, `03`, `04`가 참조할 artifact 이름과 역할을 먼저 고정한다.

즉, 이번 수정에서는 먼저 이 문서를 기준으로 schema를 얼리고, 그 다음 코드 구현에 들어간다.

---

## 1. 공통 원칙

- 학습 단위는 raw waveform 전체가 아니라 **valid cycle 1개 = 1 row** 다.
- 모든 cycle은 **동일한 차원의 fixed-length feature vector** 를 가진다.
- invalid cycle도 metadata row는 남기되, learning input에서는 제외한다.
- S3 / S4 / murmur는 hard label이 아니라 **feature space에서 표현되는 구조적 힌트** 로 다룬다.
- feature 계산은 가능하면 raw amplitude보다 **smoothed absolute envelope** 기준으로 수행한다.

---

## 2. Preprocess Output Artifact Contract

`01_preprocess.py`는 아래 artifact를 반드시 생성한다.

- `beat_features_all.csv`
- `beat_features_valid.csv`
- `record_summary.csv`
- `feature_names.json`
- `learning_input_columns.json`
- `feature_groups.json`
- `preprocess_summary.json`
- `preprocess_export.xlsx`

각 파일의 역할은 아래와 같다.

### `beat_features_all.csv`
- valid / invalid cycle 전체 row 저장
- metadata + feature 모두 포함

### `beat_features_valid.csv`
- `valid_flag == 1` 인 row만 저장
- representation learning 입력의 원본 테이블 역할

### `record_summary.csv`
- record 단위 요약
- valid/invalid count, HR summary, missingness summary 등 포함

### `feature_names.json`
- feature column 이름만 순서대로 저장
- metadata column은 제외

### `learning_input_columns.json`
- 실제 모델 입력으로 사용하는 컬럼만 저장
- metadata / boundary index / label 성격 컬럼 제외

### `feature_groups.json`
- 각 feature가 어느 그룹에 속하는지 명시
- 이후 `04_interpret_clusters.py`는 이 파일을 기준으로 group-aware summary 생성

### `preprocess_summary.json`
- 전처리 전체 요약
- valid/invalid count
- feature dimension
- missingness
- group별 feature 개수

### `preprocess_export.xlsx`
- 사람이 바로 확인할 수 있는 Excel export

---

## 3. Metadata Column Contract

아래 metadata column은 `beat_features_all.csv` 와 `beat_features_valid.csv` 에 반드시 존재해야 한다.

- `record_id`
- `source_file`
- `beat_index`
- `cycle_index`
- `valid_flag`
- `invalid_reason`
- `s1_start`
- `s1_end`
- `s2_start`
- `s2_end`
- `next_s1_start`
- `next_s1_end`
- `cycle_start`
- `cycle_end`

legacy compatibility를 위해 아래 alias도 유지한다.

- `s1_on`
- `s1_off`
- `s2_on`
- `s2_off`
- `s1_on_next`

원칙:
- metadata column은 `learning_input_columns.json` 에 포함하지 않는다.
- boundary index는 interpretation / figure generation에는 사용 가능하지만, 학습 입력에는 넣지 않는다.

---

## 4. Feature Group Contract

이번 수정에서 최종 feature group은 아래와 같이 고정한다.

- `global`
- `segment_s1`
- `segment_systole`
- `segment_s2`
- `segment_diastole`
- `zone_early_diastole`
- `zone_mid_diastole`
- `zone_late_diastole`
- `shape`
- `stat`
- `stability`
- `murmur_related`
- `s3s4_relative`

`feature_groups.json` 은 아래와 같은 구조를 가진다.

```json
{
  "global": ["global_cycle_length_ms", "global_hr_bpm"],
  "segment_s1": ["seg_s1_duration_ms", "..."],
  "segment_systole": ["seg_sys_duration_ms", "..."],
  "segment_s2": ["seg_s2_duration_ms", "..."],
  "segment_diastole": ["seg_dia_duration_ms", "..."],
  "zone_early_diastole": ["zone_ed_duration_ms", "..."],
  "zone_mid_diastole": ["zone_md_duration_ms", "..."],
  "zone_late_diastole": ["zone_ld_duration_ms", "..."],
  "shape": ["shape_s1_attack_ratio", "..."],
  "stat": ["stat_cycle_zero_crossing_rate", "..."],
  "stability": ["stab_s1_template_corr", "..."],
  "murmur_related": ["seg_sys_energy", "..."],
  "s3s4_relative": ["zone_ed_peak_rel_to_s2", "..."]
}
```

---

## 5. Global Feature Contract

global block은 cycle 전체를 설명하는 공통 feature다.

이번 수정에서 아래 컬럼을 고정한다.

- `global_cycle_length_ms`
- `global_hr_bpm`
- `global_s1_width_ratio`
- `global_s2_width_ratio`
- `global_systole_ratio`
- `global_diastole_ratio`
- `global_total_energy`
- `global_env_mean`
- `global_env_peak`
- `global_env_rms`

설명:
- 비율 계열은 cycle length 대비 상대 길이
- energy / env 계열은 cycle 전체 기준

---

## 6. Main Segment Feature Template Contract

아래 4개 segment는 동일 template를 사용한다.

- `seg_s1_*`
- `seg_sys_*`
- `seg_s2_*`
- `seg_dia_*`

공통 template는 아래로 고정한다.

- `duration_ms`
- `mean_env`
- `peak_env`
- `rms`
- `energy`
- `energy_ratio_to_cycle`
- `energy_centroid`
- `energy_spread`
- `env_occupancy`

따라서 실제 컬럼은 아래와 같이 생성된다.

### S1
- `seg_s1_duration_ms`
- `seg_s1_mean_env`
- `seg_s1_peak_env`
- `seg_s1_rms`
- `seg_s1_energy`
- `seg_s1_energy_ratio_to_cycle`
- `seg_s1_energy_centroid`
- `seg_s1_energy_spread`
- `seg_s1_env_occupancy`

### Systole
- `seg_sys_duration_ms`
- `seg_sys_mean_env`
- `seg_sys_peak_env`
- `seg_sys_rms`
- `seg_sys_energy`
- `seg_sys_energy_ratio_to_cycle`
- `seg_sys_energy_centroid`
- `seg_sys_energy_spread`
- `seg_sys_env_occupancy`

### S2
- `seg_s2_duration_ms`
- `seg_s2_mean_env`
- `seg_s2_peak_env`
- `seg_s2_rms`
- `seg_s2_energy`
- `seg_s2_energy_ratio_to_cycle`
- `seg_s2_energy_centroid`
- `seg_s2_energy_spread`
- `seg_s2_env_occupancy`

### Diastole
- `seg_dia_duration_ms`
- `seg_dia_mean_env`
- `seg_dia_peak_env`
- `seg_dia_rms`
- `seg_dia_energy`
- `seg_dia_energy_ratio_to_cycle`
- `seg_dia_energy_centroid`
- `seg_dia_energy_spread`
- `seg_dia_env_occupancy`

---

## 7. Diastole Zone Split Contract

diastole는 절대 ms 고정이 아니라 **상대 비율 기반** 으로 3분할한다.

기본 설정은 config로 둔다.

- `EARLY_DIASTOLE_RATIO = (0.00, 0.33)`
- `MID_DIASTOLE_RATIO = (0.33, 0.66)`
- `LATE_DIASTOLE_RATIO = (0.66, 1.00)`

주의:
- 실제 구현은 config 상수로 두고 조정 가능해야 한다.
- 코드 본문에 직접 숫자를 하드코딩하지 않는다.

---

## 8. Diastole Zone Feature Template Contract

아래 3개 zone은 동일 template를 사용한다.

- `zone_ed_*`
- `zone_md_*`
- `zone_ld_*`

공통 template는 아래로 고정한다.

- `duration_ms`
- `mean_env`
- `peak_env`
- `rms`
- `energy`
- `energy_ratio_to_diastole`
- `peak_timing_relative`
- `energy_centroid`
- `energy_spread`
- `env_occupancy`

따라서 zone별 공통 컬럼은 아래와 같다.

### Early Diastole
- `zone_ed_duration_ms`
- `zone_ed_mean_env`
- `zone_ed_peak_env`
- `zone_ed_rms`
- `zone_ed_energy`
- `zone_ed_energy_ratio_to_diastole`
- `zone_ed_peak_timing_relative`
- `zone_ed_energy_centroid`
- `zone_ed_energy_spread`
- `zone_ed_env_occupancy`

### Mid Diastole
- `zone_md_duration_ms`
- `zone_md_mean_env`
- `zone_md_peak_env`
- `zone_md_rms`
- `zone_md_energy`
- `zone_md_energy_ratio_to_diastole`
- `zone_md_peak_timing_relative`
- `zone_md_energy_centroid`
- `zone_md_energy_spread`
- `zone_md_env_occupancy`

### Late Diastole
- `zone_ld_duration_ms`
- `zone_ld_mean_env`
- `zone_ld_peak_env`
- `zone_ld_rms`
- `zone_ld_energy`
- `zone_ld_energy_ratio_to_diastole`
- `zone_ld_peak_timing_relative`
- `zone_ld_energy_centroid`
- `zone_ld_energy_spread`
- `zone_ld_env_occupancy`

---

## 9. S3 / S4 Relative Feature Contract

S3/S4 관련 feature는 hard rule이 아니라 relative descriptive feature로 넣는다.

다음 컬럼을 고정한다.

- `zone_ed_peak_rel_to_s2`
- `zone_ed_mean_rel_to_s2`
- `zone_ld_peak_rel_to_s1`
- `zone_ld_mean_rel_to_s1`
- `zone_md_peak_rel_to_s1s2_mean`

이 컬럼들은 `s3s4_relative` group에 속한다.

---

## 10. Murmur-Oriented Feature Contract

murmur 관련해서는 systole distribution feature를 명시적으로 강조한다.

아래 컬럼은 `segment_systole` 에도 속하지만, 동시에 `murmur_related` group에도 포함한다.

- `seg_sys_energy`
- `seg_sys_energy_ratio_to_cycle`
- `seg_sys_env_occupancy`
- `seg_sys_energy_centroid`
- `seg_sys_energy_spread`

원칙:
- 하나의 feature가 복수 group 해석에 사용될 수 있다.
- `feature_groups.json` 에서는 primary group 기준으로 저장하고,
  `preprocess_summary.json` 에 secondary tag 정보를 추가해도 된다.

---

## 11. Retained Auxiliary Feature Contract

기존 feature를 전부 버리지 않고, 아래 보조 feature는 유지한다.

### Shape
- `shape_s1_attack_ratio`
- `shape_s1_decay_ratio`
- `shape_s1_temporal_centroid_rel`
- `shape_s2_attack_ratio`
- `shape_s2_decay_ratio`
- `shape_s2_temporal_centroid_rel`

### Stat
- `stat_cycle_zero_crossing_rate`
- `stat_cycle_diff_mean_abs`
- `stat_s1_skewness`
- `stat_s1_kurtosis`
- `stat_s2_skewness`
- `stat_s2_kurtosis`

### Stability
- `stab_s1_template_corr`
- `stab_s2_template_corr`

이 보조 feature는 처음부터 과도하게 늘리지 않는다.
이번 iteration에서는 위 목록을 최소 retained set으로 고정한다.

---

## 12. Learning Input Contract

`learning_input_columns.json` 은 아래 원칙으로 생성한다.

### 포함
- `global` group의 numeric feature
- 4개 main segment block feature
- 3개 diastole zone block feature
- `shape` feature
- `stat` feature
- `stability` feature
- `murmur_related` / `s3s4_relative` 관련 numeric feature

### 제외
- metadata column
- boundary index column
- `valid_flag`
- `invalid_reason`
- source file path/name

즉 학습 입력은 **설명 가능한 numeric feature만** 포함한다.

---

## 13. Training Artifact Contract

`02_train_autoencoder.py` 는 아래 artifact를 생성한다.

- `dae_best.pt`
- `scaler.joblib`
- `training_history.csv`
- `reconstruction_summary_by_split.csv`
- `latent_train.csv`
- `latent_val.csv`
- `latent_test.csv`
- `latent_all_valid.csv`
- `split_info.json`

원칙:
- split은 record-wise
- scaler는 train에만 fit
- latent export는 split별로 분리 저장

---

## 14. Clustering Artifact Contract

`03_extract_embeddings_and_hdbscan.py` 는 아래 artifact를 생성한다.

- `latent_train.csv`
- `latent_val.csv`
- `latent_test.csv`
- `hdbscan_labels_train.csv`
- `all_valid_with_latent.csv`
- `clustering_summary.json`
- `cluster_exemplars.csv`
- `cluster_stability_summary.csv`
- `record_distribution_summary.csv`

가능하면 추가:

- `membership_probability`
- `outlier_score`
- `cluster_persistence`

원칙:
- HDBSCAN fit은 train latent에만 수행
- test는 fit에 섞지 않는다

---

## 15. Interpretation Artifact Contract

`04_interpret_clusters.py` 는 아래 artifact를 생성한다.

- `cluster_overview.csv`
- `feature_summary_by_cluster.csv`
- `top_features_per_cluster.csv`
- `feature_group_summary_by_cluster.json`
- `representative_beats.csv`
- `record_cluster_distribution.csv`
- `cluster_interpretation_report.xlsx`
- `cluster_interpretation_summary.json`
- `figures/`

원칙:
- fixed cluster count를 가정하지 않는다
- noise label `-1` 을 별도 처리한다

---

## 16. Step 0 종료 조건

이번 Step 0는 아래 조건을 만족하면 완료로 본다.

1. 최종 feature group 구조가 문서로 고정되었다.
2. `01_preprocess.py` 가 생성해야 할 artifact 이름이 고정되었다.
3. `02`, `03`, `04` 가 앞으로 참조할 입력/출력 계약이 고정되었다.
4. 이후 구현 단계에서 “컬럼명을 어떻게 할지” 때문에 다시 구조를 흔들 필요가 없어진다.

이 문서가 승인되면, 다음 단계는 [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py) 실제 구현으로 넘어간다.
