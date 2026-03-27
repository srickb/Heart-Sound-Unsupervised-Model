# End-to-End Validation 0327

## 목적

이 문서는 `Step 6: end-to-end 통합 검증` 결과를 기록한다.

검증 대상:

- [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py)
- [02_train_autoencoder.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/02_train_autoencoder.py)
- [03_extract_embeddings_and_hdbscan.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/03_extract_embeddings_and_hdbscan.py)
- [04_interpret_clusters.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/04_interpret_clusters.py)

참조 문서:

- [Pipeline Revision 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Pipeline%20Revision%200327.md)
- [Recommend Sequence 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Recommend%20Sequence%200327.md)
- [Schema Contract 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Schema%20Contract%200327.md)
- [Schema Freeze 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Schema%20Freeze%200327.md)

---

## 1. 문법 검증

아래 4개 스크립트에 대해 `python -m py_compile` 검증을 수행했다.

- [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py)
- [02_train_autoencoder.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/02_train_autoencoder.py)
- [03_extract_embeddings_and_hdbscan.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/03_extract_embeddings_and_hdbscan.py)
- [04_interpret_clusters.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/04_interpret_clusters.py)

결과:
- 문법 오류 없음

---

## 2. Stage 간 artifact 계약 검증

다음 계약이 서로 맞물리는지 확인했다.

### `01 -> 02`

`01_preprocess.py` 출력:

- `beat_features_valid.csv`
- `learning_input_columns.json`
- `feature_names.json`

`02_train_autoencoder.py` 입력:

- `BEAT_FEATURES_FILENAME = "beat_features_valid.csv"`
- `LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"`
- `FEATURE_NAMES_FILENAME = "feature_names.json"`

결론:
- 계약 일치

### `02 -> 03`

`02_train_autoencoder.py` 출력:

- `latent_train.csv`
- `latent_val.csv`
- `latent_test.csv`
- `latent_all_valid.csv`
- `split_info.json`

`03_extract_embeddings_and_hdbscan.py` 입력:

- `LATENT_TRAIN_FILENAME = "latent_train.csv"`
- `LATENT_VAL_FILENAME = "latent_val.csv"`
- `LATENT_TEST_FILENAME = "latent_test.csv"`
- `LATENT_ALL_VALID_FILENAME = "latent_all_valid.csv"`
- `SPLIT_INFO_FILENAME = "split_info.json"`

결론:
- 계약 일치

### `03 -> 04`

`03_extract_embeddings_and_hdbscan.py` 출력:

- `hdbscan_labels_train.csv`
- `all_valid_with_latent.csv`
- `clustering_summary.json`
- `cluster_exemplars.csv`
- `cluster_stability_summary.csv`
- `record_distribution_summary.csv`

`04_interpret_clusters.py` 입력:

- `HDBSCAN_LABELS_TRAIN_FILENAME = "hdbscan_labels_train.csv"`
- `ALL_VALID_WITH_LATENT_FILENAME = "all_valid_with_latent.csv"`
- `CLUSTERING_SUMMARY_FILENAME = "clustering_summary.json"`
- `CLUSTER_EXEMPLARS_FILENAME = "cluster_exemplars.csv"`
- `CLUSTER_STABILITY_SUMMARY_FILENAME = "cluster_stability_summary.csv"`
- `RECORD_DISTRIBUTION_SUMMARY_FILENAME = "record_distribution_summary.csv"`

결론:
- 계약 일치

---

## 3. 03/04 synthetic integration sanity check

실제 `hdbscan` 실행은 현재 환경 의존성 부족으로 수행하지 못했지만,
`03`과 `04`의 핵심 helper 함수는 synthetic dataframe으로 연동 검증했다.

검증 내용:

### `03_extract_embeddings_and_hdbscan.py`

확인 항목:

- train assignment frame 생성
- cluster reference table 생성
- nearest cluster distance 계산
- exemplar 선택
- cluster stability summary 생성
- record distribution summary 생성
- clustering summary 생성

결과:

- `train_assign` 에 `cluster_label`, `membership_probability`, `outlier_score`, `is_noise` 정상 생성
- reference cluster row 수: `2`
- `clustering_summary["cluster_labels_excluding_noise"] == [0, 1]`

### `04_interpret_clusters.py`

확인 항목:

- `validate_inputs()`
- `build_clustered_valid_beats()`
- dynamic cluster label 수집
- feature summary
- top features
- feature group summary
- record cluster distribution
- heuristic summary

결과:

- latent columns 검출: `['latent_00', 'latent_01']`
- dynamic cluster labels: `[-1, 0, 1]`
- top feature row 수: `6`
- feature group summary key: `cluster_-1`, `cluster_0`, `cluster_1`
- heuristic summary key: `-1`, `0`, `1`

즉:
- `noise(-1)` 포함 dynamic cluster set 해석이 가능함
- `03 -> 04` 데이터 연결이 기본적으로 성립함

---

## 4. Step 6 중 발견 및 수정한 사항

통합 검증 중 아래 연동 이슈를 발견했다.

문제:
- [04_interpret_clusters.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/04_interpret_clusters.py) 가 `predicted_cluster_label` 컬럼이 항상 존재한다고 가정하고 있었음
- 하지만 [03_extract_embeddings_and_hdbscan.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/03_extract_embeddings_and_hdbscan.py) 에서 approximate prediction이 비활성화되거나 실행 불가한 경우, 해당 컬럼이 없을 수 있음

수정:
- `build_clustered_valid_beats()` 에서 아래 optional column이 없으면 `NaN`으로 생성하도록 fallback 추가
  - `cluster_label`
  - `membership_probability`
  - `outlier_score`
  - `predicted_cluster_label`
  - `predicted_membership_probability`
  - `nearest_train_cluster_label`
  - `nearest_train_cluster_distance`
  - `split_name`

효과:
- `04`는 이제 `03`의 prediction mode 차이에 덜 민감하게 동작함

---

## 5. 현재 남아 있는 제한사항

현재 로컬 환경에는 아래 패키지가 부족하다.

- `torch`
- `sklearn`
- `hdbscan`
- `matplotlib`

따라서 아직 수행하지 못한 검증:

1. 실제 `02_train_autoencoder.py` 학습 실행 검증
2. 실제 `03_extract_embeddings_and_hdbscan.py` HDBSCAN fit 실행 검증
3. 실제 `04_interpret_clusters.py` figure export 검증

즉, 현재까지는:

- 문법 검증 완료
- artifact 계약 검증 완료
- helper-level synthetic integration 검증 완료
- 실제 ML dependency 기반 end-to-end 실행은 미검증

---

## 6. 최종 결론

현재 기준으로 다음은 완료된 상태다.

1. `01`의 frozen feature schema가 `02` 입력과 연결됨
2. `02`의 latent artifact 계약이 `03` 입력과 연결됨
3. `03`의 HDBSCAN artifact 계약이 `04` 입력과 연결됨
4. `04`는 fixed cluster count 없이 dynamic cluster / noise를 해석함
5. 통합 검증 중 발견된 optional prediction column fallback 문제를 수정함

즉, **코드 구조와 stage 간 계약은 현재 일관된 상태**로 볼 수 있다.

남은 것은 실제 의존성 환경을 갖춘 뒤,

- `01 -> 02 -> 03 -> 04`

를 한 번 실제 데이터로 돌려보는 runtime 검증이다.
