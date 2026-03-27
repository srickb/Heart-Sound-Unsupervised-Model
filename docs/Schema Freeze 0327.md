# Schema Freeze 0327

## 목적

이 문서는 `Step 2: schema 검증 및 freeze` 결과를 기록한다.

기준 문서:
- [Pipeline Revision 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Pipeline%20Revision%200327.md)
- [Recommend Sequence 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Recommend%20Sequence%200327.md)
- [Schema Contract 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Schema%20Contract%200327.md)

기준 구현:
- [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py)

---

## 검증 범위

이번 Step 2에서는 아래를 검증했다.

1. feature column이 중복 없이 고정되어 있는가
2. metadata column과 feature column이 섞이지 않았는가
3. primary feature group union이 전체 feature set과 정확히 일치하는가
4. secondary group(`murmur_related`)이 실제 feature column을 참조하는가
5. `learning_input_columns.json` 이 현재 feature schema와 일치하는가
6. export artifact가 계약대로 전부 생성되는가
7. valid cycle row가 NaN 없이 finite numeric vector를 가지는가

---

## 검증 결과

### 1. Feature / Metadata 구조

- feature dimension: `95`
- metadata dimension: `19`
- feature column 중복: 없음
- metadata column 중복: 없음
- metadata / feature overlap: 없음

### 2. Feature Group 구조

- primary group union과 `ALL_FEATURE_COLUMNS` 일치: `True`
- primary union 누락 컬럼: 없음
- primary union 초과 컬럼: 없음
- secondary group 누락 컬럼: 없음

group count:

- `global`: 10
- `segment_s1`: 9
- `segment_systole`: 9
- `segment_s2`: 9
- `segment_diastole`: 9
- `zone_early_diastole`: 10
- `zone_mid_diastole`: 10
- `zone_late_diastole`: 10
- `s3s4_relative`: 5
- `shape`: 6
- `stat`: 6
- `stability`: 2
- `murmur_related`: 5

### 3. Learning Input 계약

- `learning_input_columns.json` 길이: `95`
- 현재 구현에서는 `learning_input_columns == ALL_FEATURE_COLUMNS`
- 즉 metadata / boundary index는 학습 입력에서 제외되고, feature set 전체가 학습 입력으로 고정됨

### 4. Export 계약

아래 파일이 모두 생성되는 것을 확인했다.

- `beat_features_all.csv`
- `beat_features_valid.csv`
- `record_summary.csv`
- `feature_names.json`
- `learning_input_columns.json`
- `feature_groups.json`
- `preprocess_summary.json`
- `preprocess_export.xlsx`

또한 JSON 내용 일치도 확인했다.

- `feature_names.json` 내용 일치: `True`
- `learning_input_columns.json` 내용 일치: `True`
- `feature_groups.json` key 일치: `True`
- `preprocess_summary.json.feature_dimension == 95`

### 5. Synthetic sanity check

synthetic 예제 기준:

- total cycles: `2`
- valid cycles: `1`
- invalid cycles: `1`
- valid row의 feature NaN 존재 여부: `False`
- valid row의 finite numeric 여부: `True`

즉, valid cycle은 고정 길이 numeric feature vector로 정상 생성되고,
invalid cycle은 metadata row로 남는 현재 정책이 의도대로 동작한다.

---

## Step 2 중 발견 및 수정한 사항

검증 중 export path가 미리 생성되지 않은 경우 `export_feature_outputs()` 에서 저장이 실패할 수 있는 문제를 발견했다.

수정:
- `export_feature_outputs()` 시작 시 `preprocess_root.mkdir(parents=True, exist_ok=True)` 추가

이 수정으로 export contract가 더 안전해졌다.

---

## Freeze 결론

현재 기준으로 아래를 frozen 상태로 본다.

1. `01_preprocess.py` 의 output feature schema
2. `feature_names.json` / `learning_input_columns.json` / `feature_groups.json` 계약
3. `preprocess_summary.json` / `preprocess_export.xlsx` export 계약
4. valid / invalid cycle 저장 정책

즉, 다음 단계부터는 이 schema를 기준으로

- [02_train_autoencoder.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/02_train_autoencoder.py)
- [03_extract_embeddings_and_hdbscan.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/03_extract_embeddings_and_hdbscan.py)
- [04_interpret_clusters.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/04_interpret_clusters.py)

를 순차적으로 맞춰간다.

다음 단계는 `Step 3: 02_train_autoencoder.py 적응` 이다.
