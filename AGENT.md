# AGENT.md

이 프로젝트에서 수행할 비지도학습 파이프라인 수정 작업은 반드시 아래 원칙을 따른다.

이 문서는 단순 참고용이 아니라, 코드 수정 시 반드시 따라야 하는 작업 규칙이다.

---

## 1. 프로젝트 목적

이 프로젝트의 목적은 raw waveform 자체를 직접 시계열 모델에 넣어 학습하는 것이 아니다.

이 프로젝트는 **각 심장 주기(beat, cycle)마다 추출한 해석 가능한 수치형 feature**를 입력으로 사용하여,
**beat-level 비지도학습용 feature matrix를 구성하는 것**을 목표로 한다.

즉, 학습 단위는 전체 recording이 아니라 **개별 beat**이다.

모든 구현은 아래 목적에 맞춰야 한다.

- 각 beat를 하나의 row로 표현한다.
- 각 row는 S1/S2 boundary를 기준으로 계산한 feature vector다.
- feature는 해석 가능한 이름을 가져야 한다.
- 비지도학습은 이 feature vector를 기반으로 진행한다.
- raw waveform은 직접 end-to-end 학습 입력으로 사용하지 않는다.
- raw waveform은 feature 계산의 원천 데이터로만 사용한다.

---

## 2. 전체 파이프라인 고정

이 프로젝트의 파이프라인은 아래 구조로 고정한다.

1. beat boundary 기반 전처리
2. beat-level feature extraction
3. valid beat filtering 및 metadata 저장
4. feature matrix export
5. autoencoder 기반 representation learning
6. latent space clustering
7. cluster interpretation

현재 작업 범위에서 가장 우선되는 부분은 다음이다.

- 전처리 코드 수정
- feature engineering 구조 재설계
- feature_names / metadata / export 정리

---

## 3. 구현 단위 고정

코드는 반드시 아래 단위를 기준으로 구성한다.

- `preprocess/`
  - beat validity 검사
  - segment 추출
  - feature 계산
  - feature matrix 저장
  - metadata 저장
  - Excel/CSV export

- `training/`
  - feature matrix 로드
  - scaling
  - autoencoder 입력 차원 자동 반영
  - representation learning

- `clustering/`
  - latent vector 추출
  - clustering 수행
  - cluster label 저장

- `interpretation/`
  - feature group별 요약
  - cluster별 대표 feature 정리
  - 결과 리포트 저장

폴더 구조는 유지한다.
기존 기능을 불필요하게 깨지 말고, 현재 목적에 맞는 방식으로 내부 로직만 업그레이드한다.

---

## 4. feature engineering 원칙

feature engineering은 반드시 **beat-level feature 중심**으로 재구성한다.

다음 원칙을 반드시 따른다.

### 4.1 학습 입력은 beat-level feature만 사용
record-level summary는 beat-level clustering 입력에 넣지 않는다.

### 4.2 feature group은 prefix로 명확히 구분
모든 feature는 prefix를 강제한다.

사용할 prefix는 아래로 고정한다.

- `time_`
- `amp_`
- `shape_`
- `stat_`
- `stab_`
- `qc_`

필요한 경우 세부 분류는 아래처럼 붙인다.

- `amp_s1_...`
- `amp_s2_...`
- `shape_s1_...`
- `shape_s2_...`

### 4.3 feature 이름은 해석 가능해야 한다
압축적이거나 애매한 이름을 사용하지 않는다.

예:
- `time_s1_duration_ms`
- `time_s2_duration_ms`
- `amp_s1_peak_abs`
- `shape_s2_attack_time_ms`
- `stat_s1_kurtosis`
- `stab_s2_template_corr`

### 4.4 raw waveform은 feature 계산에만 사용
raw waveform을 직접 학습 입력 tensor로 넘기지 않는다.

### 4.5 invalid beat 처리 규칙을 명확히 유지
invalid beat는 feature를 NaN으로 저장하고 `valid_flag = 0`으로 저장한다.
학습용 matrix 생성 시 invalid beat는 제외한다.

---

## 5. beat validity 규칙 고정

각 beat는 아래 조건을 만족할 때만 valid로 처리한다.

- `0 <= s1_on < s1_off <= s2_on < s2_off < s1_on_next <= len(x)`

추가로 `cycle_length_ms > 0` 이어야 한다.

`S2 anchor` 기반 feature를 계산할 경우에는 아래 조건도 필요하다.

- `s2_off < s2_on_next <= len(x)`

위 조건을 만족하지 않으면 해당 beat는 invalid다.

invalid beat 처리 방식은 아래로 고정한다.

- 모든 beat row는 저장한다.
- invalid beat는 feature 값을 NaN으로 저장한다.
- invalid beat는 `valid_flag = 0`
- valid beat는 `valid_flag = 1`

---

## 6. 단위 규칙 고정

모든 feature 계산은 아래 단위를 따른다.

- 시간 관련 값: `ms`
- 비율 값: `unitless`
- heart rate: `bpm`
- amplitude / energy: 입력 filtered PCG `x`의 스케일 유지
- 추가 normalization은 feature 저장값에 적용하지 않는다
- template correlation 계산 시 shape 비교를 위한 z-score normalization만 허용한다

---

## 7. 전처리 feature group 고정

전처리 단계에서 계산할 feature group은 아래 다섯 개로 고정한다.

1. `Time_Timing`
2. `Amplitude_Energy`
3. `Shape`
4. `Statistics_Complexity`
5. `Stability`

record-level summary는 별도 시트 또는 별도 파일로 저장한다.

---

## 8. 입력 feature selection 규칙 고정

모델 학습용 입력 feature는 아래 규칙으로 고정한다.

### 포함
- beat-level timing feature
- beat-level amplitude/energy feature
- beat-level shape feature
- beat-level statistics/complexity feature
- beat-level stability feature

### 제외
- absolute recording position만 나타내는 feature
- record-level summary feature
- invalid beat
- label 성격의 metadata column

다음 feature는 학습 입력에서 제외한다.

- `time_s1_center_time_ms`
- `time_s2_center_time_ms`

이 두 값은 기록 내 절대 위치를 반영하므로 clustering input에서 제외한다.
단, export용 metadata에는 저장 가능하다.

또한 아래 summary feature는 학습 입력에 넣지 않는다.

- timing variability summary
- record-level mean/std/cv/rmssd summary

이 값들은 별도 summary 파일로만 저장한다.

---

## 9. scaling 규칙 고정

feature 저장값과 모델 입력값은 구분한다.

### 저장
feature는 정의된 원 단위 그대로 저장한다.

### 학습 입력
모델에 넣기 전, valid beat에 대해서만 scaling을 적용한다.

scaling은 반드시 train split 기준으로 fit 하며, 아래 중 하나를 사용한다.

- `RobustScaler`
- 또는 `StandardScaler`

기본값은 `RobustScaler`로 한다.

scaler는 저장하고, 추후 동일 방식으로 inference에 재사용한다.

---

## 10. config 구조 고정

코드 파일 상단에는 반드시 class 기반 config 구조를 둔다.

예시 이름은 아래 중 하나를 사용한다.

- `ProjectConfig`
- `PreprocessConfig`
- `TrainingConfig`

전처리 관련 코드에는 최소한 아래 항목이 반드시 있어야 한다.

- `PROJECT_ROOT`
- `DATA_ROOT`
- `OUTPUT_ROOT`
- `RUN_NAME`
- `FILE_GLOB`
- `SAMPLING_RATE`
- `MIN_CYCLE_MS`
- `MAX_CYCLE_MS`
- `ENVELOPE_SMOOTH_MS`
- `TEMPLATE_RESAMPLE_LENGTH`
- `EPS`
- `EXPORT_EXCEL`
- `EXPORT_CSV`
- `EXPORT_FEATURE_NAMES_JSON`
- `SAVE_INVALID_ROWS`
- `LEARNING_INPUT_EXCLUDE_COLUMNS`

config 수정만으로 경로와 주요 파라미터를 바꿀 수 있게 작성한다.
코드 본문에 하드코딩하지 않는다.

---

## 11. 함수 구조 고정

feature 계산 코드는 반드시 group별 함수로 분리한다.

예시:

- `validate_beat_boundaries(...)`
- `compute_time_features(...)`
- `compute_amplitude_energy_features(...)`
- `compute_shape_features(...)`
- `compute_statistics_complexity_features(...)`
- `compute_stability_features(...)`
- `build_feature_row(...)`
- `build_feature_dataframe(...)`
- `export_feature_outputs(...)`

모든 함수는 입력과 출력이 명확해야 하며,
NaN / empty segment / division by zero 처리 규칙을 내부에서 명시적으로 처리해야 한다.

---

## 12. logging 규칙

모든 주요 단계에서 logging을 남긴다.

반드시 남길 로그는 아래와 같다.

- 파일 로드 시작/완료
- beat 수
- valid / invalid beat 수
- feature 추출 완료
- export 경로
- 학습 입력 feature 수
- 제외된 column 목록
- scaler 저장 경로

print 대신 `logging`을 사용한다.

---

## 13. export 규칙

전처리 결과는 아래 형태로 저장한다.

### beat-level feature table
각 beat를 한 row로 저장한다.

포함 항목:
- metadata column
- valid_flag
- feature columns

### feature names
- `feature_names.json`

### learning input column list
- 실제 학습에 들어가는 feature 목록을 별도 json으로 저장

### record-level summary
- 별도 csv 또는 xlsx sheet로 저장

### Excel export
Excel export에는 최소한 아래 시트를 만든다.

- `beat_features_all`
- `beat_features_valid`
- `record_summary`

---

## 14. 해석 가능성 우선 원칙

이 프로젝트는 비지도학습 성능만이 아니라 **해석 가능성**이 중요하다.

따라서 아래를 반드시 지킨다.

- feature 이름은 사람이 읽고 의미를 이해할 수 있어야 한다.
- feature group별 prefix를 유지한다.
- cluster interpretation 단계에서 prefix 기반 자동 group summary가 가능해야 한다.
- feature_names.json만 읽어도 전체 구조를 이해할 수 있어야 한다.

---

## 15. 금지 사항

아래 방식은 사용하지 않는다.

- raw waveform 전체를 그대로 autoencoder 입력으로 사용하는 구조
- beat-level clustering에 record-level summary를 넣는 구조
- invalid beat를 조용히 드롭하고 흔적을 남기지 않는 방식
- feature 저장값을 임의로 무단 정규화하는 방식
- feature 이름을 축약하거나 해석 불가능하게 만드는 방식
- 코드 본문 여기저기에 경로와 파라미터를 하드코딩하는 방식

---

## 16. 이번 작업에서 반드시 수행할 내용

이번 수정 작업에서는 아래를 반드시 수행한다.

1. beat validity 로직을 명확히 정리한다.
2. beat-level feature extraction을 현재 정의된 규칙대로 재구성한다.
3. feature group별 계산 함수를 분리한다.
4. feature prefix와 이름을 표준화한다.
5. invalid beat row를 포함한 전체 feature table을 저장한다.
6. valid beat만 모은 학습 입력용 feature table을 별도로 저장한다.
7. feature_names.json과 learning_input_columns.json을 저장한다.
8. 기존 학습 파이프라인이 새 입력 차원에 자동 적응할 수 있게 연결한다.
9. 변경 내용은 파일별로 요약하여 설명한다.

---

## 17. 최종 출력 기대 형태

최종적으로 아래 상태가 되어야 한다.

- beat별 feature matrix가 생성된다.
- valid beat만 모은 학습용 feature matrix가 생성된다.
- feature group이 이름만 봐도 명확하다.
- autoencoder 입력 차원이 새 feature 수에 자동으로 맞춰진다.
- cluster interpretation에서 feature group별 요약이 가능하다.
- 경로/파라미터 변경은 config 수정만으로 가능하다.

이 문서의 규칙을 기준으로 코드 수정 작업을 수행한다.