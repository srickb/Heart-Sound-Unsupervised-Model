# Heart-Sound-Unsupervised-Model 수정 요청 문서

## 1. 목적

현재 저장소의 비지도학습 파이프라인을 다음 방향으로 수정하고 싶습니다.

기존 목적:
- beat-level numeric feature를 만들고
- denoising autoencoder로 latent representation을 학습한 뒤
- clustering으로 heart sound beat subtype을 탐색하는 것

새 목적:
- 기존의 feature-based unsupervised learning 구조는 유지한다.
- 하지만 현재 clustering stage의 IDEC + fixed KMeans cluster 구조를 그대로 유지하지 않는다.
- preprocessing 단계에서 **segment-aware feature engineering** 을 강화하고,
- representation learning은 유지하되,
- clustering은 **HDBSCAN 기반 density clustering** 으로 바꾸고,
- interpretation 단계는 fixed cluster count 가정 없이 동작하도록 수정한다.

중요:
- 이 프로젝트는 raw waveform end-to-end 모델로 바꾸려는 것이 아니다.
- 여전히 **cycle마다 fixed-length numeric feature vector** 를 만들고,
  그 feature space에서 representation learning + clustering 을 수행하는 구조를 유지한다.
- 즉 중심은 여전히 tabular feature-based unsupervised learning 이다.

---

## 2. 현재 코드에서 유지할 것

현재 저장소 구조를 기준으로 아래 사항은 유지하고 싶다.

### 2.1 유지할 전체 stage 구조
- `01_preprocess.py`
- `02_train_autoencoder.py`
- `03_extract_embeddings_and_hdbscan.py`
- `04_interpret_clusters.py`

즉 파일 분리는 유지한다.
다만 각 파일 내부 로직은 새 방향에 맞게 수정한다.

### 2.2 유지할 representation learning 방향
- `02_train_autoencoder.py`의 **record-wise split**
- train에만 fit하는 **RobustScaler**
- **denoising autoencoder**
- **early stopping**
- latent export

이 큰 틀은 유지한다.

### 2.3 유지할 beat segmentation 철학
- `01_preprocess.py`에서처럼
  S1/S2 boundary score를 이용해 valid beat를 구성하는 방식은 유지한다.
- 유효한 순서:
  `S1_start -> S1_end -> S2_start -> S2_end -> next S1_start`
- invalid beat는 완전히 버리지 말고 추적 가능하게 남기되,
  learning input에서는 제외한다.

---

## 3. 가장 중요한 수정 방향

핵심 변경점은 두 가지다.

### 3.1 feature engineering 강화
현재 feature는 beat-level global summary 중심인데,
murmur, S3, S4를 탐지하기에는 segment-specific information이 부족하다.

따라서 preprocessing 단계에서
**cycle 내부를 고정된 구간으로 나눈 뒤,
각 구간마다 동일한 템플릿의 feature block을 생성하는 방식**으로 확장한다.

### 3.2 clustering 구조 변경
현재 `03_extract_embeddings_and_hdbscan.py`는 파일명과 달리
실제로는 IDEC + fixed cluster count 기반이다.
이 구조를 바꿔서:

- train split에서 DAE latent를 추출하고
- 그 latent space에 대해 HDBSCAN을 적용하며
- test record를 동일 encoder latent space로 projection 하고
- cluster stability 및 record bias를 확인할 수 있게 바꾼다.

즉 이번 수정에서 **IDEC joint clustering은 제거**한다.
cluster 수를 고정하지 않는다.

---

## 4. preprocessing 단계 수정 요구사항 (`01_preprocess.py`)

### 4.1 핵심 원칙
기존처럼 각 cycle마다 fixed-length feature vector를 만든다.
단, 이제 feature vector는 아래 구조를 반드시 포함해야 한다.

#### A. global cycle block
cycle 전체를 설명하는 feature

#### B. main segment block 4개
- S1
- Systole
- S2
- Diastole

#### C. diastole sub-zone block 3개
- Early diastole = S3 candidate zone
- Mid diastole = neutral / NaN-expected zone
- Late diastole = S4 candidate zone

중요:
- 모든 cycle은 **같은 차원 구조**를 가져야 한다.
- 즉 각 cycle마다 위 segment와 zone에 대해 **동일 개수와 동일 이름 규칙의 parameter** 가 생성되어야 한다.

---

### 4.2 feature naming rule
feature name은 사람이 보고 의미를 바로 이해할 수 있게 명확히 만든다.

예시:
- `global_cycle_length_ms`
- `global_hr_bpm`

- `seg_s1_duration_ms`
- `seg_s1_mean_env`
- `seg_s1_peak_env`
- `seg_s1_energy`

- `seg_sys_duration_ms`
- `seg_sys_mean_env`
- `seg_sys_peak_env`
- `seg_sys_energy`

- `seg_s2_duration_ms`
- `seg_s2_mean_env`
- `seg_s2_peak_env`
- `seg_s2_energy`

- `seg_dia_duration_ms`
- `seg_dia_mean_env`
- `seg_dia_peak_env`
- `seg_dia_energy`

- `zone_ed_mean_env`
- `zone_ed_peak_env`
- `zone_ed_energy`
- `zone_ed_peak_rel_to_s2`

- `zone_md_mean_env`
- `zone_md_peak_env`
- `zone_md_energy`

- `zone_ld_mean_env`
- `zone_ld_peak_env`
- `zone_ld_energy`
- `zone_ld_peak_rel_to_s1`

---

### 4.3 main segment block에 공통으로 넣을 parameter template
S1 / Systole / S2 / Diastole 각각에 대해
동일한 template로 feature를 만든다.

최소 포함 항목:
- duration_ms
- mean_env
- peak_env
- rms
- energy
- energy_ratio_to_cycle
- energy_centroid
- energy_spread_or_entropy
- env_occupancy

즉 예를 들어 S1에는:
- `seg_s1_duration_ms`
- `seg_s1_mean_env`
- `seg_s1_peak_env`
- `seg_s1_rms`
- `seg_s1_energy`
- `seg_s1_energy_ratio_to_cycle`
- `seg_s1_energy_centroid`
- `seg_s1_energy_spread`
- `seg_s1_env_occupancy`

와 같이 만들고,
Systole / S2 / Diastole도 동일한 template를 반복한다.

---

### 4.4 diastole sub-zone 정의
diastole를 단순히 하나의 segment로만 보지 않고,
다음 3개 zone으로 나눈다.

- early diastole: S3 candidate zone
- mid diastole: neutral zone
- late diastole: S4 candidate zone

구간 분할 방식은 absolute ms 고정보다
**diastole 길이에 대한 상대 비율 기반**으로 구현한다.

예:
- early diastole: first 30~35%
- mid diastole: middle 30~40%
- late diastole: last 30~35%

구체 비율은 config로 둔다.
하드코딩하지 말고 수정 가능하게 만든다.

---

### 4.5 diastole sub-zone parameter template
각 zone(ED / MD / LD)에 대해 동일한 template를 만든다.

최소 포함 항목:
- duration_ms
- mean_env
- peak_env
- rms
- energy
- energy_ratio_to_diastole
- peak_timing_relative
- energy_centroid
- energy_spread_or_entropy
- env_occupancy

추가로 S3/S4 관련 상대 크기 파라미터:
- `zone_ed_peak_rel_to_s2`
- `zone_ed_mean_rel_to_s2`
- `zone_ld_peak_rel_to_s1`
- `zone_ld_mean_rel_to_s1`

mid zone은 필요 시:
- `zone_md_peak_rel_to_s1s2_mean`

---

### 4.6 murmur 대응 feature
murmur는 전체 beat global feature만으로는 부족하므로
특히 systole 구간에서 아래 feature가 중요하다.

반드시 추가:
- `seg_sys_energy`
- `seg_sys_energy_ratio_to_cycle`
- `seg_sys_env_occupancy`
- `seg_sys_energy_centroid`
- `seg_sys_energy_spread`

이 feature들은 systolic murmur가
S1/S2 사이에서 에너지가 비정상적으로 남아 있거나 퍼지는 현상을 반영하기 위한 것이다.

---

### 4.7 S3/S4 amplitude band 관련 요구사항
여기서 amplitude band는 hard label rule이 아니다.
즉,
- amplitude가 크다고 무조건 noise 제거
- amplitude가 작다고 무조건 S3/S4 채택
이렇게 하지 않는다.

대신 아래와 같이 **feature로 추가**한다.
- rough amplitude band consistency
- relative peak to S1 or S2
- relative mean envelope to nearby major sound

즉 amplitude band는 classifier-like rule이 아니라
clustering이 참고할 수 있는 추가 parameter로 사용한다.

가능하면 아래처럼 구현:
- `zone_ed_band_distance_to_target`
- `zone_ld_band_distance_to_target`

또는
- `zone_ed_peak_rel_to_s2`
- `zone_ld_peak_rel_to_s1`

위와 같이 상대비로 충분히 표현해도 된다.

---

### 4.8 smoothing and envelope 규칙
S3/S4 및 murmur 관련 feature는
가능하면 raw amplitude 그대로보다
**smoothed absolute envelope** 기반으로 계산한다.

즉,
- `abs(amplitude)` 생성
- moving average 또는 low-pass smoothing
- 해당 envelope 기반으로 mean / peak / occupancy / centroid / spread 계산

smoothing window는 config로 둔다.
예: 10~30 ms 범위 configurable.

---

### 4.9 기존 feature와의 관계
기존 feature를 전부 삭제하지 않는다.
다만 아래 기준으로 재정리한다.

유지:
- time block
- amp block
- shape block
- stat block
- stability/template similarity block

추가:
- main segment block
- diastole sub-zone block
- murmur-aware systole distribution block
- S3/S4 relative amplitude / energy block

주의:
동일한 의미의 중복 feature를 과도하게 늘리지 말고,
처음에는 핵심 feature 위주로 추가한 뒤
후속 단계에서 상관도 및 importance를 보고 줄일 수 있게 한다.

---

### 4.10 output requirements
`01_preprocess.py`는 기존처럼 전처리 산출물을 저장하되,
다음이 보장되어야 한다.

- valid beat feature table 저장
- learning input column list 저장
- feature name group summary 저장
- preprocess summary 저장

추가로:
- 어떤 feature가 global인지
- 어떤 feature가 main segment block인지
- 어떤 feature가 diastole zone block인지
- 어떤 feature가 murmur/S3/S4 관련인지

이걸 식별할 수 있는
`feature_groups.json` 같은 보조 파일도 저장해주면 좋다.

---

## 5. representation learning 단계 수정 요구사항 (`02_train_autoencoder.py`)

### 5.1 유지
현재 구조를 크게 유지한다.
즉:
- valid beat만 사용
- record-wise split
- train only scaler fit
- denoising autoencoder
- early stopping
- latent export

### 5.2 목적 재정의
여기서의 목적은 분류가 아니라
**segment-aware feature vector를 robust latent space로 압축하는 것**이다.

즉 autoencoder는
murmur / S3 / S4 / HR-related structure를 포함할 수 있는
고정 차원 beat feature를 더 안정적인 latent representation으로 만들기 위한 단계다.

### 5.3 architecture
복잡한 sequence model로 바꾸지 않는다.
지금처럼 MLP 기반 DAE를 유지한다.

단, input dimension이 늘어날 것이므로:
- hidden dimension
- latent dimension
- mask ratio
- dropout
등은 config로 정리하고,
기본 latent dimension은 유지하거나 약간 조정 가능하게 한다.

### 5.4 early stopping
현재처럼 validation loss 기반 early stopping 유지.
이 부분은 그대로 둔다.

### 5.5 extra outputs
기존 latent export 외에,
train / val / test split별 reconstruction summary를 저장해
새 feature block 추가 후 reconstruction stability를 비교할 수 있게 한다.

---

## 6. clustering 단계 수정 요구사항 (`03_extract_embeddings_and_hdbscan.py`)

### 6.1 가장 중요한 변경
현재 이 파일은 이름과 달리 IDEC 구조다.
이번 수정에서는 **IDEC를 제거**하고,
실제 HDBSCAN 기반 clustering script로 바꾼다.

즉 아래 요소는 제거 대상이다.
- learnable cluster centers
- fixed `NUM_CLUSTERS`
- KMeans initialization
- joint clustering loss
- target distribution update
- q-soft assignment based final label

### 6.2 새 목표
이 stage는 다음만 담당한다.

1. pretrained encoder 로 latent 추출
2. train latent에 대해 HDBSCAN 수행
3. clustering artifacts 저장
4. test latent를 동일 space에 projection
5. stability / bias analysis에 필요한 정보 저장

### 6.3 train / val / test 사용 규칙
엄격히 아래로 고정한다.

- train: HDBSCAN fit 대상
- validation: representation learning stage에서만 사용
- test: latent projection 및 후속 분석용
- all valid beats: 최종 export용, 단 cluster assignment 방식은 명확히 구분

중요:
HDBSCAN 자체는 기본적으로 inductive predictor가 아니므로,
train cluster를 기준으로 test를 어떻게 다룰지 명확히 구현해야 한다.

### 6.4 test handling rule
가능한 방향:
- HDBSCAN fit은 train latent에만 수행
- test latent는 clustering fit에 포함하지 않는다
- 대신 아래 중 하나를 선택한다.

선호안:
1. HDBSCAN의 approximate prediction 지원 사용 가능하면 사용
2. 어렵다면 test는 cluster label 강제 부여 대신
   - nearest train cluster exemplar distance
   - nearest dense region distance
   - outlier score 유사 지표
   를 저장한다.

중요:
test를 train clustering fit에 섞지 않는다.

### 6.5 output requirements
이 stage는 최소 아래를 저장해야 한다.

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
- outlier score
- membership probability
- cluster persistence
- exemplar / representative sample index

### 6.6 config requirements
HDBSCAN 관련 config를 명시적으로 추가한다.
예:
- min_cluster_size
- min_samples
- cluster_selection_epsilon
- cluster_selection_method
- prediction_data_enabled

이 값들은 코드 상단 config class 에서 한 번에 조절 가능하게 한다.

---

## 7. interpretation 단계 수정 요구사항 (`04_interpret_clusters.py`)

### 7.1 fixed cluster count 가정 제거
현재 코드는 `NUM_CLUSTERS` 와 cluster center shape를 강하게 가정한다.
이걸 제거한다.

즉:
- cluster label 집합을 파일에서 동적으로 읽어야 한다.
- noise label `-1` 을 별도 처리해야 한다.
- cluster 수는 데이터 결과에 따라 달라질 수 있다.

### 7.2 interpretation 목표
이번 interpretation은 아래를 동시에 봐야 한다.

- 어떤 cluster가 있는가
- noise 비율은 얼마인가
- 각 cluster가 어떤 feature block으로 설명되는가
- record-wise bias 가 심한가
- murmur / S3 / S4 / HR-high 와 관련된 feature가 실제로 cluster separation에 기여하는가

### 7.3 feature group aware summary
해석 단계는 단순 feature importance만 보지 말고,
아래 group별 summary를 보여줘야 한다.

- global
- S1 block
- Systole block
- S2 block
- Diastole block
- Early diastole zone
- Mid diastole zone
- Late diastole zone
- murmur-related features
- S3/S4-related relative features
- existing stat / shape / stability features

즉 “어느 cluster가 어떤 block에서 차이가 큰가”를 볼 수 있어야 한다.

### 7.4 record bias check
반드시 추가:
- cluster별 record count
- 특정 record에 cluster가 몰리는지
- record distribution entropy 같은 단순 bias 지표
- noise가 특정 record에 치우치는지

### 7.5 representative cycles
각 cluster별 representative beat는 유지하되,
가능하면 아래를 포함한다.

- 대표 cycle metadata
- 주요 segment feature values
- 주요 zone feature values
- waveform figure
- smoothed envelope figure

즉 S3/S4와 murmur 관련 구간을 사람이 직접 볼 수 있게 한다.

### 7.6 optional diagnostic summary
가능하면 interpretation output에 아래 같은 heuristic summary를 추가한다.
단 hard label로 확정하지 말고 descriptive summary로만 사용한다.

예:
- “systole energy and occupancy가 높아 murmur-like pattern 가능성”
- “early diastole zone energy가 상대적으로 높아 S3-like pattern 가능성”
- “late diastole zone peak가 상대적으로 높아 S4-like pattern 가능성”
- “cycle length가 짧고 HR이 높아 tachycardia-like pattern 가능성”

즉 clinical label 확정이 아니라
feature interpretation summary 수준으로만 작성한다.

---

## 8. 구현 시 주의사항

### 8.1 raw waveform model로 바꾸지 말 것
이번 작업은 feature engineering + DAE + HDBSCAN 구조 개선이지,
waveform end-to-end deep model 전환이 아니다.

### 8.2 feature dimension consistency 최우선
모든 cycle은 동일 차원의 feature vector여야 한다.
segment가 짧거나 zone이 애매해도
NaN-safe or fallback-safe 방식으로 같은 컬럼 구조를 유지해야 한다.

### 8.3 hard clinical labeling 금지
이번 단계는 unsupervised exploration이다.
따라서 code에서
`normal`, `murmur`, `S3`, `S4`를 정답 label처럼 확정하지 않는다.

대신:
- cluster interpretation
- heuristic summary
- feature naming
수준으로 반영한다.

### 8.4 logging 강화
각 stage에서 아래를 충분히 로그로 남긴다.
- valid / invalid beat count
- feature dimension
- missingness
- split counts
- latent dimension
- HDBSCAN cluster count
- noise ratio
- cluster persistence summary
- record bias summary

### 8.5 config class 유지
현재 코드 스타일처럼 각 파일 상단에 config class를 유지한다.
실험 반복을 쉽게 하기 위함이다.

---

## 9. 이번 수정의 핵심 성공 기준

이번 수정이 완료되었다고 볼 조건은 아래와 같다.

1. `01_preprocess.py`가
   기존 beat segmentation을 유지하면서
   segment-aware fixed-length feature vector를 생성한다.

2. 새 feature vector에
   S1 / Systole / S2 / Diastole block,
   그리고 Early/Mid/Late diastole block이 포함된다.

3. `02_train_autoencoder.py`는
   새 feature vector로 DAE를 안정적으로 학습하고
   latent를 저장한다.

4. `03_extract_embeddings_and_hdbscan.py`는
   IDEC가 아니라 실제 HDBSCAN 기반 clustering stage로 동작한다.

5. `04_interpret_clusters.py`는
   fixed cluster count 가정 없이
   dynamic cluster set + noise label 을 해석할 수 있다.

6. interpretation output에서
   murmur-like, S3-like, S4-like, HR-high-like 구조를
   feature block 차이로 읽을 수 있다.

7. record-wise bias를 별도로 점검할 수 있다.

---

## 10. Codex에게 바라는 작업 방식

- 먼저 현재 4개 파일의 구조를 읽고,
  위 요구사항과 현재 구현 차이를 정리해라.
- 그 다음 한 번에 전부 갈아엎지 말고,
  stage별로 안전하게 수정해라.
- 특히 `01_preprocess.py`의 feature schema를 먼저 안정화한 뒤,
  그 schema에 맞춰 `02`, `03`, `04`를 순서대로 수정해라.
- 새 feature 이름과 output artifact 이름은 명확하게 하라.
- 코드 주석은 충분히 달아라.
- 새 실험을 처음 돌리는 사람이 읽어도 구조를 이해할 수 있게 작성하라.

중요:
이번 작업의 핵심은 “학습을 오래 돌리는 것”이 아니라
**murmur / S3 / S4를 반영할 수 있는 feature space를 새로 설계하고,
그 feature space에서 HDBSCAN 기반 비지도 구조를 더 신뢰 가능하게 만드는 것**이다.
