"""

현재 프로젝트의 비지도학습 파이프라인(전처리 → autoencoder 학습 → embedding 추출 및 HDBSCAN → cluster 해석)을 수정하고 싶습니다.

중요한 점은, 이 프로젝트의 목적은 raw waveform 자체를 직접 시계열로 학습하는 것이 아니라,
심장 주기별로 추출 가능한 해석 가능한 수치형 feature를 입력으로 사용하여
비지도학습을 수행하는 것입니다.

따라서 현재 코드에서 사용 중인 feature 설계를 아래 방향으로 재구성해주세요.

==================================================
1. 최우선 목표
==================================================

목표는 각 심장 주기(cycle)에 대해
RS score 기반 landmark feature + time feature + amplitude feature 를 입력으로 사용하여,
cycle 간 구조적 차이를 잘 반영하는 비지도학습용 feature matrix를 만드는 것입니다.

현재 코드처럼 RS score는 단순히 cycle segmentation 용도로만 쓰지 말고,
실제로 학습 입력 feature에 직접 포함되도록 수정해주세요.

또한 현재처럼 정규화된 waveform 통계만 사용하는 것이 아니라,
raw amplitude 기반의 크기 정보도 feature에 반영되도록 해주세요.

==================================================
2. 반드시 반영할 input feature 설계
==================================================

아래 feature들은 feature group별로 명확히 구분해서 구현해주세요.
feature name prefix도 명확히 붙여주세요.
예: rs_, time_, amp_raw_, amp_norm_, ratio_, qc_

------------------------------
A. RS landmark feature
------------------------------

각 cycle 내부에서 다음 4개 event를 사용합니다.

- S1_start
- S1_end
- S2_start
- S2_end

각 event 시점에서 아래 4개 RS score 값을 모두 추출해주세요.

- S1-Start_RS_Score
- S1-End_RS_Score
- S2-Start_RS_Score
- S2-End_RS_Score

즉, event 1개당 4개 값,
총 4개 event × 4개 score = 16차원 feature가 되도록 해주세요.

예시:
- rs_at_s1_start__s1_start_score
- rs_at_s1_start__s1_end_score
- rs_at_s1_start__s2_start_score
- rs_at_s1_start__s2_end_score
- rs_at_s1_end__...
- rs_at_s2_start__...
- rs_at_s2_end__...

이 16개는 반드시 포함해주세요.

추가로 가능하다면 각 event 주변의 local window(예: ±N samples)에서
다음 요약값도 선택적으로 추출할 수 있게 해주세요.
- local max
- local mean
- local std

하지만 최우선은 “각 landmark 시점의 4개 RS score 값 = 총 16차원” 입니다.

------------------------------
B. Time feature
------------------------------

cycle 구조를 설명하는 시간 관련 feature를 포함해주세요.

필수:
- cycle_duration_sec
- s1_duration_sec
- systole_duration_sec
- s2_duration_sec
- diastole_duration_sec
- s1_to_s2_interval_sec
- s2_to_next_s1_interval_sec

추가로 아래 ratio도 포함해주세요.
- s1_ratio
- systole_ratio
- s2_ratio
- diastole_ratio
- s1_to_s2_ratio
- s2_to_next_s1_ratio

가능하다면 아래도 포함해주세요.
- s1_center_to_s2_center_sec
- s1_start_to_s2_start_sec
- s1_end_to_s2_end_sec

핵심은 각 cycle의 시간 구조를 충분히 설명할 수 있도록 하는 것입니다.

------------------------------
C. Amplitude feature
------------------------------

Amplitude feature는 반드시 두 종류로 분리해서 구현해주세요.

(1) raw amplitude 기반 feature
(절대 크기 정보를 유지하기 위한 feature)

(2) normalized amplitude 기반 feature
(shape 비교를 위한 feature)

각 구간별로 아래 영역을 나누어 feature를 계산해주세요.
- whole cycle
- S1 segment
- systole segment
- S2 segment
- diastole segment

각 구간에 대해 가능한 한 아래 feature를 포함해주세요.

필수 후보:
- mean
- std
- max
- min
- max_abs
- peak_to_peak
- rms
- energy
- abs_area

가능하면 추가:
- skewness
- kurtosis

feature prefix 예시:
- amp_raw_cycle_mean
- amp_raw_s1_std
- amp_raw_s2_peak_to_peak
- amp_norm_cycle_rms
- amp_norm_diastole_energy

중요:
현재 코드처럼 cycle mean subtraction + max_abs normalization 이후의 값만 쓰지 말고,
raw amplitude 기반 feature도 반드시 별도로 보존해주세요.

------------------------------
D. Ratio / comparative feature
------------------------------

성능과 해석력을 위해 다음과 같은 비교 feature도 추가해주세요.

예:
- amp_raw_s1_max_abs / amp_raw_s2_max_abs
- amp_raw_systole_energy / amp_raw_diastole_energy
- amp_norm_s1_rms / amp_norm_s2_rms
- s1_duration_sec / s2_duration_sec

단, 0 division 방지를 위한 safe_ratio 처리를 공통 함수로 구현해주세요.

------------------------------
E. QC / reliability feature
------------------------------

비지도학습에서 이상 cycle, annotation 불안정 cycle을 구분하는 데 도움이 되도록
품질 관련 feature도 별도 group으로 넣어주세요.

예:
- num_s1_end_candidates
- num_s2_start_candidates
- num_s2_end_candidates
- event ordering validity flag
- cycle length bounds flag

단, 기존처럼 이런 cycle을 무조건 제외만 하지 말고,
“완전 제외해야 하는 경우”와
“feature로 남겨둘 수 있는 경우”를 구분할 수 있게 구조를 설계해주세요.

==================================================
3. 코드 구조 요구사항
==================================================

코드는 반드시 파일 상단에 class 기반 설정 구조를 두고,
파일 경로 / 파라미터 / 하이퍼파라미터를 한 곳에서 관리할 수 있게 작성해주세요.

예시 스타일은 아래와 같은 구조를 따르되,
현재 비지도학습 프로젝트에 맞게 필요한 항목을 정리해주세요.

class TestConfig:
    RAW_TEST_DATA_FOLDER = r"..."
    MODEL_SAVE_FOLDER = r"..."
    NORMALIZATION_FILE_FOLDER = r"..."
    OUTPUT_FOLDER = r"..."
    OUTPUT_RESULT_FOLDER = r"..."

    SAMPLING_RATE = 4000
    PEAK_FREQUENCIES = [30, 60, 90, 120, 150]
    Q_FACTOR = 10
    ENVELOPE_WINDOW_SIZE = 200
    FRAME_SIZE = 100
    FRAME_OVERLAP = 50

    WINDOW_SIZE = 79
    GARBAGE_CLASS = 79
    OUTPUT = 80
    INPUT_DNN2 = 711

    EVENT_NAMES = ["S1-Start", "S1-End", "S2-Start", "S2-End"]
    PARAM_MIN_PEAK_DIST = 800
    PARAM_RS_PEAK_HEIGHT = 15
    PARAM_RS_PEAK_RATIO = 0.3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

위 예시처럼,
이번 수정 코드도 맨 위에 class config를 두고
다음 항목들을 명시적으로 관리할 수 있게 해주세요.

예:
- PROJECT_ROOT
- DATA_ROOT
- OUTPUT_ROOT
- RUN_NAME
- FILE_GLOB
- SAMPLING_RATE
- FIXED_LENGTH
- MIN_CYCLE_SECONDS
- MAX_CYCLE_SECONDS
- RS 관련 파라미터
- amplitude feature on/off
- normalized feature on/off
- raw feature on/off
- ratio feature on/off
- QC feature on/off
- Excel export 관련 설정

즉, 향후 내가 파라미터를 수정할 때
코드 본문을 건드리지 않고 class config만 수정하면 되도록 해주세요.

==================================================
4. 현재 코드에서 아쉬운 부분과 반드시 개선할 점
==================================================

현재 코드의 문제를 아래처럼 인식하고 반영해주세요.

(1) 현재 RS feature hook이 비어 있음
- custom_rs_feature_hook()가 실질적으로 아무 feature도 추가하지 않음
- 반드시 실제 RS landmark feature가 들어가도록 수정해주세요.

(2) 현재 amplitude 정보가 너무 정규화 중심임
- 현재는 normalized waveform 중심이라 절대 voltage 정보가 약해짐
- raw amplitude 기반 feature를 반드시 별도 포함해주세요.

(3) 현재 cycle exclusion이 너무 엄격함
- missing / multiple event가 있으면 바로 제외하는 구조인데,
  연구 목적상 이상 cycle이나 noisy cycle 자체가 의미 있을 수 있음
- 완전 제외와 보조 feature화 전략을 구분해주세요.

(4) 현재 interpretation 단계가 기존 feature 이름에 강하게 묶여 있음
- feature 이름이 바뀌어도 자동으로 group별 요약이 가능하게 구조를 개선해주세요.
- 예: rs_, time_, amp_raw_, amp_norm_, ratio_, qc_ prefix 기준 자동 요약

(5) 현재 feature engineering 실험 구조가 부족함
- feature group별 ablation이 가능하도록 해주세요.
- 예:
  - RS only
  - Time only
  - Amplitude only
  - RS + Time
  - RS + Time + Amplitude
- 이를 config에서 선택 가능하게 해주세요.

(6) 현재 비지도학습 목적에 비해 설명 가능성이 더 중요함
- feature는 가능한 한 해석 가능한 이름으로 저장해주세요.
- feature_names.json, Excel export, summary json에서 모두 확인 가능해야 합니다.

==================================================
5. 구현 범위
==================================================

우선 수정 우선순위는 아래와 같습니다.

1순위:
- 전처리 코드 수정
- feature engineering 재설계
- feature_names / metadata / excel export 정리

2순위:
- autoencoder 입력 차원이 바뀐 feature matrix에 맞게 자동 반영되도록 수정
- 기존 학습 파이프라인이 깨지지 않게 유지

3순위:
- embedding 추출 및 HDBSCAN 단계가 새 feature에 맞게 정상 동작하도록 유지

4순위:
- interpretation 단계에서 새 feature group별 요약 리포트가 가능하도록 수정

중요:
가능하면 기존 파일 구조와 저장 구조는 유지해주세요.
즉,
- preprocess/
- training/
- clustering/
- interpretation/
이 폴더 구조는 유지하되,
feature 설계만 현재 목적에 맞게 업그레이드해주세요.

==================================================
6. 구현 방식 요구사항
==================================================

- 기존 기능을 불필요하게 깨지 말 것
- 함수 단위로 역할을 분명히 나눌 것
- feature 계산 함수는 group별로 분리할 것
- feature prefix를 강제할 것
- NaN / division by zero / empty segment 처리 명확히 할 것
- 로그를 충분히 남길 것
- Excel export에서 valid cycle view에 feature들이 잘 보이게 할 것
- 코드에 주석을 충분히 달아 내가 추후 수정하기 쉽게 할 것

==================================================
7. 최종 산출물
==================================================

최종적으로 아래가 되도록 수정해주세요.

- 각 cycle별 feature matrix가
  “RS landmark + time + amplitude(raw/normalized) + ratio + QC”
  구조로 저장될 것
- feature_names.json만 봐도 feature 구성이 명확할 것
- autoencoder 입력이 새 feature 차원에 맞게 자동 반영될 것
- cluster 해석 단계에서 feature group별 요약이 가능할 것
- 코드 상단의 class config만 수정하면 경로/파라미터 변경이 가능할 것

그리고 수정 후에는
“현재 코드에서 무엇을 어떻게 바꿨는지”
를 파일별로 요약해서 설명해주세요.

필요하다면 전처리 코드부터 우선 수정하고,
그 다음 학습/클러스터링/해석 단계에 필요한 최소 수정만 이어서 적용해주세요.

"""