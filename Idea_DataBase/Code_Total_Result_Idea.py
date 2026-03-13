"""
현재 비지도학습 파이프라인(전처리 → autoencoder 학습 → embedding 추출 및 HDBSCAN → cluster 해석)의 실행 결과를 바탕으로,
각 cycle이 최종적으로 어떤 cluster(또는 noise)로 분류되었는지를 Excel로 정리해서 저장하는
새로운 standalone Python 스크립트를 만들어주세요.

중요:
기존 코드의 기능은 수정하지 말고,
기존 결과 파일들을 읽어서 “최종 군집 할당 결과만 정리하는 새 코드”를 추가해주세요.

==================================================
1. 만들고 싶은 새 코드의 목적
==================================================

각 recording file별로,
예를 들어 1_AV_RS_Score 같은 파일 안에 존재하는 각 cycle들이
최종적으로 어떤 cluster 또는 noise로 분류되었는지를
사람이 Excel에서 바로 확인할 수 있게 정리하고 싶습니다.

즉, cycle 단위로 아래 정보를 Excel에 저장하는 최종 추출 코드가 필요합니다.

- Cycle 번호
- Cycle 시작 sample
- Cycle 종료 sample
- 최종 cluster label

예를 들어 HDBSCAN 결과가 -1이면 Excel에는 "Noise"라고 쓰고,
0, 1, 2 같은 label이면 "Cluster 0", "Cluster 1", "Cluster 2" 형태로 쓰도록 해주세요.

==================================================
2. 새 파일 이름 / 구현 방식
==================================================

새 standalone 파일을 예를 들어 아래와 같은 이름으로 만들어주세요.

- 05_export_cycle_cluster_assignments.py

기존 파일들(예: 01_preprocess.py, 02_train_autoencoder.py, 03_extract_embeddings_and_hdbscan.py, 04_interpret_clusters.py)은
가능하면 수정하지 말고,
새 스크립트에서 기존 산출물을 읽어서 정리하는 방식으로 구현해주세요.

==================================================
3. 코드 상단 구조
==================================================

코드 맨 위에는 반드시 class 기반 config 구조를 사용해주세요.
파일 경로와 실행 파라미터를 한 번에 관리할 수 있게 해주세요.

예시 스타일:

class ExportConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "현재 실행 결과 폴더명"

    PREPROCESS_ROOT = OUTPUT_ROOT / RUN_NAME / "preprocess"
    CLUSTERING_ROOT = OUTPUT_ROOT / RUN_NAME / "clustering"
    INTERPRETATION_ROOT = OUTPUT_ROOT / RUN_NAME / "interpretation"

    EXPORT_ROOT = OUTPUT_ROOT / RUN_NAME / "final_exports"
    EXPORT_FILENAME = "cycle_cluster_assignments.xlsx"

    FREEZE_PANES = "A2"
    HEADER_FILL = "1F4E78"
    HEADER_FONT_COLOR = "FFFFFF"
    MAX_COLUMN_WIDTH = 30

    NOISE_LABEL_VALUE = -1
    NOISE_DISPLAY_NAME = "Noise"
    CLUSTER_PREFIX = "Cluster "

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

즉, 향후 내가 run folder만 바꾸면 바로 다시 추출할 수 있게 해주세요.

==================================================
4. 반드시 읽어야 하는 기존 결과
==================================================

새 스크립트는 기존 파이프라인 산출물을 읽어서 merge해야 합니다.

최소한 아래 정보가 필요합니다.

(1) preprocess 단계 결과
- preprocess/cycle_metadata.csv

여기서 각 cycle의 다음 정보를 읽어야 합니다.
- sample_id
- source_file
- cycle_index
- cycle_start_sample
- cycle_end_sample
- valid_flag
- feature_row_index
- waveform_row_index

(2) clustering 단계 결과
- 03_extract_embeddings_and_hdbscan.py에서 저장한 최종 cluster assignment 결과
- 각 valid cycle에 대해 cluster label이 저장된 파일을 찾아 읽어주세요

중요:
현재 프로젝트의 실제 저장 파일명을 먼저 확인해서
어떤 파일에 cluster label이 저장되어 있는지 자동으로 맞춰주세요.

즉, clustering 폴더 안에서
sample_id 또는 feature_row_index 또는 row 순서 기준으로
cycle_metadata.csv와 정확히 매칭되는 최종 cluster label 파일을 찾아 연결해주세요.

가능하면 sample_id 기준 매칭을 우선 사용하고,
불가능하면 feature_row_index 순서를 사용해주세요.

==================================================
5. Excel 결과 구조
==================================================

Excel은 사람이 보기 쉽게 정리해주세요.

가장 중요한 요구사항은 아래와 같습니다.

각 recording file별로 sheet를 하나씩 만들고,
예를 들어 source_file이 "1_AV_RS_Score.xlsx" 라면
sheet 이름은 너무 길지 않게 정리해서
예: "1_AV_RS_Score" 형태로 만들어주세요.

각 sheet의 구조는 아래처럼 해주세요.

A1 = "Cycle Num"
B1 = "Cycle Start"
C1 = "Cycle End"
D1 = "Cluster"

그리고 A2부터는 cycle 순서대로 아래처럼 채워주세요.

- Cycle1
- Cycle2
- Cycle3
- ...

B열에는 해당 cycle의 시작 sample number
C열에는 해당 cycle의 종료 sample number
D열에는 최종 cluster 결과를 넣어주세요.

예:
- Noise
- Cluster 0
- Cluster 1
- Cluster 2

즉, 예를 들어 한 recording에 cycle이 10개 있으면
A2:A11까지 Cycle1 ~ Cycle10,
B2:B11은 시작 sample,
C2:C11은 종료 sample,
D2:D11은 cluster 결과가 들어가야 합니다.

중요:
cycle 순서는 반드시 해당 recording 내에서 cycle_index 오름차순 기준으로 정렬해주세요.

==================================================
6. Cluster label 표시 규칙
==================================================

표시 규칙은 아래처럼 해주세요.

- HDBSCAN label == -1 → "Noise"
- 그 외 정수 label → "Cluster {label}"

예:
- -1 → Noise
- 0 → Cluster 0
- 1 → Cluster 1
- 2 → Cluster 2

만약 cluster label이 없는 cycle이 있다면:
- valid_flag=False 인 경우 "Excluded"
- valid_flag=True 인데 label 매칭 실패 시 "Unmatched"

처럼 명확히 구분해주세요.

==================================================
7. 추가로 넣어도 좋은 시트
==================================================

필수는 recording별 sheet이지만,
추가로 아래 요약 sheet들도 함께 만들어주세요.

(1) Overview 시트
- run_name
- total recordings
- total cycles
- valid cycles
- excluded cycles
- matched cluster rows
- unmatched rows
- noise count
- cluster별 개수 요약

(2) All_Cycles 시트
모든 recording의 cycle 정보를 한 시트에 통합해서 저장해주세요.
이 시트에는 아래 컬럼을 포함해주세요.

- source_file
- recording_id
- sample_id
- cycle_index
- cycle_num_display
- cycle_start_sample
- cycle_end_sample
- valid_flag
- cluster_label_raw
- cluster_display

단, recording별 sheet의 A~D 구조는 반드시 유지해주세요.

==================================================
8. 구현 시 주의사항
==================================================

- 기존 기능은 건드리지 말 것
- 새 파일만 추가할 것
- 현재 결과 파일 구조를 먼저 읽고, 실제 cluster label 저장 파일명을 확인해서 연결할 것
- cycle_metadata.csv와 cluster 결과의 정합성을 반드시 검증할 것
- sample_id, feature_row_index, waveform_row_index 중 가장 안정적인 키를 선택할 것
- cluster 결과 매칭이 몇 개 성공했고 몇 개 실패했는지 로그로 남길 것
- output 폴더 안에 final_exports 폴더를 만들고 Excel을 저장할 것
- Excel 서식은 기존 excel_export_utils.py를 재사용 가능하면 재사용할 것
- 헤더 스타일, freeze pane, 컬럼 너비 자동 조정 등을 적용할 것
- sheet 이름 길이 제한(Excel 31자)을 고려할 것
- source_file 이름이 중복되면 고유하게 sheet 이름을 조정할 것

==================================================
9. 최종 산출물
==================================================

최종적으로 아래가 생성되도록 해주세요.

- outputs/{RUN_NAME}/final_exports/cycle_cluster_assignments.xlsx

이 Excel에는
(1) recording별 sheet
(2) Overview 시트
(3) All_Cycles 시트
가 포함되어야 합니다.

그리고 코드 실행 후에는 로그로 아래를 출력해주세요.

- 읽은 metadata 행 수
- valid cycle 수
- cluster label 매칭 성공 수
- cluster label 매칭 실패 수
- recording별 sheet 개수
- 최종 저장 경로

==================================================
10. 마지막 요청
==================================================

코드를 작성한 뒤에는,
이 새 스크립트가 현재 프로젝트의 어떤 기존 파일을 읽고,
어떤 기준으로 cycle과 cluster를 매칭하는지
간단히 설명도 함께 남겨주세요.
"""