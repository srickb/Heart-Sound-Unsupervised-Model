가장 안전한 순서는 **`01 schema 고정 -> 01 구현 완료 -> 02 적응 -> 03 교체 -> 04 해석 확장 -> end-to-end 검증`** 입니다.  
핵심은 [Pipeline Revision 0327.md](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/docs/Pipeline%20Revision%200327.md)의 요구사항 중에서, 나머지 모든 단계가 결국 `01_preprocess.py`의 출력 컬럼과 artifact 계약에 매달려 있다는 점입니다.

**권장 작업 순서**
1. `Step 0: 계약 먼저 고정`
- 범위: [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py) 수정 전에, 최종 feature schema와 artifact schema를 먼저 문서/상수로 고정
- 여기서 확정할 것:
  - `global / seg_s1 / seg_sys / seg_s2 / seg_dia / zone_ed / zone_md / zone_ld`
  - 각 block별 feature template
  - `feature_groups.json` 구조
  - `learning_input_columns.json` 포함/제외 규칙
- 종료 기준:
  - “최종 컬럼명 목록”과 “feature group 목록”이 더 이상 흔들리지 않음

2. `Step 1: 01_preprocess.py만 먼저 완성`
- 범위: [01_preprocess.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/01_preprocess.py)만 수정
- 여기서 할 것:
  - valid cycle 유지
  - smoothed absolute envelope 기반 공통 feature 함수 작성
  - main segment block 4개 구현
  - diastole relative-ratio zone 3개 구현
  - 기존 feature 중 유지할 것과 새 feature를 함께 export
  - `beat_features_all.csv`, `beat_features_valid.csv`, `learning_input_columns.json`, `feature_names.json`, `feature_groups.json`, `preprocess_summary` 정리
- 종료 기준:
  - 모든 valid/invalid cycle에 대해 동일 차원 row 생성
  - NaN-safe 동작
  - feature group 구분 가능
  - 샘플 몇 개로 수작업 검산 가능
- 여기서 멈추는 것이 첫 번째 안전한 체크포인트입니다.

3. `Step 2: 01 결과를 검증하고 schema freeze`
- 범위: 코드 수정 최소화, 검증 중심
- 여기서 할 것:
  - missingness 확인
  - 너무 중복이 심한 feature 확인
  - 실제로 `murmur / S3 / S4` 관련 feature가 값 분산을 가지는지 점검
  - 학습 입력 컬럼이 numeric/finite한지 확인
- 종료 기준:
  - “이 schema로 02/03/04를 연결해도 되겠다”는 상태
- 이 단계가 중요한 이유:
  - 여기서 schema가 바뀌면 02, 03, 04를 다시 고쳐야 해서 비용이 큽니다.

4. `Step 3: 02_train_autoencoder.py를 schema-agnostic하게 정리`
- 범위: [02_train_autoencoder.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/02_train_autoencoder.py)
- 여기서 할 것:
  - 현재 record-wise split, RobustScaler, DAE, early stopping 유지
  - 입력 차원 자동 반영
  - hidden dim / latent dim / dropout / mask ratio를 config로 정리
  - split별 reconstruction summary 추가
  - latent export를 `train/val/test/all_valid` 기준으로 더 명확히 저장
- 종료 기준:
  - 새 feature schema를 그대로 받아 학습 가능
  - latent와 reconstruction 요약이 안정적으로 저장됨
- 여기까지가 두 번째 안전한 체크포인트입니다.

5. `Step 4: 03_extract_embeddings_and_hdbscan.py를 완전히 HDBSCAN stage로 교체`
- 범위: [03_extract_embeddings_and_hdbscan.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/03_extract_embeddings_and_hdbscan.py)
- 여기서 할 것:
  - IDEC 관련 코드 제거
  - train latent에만 HDBSCAN fit
  - val/test는 fit에 섞지 않음
  - 가능하면 approximate prediction, 아니면 distance/probability/outlier 성격의 보조값 저장
  - `latent_train.csv`, `latent_val.csv`, `latent_test.csv`, `hdbscan_labels_train.csv`, `all_valid_with_latent.csv`, `clustering_summary.json`, `cluster_exemplars.csv`, `cluster_stability_summary.csv`, `record_distribution_summary.csv` 출력
- 종료 기준:
  - cluster 수를 코드가 가정하지 않음
  - noise `-1` 포함 출력 가능
  - test handling rule이 명확함
- 이 단계는 02 출력 형식이 고정된 뒤에만 들어가는 게 안전합니다.

6. `Step 5: 04_interpret_clusters.py를 마지막에 동적 해석기로 교체`
- 범위: [04_interpret_clusters.py](/Users/ms/Desktop/Heart-Sound-Unsupervised-Model/04_interpret_clusters.py)
- 여기서 할 것:
  - fixed cluster count 제거
  - noise label `-1` 처리
  - feature group aware summary
  - record bias summary
  - representative cycles + waveform/envelope figure
  - murmur-like / S3-like / S4-like / HR-high-like heuristic summary 추가
- 종료 기준:
  - cluster 개수가 바뀌어도 동작
  - feature block별 차이를 읽을 수 있음
  - 특정 record 쏠림을 확인 가능

7. `Step 6: 마지막 end-to-end 통합 검증`
- 범위: `01 -> 02 -> 03 -> 04` 전체
- 여기서 할 것:
  - artifact 이름/경로 재검증
  - split consistency 검증
  - latent 차원 일치 검증
  - cluster/noise 비율 로그 확인
  - interpretation output이 실제로 읽히는지 확인
- 종료 기준:
  - 새 feature space와 HDBSCAN 기반 흐름이 한 번에 끝까지 돈다

**왜 이 순서가 최적이냐**
- `01`이 바뀌면 `02`, `03`, `04`가 전부 연쇄적으로 영향을 받습니다.
- `03`은 `02`의 latent output 계약이 먼저 확정돼야 안전합니다.
- `04`는 `03`의 출력 형식이 fixed cluster에서 dynamic cluster로 바뀐 뒤에야 제대로 짤 수 있습니다.
- 그래서 가장 비싼 실수는 `03`이나 `04`를 먼저 손대는 것입니다.

**실제로는 이렇게 끊는 게 제일 좋습니다**
1. `01` 설계 + 구현 + 산출물 고정  
2. `02` 적응 + reconstruction summary 추가  
3. `03` IDEC 제거 후 HDBSCAN 단독화  
4. `04` 동적 해석 + record bias + heuristic summary  
5. 전체 파이프라인 검증

한 줄로 정리하면, **이번 작업의 진짜 핵심 선행조건은 “01의 feature schema를 먼저 얼리는 것”**입니다.  
그 다음부터는 `02 -> 03 -> 04` 순서가 거의 강제라고 보시면 됩니다.

원하시면 다음 답변에서 제가 이 순서를 그대로 반영해서 **“실제 수정 작업 체크리스트”** 형태로 더 잘게 쪼개드릴게요.
