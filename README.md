# Brain Control Architecture Pipeline (Local Version)

MDSet 기반 뇌 네트워크 제어 구조 분석 파이프라인 (Lee et al., 2019)의 로컬 리팩터링 버전입니다. 본 리포지토리는 기존 [aquaman4917/metric](https://github.com/aquaman4917/metric) 리포지토리를 기반으로 하며, 일부 지표를 확장하거나 수정한 버전을 포함합니다. 특히 **Normalized Overlap Redundancy (NOR)** 지표를 새로 추가해 MDS 크기에 대한 의존성을 줄였습니다.

## 구조

metric_repo/
├── configs/
│ └── default.yaml # 설정 파일 (metrics 플래그 포함)
├── modules/
│ ├── init.py
│ ├── loader.py # 데이터 로딩 + subregion 필터링 (원본 레포에는 존재하나 이 로컬 리포에는 포함되지 않음)
│ ├── network.py # 그래프 연산 (threshold, MDSet, community, PC)
│ ├── metrics.py # DC, OCA, newDC, DC_PC, OCA_P, OCA_C, Prov_ratio, NOR 등
│ └── utils.py # config, 경로, 로깅 (필요시 추가)
└── configs/
└── default.yaml # 기본 설정

bash
코드 복사

## 설치

이 로컬 환경에서는 Python 패키지 의존성이 이미 설치되어 있습니다. 독립적으로 실행하려면 다음을 수행하세요:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
사용법
bash
코드 복사
# 기본 실행 (예: HCP_merged.mat 사용)
python main.py

# 파라미터 오버라이드
python main.py --density 0.30 --frac 0.10

# 다른 데이터 (path / 파일명 / 이름 모두 가능)
python main.py --input data/brainnetome/HCP_merged.mat
python main.py --input HCP_merged

# density × frac 일괄 실행
python main.py --batch-sweep

# 디버깅 (병렬 끄기)
python main.py --no-parallel
Metrics
Metric	정의	기준/해석
DC	top 5 % MD-node control area / 나머지	> 1이면 중앙집중적(centralized), ≤ 1이면 분산(distributed)
OCA	Σ |C(i)| / N	> 1.5이면 overlapping, ≤ 1.5이면 non-overlapping
newDC	DC=1이 되는 FRAC 값 (선형 보간)	클수록 제어가 분산됨
DC_PC	Connector control area / Provincial area	PC 기반 제어 분포 비율
OCA_P	Provincial MD‑nodes overlap	모듈 내 중첩도
OCA_C	Connector MD‑nodes overlap	모듈 간 중첩도
Prov_ratio	Provincial / 전체 MD‑nodes	논문에서 약 60 %
NOR	(OCA − 1) / (M − 1) (M = 	MDSet

NOR 지표는 OCA에서 MDS_size(= M) 의존성을 제거한 지표입니다. M ≤ 1일 때는 정의되지 않아 NaN을 반환하며, 값이 0이면 각 노드가 한 명의 MD-node에게만 지배되는 상태, 값이 커질수록 여러 MD-node에 의해 중첩 지배되는 정도가 높습니다.

Config 수정
configs/default.yaml에서 다음을 수정할 수 있습니다:

network.density: proportional thresholding 비율

network.frac: DC에서 top degree 노드 비율

metrics.*: true/false로 metric on/off. 새로 추가된 NOR 항목도 여기에서 켜거나 끌 수 있습니다.

subregion.nodes: subregion 노드 인덱스 (null = 전체 뇌)

age.bins: 수동 age bin 정의

stats.correction: 다중 비교 수정 방법 ('bonferroni' / 'fdr' / 'none')

변경 사항 요약
이 로컬 리포지토리는 기존 레포지토리를 기반으로 다음과 같은 확장을 포함합니다:

Normalized Overlap Redundancy (NOR) metric 추가. MDS_size에 의존적인 OCA를 보정하여 중첩 redundancy를 직접 측정합니다. 구현은 modules/metrics.py에서 NOR 함수를 참조하세요.

configs/default.yaml에 NOR: true 플래그 추가로 기본적으로 NOR를 계산하도록 설정했습니다.

README 업데이트: 새 지표 정의와 해석을 표에 추가하고, Config 항목에 NOR 플래그를 명시했습니다.

기타 기존 코드 구조나 사용법은 원본과 동일합니다.

yaml
코드 복사

---

위 세 파일(`modules/metrics.py`, `configs/default.yaml`, `README.md`)이 이번 수정에서 변경된 전체 내용입니다. 