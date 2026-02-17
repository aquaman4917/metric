# Brain Control Architecture Pipeline

MDSet 기반 뇌 네트워크 제어 구조 분석 파이프라인 (Lee et al., 2019).

## 구조

```
pipeline/
├── configs/
│   └── default.yaml            # 설정 파일
├── data/
│   └── brainnetome/
│       └── BNA_subregions.xlsx  # subregion 매핑
├── modules/
│   ├── __init__.py
│   ├── loader.py               # 데이터 로딩 + subregion 필터링
│   ├── network.py              # 그래프 연산 (threshold, MDSet, community, PC)
│   ├── metrics.py              # DC, OCA, newDC, DC_PC, OCA_P, OCA_C, Prov_ratio
│   ├── analysis.py             # lifespan trend, age bin, stat tests
│   ├── plotting.py             # 시각화
│   └── utils.py                # config, 경로, 로깅
├── tests/
│   └── test_network.py
├── main.py                     # 실행 진입점
├── requirements.txt
└── .gitignore
```

## 설치

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 사용법

```bash
# 기본 실행
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
```

## Metrics

| Metric | 정의 | 기준 |
|--------|------|------|
| DC | top 5% MD-node control area / 나머지 | >1: centralized, ≤1: distributed |
| OCA | Σ\|C(i)\| / N | >1.5: overlapping, ≤1.5: non-overlapping |
| newDC | DC=1이 되는 FRAC 값 (선형 보간) | 클수록 분산 |
| DC_PC | Connector control area / Provincial area | PC 기반 제어 분포 |
| OCA_P | Provincial MD-nodes overlap | 모듈 내 중첩도 |
| OCA_C | Connector MD-nodes overlap | 모듈 간 중첩도 |
| Prov_ratio | Provincial / 전체 MD-nodes | 논문 ~60% |

## Config 수정

`configs/default.yaml`에서 수정 가능:
- `network.density`: thresholding 비율
- `network.frac`: DC top% 기준
- `metrics.*`: true/false로 metric on/off
- `subregion.nodes`: subregion 노드 인덱스 (null = whole brain)
- `age.bins`: 수동 age bin 정의
- `stats.correction`: 'bonferroni' / 'fdr' / 'none'
