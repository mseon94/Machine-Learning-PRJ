<div align="center">

# 🏨 호텔 예약 취소 예측 및 손실 위험 분석

**Hotel Booking Cancellation Prediction & Loss Risk Analysis**

호텔 예약 데이터를 활용해 **예약 취소 확률을 예측**하고,  
예측 확률과 예약 금액을 결합해 **예상 손실 위험액과 LOW / MEDIUM / HIGH 위험등급**을 제공하는 머신러닝 프로젝트입니다.

단순한 분류를 넘어 호텔 운영자가 **우선 관리해야 할 예약을 판단할 수 있는 의사결정 지표**로 확장했습니다.

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | 머신러닝 기반 호텔 예약 취소 예측 및 손실 위험 분석 |
| 문제 유형 | 이진 분류(Binary Classification) |
| 예측 대상 | `is_canceled` |
| 데이터 규모 | 119,390건 / 36개 컬럼 |
| 최종 데이터 | 118,560건 |
| 최종 입력 변수 | 20개 |
| 최종 모델 | Moderate Random Forest |
| 서비스 | Streamlit 기반 예약 취소 위험 시뮬레이터 및 모델 인사이트 대시보드 |

---

## 2. Tech Stack

<div align="center">

### Language & Environment

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Data & Machine Learning

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Visualization & Service

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Model Tuning & Deployment

![RandomizedSearchCV](https://img.shields.io/badge/RandomizedSearchCV-6A5ACD?style=for-the-badge)
![GridSearchCV](https://img.shields.io/badge/GridSearchCV-8B0000?style=for-the-badge)
![joblib](https://img.shields.io/badge/joblib-4B8BBE?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

---

## 3. 프로젝트 목표

- 결측치·이상치 처리와 파생변수 생성을 통한 학습 데이터 정제
- 데이터 누수 가능성이 있는 변수 제거
- Logistic Regression / Random Forest / Gradient Boosting 성능 비교
- Random Forest 하이퍼파라미터 튜닝을 통한 과적합 완화
- Permutation Importance를 활용한 주요 변수 영향도 분석
- 취소 확률과 예약 금액을 결합한 예상 손실 위험액 산출
- LOW / MEDIUM / HIGH 위험등급 제공
- Streamlit 기반 예측 서비스 구현 및 배포

---

## 4. 데이터셋

### 데이터 출처

- Kaggle: Hotel Booking Dataset
- 원 데이터: Nuno António, Ana de Almeida, Luis Nunes, **Hotel booking demand datasets**, Data in Brief, 2019
- DOI: https://doi.org/10.1016/j.dib.2018.11.126

### 목표 변수

```text
is_canceled
0: 예약 유지
1: 예약 취소
```

### 클래스 분포

- 예약 유지: 약 **62.7%**
- 예약 취소: 약 **37.3%**

특정 클래스가 지나치게 적은 심각한 불균형 데이터는 아니라고 판단했습니다.

---

## 5. 데이터 전처리

### 5.1 주요 전처리

| 처리 항목 | 내용 |
|---|---|
| `children` 결측치 | 4건을 0으로 대체 후 정수형 변환 |
| 투숙객 0명 예약 | 180건 제거 |
| `children`, `babies` 극단값 | 5명 이상 예약 3건 제거 |
| `adr` 이상치 | 음수 1건, 1,000 초과 1건 제거 |
| 0박 예약 | `total_nights == 0` 예약 645건 제거 |
| 국가 범주 | 상위 10개 국가 + `Other`로 축소 |
| 중복 데이터 | 동일 특성의 개별 예약일 가능성을 고려하여 유지 |

최종적으로 **119,390건 → 118,560건**으로 정제했습니다.

### 5.2 데이터 누수 및 불필요 변수 제거

예측 시점에 사용할 수 없거나 결과를 직접·간접적으로 포함할 가능성이 있는 변수는 제거했습니다.

```text
reservation_status
reservation_status_date
assigned_room_type
booking_changes
```

개인정보도 학습 대상에서 제외했습니다.

```text
name
email
phone-number
credit_card
```

### 5.3 파생변수

```python
total_guests = adults + children + babies
has_children = (children + babies > 0)
has_agent = agent.notna()
has_company = company.notna()
total_nights = stays_in_weekend_nights + stays_in_week_nights
has_waiting_list = (days_in_waiting_list > 0)
```

---

## 6. 최종 입력 변수

### 범주형 7개

```text
hotel
arrival_date_month
meal
market_segment
reserved_room_type
customer_type
country_group
```

### 수치형 / 이진형 13개

```text
lead_time
is_repeated_guest
previous_cancellations
previous_bookings_not_canceled
adr
required_car_parking_spaces
total_of_special_requests
total_guests
has_children
has_agent
has_company
total_nights
has_waiting_list
```

총 **20개 입력 변수**를 사용했으며, One-Hot Encoding 이후 모델 입력 차원은 **64개**로 확장됩니다.

---

## 7. 전처리 Pipeline

수치형과 범주형 변수를 하나의 Pipeline으로 통합했습니다.

```python
categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])
```

### Pipeline 적용 목적

- 학습·평가 데이터에 동일한 전처리 기준 적용
- 데이터 누수 방지
- 모델 저장 및 서비스 배포 시 전처리 로직 일관성 확보

---

## 8. 베이스라인 모델 비교

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8020 | 0.7742 | 0.6613 | 0.7133 | 0.8772 |
| **Random Forest** | **0.8780** | **0.8620** | **0.8006** | **0.8302** | **0.9481** |
| Gradient Boosting | 0.8310 | 0.7997 | 0.7289 | 0.7627 | 0.9104 |

Random Forest가 모든 주요 지표에서 가장 높은 성능을 보여 최종 후보 모델로 선정했습니다.

---

## 9. Random Forest 과적합 및 튜닝

초기 Random Forest는 평가 성능은 높았지만 학습 성능이 지나치게 높았습니다.

```text
Train Accuracy : 0.9947
Test Accuracy  : 0.8780
Train ROC-AUC  : 0.9995
Test ROC-AUC   : 0.9481
```

이에 따라 `RandomizedSearchCV`를 이용해 트리 깊이와 노드 분할 조건을 조정했습니다.

### 단계별 결과

| Stage | Train ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9995 | 0.9481 | 0.8620 | 0.8006 | 0.8302 |
| Broad Search | 0.9878 | 0.9463 | 0.8627 | 0.7808 | 0.8197 |
| Regularized | 0.9564 | 0.9363 | 0.8660 | 0.7380 | 0.7969 |
| **Moderate Final** | **0.9732** | **0.9420** | **0.8670** | **0.7573** | **0.8085** |

### 최종 하이퍼파라미터

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="log2",
    random_state=42,
    n_jobs=-1
)
```

최종 모델은 Baseline 대비 과적합을 완화하면서도 높은 평가 성능을 유지하도록 조정했습니다.

---

## 10. 최종 모델 성능

```text
Accuracy  : 0.8663
Precision : 0.8670
Recall    : 0.7573
F1 Score  : 0.8085
ROC-AUC   : 0.9420
```

ROC-AUC를 중심 지표로 사용하여 임계값에 종속되지 않는 **예약 취소 위험 순위화 성능**을 평가했습니다.

---

## 11. Permutation Importance

모델의 예측 결과에 영향을 미치는 주요 변수를 Permutation Importance로 분석했습니다.

### 주요 변수 Top 10

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `country_group` | 0.1449 |
| 2 | `market_segment` | 0.0664 |
| 3 | `customer_type` | 0.0571 |
| 4 | `total_of_special_requests` | 0.0529 |
| 5 | `lead_time` | 0.0514 |
| 6 | `required_car_parking_spaces` | 0.0201 |
| 7 | `previous_cancellations` | 0.0147 |
| 8 | `adr` | 0.0144 |
| 9 | `arrival_date_month` | 0.0099 |
| 10 | `hotel` | 0.0098 |

중요도가 낮은 변수도 다른 변수와 결합해 예측에 기여할 수 있어 단순 중요도만으로 제거하지 않았습니다.

---

## 12. 예상 손실 위험 분석

취소 확률만 제시하는 대신 예약 금액과 결합해 운영 우선순위 지표로 확장했습니다.

### 예상 예약 금액

```text
예약금액 = ADR × 총 숙박일 수
```

### 예상 손실 위험액

```text
예상 손실 위험액 = 취소 예측확률 × 예약금액
```

> 예상 손실 위험액은 실제 회계상 손실액이 아니라 예약별 상대적 관리 우선순위를 위한 위험 지표입니다.

### 위험등급 검증

| 위험등급 | 예약 수 | 실제 취소율 | 평균 예측확률 | 평균 예상손실 |
|---|---:|---:|---:|---:|
| LOW | 9,485 | 10.3% | 13.6% | 22.269 |
| MEDIUM | 4,742 | 53.8% | 52.7% | 124.273 |
| HIGH | 4,743 | **74.5%** | 69.9% | **387.696** |

위험등급이 높아질수록 실제 취소율과 평균 예상 손실이 함께 증가하는 것을 확인했습니다.

---

## 13. Streamlit 서비스

### 예약 취소 위험 시뮬레이터

사용자가 호텔 및 예약 정보를 입력하면 다음 결과를 제공합니다.

- 예약 취소 확률
- 관리 우선순위 점수
- LOW / MEDIUM / HIGH 위험등급
- 예상 예약금액
- 예상 손실 위험액

### 모델 인사이트 대시보드

- 최종 Test ROC-AUC
- Accuracy
- HIGH 위험등급 실제 취소율
- 모델별 성능 비교
- Permutation Importance
- 위험등급별 실제 취소율
- 위험등급별 평균 예상 손실

---

## 14. 프로젝트 구조

```text
.
├── dataset/
│   └── hotel_booking.csv
│
├── model/
│   ├── hotel_canceled.pkl
│   └── risk_thresholds.pkl
│
├── hotel_ml.ipynb
├── app.py
├── requirements.txt
├── model_results.csv
├── rf_overfitting_results.csv
└── validation_loss_risk.csv
```

---

## 15. 실행 방법

### 1) 가상환경 생성

```bash
python -m venv .venv
```

### 2) 가상환경 활성화

Windows

```bash
.venv\Scripts\activate
```

### 3) 패키지 설치

```bash
pip install -r requirements.txt
```

### 4) Streamlit 실행

```bash
streamlit run app.py
```

---

## 16. 모델 저장

전처리와 모델을 포함한 전체 Pipeline을 `joblib`으로 저장했습니다.

```python
import joblib

joblib.dump(
    final_model,
    "model/hotel_canceled.pkl",
    compress=3
)
```

GitHub의 단일 파일 100MB 제한을 해결하기 위해 `compress=3` 옵션을 적용했고, 모델 파일 크기를 약 **44.86MB**까지 줄였습니다.

---

## 17. 주요 트러블슈팅

### Random Forest 과적합

- 문제: Train ROC-AUC 0.9995 / Test ROC-AUC 0.9481
- 해결: RandomizedSearchCV를 통한 트리 복잡도 규제
- 결과: 최종 Train-Test ROC-AUC 차이 약 0.0312

### GitHub 모델 파일 용량 제한

- 문제: 100MB 초과로 `GH001: Large files detected`
- 해결: `joblib.dump(..., compress=3)` 적용
- 결과: 약 44.86MB로 감소 후 배포 완료

### Streamlit 배포 환경 한글 폰트 깨짐

- 문제: Windows 로컬과 Linux 기반 Streamlit Cloud의 폰트 환경 차이
- 해결: 배포 환경에서 사용 가능한 한글 폰트 지정 및 Matplotlib 설정 수정

### GridSearchCV PicklingError

```text
PicklingError: Could not pickle the task to send it to the workers.
```

- `n_jobs=-1` 및 `n_jobs=2`에서 동일 오류 발생
- 멀티프로세싱 worker 직렬화 문제로 판단
- `n_jobs=1`로 변경하여 정상 실행

---

## 18. RandomizedSearchCV vs GridSearchCV

동일한 탐색 범위에서 두 방법을 비교했습니다.

| 항목 | RandomizedSearchCV | GridSearchCV |
|---|---:|---:|
| 탐색 조합 수 | 12 | 48 |
| Best CV ROC-AUC | 0.9335 | **0.9359** |
| Test Accuracy | 0.8613 | **0.8638** |
| Test ROC-AUC | 0.9380 | **0.9403** |
| 탐색 시간 | **217초** | 800초 |

GridSearchCV는 성능이 소폭 높았지만 약 **3.7배**의 탐색 시간이 필요했습니다. 제한된 계산 자원에서 넓은 탐색 공간을 효율적으로 탐색하기 위해 RandomizedSearchCV의 활용 가치가 높다고 판단했습니다.

---

## 19. 한계점 및 향후 발전 방향

### 데이터의 지역·시기적 한계

포르투갈의 2개 호텔에서 2015~2017년에 수집된 데이터이므로 국내 호텔이나 최근 예약 환경에 바로 일반화하기 어렵습니다.

### 예상 손실 산정 방식의 단순화

현재는 다음 식을 사용합니다.

```text
취소확률 × ADR × 숙박일수
```

향후에는 환불 비율, 취소수수료, 객실 운영비, 재판매 가능성 등을 반영할 필요가 있습니다.

### 위험등급 기준

예상 손실 분포의 50%·75% 분위수를 이용하므로 실제 호텔별 운영 정책을 반영하지 못합니다.

### 패턴 변화 대응

현재 모델은 학습 시점 이후의 예약 패턴 변화를 자동으로 반영하지 못하므로 주기적인 성능 모니터링과 재학습 체계가 필요합니다.

---

## 20. 핵심 성과

- 119,390건의 호텔 예약 데이터를 기반으로 취소 예측 모델 구축
- Random Forest 기반 최종 **ROC-AUC 0.9420** 확보
- 데이터 누수 가능 변수를 제거하고 20개 입력 변수로 정제
- Permutation Importance를 통한 주요 예측 변수 분석
- 취소확률을 예상 손실 및 위험등급으로 확장
- HIGH 위험등급 실제 취소율 **74.5%** 확인
- Streamlit 기반 예측 서비스 및 모델 인사이트 대시보드 구현
- GitHub 및 Streamlit Community Cloud 배포 과정의 기술적 문제 해결
