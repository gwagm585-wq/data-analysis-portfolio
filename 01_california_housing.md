# 주택가격 예측 — 기록 왜곡 탐지

> California Housing 회귀 분석 · 3학년 1학기 (2026 상반기) · 개인 과제 · 교내 우수과제 선정 및 발표

`sklearn`의 California Housing 데이터로 주택 중위가격을 예측한 과제입니다.
**과제로 지정된 범위를 마친 뒤, 스스로 「Part 1 추가분석」을 붙여 재설계한 것이 이 프로젝트의 핵심입니다.**

---

## 1. 과제 지정 범위에서 멈춘 지점

과제는 상관계수 상위 4개 변수만으로 Linear → Polynomial → Ridge를 구현하는 것이었습니다.

| 모델 | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.5225 | 0.5043 |
| Polynomial (degree=2) | 0.5500 | 0.5188 |
| Ridge (degree=2, α=10) | 0.5500 | 0.5188 |

과적합은 없었지만(Train−Test 차이 0.03 이내), **설명력 자체가 0.52 수준**에서 멈췄습니다.
회귀의 목적이 변수 간 관계를 설명하고 예측하는 것이라면 이 상태로는 부족하다고 판단해 추가 분석을 진행했습니다.

---

## 2. 발견 — 예측 대상 값이 상한에서 잘려 있었다

기초 통계와 히스토그램을 다시 보는 과정에서 이상한 지점을 찾았습니다.

> `MedHouseVal`의 최댓값이 정확히 5.0에서 막혀 있고, 5.0 근처에만 1,000건 가까이 몰려 있음

이는 값이 큰 관측치를 **모두 5.0으로 기록해 버린 상한 절단(capping)**입니다.
실제로는 4.8인 집과 7.0인 집이 같은 값으로 저장되어 있으므로, 모델은 서로 다른 두 집을 같은 정답으로 학습하게 됩니다.

```
제거 전: 20,640행 → 제거 후: 19,648행 (992건 제거)
```

이상치도 함께 정리했습니다. 왜도가 1을 넘는 변수(`AveRooms`, `AveBedrms`, `AveOccup`, `Population`)에 IQR × 3.0 기준을 적용해 극단값을 제거했습니다.

```
이상치 제거 후: 18,451행 (원본 대비 2,189건 제거)
```

---

## 3. 재설계 — 파생변수 · 로그변환 · 다중공선성

### 파생변수 3종

원본 변수만으로는 알 수 없는 **주거의 질과 밀집도**를 수치화했습니다.

| 변수 | 정의 | 설계 의도 |
|---|---|---|
| `BedroomRatio` | AveBedrms / AveRooms | 침실 비율이 낮을수록 거실·서재 등 다양한 방이 많다 — 방 개수만으로는 알 수 없는 집의 질 |
| `RoomsPerOccup` | AveRooms / AveOccup | 1인당 방 수. `AveRooms`만으로는 대가족의 큰 집인지 1인의 넓은 집인지 구분되지 않는다 |
| `PopPerHousehold` | Population / AveOccup | 블록의 과밀도 |

### 로그변환

왜도가 큰 8개 변수에 `log1p`를 적용했습니다. 최소제곱법은 오차를 제곱해 반영하므로, 한쪽으로 길게 늘어진 분포에서는 회귀선 전체가 극단값 방향으로 끌려갑니다.

### VIF로 중복 제거

VIF 10을 넘는 변수가 다수였지만, `Latitude`/`Longitude`는 지역 가격 정보를 담는 핵심 변수라 유지했습니다.
대신 파생변수와 정보가 겹치는 3개를 제거했습니다.

- `log_AveBedrms` — `log_AveRooms`, `log_BedroomRatio`와 중복
- `log_Population` — `log_PopPerHousehold`로 대체
- `log_AveOccup` — `log_RoomsPerOccup`, `log_PopPerHousehold`와 중복

**최종 8개 변수 / 18,451행**

| 변수 | MedHouseVal과의 상관 |
|---|---|
| log_MedInc | 0.636 |
| **log_RoomsPerOccup** | **0.392** |
| log_AveRooms | 0.262 |
| **log_BedroomRatio** | −0.231 |
| Latitude | −0.141 |
| log_PopPerHousehold | 0.100 |
| HouseAge | 0.064 |
| Longitude | −0.047 |

직접 설계한 `RoomsPerOccup`이 원본 변수보다 높은 상관을 보였습니다.

---

## 4. 모델 비교 — 7개 모델, GridSearchCV

트리 기반 모델(RF·GBM·XGBoost)은 분기 기준으로 학습해 스케일의 영향을 받지 않으므로 원본 데이터를 사용했습니다.

| 모델 | Train R² | **Test R²** | Test RMSE |
|---|---|---|---|
| **XGBoost** | 0.9184 | **0.8308** | **0.3989** |
| Gradient Boosting | 0.8917 | 0.8287 | 0.4013 |
| Random Forest | 0.9360 | 0.7950 | 0.4390 |
| Decision Tree | 0.7760 | 0.6802 | 0.5483 |
| Lasso | 0.6322 | 0.6234 | 0.5950 |
| Ridge | 0.6322 | 0.6234 | 0.5950 |
| Linear Regression | 0.6322 | 0.6234 | 0.5950 |

**XGBoost 최적 파라미터**
`colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=200, subsample=0.8`

읽히는 점 두 가지가 있습니다.

- **선형 3모델이 전부 0.6234로 동일** — 규제 방식을 바꿔도 개선되지 않는다는 것은, 남은 오차가 계수 크기의 문제가 아니라 **비선형 관계**라는 뜻입니다.
- **Random Forest가 과적합이 가장 심함** (Train−Test 0.141). 반면 GBM·XGBoost는 각각 0.063 / 0.088로 안정적이었습니다. 성능이 가장 높은 모델과 과적합이 가장 심한 모델은 다릅니다.

---

## 5. 최종 성능과 검증

```
R²   : 0.8308
RMSE : 0.3989
MAE  : 0.2729
```

- 실제값 대 예측값 산점도에서 점들이 완벽예측선 근처에 분포
- 잔차가 0을 중심으로 −1~+1에 집중 — 한 방향으로 체계적으로 틀리는 편향 없음
- **Feature Importance에서 직접 설계한 `log_RoomsPerOccup`(4위)과 `log_BedroomRatio`(5위)가 원본 변수보다 높은 중요도**

수치가 올라간 것보다, 방 개수보다 **거주 밀도와 주거의 질**이 가격을 더 잘 설명한다는 가설이 모델에서 확인된 점이 의미 있었습니다.

---

## 6. 한계

- 상한 절단된 992건을 **제거**했으므로, 이 모델은 고가 구간(5.0 이상)을 예측하지 못합니다. 검열 회귀(Tobit 등)로 살려 쓰는 방법은 다루지 않았습니다.
- IQR × 3.0이라는 이상치 기준은 관례적 선택이며, 통계적으로 도출한 값이 아닙니다.
- Feature Importance는 기여도이지 인과가 아닙니다. "소득이 높아서 집값이 비싸다"는 인과 주장은 이 분석으로 할 수 없습니다.

---

## 파일

| 파일 | 내용 |
|---|---|
| `notebooks/california_housing.ipynb` | Part 1~3(과제 지정 범위) + **Part 1 추가분석**(자발적 재설계) · 그림 8장 출력 포함 |

과제 본편에는 Breast Cancer 분류(KNN·SVM·Decision Tree) 파트도 포함되어 있습니다.

**데이터**: `sklearn.datasets.fetch_california_housing()` — 코드 실행 시 자동 로드되므로 별도 파일이 필요 없습니다.
