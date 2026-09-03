1. 주제 : Predicting Student Health Risk

2. 데이터
- test.csv : the training set, with "health_condition" as target
* id: 샘플별 고유 ID
* health_condition: 기존 모델이 예측한 at-risk, fit, unhealthy 3개 클래스
- test.csv : used to predict the category for "health_condition"
- sample_submission.csv : sample submission file in the correct format


3. 코드 흐름
(1) 테스트 데이터 전처리
- test.csv에서 id를 별도로 보관한 후 feature 데이터에서 제거
- 데이터 타입에 따라 수치형 변수와 범주형 변수를 분리
* 수치형 변수: 각 변수의 중앙값으로 결측치 대체
* 범주형 변수: 'missing'이라는 별도의 값으로 결측치 대체
* 범주형 변수: pd.get_dummies()를 사용하여 One-Hot Encoding 수행
- 이후 모든 feature를 StandardScaler()를 이용하여 표준화
- 최종적으로 295,753개의 테스트 샘플과 25개의 feature로 구성된 feature matrix를 생성

(2) KNN 기반 공간 구조 생성
- 기존 최고 모델의 health_condition 예측값을 다음과 같이 숫자로 변환
* at-risk → 0
* fit → 1
* unhealthy → 2

- 전처리된 테스트 feature에 대해 NearestNeighbors를 적용하여 각 샘플마다 가장 가까운 15개의 이웃을 탐색

nn = NearestNeighbors(
    n_neighbors=15,
    algorithm='auto',
    n_jobs=-1
)



(3) KNN Neighborhood Probability 생성
- 각 샘플의 15개 이웃이 어떤 클래스로 예측되었는지를 확인하고, 클래스별 개수를 계산


(4) 기존 모델 예측과 KNN 확률의 비대칭적 결합
- 기존 모델의 hard label을 One-Hot Encoding하여 확률 형태로 변환
- 이후 기존 모델의 예측에는 85%, KNN 기반 확률에는 15%의 가중치를 부여하여 최종 확률을 계산. 즉, 기존 모델의 예측을 완전히 변경하기보다는 기존 예측을 유지하면서 주변 데이터의 feature 구조를 일부 반영하는 방식으로 설계

- 최종적으로 가장 높은 확률을 가진 클래스를 선택하여 최종 health_condition을 결정함.

(5) 결과 확인 및 제출 파일 생성
- 기존 모델의 hard label과 KNN smoothing 이후의 최종 label을 비교하여 변경된 샘플의 개수를 확인
- 최종적으로 id와 health_condition을 결합하여 submission.csv로 저장


3-1. 주요 코드
1. KNN 모델 생성 및 이웃 탐색
nn = NearestNeighbors(n_neighbors=15, algorithm='auto', n_jobs=-1)
nn.fit(X_test_scaled)

distances, indices = nn.kneighbors(X_test_scaled)


2. KNN 기반 확률 생성 및 기존 예측과 KNN 확률의 Blending
for i in range(N):
    neighbors = indices[i]
    neighbor_labels = y_hard[neighbors]
    
    counts = np.bincount(neighbor_labels, minlength=3)
    knn_probs[i] = counts / np.sum(counts)

y_one_hot = np.zeros((N, 3))
y_one_hot[np.arange(N), y_hard] = 1.0

blend_weight_top = 0.85
blend_weight_knn = 0.15

blended_probs = (y_one_hot * blend_weight_top) + (knn_probs * blend_weight_knn)
final_preds = np.argmax(blended_probs, axis=1)


4. 새롭게 알게 된 내용 / 어려운 내용 / 배울 점
- KNN을 후처리 단계에서 활용하는 방법
: KNN은 일반적으로 새로운 데이터를 분류하거나 예측하는 모델로 생각했지만, 이번 과정에서는 기존 모델의 예측 결과를 보정하는 후처리 방법으로 활용할 수 있다는 점을 알게 됨
: 특히 feature 공간에서 가까운 데이터들은 비슷한 특성을 가지고 있을 가능성이 있다는 점을 활용하여, 기존 모델의 예측값 주변에 존재하는 다른 샘플들의 label 분포를 확인할 수 있었음


- Hard Label과 Probability의 결합
: 기존 모델은 at-risk, fit, unhealthy와 같은 하나의 hard label만 제공하지만, KNN을 이용하면 주변 이웃의 비율을 통해 확률 분포를 만들 수 있음
: 이를 기존 모델의 One-Hot 확률과 결합함으로써 기존 모델의 강한 예측을 유지하면서 feature 공간의 local information을 반영할 수 있다는 것을 학습함
