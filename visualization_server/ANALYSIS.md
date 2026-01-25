# Latent Space 군집 분석 가이드

## 문제: KL loss는 낮은데 군집이 구분되지 않음

### 가능한 원인들:

1. **PCA 정보 손실**
   - 32차원 latent space를 2D로 projection하면서 중요한 정보가 손실됨
   - **해결책**: t-SNE 사용 (이미 적용됨)

2. **Posterior Collapse**
   - KL weight가 너무 높아서 모든 샘플이 비슷한 latent z로 수렴
   - **확인 방법**: Latent z의 분산 확인 (서버 로그에서 확인 가능)
   - **해결책**: KL weight를 낮추거나 (예: 0.1~0.5), β-VAE annealing 사용

3. **Trajectory 분류 문제**
   - 분류 함수가 제대로 작동하지 않아서 실제로는 다른 trajectory들이 같은 라벨을 받음
   - **확인 방법**: 실제 trajectory를 확인하여 분류가 올바른지 검증

4. **모델이 의미있는 특징을 학습하지 못함**
   - VAE가 단순히 reconstruction만 잘하고, 의미있는 representation을 학습하지 못함
   - **해결책**: 
     - 더 깊은 encoder/decoder 사용
     - 더 많은 데이터로 학습
     - 다른 loss 함수 사용 (예: perceptual loss)

### 디버깅 방법:

1. **Latent z 분산 확인**
   - 서버 로그에서 "Latent z std" 확인
   - 만약 std가 매우 작다면 (예: < 0.1) posterior collapse 가능성

2. **t-SNE vs PCA 비교**
   - 현재 t-SNE로 변경됨
   - t-SNE가 더 나은 클러스터링을 보여줄 수 있음

3. **더 많은 샘플 확인**
   - max_samples를 늘려서 더 많은 데이터로 확인

4. **실제 trajectory 확인**
   - 같은 색상의 trajectory들이 실제로 비슷한지 확인
   - 분류 함수가 올바르게 작동하는지 검증

### 개선 제안:

1. **KL weight 조정**
   - 현재: 1.0
   - 제안: 0.1~0.5로 낮춰서 실험

2. **β-VAE Annealing**
   - 학습 초기에는 KL weight를 낮게 시작하고 점진적으로 증가

3. **다른 시각화 방법**
   - UMAP 사용 고려
   - 3D visualization 고려

4. **분류 함수 개선**
   - 더 정교한 trajectory 분류 방법 사용
   - 속도, 곡률 등 추가 특징 사용
