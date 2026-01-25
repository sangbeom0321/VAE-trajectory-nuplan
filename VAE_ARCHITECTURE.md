# VAE-Planner Architecture Documentation

VAE-Planner의 인코더-디코더 구조 상세 설명

## 📋 개요

VAE-Planner는 Variational Autoencoder (VAE)를 사용하여 자율주행 차량의 미래 궤적을 예측하는 모델입니다. 8초간의 궤적 데이터를 32차원의 latent space로 압축하고, 이를 통해 다양한 주행 시나리오를 표현합니다.

## 🏗️ 전체 아키텍처

```
Input Trajectory (80 timesteps × 2D)
    ↓
Flatten: (batch, 80, 2) → (batch, 160)
    ↓
┌─────────────────────────────────┐
│         Encoder                 │
│  160 → 512 → 256 → 128          │
│         ↓                        │
│    μ, logvar (32차원)           │
└─────────────────────────────────┘
    ↓
Reparameterization Trick
    ↓
Latent z (32차원)
    ↓
┌─────────────────────────────────┐
│         Decoder                 │
│  32 → 128 → 256 → 512 → 160     │
└─────────────────────────────────┘
    ↓
Unflatten: (batch, 160) → (batch, 80, 2)
    ↓
Reconstructed Trajectory (80 timesteps × 2D)
```

## 🔷 Encoder 구조

### 입력
- **Shape**: `(batch, 160)`
- **내용**: 80 타임스텝 × 2차원 (x, y) 좌표를 flatten한 벡터
- **정규화**: 데이터셋의 평균과 표준편차로 정규화됨

### 아키텍처

```python
VAEEncoder(
    input_dim=160,
    latent_dim=32,
    hidden_dims=[512, 256, 128]
)
```

### 레이어 구성

1. **Input Layer**: 160차원
2. **Hidden Layer 1**: 160 → 512
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
3. **Hidden Layer 2**: 512 → 256
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
4. **Hidden Layer 3**: 256 → 128
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
5. **Latent Distribution Heads**:
   - **μ (mean)**: 128 → 32 (Linear)
   - **logvar (log variance)**: 128 → 32 (Linear)

### 출력
- **μ**: `(batch, 32)` - Latent distribution의 평균
- **logvar**: `(batch, 32)` - Latent distribution의 로그 분산

### 특징
- **Batch Normalization**: 각 hidden layer에 적용되어 학습 안정화
- **Dropout**: 0.1 비율로 과적합 방지
- **대칭 구조**: Decoder와 대칭적인 구조 (160→512→256→128→32)

## 🔶 Decoder 구조

### 입력
- **Shape**: `(batch, 32)`
- **내용**: Reparameterization trick으로 샘플링된 latent variable z
- **분포**: N(μ, σ²)에서 샘플링 (학습 시) 또는 N(0, I)에서 샘플링 (추론 시)

### 아키텍처

```python
VAEDecoder(
    latent_dim=32,
    output_dim=160,
    hidden_dims=[128, 256, 512]
)
```

### 레이어 구성

1. **Input Layer**: 32차원 (latent z)
2. **Hidden Layer 1**: 32 → 128
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
3. **Hidden Layer 2**: 128 → 256
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
4. **Hidden Layer 3**: 256 → 512
   - Linear Transformation
   - Batch Normalization
   - ReLU Activation
   - Dropout (0.1)
5. **Output Layer**: 512 → 160
   - Linear Transformation
   - **No activation** (선형 출력)
   - **No normalization** (원본 스케일 복원)

### 출력
- **Shape**: `(batch, 160)`
- **내용**: 복원된 궤적 벡터 (80 타임스텝 × 2차원)
- **후처리**: Reshape하여 `(batch, 80, 2)` 형태로 변환

### 특징
- **대칭 구조**: Encoder와 대칭적인 구조 (32→128→256→512→160)
- **출력 레이어**: 활성화 함수 없음 (선형 출력으로 원본 스케일 유지)
- **정규화 없음**: 출력 레이어에는 Batch Normalization 적용 안 함

## 🔄 Reparameterization Trick

VAE의 핵심 기술로, 샘플링 과정을 미분 가능하게 만듭니다.

### 수식

```
z = μ + σ × ε
where:
  σ = exp(0.5 × logvar)
  ε ~ N(0, I)
```

### 구현

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### 특징
- **미분 가능**: 역전파가 가능하도록 샘플링
- **확률적**: 각 forward pass마다 다른 z 샘플링
- **학습 안정화**: KL divergence와 함께 사용하여 latent space 정규화

## 📊 데이터 흐름

### 학습 시 (Training)

```
Input: (batch, 80, 2)
  ↓ flatten
(batch, 160)
  ↓ Encoder
μ: (batch, 32), logvar: (batch, 32)
  ↓ Reparameterization
z: (batch, 32)
  ↓ Decoder
(batch, 160)
  ↓ unflatten
Output: (batch, 80, 2)
```

### 추론 시 (Inference)

#### 1. Reconstruction (복원)
```
Input Trajectory → Encoder → z → Decoder → Reconstructed Trajectory
```

#### 2. Sampling (샘플링)
```
Prior z ~ N(0, I) → Decoder → Generated Trajectory
```

## 🎯 주요 파라미터

### 입력 차원
- **future_horizon**: 80 (8초 × 10Hz)
- **future_dim**: 2 ([x, y])
- **input_dim**: 160 (80 × 2)

### Latent Space
- **latent_dim**: 32
- **분포**: N(μ, σ²) (학습 시), N(0, I) (추론 시)

### Hidden Layers
- **Encoder**: [512, 256, 128]
- **Decoder**: [128, 256, 512]
- **총 파라미터 수**: 약 500K (모델에 따라 다름)

## 🔧 구현 세부사항

### Flatten/Unflatten

```python
# Flatten: (batch, 80, 2) → (batch, 160)
flattened = trajectory.reshape(batch_size, -1)

# Unflatten: (batch, 160) → (batch, 80, 2)
trajectory = flattened.reshape(batch_size, 80, 2)
```

### Forward Pass

```python
# 1. Flatten input
flattened_input = self._flatten_input(ego_future_trajectory)

# 2. Encode
mu, logvar = self.vae_encoder(flattened_input)

# 3. Reparameterize
z = reparameterize(mu, logvar)

# 4. Decode
reconstructed_flat = self.vae_decoder(z)

# 5. Unflatten
reconstructed = self._unflatten_output(reconstructed_flat)
```

## 📈 학습 목표

### Loss Function

```
Total Loss = Reconstruction Loss + β × KL Divergence Loss
```

1. **Reconstruction Loss**: MSE between input and reconstructed trajectory
   ```
   L_recon = MSE(x, x_recon)
   ```

2. **KL Divergence Loss**: Regularization term
   ```
   L_KL = KL(q(z|x) || N(0, I))
        = 0.5 × Σ(exp(logvar) + μ² - 1 - logvar)
   ```

3. **β-VAE Annealing**: β 값을 점진적으로 증가
   ```
   β(epoch) = start_weight + (end_weight - start_weight) × progress
   ```

## 🎨 Multi-modal 예측

### 샘플링 방법

```python
# Prior에서 여러 샘플 생성
z_samples = torch.randn(batch_size, num_samples, latent_dim)

# 각 샘플을 디코딩
trajectories = []
for z in z_samples:
    traj = decoder(z)
    trajectories.append(traj)
```

### 활용
- 다양한 주행 시나리오 생성
- 불확실성 모델링
- 다중 궤적 예측

## 🔍 모델 특징

### 장점
1. **압축 표현**: 160차원 → 32차원으로 효율적 압축
2. **확률적 모델링**: 불확실성을 latent distribution으로 표현
3. **다중 궤적 생성**: 샘플링을 통한 다양한 경로 생성
4. **정규화**: KL divergence로 latent space 정규화

### 설계 선택
1. **Deep MLP**: CNN 대신 MLP 사용 (시계열 특성 활용)
2. **대칭 구조**: Encoder-Decoder 대칭으로 정보 손실 최소화
3. **Batch Normalization**: 학습 안정화
4. **Dropout**: 과적합 방지

## 📚 참고

- **입력 데이터**: nuPlan 데이터셋의 8초 궤적
- **좌표계**: 로컬 좌표계 (시작점이 (0, 0))
- **정규화**: 데이터셋 평균/표준편차로 정규화
- **배치 크기**: 기본 32 (설정 가능)

## 🔗 관련 파일

- `models/vae.py`: VAEEncoder, VAEDecoder 클래스 정의
- `models/trajectory_predictor.py`: 전체 모델 통합
- `train/config.yaml`: 모델 설정 파일
- `utils/loss.py`: Loss 함수 구현
