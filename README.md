# VAE-Planner

VAE (Variational Autoencoder) 기반 자율주행 차량 경로 예측 모델

## 📋 프로젝트 개요

VAE-Planner는 Variational Autoencoder를 사용하여 자율주행 차량의 미래 경로를 예측하는 딥러닝 모델입니다. nuPlan 데이터셋에서 추출한 8초 경로 데이터로 학습하며, 다양한 주행 시나리오에 대한 다중 경로를 생성할 수 있습니다.

### 주요 기능

- **VAE 기반 불확실성 모델링**: Latent variable 샘플링을 통한 다중 경로 예측
- **8초 경로 데이터셋**: 160차원 벡터 (80 타임스텝 × 2차원 [x, y])
- **인터랙티브 시각화**: 웹 기반 시각화 서버로 latent space 탐색 및 경로 시각화
- **nuPlan 데이터셋 지원**: 대규모 자율주행 데이터셋 활용

## 🏗️ 모델 아키텍처

![Model Architecture](assets/model_architecture.png)

VAE-Planner는 다음과 같은 구조로 구성됩니다:

1. **Encoder**: 160차원 경로 벡터를 32차원 latent space로 인코딩
   - 아키텍처: 160 → 512 → 256 → 128 → 32 (μ, logvar)
   - Batch Normalization 및 Dropout 포함
   
2. **Reparameterization Trick**: μ와 logvar에서 latent variable z 샘플링
   - z = μ + σ × ε, where ε ~ N(0, I)

3. **Decoder**: 32차원 latent z에서 160차원 경로 벡터 복원
   - 아키텍처: 32 → 128 → 256 → 512 → 160
   - 정규화된 데이터의 경우 Tanh 활성화 함수로 [-1, 1] 범위 제한

### 학습 결과 예시

<img src="assets/VAE_results_1.gif" alt="VAE Results" width="800"/>

## 📦 설치

### 요구사항

- Python 3.9 이상
- CUDA 지원 GPU (권장)
- Node.js 14 이상 (시각화 서버용)

### Python 패키지 설치

```bash
# 프로젝트 루트에서
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn pyyaml tqdm matplotlib
pip install wandb tensorboard  # 선택사항: 학습 로깅용
```

### nuPlan 의존성

이 프로젝트는 **nuPlan** 데이터셋을 사용하며, 데이터 추출을 위해 nuPlan devkit이 필요합니다. `data/extract_8s_trajectories.py` 스크립트가 nuPlan 라이브러리에 의존합니다.

**설치 방법:**

1. 공식 nuPlan 설치 가이드를 따르세요: [nuPlan-devkit Documentation](https://github.com/motional/nuplan-devkit)

2. 기본 설치 단계:
   ```bash
   # nuPlan devkit 클론
   git clone https://github.com/motional/nuplan-devkit.git
   cd nuplan-devkit
   
   # 의존성 설치
   pip install -e .
   ```

3. 상세 설치 가이드 (Docker 설정 및 데이터셋 다운로드 포함)는 다음을 참조하세요:
   - [nuPlan-devkit README](https://github.com/motional/nuplan-devkit#installation)
   - [nuPlan Documentation](https://nuplan-devkit.readthedocs.io/)

**참고**: nuPlan devkit은 특정 시스템 의존성이 필요하며 추가 설정이 필요할 수 있습니다. 공식 문서를 참조하세요.

### 시각화 서버 의존성

```bash
cd visualization_server
pip install -r requirements.txt
npm install
```

**선택사항**: UMAP 시각화 지원:
```bash
pip install umap-learn
```

## 📊 데이터 준비

### nuPlan 데이터셋에서 경로 추출

1. nuPlan 데이터셋을 다운로드하고 경로를 설정하세요.

2. `data/extract_8s_trajectories.sh` 파일을 수정하여 데이터 경로를 설정하세요:

```bash
# 환경 변수로 경로 설정 (기본값 사용 가능)
export NUPLAN_DATA_PATH="$HOME/99_dataset/01_nuplan/dataset/nuplan-v1.1/splits/trainval"
export NUPLAN_MAP_PATH="$HOME/99_dataset/01_nuplan/dataset/maps"
export TRAJECTORY_SAVE_PATH="$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz"
export NUM_SAMPLES=100000  # 추출할 샘플 수
```

또는 스크립트 내에서 직접 수정:

```bash
NUPLAN_DATA_PATH="${NUPLAN_DATA_PATH:-$HOME/99_dataset/01_nuplan/dataset/nuplan-v1.1/splits/trainval}"
NUPLAN_MAP_PATH="${NUPLAN_MAP_PATH:-$HOME/99_dataset/01_nuplan/dataset/maps}"
TRAJECTORY_SAVE_PATH="${TRAJECTORY_SAVE_PATH:-$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz}"
NUM_SAMPLES=100000
```

3. 경로 추출 스크립트 실행:

```bash
cd data
chmod +x extract_8s_trajectories.sh
./extract_8s_trajectories.sh
```

추출된 데이터는 `.npz` 형식으로 저장되며, 각 샘플은 160차원 벡터 (80 타임스텝 × 2차원 [x, y])입니다.

### 데이터 시각화 (전처리 후 확인)

추출된 데이터를 시각화하여 확인할 수 있습니다:

![Input Data](assets/input_data.png)

```bash
cd data
./visualize_trajectories.sh
```

또는 Python으로 직접 실행:

```bash
python data/visualize_trajectories.py \
    --data_path "$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz" \
    --num_samples 100000 \
    --save_dir ./trajectory_visualizations
```

### Planning Vocabulary 생성 (선택사항)

K-means 클러스터링을 사용하여 Planning Vocabulary를 생성할 수 있습니다:

```bash
cd data
./create_planning_vocabulary.sh
```

또는 Python으로 직접 실행:

```bash
python data/create_planning_vocabulary.py \
    --data_path "$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz" \
    --k 5 \
    --num_samples 100000 \
    --save_dir ./planning_vocabulary
```

## 🚀 사용 방법

### 1. 설정 파일 구성

`train/config.yaml` 파일을 열어 데이터 경로와 학습 설정을 수정하세요:

```yaml
data:
  trajectory_data_path: "$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz"
  trajectory_norm_params_path: "$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s_norm_params.json"
  normalize: true  # 데이터 정규화 여부
  max_samples: 100000  # 사용할 최대 샘플 수

model:
  future_horizon: 80  # 미래 프레임 수 (8초 × 10Hz = 80)
  future_dim: 2  # [x, y]
  latent_dim: 32  # Latent space 차원
  kl_weight: 0.0001  # KL Divergence Loss 가중치

training:
  batch_size: 32
  num_epochs: 500
  learning_rate: 1.5e-4
  weight_decay: 1e-5
  gradient_clip: 1.0
  lr_scheduler:
    enabled: true
    mode: "min"
    factor: 0.7
    patience: 8
    min_lr: 1e-7
```

### 2. 학습

```bash
cd train
python train.py --config config.yaml --name vae-planner-training
```

#### 주요 옵션

- `--config`: 설정 파일 경로 (기본값: `config.yaml`)
- `--resume`: 체크포인트 경로 (학습 재개용)
- `--name`: 실험 이름 (기본값: `vae-planner-training`)
- `--num_workers`: 데이터 로딩 워커 수 (기본값: 8)
- `--use_wandb`: Wandb 로깅 사용 여부 (기본값: True)

#### Wandb 설정 (선택사항)

Wandb를 사용하여 학습 로깅을 하려면 API 키를 설정하세요:

```bash
# 방법 1: 명령어로 로그인 (권장)
wandb login

# 방법 2: 환경 변수 설정
export WANDB_API_KEY=your_api_key_here
```

API 키는 [https://wandb.ai/settings](https://wandb.ai/settings)에서 확인할 수 있습니다.

### 3. 학습 결과 확인

학습 결과는 `train/train_output/<experiment_name>/<timestamp>/` 디렉토리에 저장됩니다:

- `checkpoints/`: 모델 체크포인트 파일 (`.pth`)
- `logs/`: TensorBoard 로그 파일
- `latent_analysis/`: Latent space 분석 결과 (PCA 시각화 포함)
- `original_trajectories.npz`: 학습에 사용된 원본 경로 데이터

TensorBoard로 학습 진행 상황 확인:

```bash
tensorboard --logdir train/train_output
```

## 🎨 시각화 서버

학습된 모델의 latent space를 탐색하고 경로를 시각화할 수 있는 웹 애플리케이션입니다.

### 빌드 및 실행

1. **클라이언트 빌드** (최초 1회 또는 클라이언트 코드 변경 시):

```bash
cd visualization_server
./build_client.sh
```

2. **서버 시작**:

```bash
./start_server.sh
```

또는:

```bash
python app.py --config ../train/config.yaml --port 5000
```

3. **브라우저에서 접속**:

```
http://localhost:5000
```

서버는 자동으로:
- `train/train_output`에서 최신 체크포인트를 찾아 사용
- Config 파일에서 데이터셋 경로를 읽어 로드
- 통합된 React 클라이언트 서빙
- 성능을 위해 기본적으로 5,000개 샘플로 제한

### 수동 체크포인트 지정

```bash
python app.py --checkpoint <checkpoint_path> --config ../train/config.yaml --port 5000
```

### 사용 가이드

1. **Projection Method 선택**: 헤더에서 PCA, t-SNE, 또는 UMAP 선택
   - PCA: Generate 모드에 최적 (정확한 역변환)
   - t-SNE/UMAP: Browse 모드에서 클러스터 시각화에 더 좋음

2. **Browse Mode**:
   - Latent space 위에서 마우스를 움직이면 기존 경로 확인
   - 경로는 타입별로 색상 구분 (정지, 좌회전, 우회전, 직진)
   - 데이터 포인트를 클릭하여 경로 뷰 고정

3. **Generate Mode**:
   - Latent space의 아무 곳이나 클릭하여 새로운 경로 생성
   - 2D 좌표와 32차원 latent z 벡터 확인
   - 생성된 경로는 ✨ 표시로 표시
   - **참고**: PCA는 정확한 역변환을 제공하지만, t-SNE/UMAP은 보간을 사용

4. **Trajectory Information**:
   - 생성된 경로는 2D projection 좌표와 전체 latent z 벡터를 표시
   - 경로 분류 (정지/좌회전/우회전/직진)이 자동으로 표시됨

자세한 내용은 [visualization_server/README.md](visualization_server/README.md)를 참조하세요.

### GitHub Pages 배포

시각화 서버를 GitHub Pages에 배포하여 공개 접근이 가능합니다. 프론트엔드는 GitHub Pages에 호스팅되고, 백엔드 API는 별도로 호스팅해야 합니다 (예: Render, Railway, Heroku).

자세한 배포 가이드는 [.github/workflows/README.md](.github/workflows/README.md)를 참조하세요.

## 📁 프로젝트 구조

```
VAE-Planner/
├── data/                          # 데이터 전처리 및 로더
│   ├── __init__.py
│   ├── trajectory_dataset.py      # Trajectory 데이터셋 클래스
│   ├── extract_8s_trajectories.py  # nuPlan에서 8초 경로 추출
│   ├── extract_8s_trajectories.sh
│   ├── visualize_trajectories.py  # 데이터 시각화 (전처리 후 확인)
│   ├── visualize_trajectories.sh
│   ├── create_planning_vocabulary.py  # Planning Vocabulary 생성
│   └── create_planning_vocabulary.sh
├── models/                        # 모델 정의
│   ├── __init__.py
│   ├── vae.py                    # VAE 모듈 (Encoder, Decoder)
│   ├── trajectory_predictor.py   # 통합 모델
│   ├── loss.py                   # Loss 함수
│   └── metrics.py                # 평가 지표
├── train/                         # 학습 스크립트
│   ├── config.yaml               # 학습 설정 파일
│   ├── train.py                  # 학습 스크립트
│   └── train_output/             # 학습 결과 디렉토리
│       └── <experiment_name>/
│           └── <timestamp>/
│               ├── checkpoints/  # 모델 체크포인트
│               ├── logs/         # TensorBoard 로그
│               └── latent_analysis/  # Latent space 분석 결과
├── visualization_server/          # 시각화 웹 서버
│   ├── app.py                    # Flask 백엔드 API 서버
│   ├── src/                      # React 프론트엔드
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── LatentSpacePlot.jsx
│   │       └── TrajectoryCanvas.jsx
│   ├── requirements.txt          # Python 의존성
│   ├── package.json              # Node.js 의존성
│   ├── build_client.sh           # 클라이언트 빌드 스크립트
│   └── start_server.sh           # 서버 시작 스크립트
├── assets/                       # 에셋 (이미지, gif)
│   ├── model_architecture.png
│   ├── VAE_results_1.gif
│   ├── input_data.png            # 입력 데이터 시각화
│   ├── pca.png                   # PCA 기반 latent space 시각화
│   └── tsne.png                  # t-SNE 기반 latent space 시각화
├── requirements.txt              # 메인 Python 의존성
└── README.md                     # 이 파일
```

## 🔧 주요 기능 설명

### 데이터 정규화

경로 데이터를 정규화하여 데이터셋의 평균과 표준편차를 계산합니다:

- 정규화 파라미터는 `trajectory_norm_params_path`에서 자동으로 계산되거나 로드됩니다
- 정규화된 데이터는 `_normalized.npz` 형식으로 저장할 수 있습니다
- 모든 경로는 시작점이 (0, 0)으로 정규화됩니다 (로컬 좌표계)

### Latent Space 분석

학습 후 자동으로 latent space 분석을 수행합니다:

- 경로를 정지, 좌회전, 우회전, 직진으로 분류
- PCA를 사용하여 latent space를 2D로 projection
- 카테고리별 경로 샘플 시각화

#### PCA 기반 Latent Space 시각화

![PCA Latent Space](assets/pca.png)

PCA (Principal Component Analysis)를 사용하여 32차원 latent space를 2D로 투영한 결과입니다. 각 색상은 경로 분류를 나타냅니다.

#### t-SNE 기반 Latent Space 시각화

![t-SNE Latent Space](assets/tsne.png)

t-SNE (t-Distributed Stochastic Neighbor Embedding)를 사용하여 latent space의 클러스터 구조를 시각화한 결과입니다. 유사한 경로 패턴이 가까이 모여 있는 것을 확인할 수 있습니다.

### 분류 조건 설명

경로는 다음과 같은 조건에 따라 분류됩니다:

| 분류 | 조건 | 설명 |
|------|------|------|
| **Stop (정지)** | 경로 길이 < 2.0m | 시작점과 끝점 사이의 직선 거리가 2m 미만 |
| **Straight (직진)** | -10° ≤ 각도 ≤ 10° | 시작점-끝점 각도가 거의 직진 |
| **Straight(sharp) (급커브 직진)** | 직진 + 급커브 | 직진 방향이지만 평균 곡률 > 0.15 rad 또는 최대 곡률 > 0.3 rad |
| **Straight(slow) (느린 직진)** | 직진 + 느린 속도 | 직진 방향이지만 평균 속도 < 5 m/s |
| **Left Turn (좌회전)** | 각도 > 10° | 시작점-끝점 각도가 좌측 방향 |
| **Left Turn(Slow) (느린 좌회전)** | 좌회전 + 느린 속도 | 좌회전 방향이지만 평균 속도 < 5 m/s |
| **Right Turn (우회전)** | 각도 < -10° | 시작점-끝점 각도가 우측 방향 |
| **Right Turn(Slow) (느린 우회전)** | 우회전 + 느린 속도 | 우회전 방향이지만 평균 속도 < 5 m/s |

## 📝 참고사항

- 학습에 사용된 원본 경로는 `train_output/<experiment_name>/<timestamp>/original_trajectories.npz`에 저장됩니다
- 모든 경로는 시작점이 (0, 0)으로 정규화됩니다 (로컬 좌표계)
- GPU 메모리가 부족한 경우 `batch_size`를 줄이거나 `num_workers`를 조정하세요
- 프로젝트는 경로 설정을 위해 환경 변수를 사용합니다 - 스크립트를 참조하세요

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

## 🙏 감사의 말

- nuPlan Dataset: [nuPlan-devkit](https://github.com/motional/nuplan-devkit)