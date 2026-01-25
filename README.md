# VAE-Planner

VAE(Variational Autoencoder) 기반 자율주행 차량 궤적 예측 모델

## 📋 프로젝트 개요

VAE-Planner는 VAE를 활용하여 자율주행 차량의 미래 궤적을 예측하는 딥러닝 모델입니다. nuPlan 데이터셋의 8초 궤적 데이터를 학습하여 다양한 주행 시나리오에 대한 다중 궤적을 생성할 수 있습니다.

### 주요 특징

- **VAE 기반 불확실성 모델링**: Latent variable 샘플링을 통한 multi-modal 궤적 예측
- **β-VAE Annealing**: KL divergence weight를 점진적으로 증가시켜 posterior collapse 방지
- **Interactive Visualization**: 웹 기반 시각화 서버를 통한 latent space 탐색 및 궤적 시각화
- **nuPlan 데이터셋 지원**: 대규모 자율주행 데이터셋 활용

## 🏗️ 모델 아키텍처

![Model Architecture](assets/model_architecture.png)

VAE-Planner는 다음과 같은 구조로 구성됩니다:

1. **Encoder**: 160차원 궤적 벡터(80 타임스텝 × 2차원)를 32차원 latent space로 인코딩
   - 구조: 160 → 512 → 256 → 128 → 32 (μ, logvar)
   
2. **Reparameterization Trick**: μ와 logvar로부터 latent variable z 샘플링
   - z = μ + σ × ε, where ε ~ N(0, I)

3. **Decoder**: 32차원 latent z를 160차원 궤적 벡터로 복원
   - 구조: 32 → 128 → 256 → 512 → 160

### 학습 결과 예시

![VAE Results](assets/VAE_results_1.gif)

## 📦 설치

### 필수 요구사항

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

### 시각화 서버 의존성 설치

```bash
cd visualization_server
pip install -r requirements.txt
npm install
```

## 📊 데이터 준비

### nuPlan 데이터셋에서 궤적 추출

1. nuPlan 데이터셋을 다운로드하고 경로를 설정합니다.

2. `data_process/extract_8s_trajectories.sh` 파일을 수정하여 데이터 경로를 설정합니다:

```bash
NUPLAN_DATA_PATH="$HOME/99_dataset/01_nuplan/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="$HOME/99_dataset/01_nuplan/dataset/maps"
TRAJECTORY_SAVE_PATH="$HOME/99_dataset/01_nuplan/dataset/exp2/trajectories_8s.npz"
NUM_SAMPLES=100000  # 추출할 샘플 수
```

3. 궤적 추출 스크립트 실행:

```bash
cd data_process
chmod +x extract_8s_trajectories.sh
./extract_8s_trajectories.sh
```

추출된 데이터는 `.npz` 형식으로 저장되며, 각 샘플은 160차원 벡터(80 타임스텝 × 2차원 [x, y])로 구성됩니다.

## 🚀 사용 방법

### 1. 설정 파일 수정

`train/config.yaml` 파일을 열어 데이터 경로와 학습 설정을 수정합니다:

```yaml
data:
  trajectory_data_path: "$HOME/99_dataset/01_nuplan/dataset/exp2/trajectories_8s.npz"
  normalize: true  # 데이터 정규화 여부
  max_samples: 100000  # 사용할 최대 샘플 수

model:
  future_horizon: 80  # 미래 프레임 수 (8초 × 10Hz)
  future_dim: 2  # [x, y]
  latent_dim: 32  # Latent space 차원
  kl_weight: 0.5  # KL divergence 가중치
  kl_annealing:
    enabled: true  # Annealing 사용 여부
    start_weight: 0.01  # 시작 KL weight
    end_weight: 0.5  # 최종 KL weight
    annealing_type: "linear"  # "linear" 또는 "cosine"
    warmup_epochs: 2  # Warmup 에포크 수

training:
  batch_size: 32
  num_epochs: 5
  learning_rate: 1e-4
  weight_decay: 1e-5
  gradient_clip: 1.0
```

### 2. 학습 실행

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

학습 로깅을 위해 Wandb를 사용하려면 API 키를 설정해야 합니다:

```bash
# 방법 1: 명령어로 로그인 (권장)
wandb login

# 방법 2: 환경 변수로 설정
export WANDB_API_KEY=your_api_key_here
```

API 키는 [https://wandb.ai/settings](https://wandb.ai/settings)에서 확인할 수 있습니다.

### 3. 학습 결과 확인

학습 결과는 `train/train_output/<실험명>/<타임스탬프>/` 디렉토리에 저장됩니다:

- `checkpoints/`: 모델 체크포인트 파일 (`.pth`)
- `logs/`: TensorBoard 로그 파일
- `latent_analysis/`: Latent space 분석 결과 (PCA 시각화 포함)
- `original_trajectories.npz`: 학습에 사용된 원본 궤적 데이터

TensorBoard로 학습 과정 확인:

```bash
tensorboard --logdir train/train_output
```

## 🎨 시각화 서버

학습된 모델의 latent space를 브라우저에서 탐색하고 궤적을 시각화할 수 있는 웹 애플리케이션입니다.

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

또는

```bash
python app.py --config ../train/config.yaml --port 5000
```

3. **브라우저에서 접속**:

```
http://localhost:5000
```

서버가 자동으로:
- `train/train_output`에서 최신 체크포인트를 찾아 사용
- Config 파일에서 데이터셋 경로를 읽어 로드
- 통합된 React 클라이언트 서빙

### 수동 체크포인트 지정

```bash
python app.py --checkpoint <체크포인트_경로> --config ../train/config.yaml --port 5000
```

### 주요 기능

- **Latent Space 시각화**: 데이터셋의 궤적들이 latent space에 2D로 projection되어 표시 (PCA 또는 t-SNE)
- **Interactive Hover**: Latent space에서 마우스를 움직이면 가장 가까운 latent z가 강조되고, 해당 z에 매칭되는 원본 입력 궤적이 표시됩니다
- **Trajectory 시각화**: 원본 궤적이 시작점(녹색), 끝점(빨강), 경로(빨간 선)로 시각화됩니다

자세한 내용은 [visualization_server/README.md](visualization_server/README.md)를 참고하세요.

## 📁 프로젝트 구조

```
VAE-Planner/
├── data/                          # 데이터 로더
│   ├── __init__.py
│   └── trajectory_dataset.py      # 궤적 데이터셋 클래스
├── data_process/                  # 데이터 전처리
│   ├── extract_8s_trajectories.py
│   └── extract_8s_trajectories.sh
├── models/                        # 모델 정의
│   ├── __init__.py
│   ├── vae.py                    # VAE 모듈 (Encoder, Decoder)
│   └── trajectory_predictor.py   # 전체 모델 통합
├── train/                         # 학습 스크립트
│   ├── config.yaml               # 학습 설정 파일
│   ├── train.py                  # 학습 스크립트
│   └── train_output/             # 학습 결과 저장 디렉토리
│       └── <실험명>/
│           └── <타임스탬프>/
│               ├── checkpoints/  # 모델 체크포인트
│               ├── logs/         # TensorBoard 로그
│               └── latent_analysis/  # Latent space 분석 결과
├── utils/                         # 유틸리티 함수
│   ├── loss.py                   # Loss 함수 (MSE, KL Divergence)
│   └── metrics.py                # 평가 지표
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
├── trajectory_visualizations/     # 궤적 시각화 결과
└── README.md                      # 이 파일
```

## 🔧 주요 기능 설명

### β-VAE Annealing

학습 초기에 reconstruction에 집중하고, 점진적으로 KL divergence를 증가시켜 posterior collapse를 방지합니다:

- **Warmup 단계**: 처음 N 에포크 동안 낮은 KL weight 유지
- **Annealing 단계**: KL weight를 선형 또는 코사인 방식으로 증가
- 설정: `config.yaml`의 `model.kl_annealing` 섹션

### 데이터 정규화

데이터셋의 평균과 표준편차를 계산하여 궤적 데이터를 정규화합니다:

- 정규화 파라미터는 자동으로 계산되거나 `trajectory_norm_params_path`에서 로드됩니다
- 정규화된 데이터는 `_normalized.npz` 형식으로 저장할 수 있습니다

### Latent Space 분석

학습 완료 후 자동으로 latent space 분석을 수행합니다:

- 궤적을 stop, left turn, right turn, straight로 분류
- PCA를 사용하여 latent space를 2D로 projection
- 분류별 궤적 샘플 시각화

## 📝 참고사항

- 학습에 사용된 원본 궤적은 `train_output/<실험명>/<타임스탬프>/original_trajectories.npz`에 저장됩니다
- 모든 궤적의 시작점은 (0, 0)으로 정규화됩니다 (로컬 좌표계)
- GPU 메모리가 부족한 경우 `batch_size`를 줄이거나 `num_workers`를 조정하세요

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

## 🙏 감사의 말

- nuPlan 데이터셋: [nuPlan-devkit](https://github.com/motional/nuplan-devkit)
- 시각화 서버는 CVAE-Planner의 visualization_server를 참고하여 제작되었습니다
