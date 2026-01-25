# VAE-Planner Latent Space Explorer

VAE-Planner 모델의 latent space를 브라우저에서 탐색하고, latent z에 hover하면 해당 z에 매칭되는 원본 입력 경로를 시각화하는 웹 애플리케이션입니다.

## 기능

- **Latent Space 시각화**: 데이터셋의 trajectory들이 latent space에 2D로 projection되어 표시됩니다 (PCA 또는 t-SNE)
- **Interactive Hover**: Latent space에서 마우스를 움직이면 가장 가까운 latent z가 강조되고, 해당 z에 매칭되는 원본 입력 경로가 오른쪽에 표시됩니다
- **Trajectory 시각화**: 원본 경로가 시작점(녹색), 끝점(빨강), 경로(빨간 선)로 시각화됩니다

## 설치

### Python 의존성
```bash
cd visualization_server
pip install -r requirements.txt
```

### Node.js 의존성 (클라이언트 빌드용)
```bash
npm install
```

## 빌드 및 실행

### 1. 클라이언트 빌드 (최초 1회 또는 클라이언트 코드 변경 시)
```bash
./build_client.sh
```

### 2. 서버 시작
```bash
./start_server.sh
```

또는

```bash
python visualization_server/app.py --config train/config.yaml --port 5000
```

### 3. 브라우저에서 접속
```
http://localhost:5000
```

서버가 자동으로:
- `train/train_output`에서 최신 체크포인트를 찾아 사용
- Config 파일에서 데이터셋 경로를 읽어 로드
- 통합된 React 클라이언트 서빙

### 수동 체크포인트 지정
```bash
python visualization_server/app.py --checkpoint <체크포인트_경로> --config train/config.yaml --port 5000
```

## API 엔드포인트

### GET /api/health
서버 상태 확인

### GET /api/dataset-info
데이터셋 정보 반환 (샘플 수, 차원 등)

### GET /api/trajectory/<int:trajectory_idx>
특정 trajectory의 원본 경로 반환

### GET /api/trajectory/<int:trajectory_idx>/latent
특정 trajectory의 latent z 반환

### POST /api/latent-space
여러 trajectory의 Latent Space를 계산하고 2D로 projection
Body: `{"method": "pca", "max_samples": 1000}`

### POST /api/find-nearest-trajectory
주어진 latent z에 가장 가까운 trajectory 찾기
Body: `{"latent": [latent_vector], "max_search": 1000}`

## 사용 방법

1. **서버 시작**: `./start_server.sh` 실행
2. **브라우저 접속**: `http://localhost:5000` 열기
3. **Latent Space 탐색**: 왼쪽에 표시된 latent space에서 마우스를 움직이세요
4. **Trajectory 확인**: Hover하면 오른쪽에 해당 latent z에 매칭되는 원본 입력 경로가 표시됩니다

## 개발 모드 (React 개발 서버 사용)

클라이언트 코드를 수정할 때는 React 개발 서버를 사용할 수 있습니다:

1. 서버 시작 (터미널 1):
```bash
./start_server.sh
```

2. React 개발 서버 시작 (터미널 2):
```bash
npm start
```

브라우저에서 `http://localhost:3000`을 열면 개발 모드로 실행됩니다.

## 구조

```
VAE-Planner/
├── visualization_server/
│   ├── app.py              # Flask 백엔드 API 서버
│   ├── requirements.txt    # Python 의존성
│   ├── start_server.sh     # 서버 시작 스크립트
│   ├── build_client.sh     # 클라이언트 빌드 스크립트
│   ├── src/
│   │   ├── App.jsx         # 메인 앱 컴포넌트
│   │   └── components/     # React 컴포넌트들
│   │       ├── LatentSpacePlot.jsx    # Latent space 시각화
│   │       └── TrajectoryCanvas.jsx    # Trajectory 시각화
│   └── package.json        # Node.js 의존성
```

## 참고

- CVAE-Planner의 visualization_server를 참고하여 제작되었습니다
- VAE-Latent-Space-Explorer의 hover 기능을 참고하여 구현되었습니다
