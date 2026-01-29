"""
VAE-Planner 시각화 서버
브라우저에서 latent space를 탐색하고 원본 경로를 시각화할 수 있는 API 제공
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import numpy as np
import os
import sys
import json
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# 프로젝트 루트를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.trajectory_predictor import TrajectoryPredictor
from models.vae import reparameterize
from data.trajectory_dataset import TrajectoryDataset

# Trajectory 분류 함수들
def train_kmeans_model(k=5, num_samples=1000, random_seed=42):
    """
    K-means 모델 학습
    
    Args:
        k: 클러스터 수
        num_samples: 학습에 사용할 샘플 수
        random_seed: 랜덤 시드
    """
    global kmeans_model, kmeans_k, dataset
    
    if dataset is None:
        print("Error: Dataset not loaded")
        return False
    
    print(f"Training K-means model (k={k}, num_samples={num_samples})...")
    
    # 샘플링
    np.random.seed(random_seed)
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    if num_samples < total_samples:
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
    
    # 궤적 데이터 수집
    trajectories_list = []
    for idx in sample_indices:
        traj_xy = dataset.get_trajectory_as_xy(idx, denormalize=True)
        traj_flat = traj_xy.reshape(-1).astype(np.float32)
        trajectories_list.append(traj_flat)
    
    trajectories_array = np.array(trajectories_list)  # (num_samples, 160)
    
    # K-means 학습
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
    kmeans_model.fit(trajectories_array)
    kmeans_k = k
    
    print(f"K-means model trained successfully (k={k})")
    return True

def classify_trajectory_kmeans(trajectory_xy):
    """
    K-means 클러스터링 기반 분류
    
    Args:
        trajectory_xy: (80, 2) - [x, y] coordinate array
        
    Returns:
        label: 클러스터 번호를 문자열로 반환 (예: 'cluster_0', 'cluster_1', ...)
    """
    global kmeans_model
    
    if kmeans_model is None:
        # K-means 모델이 없으면 기본값 반환
        return 'cluster_unknown'
    
    # trajectory_xy를 (160,) 형태로 변환
    trajectory_flat = trajectory_xy.reshape(-1).astype(np.float32)
    
    # K-means 예측
    cluster_label = kmeans_model.predict(trajectory_flat.reshape(1, -1))[0]
    
    return f'cluster_{cluster_label}'

# Trajectory 분류 함수 (개선된 다차원 분류) - 룰 기반
def classify_trajectory(trajectory_xy):
    """
    개선된 분류 로직: 여러 특성을 조합하여 더 의미있는 분류
    
    분류 기준:
    1. 정지: 경로 길이 < 2m
    2. 방향: 각도 기반 (직진/좌/우)
    3. 곡률: 경로의 곡률 정도 (급커브/완만한)
    4. 속도: 평균 속도 (빠른/보통/느린)
    
    Args:
        trajectory_xy: (80, 2) - [x, y] coordinate array
        
    Returns:
        label: 'stop', 'straight', 'left', 'right', 
               'straight_sharp', 'left_sharp', 'right_sharp',
               'straight_slow', 'left_slow', 'right_slow'
    """
    import numpy as np
    
    # Start and end points
    start = trajectory_xy[0]
    end = trajectory_xy[-1]
    
    # Total distance traveled
    total_distance = np.linalg.norm(end - start)
    
    # 정지: 경로 길이 < 2m
    if total_distance < 2.0:
        return 'stop'
    
    # Ego 정면 기준 각도 계산 (x축이 정면, y축이 좌측)
    angle_rad = np.arctan2(end[1] - start[1], end[0] - start[0])
    angle_deg = np.degrees(angle_rad)
    
    # 방향 분류
    if -10.0 <= angle_deg <= 10.0:
        direction = 'straight'
    elif angle_deg > 10.0:
        direction = 'left'
    else:
        direction = 'right'
    
    # 곡률 계산 (경로의 곡률 정도)
    deltas = np.diff(trajectory_xy, axis=0)  # (79, 2)
    velocities = deltas  # 각 타임스텝의 속도 벡터
    
    # 곡률: 연속된 속도 벡터 간의 각도 변화
    curvatures = []
    for i in range(len(velocities) - 1):
        v1 = velocities[i]
        v2 = velocities[i + 1]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            continue
        
        # 정규화
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        
        # 각도 변화 (dot product로 계산)
        dot = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle_change = np.arccos(dot)
        curvatures.append(angle_change)
    
    if len(curvatures) > 0:
        avg_curvature = np.mean(curvatures)
        max_curvature = np.max(curvatures)
        total_curvature = np.sum(curvatures)
    else:
        avg_curvature = 0
        max_curvature = 0
        total_curvature = 0
    
    # 곡률 분류: 급커브 vs 완만한
    # 평균 곡률이 크거나 최대 곡률이 크면 급커브
    is_sharp = avg_curvature > 0.15 or max_curvature > 0.3  # 라디안 기준
    
    # 속도 분류: 평균 속도 계산
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds[speeds > 1e-6]) if np.any(speeds > 1e-6) else 0
    
    # 속도 분류 (임계값은 데이터에 따라 조정 필요)
    # 8초 경로이므로 평균 속도가 낮으면 느린 경로
    is_slow = avg_speed < 0.5  # m/step (10Hz 기준이므로 0.5 m/step = 5 m/s)
    
    # 조합된 라벨 생성
    if is_slow:
        return f'{direction}_slow'
    elif is_sharp:
        return f'{direction}_sharp'
    else:
        return direction

app = Flask(__name__, static_folder='client_build', static_url_path='')
CORS(app)

# 전역 변수
model = None
config = None
dataset = None
device = None
current_checkpoint_path = None  # 현재 로드된 체크포인트 경로
kmeans_model = None  # K-means 클러스터링 모델 (k=5)
kmeans_k = 5  # K-means 클러스터 수
latent_z_cache = {}  # 인덱스 -> latent z 매핑
trajectory_cache = {}  # 인덱스 -> 원본 trajectory 매핑
reducer_cache = {}  # method -> reducer 매핑 (PCA, t-SNE, UMAP)
latent_matrix_cache = {}  # method -> latent_matrix 매핑
projected_cache = {}  # method -> projected 매핑


def load_model(checkpoint_path, config_path):
    """모델 로드"""
    global model, config, device, current_checkpoint_path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Config 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 확장 ($HOME 등)
    if 'data' in config and 'trajectory_data_path' in config['data']:
        config['data']['trajectory_data_path'] = os.path.expandvars(config['data']['trajectory_data_path'])
    if 'data' in config and 'trajectory_norm_params_path' in config['data']:
        config['data']['trajectory_norm_params_path'] = os.path.expandvars(config['data']['trajectory_norm_params_path'])
    
    # 모델 생성
    model = TrajectoryPredictor(config)
    model.to(device)
    model.eval()
    
    # 체크포인트 로드
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f'Loaded model from {checkpoint_path}')
            current_checkpoint_path = checkpoint_path
        except RuntimeError as e:
            print(f'Warning: Some keys could not be loaded: {e}')
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            if missing_keys:
                print(f'Missing keys (will use random initialization): {missing_keys[:5]}...')
            if unexpected_keys:
                print(f'Unexpected keys (ignored): {unexpected_keys[:5]}...')
            print(f'Model loaded with partial state dict from {checkpoint_path}')
            current_checkpoint_path = checkpoint_path
    else:
        print(f'Warning: Checkpoint not found at {checkpoint_path}, using untrained model')
        current_checkpoint_path = None
    
    return model, config


def load_dataset(config):
    """데이터셋 로드"""
    global dataset
    
    data_cfg = config['data']
    data_path = data_cfg['trajectory_data_path']
    norm_params_path = data_cfg.get('trajectory_norm_params_path', None)
    normalize = data_cfg.get('normalize', True)
    max_samples = data_cfg.get('max_samples', None)
    
    # 원본 데이터가 정규화되지 않았다면 normalize=False로 설정
    # 파일명에 '_normalized'가 없으면 정규화되지 않은 데이터로 간주
    if '_normalized' not in os.path.basename(data_path):
        normalize = False
        print(f'원본 데이터가 정규화되지 않았습니다. normalize=False로 설정합니다.')
    
    dataset = TrajectoryDataset(
        data_path=data_path,
        norm_params_path=norm_params_path,
        normalize=normalize,
        max_samples=max_samples
    )
    
    print(f'Loaded dataset with {len(dataset)} samples from {data_path}')
    print(f'  Normalize: {dataset.normalize}')
    
    return dataset


def compute_latent_for_trajectory(trajectory_idx):
    """특정 trajectory의 latent z 계산 및 캐싱"""
    global latent_z_cache, trajectory_cache
    
    if trajectory_idx in latent_z_cache:
        return latent_z_cache[trajectory_idx], trajectory_cache[trajectory_idx]
    
    # 데이터 로드
    sample = dataset[trajectory_idx]
    trajectory_tensor = sample['trajectory'].unsqueeze(0).to(device)  # (1, 160)
    trajectory_reshaped = trajectory_tensor.reshape(1, 80, 2)  # (1, 80, 2)
    
    # Latent 인코딩
    with torch.no_grad():
        flattened = model._flatten_input(trajectory_reshaped)
        mu, logvar = model.vae_encoder(flattened)
        z = reparameterize(mu, logvar)
    
    # 원본 trajectory 가져오기 (정규화되지 않은 데이터면 denormalize=False)
    denormalize = dataset.normalize  # 데이터셋이 정규화되어 있으면 역정규화, 아니면 그대로 사용
    trajectory_xy = dataset.get_trajectory_as_xy(trajectory_idx, denormalize=denormalize)  # (80, 2)
    
    # 캐싱
    latent_z_cache[trajectory_idx] = z.cpu().numpy()[0].tolist()
    trajectory_cache[trajectory_idx] = trajectory_xy.tolist()
    
    return latent_z_cache[trajectory_idx], trajectory_cache[trajectory_idx]


def find_available_checkpoints():
    """사용 가능한 체크포인트 목록 찾기"""
    train_output_dir = os.path.join(project_root, 'train', 'train_output', 'vae-planner-training')
    checkpoints = []
    
    # 검색할 디렉토리 목록
    search_dirs = ['beta1', 'beta4', 'betapoint1']
    
    for dir_name in search_dirs:
        dir_path = os.path.join(train_output_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
        
        checkpoint_files = []
        
        # 1. checkpoints 폴더 안에서 찾기
        checkpoints_dir = os.path.join(dir_path, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pth') and ('checkpoint' in file or 'best_model' in file):
                    file_path = os.path.join(checkpoints_dir, file)
                    mtime = os.path.getmtime(file_path)
                    checkpoint_files.append({
                        'path': file_path,
                        'name': file,
                        'mtime': mtime,
                        'dir': dir_name
                    })
        
        # 2. checkpoints 폴더가 없으면 디렉토리 자체에서 찾기
        if not checkpoint_files:
            for file in os.listdir(dir_path):
                if file.endswith('.pth') and ('checkpoint' in file or 'best_model' in file):
                    file_path = os.path.join(dir_path, file)
                    mtime = os.path.getmtime(file_path)
                    checkpoint_files.append({
                        'path': file_path,
                        'name': file,
                        'mtime': mtime,
                        'dir': dir_name
                    })
        
        # 3. 재귀적으로 하위 디렉토리에서도 찾기
        if not checkpoint_files:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.pth') and ('checkpoint' in file or 'best_model' in file):
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        checkpoint_files.append({
                            'path': file_path,
                            'name': file,
                            'mtime': mtime,
                            'dir': dir_name
                        })
        
        if checkpoint_files:
            # 최신 체크포인트만 선택
            latest = max(checkpoint_files, key=lambda x: x['mtime'])
            checkpoints.append({
                'name': dir_name,
                'display_name': f'{dir_name} (Latest)',
                'path': latest['path'],
                'file': latest['name'],
                'mtime': latest['mtime'],
                'all_checkpoints': sorted(checkpoint_files, key=lambda x: x['mtime'], reverse=True)
            })
    
    # mtime 기준으로 정렬 (최신순)
    checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    
    return checkpoints


@app.route('/api/checkpoints', methods=['GET'])
def get_checkpoints():
    """사용 가능한 체크포인트 목록 반환"""
    try:
        checkpoints = find_available_checkpoints()
        return jsonify({
            'checkpoints': checkpoints,
            'current': current_checkpoint_path
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/load-checkpoint', methods=['POST'])
def load_checkpoint():
    """체크포인트 변경 및 모델 재로드"""
    global model, config, latent_z_cache, trajectory_cache, reducer_cache, latent_matrix_cache, projected_cache
    
    try:
        data = request.json
        checkpoint_path = data.get('checkpoint_path', None)
        config_path = data.get('config_path', None)
        
        if not checkpoint_path:
            return jsonify({'error': 'checkpoint_path required'}), 400
        
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'Checkpoint not found: {checkpoint_path}'}), 404
        
        # Config 경로 설정
        if config_path is None:
            config_path = os.path.join(project_root, 'train', 'config.yaml')
        
        if not os.path.exists(config_path):
            return jsonify({'error': f'Config not found: {config_path}'}), 404
        
        # 캐시 초기화 (새 모델이므로 기존 캐시는 무효)
        latent_z_cache = {}
        trajectory_cache = {}
        reducer_cache = {}
        latent_matrix_cache = {}
        projected_cache = {}
        
        # 모델 재로드
        print(f'Reloading model with checkpoint: {checkpoint_path}')
        load_model(checkpoint_path, config_path)
        
        return jsonify({
            'status': 'success',
            'checkpoint_path': current_checkpoint_path,
            'message': 'Model reloaded successfully'
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None, 
        'dataset_loaded': dataset is not None,
        'current_checkpoint': current_checkpoint_path
    })


@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """데이터셋 정보 반환"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    return jsonify({
        'num_samples': len(dataset),
        'future_horizon': config['model']['future_horizon'],
        'future_dim': config['model']['future_dim'],
        'latent_dim': config['model']['latent_dim']
    })


@app.route('/api/trajectory/<int:trajectory_idx>', methods=['GET'])
def get_trajectory(trajectory_idx):
    """특정 trajectory의 원본 경로 반환"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    if trajectory_idx >= len(dataset):
        return jsonify({'error': 'Invalid trajectory index'}), 400
    
    # 원본 trajectory 가져오기 (정규화되지 않은 데이터면 denormalize=False)
    denormalize = dataset.normalize  # 데이터셋이 정규화되어 있으면 역정규화, 아니면 그대로 사용
    trajectory_xy = dataset.get_trajectory_as_xy(trajectory_idx, denormalize=denormalize)  # (80, 2)
    
    return jsonify({
        'trajectory': trajectory_xy.tolist(),
        'index': trajectory_idx
    })


@app.route('/api/trajectory/<int:trajectory_idx>/latent', methods=['GET'])
def get_trajectory_latent(trajectory_idx):
    """특정 trajectory의 latent z 반환"""
    if model is None or dataset is None:
        return jsonify({'error': 'Model or dataset not loaded'}), 500
    
    if trajectory_idx >= len(dataset):
        return jsonify({'error': 'Invalid trajectory index'}), 400
    
    latent_z, trajectory_xy = compute_latent_for_trajectory(trajectory_idx)
    
    return jsonify({
        'latent': latent_z,
        'trajectory': trajectory_xy,
        'index': trajectory_idx
    })


@app.route('/api/latent-space', methods=['POST'])
def compute_latent_space():
    """여러 trajectory의 Latent Space를 계산하고 2D로 projection"""
    try:
        if model is None or dataset is None:
            return jsonify({'error': 'Model or dataset not loaded'}), 500
        
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid request data'}), 400
        
        trajectory_indices = data.get('trajectory_indices', None)
        method = data.get('method', 'pca')  # 'pca', 'tsne', or 'umap'
        max_samples = data.get('max_samples', None)  # 기본값: None (전체 사용)
        classification_method = data.get('classification_method', 'rule')  # 'rule' or 'kmeans'
        
        # UMAP 사용 가능 여부 확인
        if method == 'umap' and not UMAP_AVAILABLE:
            return jsonify({'error': 'UMAP is not available. Install with: pip install umap-learn'}), 400
        
        # 샘플 수를 5,000개로 고정
        MAX_SAMPLES = 5000
        
        # trajectory_indices가 없으면 샘플링
        if trajectory_indices is None:
            if max_samples is None:
                # 기본값: 10,000개로 고정
                num_samples = min(MAX_SAMPLES, len(dataset))
                print(f'샘플 수를 {num_samples}개로 고정합니다 (전체: {len(dataset)}개)')
                trajectory_indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
            else:
                # max_samples가 지정된 경우, 최대값으로 제한
                if max_samples > MAX_SAMPLES:
                    print(f'⚠️  샘플 수를 {MAX_SAMPLES}개로 제한합니다 (요청: {max_samples}개)')
                    max_samples = MAX_SAMPLES
                
                num_samples = min(max_samples, len(dataset))
                trajectory_indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
        
        print(f'Processing {len(trajectory_indices)} trajectories with {method}...')
        
        # 각 trajectory의 latent z 계산
        latent_vectors = []
        valid_indices = []
        
        for idx in trajectory_indices:
            if idx >= len(dataset):
                continue
            
            try:
                latent_z, _ = compute_latent_for_trajectory(idx)
                latent_vectors.append(latent_z)
                valid_indices.append(idx)
            except Exception as e:
                print(f'Error processing trajectory {idx}: {e}')
                continue
        
        if len(latent_vectors) == 0:
            return jsonify({'error': 'No valid trajectories processed'}), 400
        
        print(f'Successfully processed {len(latent_vectors)} trajectories')
        
        latent_matrix = np.array(latent_vectors)
        
        # 2D Projection
        if method == 'pca':
            reducer = PCA(n_components=2)
            projected = reducer.fit_transform(latent_matrix)
            explained_variance = reducer.explained_variance_ratio_
            print(f'PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}')
        elif method == 'tsne':
            # t-SNE는 더 많은 정보를 보존하지만 계산이 느림
            perplexity = min(30, max(5, len(latent_vectors) // 4))  # 적절한 perplexity 설정
            print(f'Running t-SNE with perplexity={perplexity}...')
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
            projected = reducer.fit_transform(latent_matrix)
            print(f't-SNE projection completed with perplexity={perplexity}')
        elif method == 'umap':
            # UMAP은 빠르고 좋은 클러스터링 결과를 제공
            n_neighbors = min(15, max(5, len(latent_vectors) // 10))  # 적절한 n_neighbors 설정
            min_dist = 0.1  # 클러스터 간 최소 거리
            print(f'Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...')
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            projected = reducer.fit_transform(latent_matrix)
            print(f'UMAP projection completed')
        else:
            return jsonify({'error': 'Invalid method. Use "pca", "tsne", or "umap"'}), 400
        
        # Reducer와 데이터를 캐시에 저장 (나중에 클릭한 위치에서 역변환하기 위해)
        reducer_cache[method] = reducer
        latent_matrix_cache[method] = latent_matrix
        projected_cache[method] = projected
        
        # Latent z의 실제 분산 확인 (디버깅용 - Posterior Collapse 감지)
        latent_std = np.std(latent_matrix, axis=0)
        latent_mean = np.mean(latent_matrix, axis=0)
        overall_std = np.std(latent_matrix)
        overall_mean = np.mean(latent_matrix)
        
        print(f'\n=== Latent Space Analysis ===')
        print(f'Latent z dimension-wise std: mean={np.mean(latent_std):.4f}, min={np.min(latent_std):.4f}, max={np.max(latent_std):.4f}')
        print(f'Latent z overall: mean={overall_mean:.4f}, std={overall_std:.4f}')
        print(f'Latent z range: min={np.min(latent_matrix):.4f}, max={np.max(latent_matrix):.4f}')
        
        # Posterior Collapse 감지
        if np.mean(latent_std) < 0.1:
            print(f'⚠️  WARNING: Very low latent std ({np.mean(latent_std):.4f}) - Possible POSTERIOR COLLAPSE!')
            print(f'   This means all trajectories map to similar latent z values.')
            print(f'   Solution: Reduce KL weight (currently {config["model"]["kl_weight"]}) or use β-VAE annealing.')
        elif np.mean(latent_std) < 0.5:
            print(f'⚠️  CAUTION: Low latent std ({np.mean(latent_std):.4f}) - May indicate weak latent representation.')
        else:
            print(f'✓ Latent std looks healthy ({np.mean(latent_std):.4f})')
        print('=' * 30 + '\n')
        
        # 결과 반환
        result = []
        for i, idx in enumerate(valid_indices):
            trajectory_xy = trajectory_cache.get(idx, None)
            if trajectory_xy is None:
                denormalize = dataset.normalize  # 데이터셋이 정규화되어 있으면 역정규화, 아니면 그대로 사용
                trajectory_xy_np = dataset.get_trajectory_as_xy(idx, denormalize=denormalize)
                trajectory_xy = trajectory_xy_np.tolist()
                trajectory_cache[idx] = trajectory_xy
            else:
                trajectory_xy_np = np.array(trajectory_xy)
            
            # Trajectory 분류 (방법 선택)
            if classification_method == 'kmeans':
                # K-means 모델이 없으면 학습
                if kmeans_model is None:
                    train_kmeans_model(k=5, num_samples=1000)
                label = classify_trajectory_kmeans(trajectory_xy_np)
            else:  # 'rule' (기본값)
                label = classify_trajectory(trajectory_xy_np)
            
            # latent_vectors[i]는 이미 list (compute_latent_for_trajectory에서 .tolist() 호출)
            latent_z_value = latent_vectors[i]
            if isinstance(latent_z_value, np.ndarray):
                latent_z_value = latent_z_value.tolist()
            
            result.append({
                'index': idx,
                'x': float(projected[i, 0]),
                'y': float(projected[i, 1]),
                'latent': latent_z_value,  # 이미 list이거나 numpy array를 list로 변환
                'trajectory': trajectory_xy,  # 원본 trajectory 포함
                'label': label,  # 분류 라벨 추가
                'classification_method': classification_method  # 사용된 분류 방법
            })
        
        response_data = {
            'projected_points': result,
            'method': method,
            'reducer_info': {
                'mean': reducer.mean_.tolist() if hasattr(reducer, 'mean_') else None,
                'components': reducer.components_.tolist() if hasattr(reducer, 'components_') else None
            },
            'latent_stats': {
                'mean_std': float(np.mean(latent_std)),
                'min_std': float(np.min(latent_std)),
                'max_std': float(np.max(latent_std)),
                'overall_mean': float(overall_mean),
                'overall_std': float(overall_std),
                'posterior_collapse_warning': bool(np.mean(latent_std) < 0.1)  # numpy bool_를 Python bool로 변환
            },
            'num_samples': len(result),
            'total_dataset_size': len(dataset)
        }
        
        print(f'Returning response with {len(result)} points')
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f'Error in compute_latent_space: {error_msg}')
        print(traceback_str)
        return jsonify({
            'error': f'Server error: {error_msg}',
            'traceback': traceback_str
        }), 500


@app.route('/api/generate-trajectory-from-point', methods=['POST'])
def generate_trajectory_from_point():
    """클릭한 2D 좌표에서 latent z를 복원하고 trajectory 생성"""
    if model is None or dataset is None:
        return jsonify({'error': 'Model or dataset not loaded'}), 500
    
    try:
        data = request.json
        x_2d = data.get('x', None)
        y_2d = data.get('y', None)
        method = data.get('method', 'pca')
        
        if x_2d is None or y_2d is None:
            return jsonify({'error': 'x and y coordinates required'}), 400
        
        # Reducer와 데이터 가져오기
        if method not in reducer_cache:
            return jsonify({'error': f'No cached data for method {method}. Please load latent space first.'}), 400
        
        reducer = reducer_cache[method]
        latent_matrix = latent_matrix_cache[method]
        projected = projected_cache[method]
        
        # 2D 좌표를 latent z로 역변환
        point_2d = np.array([[x_2d, y_2d]], dtype=np.float32)
        
        if method == 'pca':
            # PCA는 선형 변환이라 역변환이 가능 - 클릭한 위치를 직접 사용
            latent_z_approx = reducer.inverse_transform(point_2d)[0]  # (latent_dim,)
            
            # PCA 역변환 결과가 학습 데이터의 latent z 범위를 벗어나는지 확인
            latent_z_min = np.min(latent_matrix, axis=0)
            latent_z_max = np.max(latent_matrix, axis=0)
            latent_z_range = latent_z_max - latent_z_min
            
            # 범위를 벗어난 정도 계산
            out_of_range_mask = (latent_z_approx < latent_z_min) | (latent_z_approx > latent_z_max)
            if np.any(out_of_range_mask):
                # 범위를 벗어난 차원들을 학습 데이터 범위로 클리핑
                latent_z_approx = np.clip(latent_z_approx, latent_z_min, latent_z_max)
                print(f'Warning: PCA inverse transform resulted in out-of-range latent z. Clipped to valid range.')
            
            print(f'PCA inverse transform: clicked 2D=[{x_2d:.4f}, {y_2d:.4f}]')
        else:
            # t-SNE/UMAP은 비선형 변환이라 역변환이 불가능
            # 클릭한 위치에서 가장 가까운 점들의 latent z를 거리 기반 가중 평균으로 보간
            distances = np.sum((projected - point_2d) ** 2, axis=1)
            
            # 가장 가까운 k개 점 사용 (더 많은 점을 사용하면 더 부드러운 보간)
            # 거리가 멀수록 더 많은 점을 사용하여 보간의 안정성 향상
            k = min(30, len(projected))  # 최대 30개까지 사용
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            
            # 가장 가까운 점까지의 거리 확인
            min_distance = np.sqrt(nearest_distances[0])
            
            # 학습 데이터의 평균 거리를 기준으로 거리 정규화
            if len(projected) > 1:
                # 학습 데이터 간 평균 거리 계산 (캐싱 가능하지만 일단 매번 계산)
                sample_distances = []
                sample_size = min(1000, len(projected))
                sample_indices = np.random.choice(len(projected), sample_size, replace=False)
                for i in range(min(100, sample_size)):
                    for j in range(i+1, min(i+10, sample_size)):
                        dist = np.sqrt(np.sum((projected[sample_indices[i]] - projected[sample_indices[j]]) ** 2))
                        sample_distances.append(dist)
                avg_distance = np.mean(sample_distances) if sample_distances else 1.0
                
                # 거리 정규화 (평균 거리 기준)
                normalized_distances = np.sqrt(nearest_distances) / (avg_distance + 1e-10)
                
                # 거리가 멀수록 가중치를 더 부드럽게 감소시키는 함수 사용
                # 가우시안 커널 기반 가중치: exp(-d^2 / (2*sigma^2))
                # 거리가 멀수록 sigma를 크게 하여 더 많은 점의 영향을 받도록
                if min_distance > avg_distance * 2.0:
                    # 거리가 멀면 더 넓은 범위의 점들을 고려 (sigma를 크게)
                    sigma = avg_distance * 2.0
                    print(f'Warning: Clicked point is far from data manifold (distance: {min_distance:.4f}, avg: {avg_distance:.4f})')
                    print(f'  Using wider interpolation kernel (sigma={sigma:.4f})')
                else:
                    # 거리가 가까우면 좁은 범위의 점들만 고려 (sigma를 작게)
                    sigma = avg_distance * 0.5
                
                # 가우시안 커널 기반 가중치 계산
                weights = np.exp(-normalized_distances ** 2 / (2 * (sigma / avg_distance) ** 2))
                weights = weights / (np.sum(weights) + 1e-10)  # 정규화
                
                # 가중 평균으로 latent z 계산 (보간)
                latent_z_approx = np.sum(latent_matrix[nearest_indices] * weights[:, np.newaxis], axis=0)
                
                print(f'Gaussian kernel interpolation ({method}): clicked 2D=[{x_2d:.4f}, {y_2d:.4f}], using {k} nearest points')
                print(f'  Nearest point distance: {min_distance:.4f}, avg distance: {avg_distance:.4f}')
                print(f'  Weight distribution: min={np.min(weights):.4f}, max={np.max(weights):.4f}, mean={np.mean(weights):.4f}')
            else:
                # 데이터가 부족한 경우 가장 가까운 점 사용
                latent_z_approx = latent_matrix[nearest_indices[0]].copy()
                print(f'Direct interpolation ({method}): using nearest point only (insufficient data)')
        
        # 디코더로 trajectory 생성
        with torch.no_grad():
            latent_z_tensor = torch.FloatTensor(latent_z_approx).unsqueeze(0).to(device)  # (1, latent_dim)
            reconstructed_flat = model.vae_decoder(latent_z_tensor)  # (1, 160) - 정규화된 출력
            reconstructed = model._unflatten_output(reconstructed_flat)  # (1, 80, 2)
            trajectory_np = reconstructed[0].cpu().numpy()  # (80, 2) - 정규화된 상태
        
        print(f'Generated trajectory (normalized): range x=[{np.min(trajectory_np[:, 0]):.2f}, {np.max(trajectory_np[:, 0]):.2f}], y=[{np.min(trajectory_np[:, 1]):.2f}, {np.max(trajectory_np[:, 1]):.2f}]')
        
        # 역정규화 (디코더 출력은 정규화된 상태이므로 역정규화 필요)
        if dataset.normalize:
            trajectory_flat = trajectory_np.flatten()  # (160,)
            trajectory_denorm = dataset._denormalize_trajectory(trajectory_flat)
            trajectory_np = trajectory_denorm.reshape(80, 2)
            print(f'Generated trajectory (denormalized): range x=[{np.min(trajectory_np[:, 0]):.2f}, {np.max(trajectory_np[:, 0]):.2f}], y=[{np.min(trajectory_np[:, 1]):.2f}, {np.max(trajectory_np[:, 1]):.2f}]')
        else:
            print('Dataset is not normalized, skipping denormalization')
        
        # 시작점을 (0, 0)으로 강제 설정 (로컬 좌표계)
        trajectory_np[0, 0] = 0.0
        trajectory_np[0, 1] = 0.0
        
        # 생성된 trajectory 검증
        # 1. 경로 길이 확인 (너무 짧으면 문제)
        path_lengths = np.linalg.norm(np.diff(trajectory_np, axis=0), axis=1)
        total_length = np.sum(path_lengths)
        
        # 2. 모든 점이 (0,0)에 가까운지 확인 (비정상적인 경우)
        distances_from_origin = np.linalg.norm(trajectory_np, axis=1)
        max_distance = np.max(distances_from_origin)
        
        # 3. NaN이나 Inf 확인
        has_nan = np.any(np.isnan(trajectory_np))
        has_inf = np.any(np.isinf(trajectory_np))
        
        print(f'Trajectory validation: length={total_length:.2f}m, max_distance={max_distance:.2f}m, has_nan={has_nan}, has_inf={has_inf}')
        
        # 비정상적인 trajectory 감지
        if has_nan or has_inf:
            return jsonify({
                'error': 'Generated trajectory contains invalid values (NaN or Inf). Please try clicking closer to existing data points.',
                'is_invalid': True
            }), 400
        
        if total_length < 0.1:  # 10cm 미만이면 너무 짧음
            return jsonify({
                'error': f'Generated trajectory is too short ({total_length:.2f}m). This may happen when clicking far from the data manifold. Please try clicking closer to existing data points.',
                'is_invalid': True,
                'trajectory_length': float(total_length)
            }), 400
        
        if max_distance < 0.5:  # 모든 점이 원점에서 50cm 이내면 비정상
            return jsonify({
                'error': f'Generated trajectory is too close to origin (max distance: {max_distance:.2f}m). This may happen when clicking far from the data manifold. Please try clicking closer to existing data points.',
                'is_invalid': True,
                'max_distance': float(max_distance)
            }), 400
        
        # Trajectory 분류 (기본값: rule-based)
        # 생성된 trajectory는 항상 rule-based 분류 사용
        label = classify_trajectory(trajectory_np)
        
        # 2D 좌표 계산 (projected space에서의 위치)
        if method in projected_cache:
            # 가장 가까운 점의 2D 좌표 사용 (또는 직접 계산)
            if method == 'pca':
                # PCA는 직접 역변환 가능하므로 2D 좌표를 그대로 사용
                point_2d = np.array([[x_2d, y_2d]], dtype=np.float32)
            else:
                # t-SNE/UMAP은 가장 가까운 점의 2D 좌표 사용
                point_2d = np.array([[x_2d, y_2d]], dtype=np.float32)
        else:
            point_2d = np.array([[x_2d, y_2d]], dtype=np.float32)
        
        return jsonify({
            'trajectory': trajectory_np.tolist(),
            'latent_z': latent_z_approx.tolist(),
            'latent_z_2d': [float(x_2d), float(y_2d)],  # 2D 좌표
            'label': label,
            'method': method,
            'is_generated': True  # 생성된 trajectory임을 표시
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f'Error in generate_trajectory_from_point: {error_msg}')
        print(traceback_str)
        return jsonify({
            'error': f'Server error: {error_msg}',
            'traceback': traceback_str
        }), 500


@app.route('/api/find-nearest-trajectory', methods=['POST'])
def find_nearest_trajectory():
    """주어진 latent z에 가장 가까운 trajectory 찾기"""
    if model is None or dataset is None:
        return jsonify({'error': 'Model or dataset not loaded'}), 500
    
    data = request.json
    target_z = data.get('latent', None)
    max_search = data.get('max_search', 1000)  # 검색할 최대 샘플 수
    
    if target_z is None:
        return jsonify({'error': 'latent z required'}), 400
    
    target_z_array = np.array(target_z, dtype=np.float32)
    
    # 샘플링하여 latent z 계산
    num_samples = min(max_search, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    min_dist = float('inf')
    nearest_idx = None
    nearest_trajectory = None
    
    for idx in sample_indices:
        try:
            latent_z, trajectory_xy = compute_latent_for_trajectory(idx)
            latent_z_array = np.array(latent_z, dtype=np.float32)
            
            # 유클리드 거리 계산
            dist = np.sqrt(np.sum((latent_z_array - target_z_array) ** 2))
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
                nearest_trajectory = trajectory_xy
        except Exception as e:
            continue
    
    if nearest_idx is None:
        return jsonify({'error': 'No trajectory found'}), 404
    
    return jsonify({
        'index': nearest_idx,
        'trajectory': nearest_trajectory,
        'distance': float(min_dist)
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VAE-Planner Visualization Server')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (.pth file). If not provided, will search in train/train_output')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(project_root, 'train', 'config.yaml'),
                       help='Path to config file')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (0.0.0.0 for external access)')
    
    args = parser.parse_args()
    
    # 체크포인트 자동 탐색 (지정되지 않은 경우)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 기본 체크포인트 경로 시도
        default_checkpoint = os.path.join(
            project_root, 
            'train', 
            'train_output', 
            'vae-planner-training', 
            '2026-01-28-12:56:00', 
            'checkpoints', 
            'checkpoint_epoch_234.pth'
        )
        
        if os.path.exists(default_checkpoint):
            checkpoint_path = default_checkpoint
            print(f'Using default checkpoint: {checkpoint_path}')
        else:
            # 자동 탐색: train_output 디렉토리에서 최신 체크포인트 찾기
            train_output_dir = os.path.join(project_root, 'train', 'train_output')
            if os.path.exists(train_output_dir):
                checkpoint_files = []
                for root, dirs, files in os.walk(train_output_dir):
                    for file in files:
                        if file.endswith('.pth') and ('best_model' in file or 'checkpoint' in file):
                            checkpoint_files.append(os.path.join(root, file))
                
                if checkpoint_files:
                    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
                    print(f'Auto-detected checkpoint: {checkpoint_path}')
                else:
                    print('Warning: No checkpoint found in train/train_output, using untrained model')
            else:
                print('Warning: train/train_output directory not found, using untrained model')
    
    # 모델 로드
    print('Loading model...')
    load_model(checkpoint_path, args.config)
    
    # 데이터셋 로드
    print('Loading dataset...')
    load_dataset(config)
    
    # React 빌드 결과를 정적 파일로 서빙
    client_build_path = os.path.join(script_dir, 'client_build')
    if os.path.exists(client_build_path):
        from flask import send_from_directory
        
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_client(path):
            # API 경로는 제외
            if path.startswith('api/'):
                return jsonify({'error': 'Not found'}), 404
            
            if path != "" and os.path.exists(os.path.join(client_build_path, path)):
                return send_from_directory(client_build_path, path)
            else:
                return send_from_directory(client_build_path, 'index.html')
        
        print(f'Server starting on http://{args.host}:{args.port}')
        print(f'Serving integrated React client from {client_build_path}')
    else:
        print(f'Server starting on http://{args.host}:{args.port}')
        print('Note: React client not built. Run "./build_client.sh" to build the client.')
        print('      Or use API endpoints directly at http://localhost:5000/api/...')
    
    # Flask 개발 서버 시작
    import socket
    
    def find_free_port(start_port, max_attempts=10):
        """사용 가능한 포트 찾기"""
        for i in range(max_attempts):
            port = start_port + i
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return None
    
    # 포트가 사용 가능한지 먼저 확인
    actual_port = args.port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', args.port))
        sock.close()
    except OSError:
        # 포트가 사용 중이면 사용 가능한 포트 찾기
        print(f"\n⚠️  Port {args.port} is already in use.")
        free_port = find_free_port(args.port)
        if free_port:
            print(f"   Using alternative port {free_port}...")
            actual_port = free_port
        else:
            print(f"   Could not find a free port starting from {args.port}.")
            print(f"   Please either:")
            print(f"   1. Stop the process using port {args.port}")
            print(f"   2. Use a different port: python visualization_server/app.py --port 5002")
            sys.exit(1)
    
    # 서버 시작
    print(f"Starting server on port {actual_port}...")
    app.run(host=args.host, port=actual_port, debug=True, use_reloader=False)
