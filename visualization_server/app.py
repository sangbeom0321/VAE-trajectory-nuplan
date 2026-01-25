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

# Trajectory 분류 함수 (개선된 다차원 분류)
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
latent_z_cache = {}  # 인덱스 -> latent z 매핑
trajectory_cache = {}  # 인덱스 -> 원본 trajectory 매핑


def load_model(checkpoint_path, config_path):
    """모델 로드"""
    global model, config, device
    
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
    else:
        print(f'Warning: Checkpoint not found at {checkpoint_path}, using untrained model')
    
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


@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'dataset_loaded': dataset is not None})


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
        
        # UMAP 사용 가능 여부 확인
        if method == 'umap' and not UMAP_AVAILABLE:
            return jsonify({'error': 'UMAP is not available. Install with: pip install umap-learn'}), 400
        
        # 샘플 수를 10,000개로 고정
        MAX_SAMPLES = 10000
        
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
            
            # Trajectory 분류
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
                'label': label  # 분류 라벨 추가
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
    
    app.run(host=args.host, port=args.port, debug=True)
