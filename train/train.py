"""
학습 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import sys
from tqdm import tqdm
import argparse
from datetime import datetime
import wandb
import numpy as np

# 프로젝트 루트를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.trajectory_predictor import TrajectoryPredictor
from models.vae import reparameterize
from data.trajectory_dataset import TrajectoryDataset, collate_fn
from utils.loss import compute_loss


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 처리 ($HOME 등)
    def expand_path(path):
        if isinstance(path, str):
            return os.path.expandvars(path)
        return path
    
    # 재귀적으로 모든 경로 확장
    def expand_paths(obj):
        if isinstance(obj, dict):
            return {k: expand_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_paths(item) for item in obj]
        elif isinstance(obj, str):
            return expand_path(obj)
        return obj
    
    config = expand_paths(config)
    return config


def get_kl_weight(config, epoch=None, num_epochs=None, best_recon_loss=None, current_recon_loss=None, kl_weight_state=None):
    """
    KL weight 반환 (어닐링 없이 고정값 사용)
    
    Args:
        config: 설정 딕셔너리
        epoch: 사용하지 않음 (하위 호환성을 위해 유지)
        num_epochs: 사용하지 않음 (하위 호환성을 위해 유지)
        best_recon_loss: 사용하지 않음 (하위 호환성을 위해 유지)
        current_recon_loss: 사용하지 않음 (하위 호환성을 위해 유지)
        kl_weight_state: 사용하지 않음 (하위 호환성을 위해 유지)
    
    Returns:
        kl_weight: config에서 지정한 고정 KL weight
        None: 상태 없음
    """
    model_cfg = config.get('model', {})
    kl_weight = float(model_cfg.get('kl_weight', 1.0))
    return kl_weight, None


def train_epoch(model, dataloader, optimizer, device, config, epoch, num_epochs, writer=None, use_wandb=False, global_scenario_count=0, log_interval_scenarios=1000, batch_size=32, last_logged_scenario=0, kl_weight=None):
    """한 에포크 학습"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    current_scenario_count = global_scenario_count  # 현재 에포크에서의 시나리오 카운터
    last_log_scenario = last_logged_scenario  # 마지막 로깅 시나리오 수
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        # TrajectoryDataset 또는 NuPlanDataset에 따라 다른 키 사용
        if 'trajectories' in batch:
            # TrajectoryDataset: (batch, 160) -> (batch, 80, 2)로 reshape
            trajectories = batch['trajectories'].to(device)  # (batch, 160)
            ego_future_trajectory = trajectories.reshape(-1, 80, 2)  # (batch, 80, 2)
        else:
            # NuPlanDataset: (batch, future_horizon, future_dim)
            ego_future_trajectory = batch['ego_future_trajectory'].to(device)
        
        # Forward pass
        output = model(
            ego_future_trajectory=ego_future_trajectory,
            mode='train'
        )
        
        # Loss 계산
        # 정규화된 경우: 정규화된 값끼리 비교
        # 정규화되지 않은 경우: 원본 스케일끼리 비교
        # 모델 출력은 이미 적절한 스케일로 나옴 (정규화된 경우 정규화된 값, 아닌 경우 원본 스케일)
        
        # Compute loss
        # KL weight는 config에서 고정값 사용
        if kl_weight is None:
            # Fallback: kl_weight가 전달되지 않은 경우
            kl_weight = get_kl_weight(config)[0]
        
        loss, recon_loss, kl_loss = compute_loss(
            reconstructed_ego_future_trajectory=output['reconstructed_ego_future_trajectory'],
            ego_future_trajectory=ego_future_trajectory,
            mu=output['mu'],
            logvar=output['logvar'],
            kl_weight=kl_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        # YAML에서 float 값이 문자열로 읽힐 수 있으므로 float로 변환
        gradient_clip = float(config['training']['gradient_clip'])
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip
        )
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # 실제 배치 크기 확인 (마지막 배치는 작을 수 있음)
        actual_batch_size = ego_future_trajectory.shape[0]
        
        # 시나리오 수 업데이트 (실제 배치 크기만큼 증가 - 지금까지 진행한 총 시나리오 수)
        current_scenario_count += actual_batch_size
        
        # 실시간 로깅 (1000개 시나리오마다)
        # 마지막 로깅 이후 1000개 이상의 시나리오가 지나갔는지 확인
        scenarios_since_last_log = current_scenario_count - last_log_scenario
        if scenarios_since_last_log >= log_interval_scenarios or batch_idx == 0:
            # 시나리오 단위 step 사용 (지금까지 진행한 총 시나리오 수)
            scenario_step = current_scenario_count
            
            # TensorBoard 로깅
            if writer is not None:
                writer.add_scalar('Loss/Train_Batch', loss.item(), scenario_step)
                writer.add_scalar('Loss/Train_Recon_Batch', recon_loss.item(), scenario_step)
                writer.add_scalar('Loss/Train_KL_Batch', kl_loss.item(), scenario_step)
                writer.add_scalar('Learning_Rate_Batch', optimizer.param_groups[0]['lr'], scenario_step)
            
            # Wandb 로깅 (시나리오 단위 step 사용 - 지금까지 진행한 총 시나리오 수)
            if use_wandb:
                log_dict = {
                    'train_loss/loss_batch': loss.item(),
                    'train_loss/recon_batch': recon_loss.item(),
                    'train_loss/kl_batch': kl_loss.item(),
                    'lr/lr_batch': optimizer.param_groups[0]['lr'],
                }
                wandb.log(log_dict, step=scenario_step)
            
            # 마지막 로깅 시나리오 수 업데이트
            last_log_scenario = current_scenario_count
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'scenarios': current_scenario_count
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss, current_scenario_count, last_log_scenario


def boolean(v):
    """Boolean argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def classify_trajectory(trajectory_xy):
    """
    Classify trajectory into stop, left turn, right turn, or straight
    
    Args:
        trajectory_xy: (80, 2) - [x, y] coordinate array
        
    Returns:
        label: 'stop', 'left', 'right', 'straight'
    """
    # Start and end points
    start = trajectory_xy[0]
    end = trajectory_xy[-1]
    
    # Total distance traveled
    total_distance = np.linalg.norm(end - start)
    
    # Stop detection: very short distance
    if total_distance < 1.5:  # Less than 1.5m
        return 'stop'
    
    # Calculate direction changes at each timestep
    deltas = np.diff(trajectory_xy, axis=0)  # (79, 2)
    
    # Filter out zero movements
    non_zero_mask = np.linalg.norm(deltas, axis=1) > 1e-6
    if np.sum(non_zero_mask) < 5:  # Too few valid movements
        return 'stop'
    
    # Calculate signed angles between consecutive velocity vectors
    angles = []
    for i in range(len(deltas) - 1):
        v1 = deltas[i]
        v2 = deltas[i + 1]
        
        # Skip if either vector is too small
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            continue
        
        # Normalize
        v1_norm = v1 / v1_norm
        v2_norm = v2 / v2_norm
        
        # Cross product (z-component) - signed curvature
        # Positive: left turn, Negative: right turn
        cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        angles.append(cross)
    
    if len(angles) == 0:
        return 'straight'
    
    angles = np.array(angles)
    
    # Mean rotation direction (positive: left turn, negative: right turn)
    mean_rotation = np.mean(angles)
    
    # Cumulative rotation (sum of all angle changes)
    cumulative_rotation = np.sum(angles)
    
    # Rotation strength (absolute mean)
    rotation_strength_mean = np.abs(mean_rotation)
    
    # Rotation strength (absolute cumulative)
    rotation_strength_cum = np.abs(cumulative_rotation)
    
    # Also check end point position relative to start
    # In local coordinates, left turns typically have positive y displacement
    # Right turns typically have negative y displacement
    y_displacement = end[1] - start[1]
    
    # Straight detection: very little rotation
    # Use multiple criteria for robustness
    is_straight = (
        rotation_strength_mean < 0.02 and  # Low mean rotation
        rotation_strength_cum < 0.3 and    # Low cumulative rotation
        np.abs(y_displacement) < 2.0       # Small lateral displacement
    )
    
    if is_straight:
        return 'straight'
    
    # Left vs Right turn
    # Use both cumulative rotation and y displacement for robust detection
    # Cumulative rotation is more reliable for turns
    if cumulative_rotation > 0.1 or (cumulative_rotation > 0 and y_displacement > 1.0):
        return 'left'
    elif cumulative_rotation < -0.1 or (cumulative_rotation < 0 and y_displacement < -1.0):
        return 'right'
    else:
        # Fallback: use mean rotation
        if mean_rotation > 0:
            return 'left'
        else:
            return 'right'


def save_original_trajectories(dataset, save_path):
    """
    학습에 사용된 원본 trajectory 저장
    
    Args:
        dataset: TrajectoryDataset 인스턴스
        save_path: 저장 경로
    """
    print(f'\nSaving original trajectories...')
    
    # Collect original trajectories (before normalization)
    original_trajectories = []
    for idx in range(len(dataset)):
        trajectory_xy = dataset.get_trajectory_as_xy(idx, denormalize=True)  # Original scale
        original_trajectories.append(trajectory_xy)
    
    original_trajectories = np.array(original_trajectories)  # (N, 80, 2)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, trajectories=original_trajectories)
    print(f'Original trajectories saved: {save_path} ({len(original_trajectories)} samples)')


def analyze_latent_space_after_training(model, original_trajectories_path, dataset, device, save_dir, best_checkpoint_path=None):
    """
    학습 완료 후 latent space 분석 수행
    
    Args:
        model: 학습된 모델
        original_trajectories_path: 원본 trajectory 파일 경로
        dataset: 데이터셋 (정규화 정보 포함)
        device: 디바이스
        save_dir: 결과 저장 디렉토리
        best_checkpoint_path: 최고 모델 체크포인트 경로 (None이면 현재 모델 사용)
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load best model if available
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f'Loading best model: {best_checkpoint_path}')
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load original trajectories
    print('Loading original trajectories...')
    data = np.load(original_trajectories_path)
    original_trajectories = data['trajectories']  # (N, 80, 2)
    print(f'Loaded: {len(original_trajectories)} samples')
    
    # Classify each trajectory
    print('Classifying trajectories...')
    labels = []
    for traj in original_trajectories:
        label = classify_trajectory(traj)
        labels.append(label)
    
    labels = np.array(labels)
    print(f'Classification results:')
    for label in ['stop', 'left', 'right', 'straight']:
        count = np.sum(labels == label)
        print(f'  {label}: {count} ({count/len(labels):.1%})')
    
    # Encode to latent space
    print('\nEncoding to latent space...')
    latent_z_list = []
    latent_mu_list = []
    
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(original_trajectories), batch_size), desc="Encoding"):
            batch_trajectories = original_trajectories[i:i+batch_size]  # (batch, 80, 2)
            
            # Apply normalization (using dataset's normalization function)
            batch_normalized = []
            for traj in batch_trajectories:
                traj_flat = traj.flatten()  # (160,)
                if dataset.normalize:
                    traj_normalized = dataset._normalize_trajectory(traj_flat)
                else:
                    traj_normalized = traj_flat
                batch_normalized.append(traj_normalized)
            
            batch_normalized = np.array(batch_normalized)  # (batch, 160)
            batch_tensor = torch.FloatTensor(batch_normalized).to(device)
            
            # Reshape to (batch, 80, 2)
            batch_reshaped = batch_tensor.reshape(-1, 80, 2)
            
            # Encode to latent space
            flattened = model._flatten_input(batch_reshaped)
            mu, logvar = model.vae_encoder(flattened)
            z = reparameterize(mu, logvar)
            
            latent_z_list.append(z.cpu().numpy())
            latent_mu_list.append(mu.cpu().numpy())
    
    latent_z = np.concatenate(latent_z_list, axis=0)  # (N, 32)
    latent_mu = np.concatenate(latent_mu_list, axis=0)  # (N, 32)
    
    print(f'Encoding complete: {latent_z.shape}')
    
    # Dimensionality reduction with PCA
    print('\nReducing dimensions with PCA...')
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_z)
    
    explained_variance = pca.explained_variance_ratio_
    print(f'PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}')
    
    # Visualization
    print('\nVisualizing...')
    label_colors = {
        'stop': 'red',
        'left': 'blue',
        'right': 'green',
        'straight': 'orange'
    }
    
    label_names = {
        'stop': 'Stop',
        'left': 'Left Turn',
        'right': 'Right Turn',
        'straight': 'Straight'
    }
    
    # Create figure with subplots for trajectory samples
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main latent space plot
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Trajectory samples subplot
    ax_samples = fig.add_subplot(gs[0, 1])
    ax_samples.set_title('Trajectory Samples', fontsize=12)
    ax_samples.grid(True, alpha=0.3)
    
    # Calculate reasonable axis limits based on data
    all_x = original_trajectories[:, :, 0].flatten()
    all_y = original_trajectories[:, :, 1].flatten()
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    
    # Print statistics for debugging
    print(f"\nTrajectory coordinate statistics:")
    print(f"  X range: [{np.min(all_x):.2f}, {np.max(all_x):.2f}] (span: {x_range:.2f})")
    print(f"  Y range: [{np.min(all_y):.2f}, {np.max(all_y):.2f}] (span: {y_range:.2f})")
    
    # Set reasonable limits (with some padding)
    x_margin = max(x_range * 0.1, 5.0)
    y_margin = max(y_range * 0.1, 5.0)
    ax_samples.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
    ax_samples.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    ax_samples.set_aspect('equal', adjustable='box')
    
    # Sample indices for each label
    num_samples_per_label = 5
    sample_indices = {}
    for label in ['stop', 'left', 'right', 'straight']:
        mask = labels == label
        if np.any(mask):
            indices = np.where(mask)[0]
            # Randomly sample
            n_samples = min(num_samples_per_label, len(indices))
            if n_samples > 0:
                selected_indices = np.random.choice(indices, n_samples, replace=False)
                sample_indices[label] = selected_indices
    
    # Plot all points in latent space
    for label in ['stop', 'left', 'right', 'straight']:
        mask = labels == label
        if np.any(mask):
            count = np.sum(mask)
            ax_main.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                          c=label_colors[label], label=f'{label_names[label]} (n={count})',
                          alpha=0.4, s=15)
    
    # Highlight sample trajectories
    for label in ['stop', 'left', 'right', 'straight']:
        if label in sample_indices:
            for idx in sample_indices[label]:
                # Plot trajectory sample
                traj = original_trajectories[idx]
                ax_samples.plot(traj[:, 0], traj[:, 1], 
                               color=label_colors[label], 
                               linewidth=1.5, alpha=0.7,
                               label=label_names[label] if idx == sample_indices[label][0] else '')
                
                # Mark corresponding point in latent space
                ax_main.scatter(latent_2d[idx, 0], latent_2d[idx, 1],
                              c=label_colors[label], s=100, marker='*', 
                              edgecolors='black', linewidths=1.5, zorder=10,
                              label=f'{label_names[label]} samples' if idx == sample_indices[label][0] else '')
    
    ax_main.set_xlabel(f'PC1 (explained variance: {explained_variance[0]:.2%})', fontsize=12)
    ax_main.set_ylabel(f'PC2 (explained variance: {explained_variance[1]:.2%})', fontsize=12)
    ax_main.set_title('Latent Space Visualization (PCA)', fontsize=14)
    ax_main.legend(fontsize=9, loc='best')
    ax_main.grid(True, alpha=0.3)
    
    ax_samples.set_xlabel('X (m)', fontsize=10)
    ax_samples.set_ylabel('Y (m)', fontsize=10)
    ax_samples.legend(fontsize=8, loc='best')
    
    # Category-wise trajectory visualization
    ax_categories = fig.add_subplot(gs[1, 1])
    ax_categories.set_title('Trajectories by Category', fontsize=12)
    ax_categories.grid(True, alpha=0.3)
    
    # Use same limits as samples plot
    ax_categories.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
    ax_categories.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    ax_categories.set_aspect('equal', adjustable='box')
    
    for label in ['stop', 'left', 'right', 'straight']:
        mask = labels == label
        if np.any(mask):
            # Plot a few representative trajectories
            indices = np.where(mask)[0]
            n_plot = min(10, len(indices))
            plot_indices = np.random.choice(indices, n_plot, replace=False)
            
            for idx in plot_indices:
                traj = original_trajectories[idx]
                ax_categories.plot(traj[:, 0], traj[:, 1], 
                                  color=label_colors[label], 
                                  linewidth=1.0, alpha=0.5)
    
    ax_categories.set_xlabel('X (m)', fontsize=10)
    ax_categories.set_ylabel('Y (m)', fontsize=10)
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=label_colors[label], label=label_names[label]) 
                     for label in ['stop', 'left', 'right', 'straight'] 
                     if np.any(labels == label)]
    ax_categories.legend(handles=legend_elements, fontsize=8, loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'latent_space_pca.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Visualization saved: {save_path}')
    plt.close()
    
    # Save results
    print('\nSaving results...')
    np.savez(os.path.join(save_dir, 'latent_mappings.npz'),
             latent_z=latent_z,
             latent_mu=latent_mu,
             latent_2d=latent_2d,
             labels=labels,
             trajectories=original_trajectories,
             pca_explained_variance=explained_variance)
    
    print(f'All analysis complete! Results saved to {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='VAE-Planner Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path (relative to train directory)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
    parser.add_argument('--name', type=str, default='vae-planner-training', help='Experiment name')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers (4 → 8로 증가하여 RAM 활용)')
    parser.add_argument('--pin_mem', action='store_true', default=True, help='Pin CPU memory in DataLoader')
    parser.add_argument('--use_wandb', type=boolean, default=True, help='Use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='vae-planner', help='Wandb project name')
    parser.add_argument('--notes', type=str, default='', help='Notes for wandb')
    args = parser.parse_args()
    
    # Load config
    # config 경로 처리 (train 디렉토리 기준)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    config = load_config(config_path)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 학습 결과 저장 디렉토리 설정 (train/train_output)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_output_dir = os.path.join(script_dir, 'train_output')
    
    # 타임스탬프로 서브디렉토리 생성
    if args.resume:
        save_path = args.resume.rsplit('/', 1)[0] if '/' in args.resume else train_output_dir
    else:
        time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        save_path = os.path.join(train_output_dir, args.name, time_str)
    
    # train_output 디렉토리 내부에 하위 디렉토리 생성
    log_dir = os.path.join(save_path, 'logs')
    save_dir = os.path.join(save_path, 'checkpoints')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # config에 저장 경로 업데이트
    config['logging']['log_dir'] = log_dir
    config['logging']['save_dir'] = save_dir
    
    print(f'학습 결과 저장 경로: {save_path}')
    print(f'  - 로그: {log_dir}')
    print(f'  - 체크포인트: {save_dir}')
    
    # Dataset 선택: trajectories_8s.npz가 있으면 TrajectoryDataset 사용, 아니면 NuPlanDataset 사용
    data_cfg = config.get('data', {})
    trajectory_data_path = data_cfg.get('trajectory_data_path', None)
    use_trajectory_dataset = False
    
    # 정규화된 파일이 없으면 원본 파일 확인
    if trajectory_data_path:
        if not os.path.exists(trajectory_data_path):
            # 정규화된 파일이 없으면 원본 파일 경로 시도
            original_path = trajectory_data_path.replace('_normalized.npz', '.npz')
            if os.path.exists(original_path):
                trajectory_data_path = original_path
                print(f"정규화된 파일이 없어 원본 파일 사용: {original_path}")
    
    if trajectory_data_path and os.path.exists(trajectory_data_path):
        use_trajectory_dataset = True
        
        # 정규화 설정
        normalize = data_cfg.get('normalize', True)
        norm_params_path = data_cfg.get('trajectory_norm_params_path', None)
        
        # norm_params_path가 없거나 존재하지 않으면 자동으로 찾기
        if normalize and (not norm_params_path or not os.path.exists(norm_params_path)):
            # 같은 디렉토리에서 _norm_params.json 파일 찾기
            base_path = trajectory_data_path.replace('_normalized.npz', '').replace('.npz', '')
            possible_norm_paths = [
                f"{base_path}_norm_params.json",
                trajectory_data_path.replace('.npz', '_norm_params.json'),
                os.path.join(os.path.dirname(trajectory_data_path), 'trajectories_8s_norm_params.json')
            ]
            for path in possible_norm_paths:
                if os.path.exists(path):
                    norm_params_path = path
                    break
            else:
                norm_params_path = None
        
        print(f"TrajectoryDataset 사용: {trajectory_data_path}")
        print(f"  정규화: {'사용' if normalize else '사용하지 않음'}")
        if normalize and norm_params_path:
            print(f"  정규화 파라미터: {norm_params_path}")
        elif normalize:
            print(f"  정규화 파라미터: 자동 계산됨")
        
        # 사용할 최대 샘플 수 설정
        max_samples = data_cfg.get('max_samples', None)
        if max_samples is not None:
            print(f"  최대 샘플 수: {max_samples}")
        
        train_dataset = TrajectoryDataset(
            data_path=trajectory_data_path,
            norm_params_path=norm_params_path,
            normalize=normalize,
            max_samples=max_samples  # 샘플 수 제한
        )
        collate_fn_to_use = collate_fn
    else:
        raise ValueError("trajectory_data_path가 설정되지 않았거나 파일이 존재하지 않습니다. config.yaml에서 trajectory_data_path를 확인하세요.")
    
    # 데이터 shape 검증
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\n" + "="*60)
        print("데이터 Shape 검증:")
        print("="*60)
        
        # TrajectoryDataset: (160,) 형태
        print(f"trajectory: {sample['trajectory'].shape} (expected: (160,))")
        print("="*60 + "\n")
        assert sample['trajectory'].shape == (160,), \
            f"trajectory shape mismatch: {sample['trajectory'].shape} != (160,)"
        
        print("✓ 모든 데이터 shape이 올바릅니다.\n")
    
    # DataLoader 설정 (CPU 병렬 처리)
    batch_size = config['training']['batch_size']
    num_workers = args.num_workers
    pin_memory = args.pin_mem and device.type == 'cuda'  # GPU 사용 시에만 pin_memory 활성화
    
    print(f'DataLoader 설정:')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Num workers: {num_workers}')
    print(f'  - Pin memory: {pin_memory}')
    print()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_to_use,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,  # VRAM 절약을 위해 False로 변경 (에포크마다 워커 재생성)
        prefetch_factor=2 if num_workers > 0 else None  # VRAM 절약을 위해 4 → 2로 감소
    )
    
    # 학습에 사용된 원본 trajectory 저장
    original_trajectories_path = os.path.join(save_path, 'original_trajectories.npz')
    save_original_trajectories(train_dataset, original_trajectories_path)
    
    # Model
    model = TrajectoryPredictor(config).to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {total_params:,} (학습 가능: {trainable_params:,})")
    print()
    
    # Optimizer
    # YAML에서 과학적 표기법이 문자열로 읽힐 수 있으므로 float로 변환
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning Rate Scheduler: Reconstruction error 개선이 없을 때 학습률 감소
    scheduler_cfg = config['training'].get('lr_scheduler', {})
    if scheduler_cfg.get('enabled', True):
        scheduler_mode = scheduler_cfg.get('mode', 'min')  # 'min' for reconstruction error
        scheduler_factor = float(scheduler_cfg.get('factor', 0.5))  # 학습률 감소 비율
        scheduler_patience = int(scheduler_cfg.get('patience', 5))  # 개선 없을 때 기다릴 epoch 수
        scheduler_threshold = float(scheduler_cfg.get('threshold', 1e-4))  # 개선으로 간주할 최소 변화량
        scheduler_min_lr = float(scheduler_cfg.get('min_lr', 1e-6))  # 최소 학습률
        scheduler_cooldown = int(scheduler_cfg.get('cooldown', 0))  # 학습률 감소 후 대기 epoch 수
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            min_lr=scheduler_min_lr,
            cooldown=scheduler_cooldown
        )
        print(f'Learning Rate Scheduler 설정:')
        print(f'  - 모드: {scheduler_mode} (reconstruction error 감소 모니터링)')
        print(f'  - 감소 비율: {scheduler_factor}')
        print(f'  - Patience: {scheduler_patience} epochs')
        print(f'  - 최소 학습률: {scheduler_min_lr}')
        print()
    else:
        scheduler = None
        print('Learning Rate Scheduler 비활성화됨')
    
    # Resume from checkpoint
    start_epoch = 0
    wandb_id = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        wandb_id = checkpoint.get('wandb_id', None)
        # Scheduler state 복원 (있는 경우)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'Resumed from epoch {start_epoch}')
    
    # TensorBoard
    writer = SummaryWriter(config['logging']['log_dir'])
    
    # Wandb 초기화
    if args.use_wandb:
        # Wandb 모드 설정 (online/offline)
        os.environ["WANDB_MODE"] = "online"
        
        # Wandb 로그인 상태 확인
        wandb_logged_in = False
        try:
            # netrc 파일이나 환경 변수에서 API 키 확인
            api_key = os.environ.get('WANDB_API_KEY') or wandb.api.api_key
            if api_key:
                wandb_logged_in = True
            else:
                # netrc 파일 확인
                netrc_path = os.path.expanduser('~/.netrc')
                if os.path.exists(netrc_path):
                    with open(netrc_path, 'r') as f:
                        if 'api.wandb.ai' in f.read():
                            wandb_logged_in = True
        except Exception:
            pass
        
        if not wandb_logged_in:
            print("\n" + "="*60)
            print("⚠️  Wandb API 키가 설정되지 않았습니다!")
            print("="*60)
            print("다음 중 하나의 방법으로 설정하세요:")
            print("\n1. 명령어로 로그인 (권장):")
            print("   wandb login")
            print("\n2. 환경 변수로 설정:")
            print("   export WANDB_API_KEY=your_api_key_here")
            print("\nAPI 키는 https://wandb.ai/settings 에서 확인할 수 있습니다.")
            print("="*60 + "\n")
            
            # API 키 입력 요청
            api_key = input("Wandb API 키를 입력하세요 (또는 Enter로 건너뛰기): ").strip()
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
                print("✓ API 키가 설정되었습니다.\n")
            else:
                print("⚠️  Wandb를 사용하지 않고 계속 진행합니다.\n")
                args.use_wandb = False
        else:
            print("✓ Wandb 로그인 상태 확인됨")
        
        if args.use_wandb:
            try:
                # Wandb 초기화 (온라인 모드, TensorBoard 동기화 활성화)
                wandb.init(
                    project=args.wandb_project,
                    name=args.name,
                    notes=args.notes,
                    resume='allow' if wandb_id else None,
                    id=wandb_id,
                    sync_tensorboard=True,  # TensorBoard와 자동 동기화
                    dir=save_path
                )
                
                # Config 업데이트 (모든 설정값 저장)
                wandb.config.update({
                    **config,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'device': str(device),
                })
                
                print(f'✓ Wandb initialized: {wandb.run.name} (온라인 모드)')
                print(f'  프로젝트: {args.wandb_project}')
                print(f'  Run ID: {wandb.run.id}')
            except Exception as e:
                print(f'⚠️  Wandb 초기화 실패: {e}')
                print('Wandb 없이 계속 진행합니다.\n')
                args.use_wandb = False
    
    # Training loop
    best_train_loss = float('inf')
    best_recon_loss = float('inf')  # 최고 reconstruction loss 추적
    best_epoch = -1  # 최고 성능 에포크 추적
    global_scenario_count = 0  # 전체 처리된 시나리오 수 추적 (step으로 사용)
    last_logged_scenario = 0  # 마지막 로깅 시나리오 수
    log_interval_scenarios = 1000  # 1000개 시나리오마다 로깅
    batch_size = config['training']['batch_size']
    
    num_epochs = config['training']['num_epochs']
    # KL weight는 고정값 사용 (어닐링 없음)
    current_kl_weight = get_kl_weight(config)[0]
    print(f'Using fixed KL weight = {current_kl_weight:.6f}')
    
    for epoch in range(start_epoch, num_epochs):
        
        # Train
        train_loss, train_recon_loss, train_kl_loss, current_scenario_count, last_log_scenario = train_epoch(
            model, train_loader, optimizer, device, config, epoch, num_epochs,
            writer=writer, use_wandb=args.use_wandb, 
            global_scenario_count=global_scenario_count, 
            log_interval_scenarios=log_interval_scenarios,
            batch_size=batch_size,
            last_logged_scenario=last_logged_scenario,
            kl_weight=current_kl_weight  # 계산된 KL weight 전달
        )

        # global_scenario_count 업데이트 (에포크가 끝난 후)
        global_scenario_count = current_scenario_count
        last_logged_scenario = last_log_scenario

        # VRAM 메모리 정리 (에포크 끝나고 캐시 정리)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Best reconstruction loss 업데이트
        if train_recon_loss < best_recon_loss:
            best_recon_loss = train_recon_loss
        
        # KL weight는 고정값 사용 (어닐링 없음)
        
        # Logging (카테고리별로 구분된 메트릭) - 에포크 평균값
        log_dict = {
            'train_loss/loss': train_loss,
            'train_loss/recon': train_recon_loss,
            'train_loss/kl': train_kl_loss,
            'lr/lr': current_lr,
            'kl_weight/current': current_kl_weight,  # 현재 KL weight 로깅
            'epoch': epoch
        }
        
        # 시나리오 단위 step 사용 (지금까지 진행한 총 시나리오 수)
        scenario_step = global_scenario_count
        
        # TensorBoard (에포크 평균값, 시나리오 단위 step 사용 - 총 시나리오 수)
        writer.add_scalar('Loss/Train', train_loss, scenario_step)
        writer.add_scalar('Loss/Train_Recon', train_recon_loss, scenario_step)
        writer.add_scalar('Loss/Train_KL', train_kl_loss, scenario_step)
        writer.add_scalar('Learning_Rate', current_lr, scenario_step)
        writer.add_scalar('KL_Weight/Current', current_kl_weight, scenario_step)  # KL weight 추적
        
        # Wandb (카테고리별 메트릭 기록, step은 지금까지 진행한 총 시나리오 수 사용)
        if args.use_wandb:
            wandb.log(log_dict, step=scenario_step)
        
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Learning Rate Scheduler 업데이트 (reconstruction error 기준)
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(train_recon_loss)  # reconstruction error를 기준으로 학습률 조정
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f'  ⚠️  학습률 조정: {old_lr:.6f} → {new_lr:.6f} (Recon Loss: {train_recon_loss:.4f})')
        
        # Save checkpoint
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(
                config['logging']['save_dir'],
                f'best_model_epoch_{epoch}.pth'
            )
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'wandb_id': wandb.run.id if (args.use_wandb and wandb.run) else None,
            }
            # Scheduler state 저장 (있는 경우)
            if scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)
            print(f'  Saved best model to {checkpoint_path}')
            
            if args.use_wandb and wandb.run:
                wandb.run.summary['best_train_loss'] = best_train_loss
                wandb.run.summary['best_epoch'] = epoch
        
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['save_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'wandb_id': wandb.run.id if (args.use_wandb and wandb.run) else None,
            }
            # Scheduler state 저장 (있는 경우)
            if scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)
            print(f'  Saved checkpoint to {checkpoint_path}')
    
    writer.close()
    if args.use_wandb:
        wandb.finish()
    print('Training completed!')
    
    # Analyze latent space after training
    print('\n' + '='*60)
    print('Starting Latent Space Analysis')
    print('='*60)
    
    # Check if original trajectories are saved
    original_trajectories_path = os.path.join(save_path, 'original_trajectories.npz')
    if os.path.exists(original_trajectories_path):
        print(f'Found original trajectory file: {original_trajectories_path}')
        best_checkpoint_path = None
        if best_epoch >= 0:
            best_checkpoint_path = os.path.join(config['logging']['save_dir'], f'best_model_epoch_{best_epoch}.pth')
        
        analyze_latent_space_after_training(
            model=model,
            original_trajectories_path=original_trajectories_path,
            dataset=train_dataset,
            device=device,
            save_dir=os.path.join(save_path, 'latent_analysis'),
            best_checkpoint_path=best_checkpoint_path
        )
    else:
        print(f'Original trajectory file not found: {original_trajectories_path}')
        print('Trajectories were not saved during training.')


if __name__ == '__main__':
    main()
