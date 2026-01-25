"""
Latent Space 분석 스크립트
- 각 경로가 latent space의 어디로 매핑되는지 확인
- 군집 분석: 정지, 좌회전, 우회전, 직진 분류
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json

# 프로젝트 루트를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.trajectory_predictor import TrajectoryPredictor
from models.vae import reparameterize
from data.trajectory_dataset import TrajectoryDataset
from torch.utils.data import DataLoader


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


def extract_latent_representations(model, dataloader, device='cuda', max_samples=None):
    """
    데이터셋의 모든 경로를 latent space에 인코딩
    
    Args:
        model: 학습된 VAE 모델
        dataloader: 데이터 로더
        device: 디바이스
        
    Returns:
        latent_z: (N, 32) - 각 경로의 latent representation
        latent_mu: (N, 32) - 각 경로의 latent mean
        trajectories_xy: List of (80, 2) - 원본 경로들
        labels: List of str - 각 경로의 분류 라벨
    """
    model.eval()
    
    latent_z_list = []
    latent_mu_list = []
    trajectories_xy_list = []
    labels_list = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Latent 인코딩")):
            # 데이터 로드
            # TrajectoryDataset의 collate_fn은 'trajectories' 키를 사용
            if 'trajectories' in batch:
                trajectories = batch['trajectories'].to(device)  # (batch, 160)
                trajectories_reshaped = trajectories.reshape(-1, 80, 2)  # (batch, 80, 2)
            elif 'trajectory' in batch:
                # 단일 샘플인 경우
                trajectories = batch['trajectory'].to(device)  # (batch, 160)
                trajectories_reshaped = trajectories.reshape(-1, 80, 2)  # (batch, 80, 2)
            elif 'ego_future_trajectory' in batch:
                trajectories_reshaped = batch['ego_future_trajectory'].to(device)
            else:
                raise KeyError(f"Batch에 'trajectories' 또는 'ego_future_trajectory' 키가 없습니다. 사용 가능한 키: {batch.keys()}")
            
            batch_size = trajectories_reshaped.shape[0]
            
            # Latent 인코딩
            flattened = model._flatten_input(trajectories_reshaped)
            mu, logvar = model.vae_encoder(flattened)
            z = reparameterize(mu, logvar)
            
            # CPU로 이동 및 numpy 변환
            z_np = z.cpu().numpy()
            mu_np = mu.cpu().numpy()
            trajectories_np = trajectories_reshaped.cpu().numpy()
            
            # 각 샘플에 대해 처리
            for i in range(batch_size):
                trajectory_xy = trajectories_np[i]  # (80, 2)
                
                # 정규화된 데이터이므로 역정규화하여 분류 (분류는 원본 스케일에서 수행)
                # 데이터셋에서 역정규화 함수 사용
                if hasattr(dataloader.dataset, 'normalize') and dataloader.dataset.normalize:
                    trajectory_flat = trajectory_xy.flatten()  # (160,)
                    trajectory_denorm = dataloader.dataset._denormalize_trajectory(trajectory_flat)
                    trajectory_xy = trajectory_denorm.reshape(80, 2)  # (80, 2)
                
                # 경로 분류
                label = classify_trajectory(trajectory_xy)
                
                latent_z_list.append(z_np[i])
                latent_mu_list.append(mu_np[i])
                trajectories_xy_list.append(trajectory_xy)
                labels_list.append(label)
                
                sample_count += 1
                
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
    
    return np.array(latent_z_list), np.array(latent_mu_list), trajectories_xy_list, labels_list


def visualize_latent_space(latent_z, labels, trajectories_xy=None, save_dir='./latent_analysis', use_tsne=False, num_samples_per_label=5):
    """
    Latent space 시각화 (PCA 중심)
    
    Args:
        latent_z: (N, 32) - Latent representations
        labels: List of str - 각 샘플의 라벨
        trajectories_xy: List of (80, 2) - 원본 trajectory들 (선택사항, 있으면 샘플 표시)
        save_dir: 저장 디렉토리
        use_tsne: t-SNE도 함께 사용할지 여부 (기본값: False, PCA만 사용)
        num_samples_per_label: 각 라벨당 표시할 trajectory 샘플 수
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Label colors
    label_colors = {
        'stop': 'red',
        'left': 'blue',
        'right': 'green',
        'straight': 'orange'
    }
    
    # Label names
    label_names = {
        'stop': 'Stop',
        'left': 'Left Turn',
        'right': 'Right Turn',
        'straight': 'Straight'
    }
    
    # Dimensionality reduction with PCA
    print("Reducing dimensions with PCA...")
    
    # Check for invalid values before PCA
    if np.any(np.isnan(latent_z)) or np.any(np.isinf(latent_z)):
        print("Warning: Invalid values detected before PCA. Cleaning...")
        latent_z = np.nan_to_num(latent_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_z)
    
    explained_variance = pca.explained_variance_ratio_
    print(f'PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}')
    
    # 시각화
    if use_tsne:
        # t-SNE도 함께 사용하는 경우
        print("t-SNE로 차원 축소 중...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        latent_2d_tsne = tsne.fit_transform(latent_z)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # PCA 시각화
        ax1 = axes[0]
        for label in ['stop', 'left', 'right', 'straight']:
            mask = np.array(labels) == label
            if np.any(mask):
                ax1.scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1],
                           c=label_colors[label], label=label_names[label],
                           alpha=0.6, s=20)
        ax1.set_xlabel(f'PC1 (설명 분산: {explained_variance[0]:.2%})', fontsize=12)
        ax1.set_ylabel(f'PC2 (설명 분산: {explained_variance[1]:.2%})', fontsize=12)
        ax1.set_title('Latent Space Visualization (PCA)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # t-SNE 시각화
        ax2 = axes[1]
        for label in ['stop', 'left', 'right', 'straight']:
            mask = np.array(labels) == label
            if np.any(mask):
                ax2.scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1],
                           c=label_colors[label], label=label_names[label],
                           alpha=0.6, s=20)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.set_title('Latent Space Visualization (t-SNE)', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'latent_space_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 저장: {save_path}")
        plt.close()
        
        return latent_2d_tsne, latent_2d_pca
    else:
        # PCA만 사용
        if trajectories_xy is not None:
            # Convert to numpy array if it's a list
            if isinstance(trajectories_xy, list):
                trajectories_xy = np.array(trajectories_xy)
            
            # Trajectory 샘플과 함께 시각화
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Main latent space plot
            ax_main = fig.add_subplot(gs[:, 0])
            
            # Trajectory samples subplot
            ax_samples = fig.add_subplot(gs[0, 1])
            ax_samples.set_title('Trajectory Samples', fontsize=12)
            ax_samples.grid(True, alpha=0.3)
            
            # Calculate reasonable axis limits based on data
            all_x = trajectories_xy[:, :, 0].flatten()
            all_y = trajectories_xy[:, :, 1].flatten()
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
            sample_indices = {}
            for label in ['stop', 'left', 'right', 'straight']:
                mask = np.array(labels) == label
                if np.any(mask):
                    indices = np.where(mask)[0]
                    # Randomly sample
                    n_samples = min(num_samples_per_label, len(indices))
                    if n_samples > 0:
                        selected_indices = np.random.choice(indices, n_samples, replace=False)
                        sample_indices[label] = selected_indices
            
            # Plot all points in latent space
            for label in ['stop', 'left', 'right', 'straight']:
                mask = np.array(labels) == label
                if np.any(mask):
                    count = np.sum(mask)
                    ax_main.scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1],
                                  c=label_colors[label], label=f'{label_names[label]} (n={count})',
                                  alpha=0.4, s=15)
            
            # Highlight sample trajectories
            for label in ['stop', 'left', 'right', 'straight']:
                if label in sample_indices:
                    for idx in sample_indices[label]:
                        # Plot trajectory sample
                        traj = trajectories_xy[idx]
                        ax_samples.plot(traj[:, 0], traj[:, 1], 
                                       color=label_colors[label], 
                                       linewidth=1.5, alpha=0.7,
                                       label=label_names[label] if idx == sample_indices[label][0] else '')
                        
                        # Mark corresponding point in latent space
                        ax_main.scatter(latent_2d_pca[idx, 0], latent_2d_pca[idx, 1],
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
                mask = np.array(labels) == label
                if np.any(mask):
                    # Plot a few representative trajectories
                    indices = np.where(mask)[0]
                    n_plot = min(10, len(indices))
                    plot_indices = np.random.choice(indices, n_plot, replace=False)
                    
                    for idx in plot_indices:
                        traj = trajectories_xy[idx]
                        ax_categories.plot(traj[:, 0], traj[:, 1], 
                                          color=label_colors[label], 
                                          linewidth=1.0, alpha=0.5)
            
            ax_categories.set_xlabel('X (m)', fontsize=10)
            ax_categories.set_ylabel('Y (m)', fontsize=10)
            
            # Add legend for categories
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=label_colors[label], label=label_names[label]) 
                             for label in ['stop', 'left', 'right', 'straight'] 
                             if np.any(np.array(labels) == label)]
            ax_categories.legend(handles=legend_elements, fontsize=8, loc='best')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, 'latent_space_pca.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
            plt.close()
        else:
            # Simple visualization without trajectory samples
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            for label in ['stop', 'left', 'right', 'straight']:
                mask = np.array(labels) == label
                if np.any(mask):
                    count = np.sum(mask)
                    ax.scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1],
                              c=label_colors[label], label=f'{label_names[label]} (n={count})',
                              alpha=0.6, s=20)
            
            ax.set_xlabel(f'PC1 (explained variance: {explained_variance[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 (explained variance: {explained_variance[1]:.2%})', fontsize=12)
            ax.set_title('Latent Space Visualization (PCA)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, 'latent_space_pca.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
            plt.close()
        
        return None, latent_2d_pca


def perform_clustering(latent_z, n_clusters=4, save_dir='./latent_analysis'):
    """
    K-means 클러스터링 수행
    
    Args:
        latent_z: (N, 32) - Latent representations
        n_clusters: 클러스터 수
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"K-means 클러스터링 수행 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_z)
    
    # 클러스터 중심점
    cluster_centers = kmeans.cluster_centers_
    
    # t-SNE로 시각화
    print("t-SNE로 차원 축소 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_2d = tsne.fit_transform(latent_z)
    
    # 클러스터 중심은 PCA로 시각화 (샘플 수가 적어서 t-SNE 사용 불가)
    if len(cluster_centers) < 30:
        print("클러스터 중심은 PCA로 시각화 (샘플 수가 적음)...")
        pca = PCA(n_components=2)
        centers_2d = pca.fit_transform(cluster_centers)
    else:
        tsne_centers = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cluster_centers)-1), max_iter=1000)
        centers_2d = tsne_centers.fit_transform(cluster_centers)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', 
                         alpha=0.6, s=20)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
               c='red', marker='x', s=200, linewidths=3, 
               label='Cluster Centers')
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'K-means Clustering (k={n_clusters})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'kmeans_clustering.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"클러스터링 시각화 저장: {save_path}")
    plt.close()
    
    return cluster_labels, cluster_centers


def analyze_cluster_statistics(labels, cluster_labels, save_dir='./latent_analysis'):
    """
    클러스터별 통계 분석
    
    Args:
        labels: List of str - 각 샘플의 분류 라벨 (stop, left, right, straight)
        cluster_labels: (N,) - K-means 클러스터 라벨
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # 클러스터별 분류 라벨 분포
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_labels_subset = np.array(labels)[mask]
        
        stats = {
            'total': len(cluster_labels_subset),
            'stop': np.sum(cluster_labels_subset == 'stop'),
            'left': np.sum(cluster_labels_subset == 'left'),
            'right': np.sum(cluster_labels_subset == 'right'),
            'straight': np.sum(cluster_labels_subset == 'straight')
        }
        
        # 비율 계산
        if stats['total'] > 0:
            stats['stop_ratio'] = stats['stop'] / stats['total']
            stats['left_ratio'] = stats['left'] / stats['total']
            stats['right_ratio'] = stats['right'] / stats['total']
            stats['straight_ratio'] = stats['straight'] / stats['total']
        else:
            stats['stop_ratio'] = 0
            stats['left_ratio'] = 0
            stats['right_ratio'] = 0
            stats['straight_ratio'] = 0
        
        cluster_stats[cluster_id] = stats
    
    # 결과 출력
    print("\n" + "="*80)
    print("클러스터별 통계")
    print("="*80)
    for cluster_id in sorted(unique_clusters):
        stats = cluster_stats[cluster_id]
        print(f"\n클러스터 {cluster_id}:")
        print(f"  총 샘플 수: {stats['total']}")
        print(f"  정지: {stats['stop']} ({stats['stop_ratio']:.1%})")
        print(f"  좌회전: {stats['left']} ({stats['left_ratio']:.1%})")
        print(f"  우회전: {stats['right']} ({stats['right_ratio']:.1%})")
        print(f"  직진: {stats['straight']} ({stats['straight_ratio']:.1%})")
    
    # JSON으로 저장 (numpy 타입을 Python 기본 타입으로 변환)
    def convert_to_python_types(obj):
        """numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    cluster_stats_python = convert_to_python_types(cluster_stats)
    save_path = os.path.join(save_dir, 'cluster_statistics.json')
    with open(save_path, 'w') as f:
        json.dump(cluster_stats_python, f, indent=2)
    print(f"\n통계 저장: {save_path}")
    print("="*80)
    
    return cluster_stats


def save_latent_mappings(latent_z, latent_mu, labels, trajectories_xy, save_dir='./latent_analysis'):
    """
    Latent 매핑 결과 저장
    
    Args:
        latent_z: (N, 32) - Latent representations
        latent_mu: (N, 32) - Latent means
        labels: List of str - 각 샘플의 분류 라벨
        trajectories_xy: List of (80, 2) - 원본 경로들
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # NumPy 배열로 저장
    save_path = os.path.join(save_dir, 'latent_mappings.npz')
    np.savez(save_path,
             latent_z=latent_z,
             latent_mu=latent_mu,
             labels=np.array(labels),
             trajectories=np.array(trajectories_xy))
    
    print(f"Latent 매핑 저장: {save_path}")
    
    # 요약 정보 저장
    summary = {
        'num_samples': len(labels),
        'latent_dim': latent_z.shape[1],
        'label_distribution': {
            'stop': int(np.sum(np.array(labels) == 'stop')),
            'left': int(np.sum(np.array(labels) == 'left')),
            'right': int(np.sum(np.array(labels) == 'right')),
            'straight': int(np.sum(np.array(labels) == 'straight'))
        }
    }
    
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"요약 정보 저장: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Latent Space Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data_path', type=str, default=None, help='Dataset .npz file path (optional if using --original_trajectories)')
    parser.add_argument('--original_trajectories', type=str, default=None, help='Original trajectories .npz file path (saved during training)')
    parser.add_argument('--norm_params_path', type=str, default=None, help='Normalization parameters JSON file path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to analyze (None for all)')
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of K-means clusters')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./latent_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if either data_path or original_trajectories is provided
    if not args.data_path and not args.original_trajectories:
        parser.error("Either --data_path or --original_trajectories must be provided")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 모델 로드
    print("모델 로드 중...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 처리
    def expand_paths(obj):
        if isinstance(obj, dict):
            return {k: expand_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_paths(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    config = expand_paths(config)
    
    model = TrajectoryPredictor(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: {args.checkpoint}")
    
    # Load trajectories
    if args.original_trajectories:
        # Use saved original trajectories directly
        print("Loading original trajectories...")
        data = np.load(args.original_trajectories)
        original_trajectories = data['trajectories']  # (N, 80, 2)
        print(f"Loaded: {len(original_trajectories)} samples")
        
        # Need dataset for normalization info
        if args.data_path:
            dataset = TrajectoryDataset(
                data_path=args.data_path,
                norm_params_path=args.norm_params_path,
                normalize=True
            )
        else:
            # Try to find norm params from various locations
            checkpoint_dir = os.path.dirname(args.checkpoint)
            original_traj_dir = os.path.dirname(args.original_trajectories)
            
            # Expand environment variables
            config_norm_path = args.norm_params_path
            if config_norm_path:
                config_norm_path = os.path.expandvars(config_norm_path)
            
            possible_norm_paths = [
                config_norm_path,
                os.path.join(original_traj_dir, 'trajectories_8s_norm_params.json'),
                os.path.join(checkpoint_dir, '..', '..', 'trajectories_8s_norm_params.json'),
                os.path.expandvars('$HOME/99_dataset/01_nuplan/dataset/exp2/trajectories_8s_norm_params.json'),
            ]
            
            norm_path = None
            for path in possible_norm_paths:
                if path and os.path.exists(path):
                    norm_path = path
                    print(f"Found normalization parameters: {norm_path}")
                    break
            
            if norm_path:
                # Create a minimal dataset-like object for normalization
                # Load norm params manually
                with open(norm_path, 'r') as f:
                    norm_params = json.load(f)
                
                # Create a simple object with normalization method
                class SimpleNormalizer:
                    def __init__(self, norm_params):
                        self.norm_params = norm_params
                        self.normalize = True
                    
                    def _normalize_trajectory(self, trajectory):
                        traj_min = np.array(self.norm_params['min'], dtype=np.float32)
                        traj_max = np.array(self.norm_params['max'], dtype=np.float32)
                        traj_range = traj_max - traj_min
                        traj_range = np.where(traj_range < 1e-6, 1.0, traj_range)
                        
                        normalized = (trajectory - traj_min) / traj_range * 2.0 - 1.0
                        normalized[0] = 0.0  # x_0 = 0
                        normalized[1] = 0.0  # y_0 = 0
                        return normalized
                
                dataset = SimpleNormalizer(norm_params)
            else:
                print("Warning: Normalization parameters not found!")
                print("Please provide --data_path or --norm_params_path for proper normalization.")
                print("Attempting to proceed without normalization (may cause errors)...")
                dataset = None
        
        # Classify trajectories
        print("Classifying trajectories...")
        labels = []
        for traj in original_trajectories:
            label = classify_trajectory(traj)
            labels.append(label)
        labels = np.array(labels)
        
        # Encode to latent space
        print(f"\nAnalyzing {len(original_trajectories)} samples")
        latent_z_list = []
        latent_mu_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(original_trajectories), args.batch_size), desc="Encoding"):
                batch_trajectories = original_trajectories[i:i+args.batch_size]  # (batch, 80, 2)
                
                # Apply normalization if dataset is available
                batch_normalized = []
                for traj in batch_trajectories:
                    traj_flat = traj.flatten()  # (160,)
                    if dataset and hasattr(dataset, 'normalize') and dataset.normalize:
                        traj_normalized = dataset._normalize_trajectory(traj_flat)
                    else:
                        traj_normalized = traj_flat
                    batch_normalized.append(traj_normalized)
                
                batch_normalized = np.array(batch_normalized)  # (batch, 160)
                
                # Check for invalid values
                if np.any(np.isnan(batch_normalized)) or np.any(np.isinf(batch_normalized)):
                    print(f"Warning: Invalid values detected in batch {i//args.batch_size}")
                    print(f"  NaN count: {np.sum(np.isnan(batch_normalized))}")
                    print(f"  Inf count: {np.sum(np.isinf(batch_normalized))}")
                    # Replace invalid values with 0
                    batch_normalized = np.nan_to_num(batch_normalized, nan=0.0, posinf=0.0, neginf=0.0)
                
                batch_tensor = torch.FloatTensor(batch_normalized).to(device)
                
                # Reshape to (batch, 80, 2)
                batch_reshaped = batch_tensor.reshape(-1, 80, 2)
                
                # Encode to latent space
                flattened = model._flatten_input(batch_reshaped)
                mu, logvar = model.vae_encoder(flattened)
                z = reparameterize(mu, logvar)
                
                # Check for invalid values in latent space
                z_np = z.cpu().numpy()
                mu_np = mu.cpu().numpy()
                
                if np.any(np.isnan(z_np)) or np.any(np.isinf(z_np)):
                    print(f"Warning: Invalid values in latent z for batch {i//args.batch_size}")
                    z_np = np.nan_to_num(z_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                if np.any(np.isnan(mu_np)) or np.any(np.isinf(mu_np)):
                    print(f"Warning: Invalid values in latent mu for batch {i//args.batch_size}")
                    mu_np = np.nan_to_num(mu_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                latent_z_list.append(z_np)
                latent_mu_list.append(mu_np)
                
                if args.max_samples and len(latent_z_list) * args.batch_size >= args.max_samples:
                    break
        
        latent_z = np.concatenate(latent_z_list, axis=0)  # (N, 32)
        latent_mu = np.concatenate(latent_mu_list, axis=0)  # (N, 32)
        trajectories_xy = original_trajectories[:len(latent_z)]
        labels = labels[:len(latent_z)]
        
        # Final check for invalid values
        invalid_z = np.isnan(latent_z) | np.isinf(latent_z)
        invalid_mu = np.isnan(latent_mu) | np.isinf(latent_mu)
        
        if np.any(invalid_z) or np.any(invalid_mu):
            print(f"Warning: Found invalid values in final latent representations")
            print(f"  Invalid z: {np.sum(invalid_z)} values")
            print(f"  Invalid mu: {np.sum(invalid_mu)} values")
            # Replace invalid values
            latent_z = np.nan_to_num(latent_z, nan=0.0, posinf=0.0, neginf=0.0)
            latent_mu = np.nan_to_num(latent_mu, nan=0.0, posinf=0.0, neginf=0.0)
        
    else:
        # Use dataset loader (original method)
        print("Loading dataset...")
        from data.trajectory_dataset import collate_fn
        
        dataset = TrajectoryDataset(
            data_path=args.data_path,
            norm_params_path=args.norm_params_path,
            normalize=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Extract latent representations
        print(f"\nAnalyzing {args.max_samples if args.max_samples else len(dataset)} samples out of {len(dataset)} total")
        latent_z, latent_mu, trajectories_xy, labels = extract_latent_representations(
            model, dataloader, device=device, max_samples=args.max_samples
        )
    
    print(f"\nExtraction complete: {len(latent_z)} samples")
    print(f"  Latent z shape: {latent_z.shape}")
    print(f"  Latent mu shape: {latent_mu.shape}")
    
    # Print label distribution
    print("\nLabel distribution:")
    for label in ['stop', 'left', 'right', 'straight']:
        count = np.sum(np.array(labels) == label)
        print(f"  {label}: {count} ({count/len(labels):.1%})")
    
    # Visualize latent space (PCA-based)
    print("\nVisualizing latent space...")
    # trajectories_xy should be defined in both branches above
    if 'trajectories_xy' not in locals():
        trajectories_xy = None
    _, latent_2d_pca = visualize_latent_space(
        latent_z, labels, 
        trajectories_xy=trajectories_xy,
        save_dir=args.save_dir, 
        use_tsne=False,
        num_samples_per_label=5
    )
    
    # K-means clustering
    print("\nPerforming K-means clustering...")
    cluster_labels, cluster_centers = perform_clustering(
        latent_z, n_clusters=args.n_clusters, save_dir=args.save_dir
    )
    
    # Cluster statistics analysis
    print("\nAnalyzing cluster statistics...")
    cluster_stats = analyze_cluster_statistics(labels, cluster_labels, save_dir=args.save_dir)
    
    # Save results
    print("\nSaving results...")
    save_latent_mappings(latent_z, latent_mu, labels, trajectories_xy, save_dir=args.save_dir)
    
    print(f"\nAll analysis complete! Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
