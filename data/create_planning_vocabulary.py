"""
Planning Vocabulary 생성 스크립트
- 샘플링된 1000개의 궤적을 대상으로 K-means 클러스터링 수행
- 클러스터링의 중심점(centers)들이 Planning Vocabulary가 됨
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
import json

# 환경 변수에서 기본 경로 가져오기
DEFAULT_DATA_PATH = os.getenv('DATA_PATH', os.path.expanduser('~/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz'))


def create_planning_vocabulary(data_path, k=100, num_samples=1000, random_seed=42, save_dir='./planning_vocabulary'):
    """
    Planning Vocabulary 생성
    
    Args:
        data_path: .npz 파일 경로
        k: 어휘집 크기 (클러스터 수)
        num_samples: 샘플링할 궤적 수
        random_seed: 랜덤 시드
        save_dir: 결과 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 로드
    print(f"데이터 로드 중: {data_path}")
    data = np.load(data_path)
    trajectories = data['trajectories'].astype(np.float32)
    
    print(f"로드된 데이터 shape: {trajectories.shape}")
    
    # 240차원인 경우 160차원으로 변환
    if trajectories.shape[1] == 240:
        print("240차원 데이터를 160차원으로 변환 중...")
        trajectories_reshaped = trajectories.reshape(-1, 80, 3)
        trajectories_xy = trajectories_reshaped[:, :, :2]  # x, y만 추출
        trajectories = trajectories_xy.reshape(-1, 160).astype(np.float32)
        print(f"변환 후 shape: {trajectories.shape}")
    
    # 샘플링: 1000개 궤적 랜덤 선택
    np.random.seed(random_seed)
    total_samples = len(trajectories)
    num_samples = min(num_samples, total_samples)
    
    if num_samples < total_samples:
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        sampled_trajectories = trajectories[sample_indices]
        print(f"랜덤 샘플링: {num_samples}개 궤적 선택 (전체 {total_samples}개 중)")
    else:
        sampled_trajectories = trajectories
        sample_indices = np.arange(total_samples)
        print(f"전체 {total_samples}개 궤적 사용")
    
    # K-means 클러스터링 수행
    print(f"\nK-means 클러스터링 수행 중 (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(sampled_trajectories)
    
    # 클러스터 중심점 (Planning Vocabulary)
    vocabulary = kmeans.cluster_centers_  # (k, 160)
    
    print(f"Planning Vocabulary 생성 완료!")
    print(f"  어휘집 크기: {k}")
    print(f"  Vocabulary shape: {vocabulary.shape}")
    print(f"  샘플링된 궤적 수: {num_samples}")
    
    # 클러스터별 통계
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n클러스터별 궤적 수:")
    print(f"  평균: {np.mean(counts):.1f}")
    print(f"  최소: {np.min(counts)}")
    print(f"  최대: {np.max(counts)}")
    print(f"  표준편차: {np.std(counts):.1f}")
    
    # Vocabulary 저장
    print(f"\nPlanning Vocabulary 저장 중...")
    
    # NumPy 배열로 저장
    vocab_path = os.path.join(save_dir, f'planning_vocabulary_k{k}.npz')
    np.savez(vocab_path,
             vocabulary=vocabulary,
             cluster_labels=cluster_labels,
             sample_indices=sample_indices,
             k=k,
             num_samples=num_samples)
    print(f"Vocabulary 저장: {vocab_path}")
    
    # JSON 메타데이터 저장
    metadata = {
        'k': int(k),
        'vocabulary_size': int(k),
        'num_samples': int(num_samples),
        'total_trajectories': int(total_samples),
        'vocabulary_shape': list(vocabulary.shape),
        'cluster_statistics': {
            'mean_cluster_size': float(np.mean(counts)),
            'min_cluster_size': int(np.min(counts)),
            'max_cluster_size': int(np.max(counts)),
            'std_cluster_size': float(np.std(counts))
        },
        'random_seed': int(random_seed)
    }
    
    metadata_path = os.path.join(save_dir, f'planning_vocabulary_k{k}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"메타데이터 저장: {metadata_path}")
    
    # 시각화
    print(f"\n시각화 생성 중...")
    
    # Vocabulary를 (k, 80, 2) 형태로 변환
    vocabulary_xy = vocabulary.reshape(k, 80, 2)
    
    # 1. Vocabulary 전체 시각화
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 각 vocabulary 항목을 다른 색상으로 표시
    colors = plt.cm.tab20(np.linspace(0, 1, min(k, 20)))
    
    for i in range(k):
        vocab_traj = vocabulary_xy[i]
        color = colors[i % len(colors)]
        ax.plot(vocab_traj[:, 0], vocab_traj[:, 1], 
               color=color, linewidth=2, alpha=0.7, 
               label=f'Vocab {i}' if i < 10 else '')
        # 시작점과 끝점 표시
        ax.scatter(vocab_traj[0, 0], vocab_traj[0, 1], 
                  c=color, s=50, marker='o', zorder=5, alpha=0.8)
        ax.scatter(vocab_traj[-1, 0], vocab_traj[-1, 1], 
                  c=color, s=50, marker='s', zorder=5, alpha=0.8)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Planning Vocabulary (k={k})', fontsize=14)
    if k <= 10:
        ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    save_path = os.path.join(save_dir, f'planning_vocabulary_k{k}_overlay.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Vocabulary 오버레이 저장: {save_path}")
    plt.close()
    
    # 2. Vocabulary 그리드 시각화
    grid_size = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(k):
        vocab_traj = vocabulary_xy[i]
        ax = axes[i]
        
        # Vocabulary trajectory 그리기
        ax.plot(vocab_traj[:, 0], vocab_traj[:, 1], 'b-', linewidth=2)
        ax.scatter(vocab_traj[0, 0], vocab_traj[0, 1], 
                  c='green', s=50, marker='o', zorder=5)
        ax.scatter(vocab_traj[-1, 0], vocab_traj[-1, 1], 
                  c='red', s=50, marker='s', zorder=5)
        
        # 클러스터 크기 표시
        cluster_size = counts[i]
        ax.set_title(f'Vocab {i} (n={cluster_size})', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # 빈 subplot 숨기기
    for i in range(k, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Planning Vocabulary Grid (k={k})', fontsize=16, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'planning_vocabulary_k{k}_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Vocabulary 그리드 저장: {save_path}")
    plt.close()
    
    # 3. 클러스터 크기 분포 히스토그램
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=min(50, k), alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Cluster Size', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Cluster Size Distribution (k={k})', fontsize=14)
    ax.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'planning_vocabulary_k{k}_cluster_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"클러스터 분포 히스토그램 저장: {save_path}")
    plt.close()
    
    # 4. 전체 경로를 클러스터 군집색으로 시각화
    print(f"\n전체 경로 클러스터 시각화 생성 중...")
    
    # 샘플링된 궤적을 (num_samples, 80, 2) 형태로 변환
    sampled_trajectories_xy = sampled_trajectories.reshape(num_samples, 80, 2)
    
    # 클러스터별 색상 생성
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, min(k, 20)))
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 각 궤적을 해당 클러스터 색상으로 그리기
    for i in tqdm(range(num_samples), desc="궤적 그리기"):
        traj = sampled_trajectories_xy[i]
        cluster_id = cluster_labels[i]
        color = cluster_colors[cluster_id % len(cluster_colors)]
        ax.plot(traj[:, 0], traj[:, 1], 
               color=color, linewidth=0.5, alpha=0.3)
    
    # Vocabulary 중심점도 함께 표시 (더 진하게)
    for i in range(k):
        vocab_traj = vocabulary_xy[i]
        color = cluster_colors[i % len(cluster_colors)]
        ax.plot(vocab_traj[:, 0], vocab_traj[:, 1], 
               color=color, linewidth=3, alpha=0.9, 
               label=f'Cluster {i}' if i < 10 else '')
        # 시작점 표시
        ax.scatter(vocab_traj[0, 0], vocab_traj[0, 1], 
                  c=color, s=100, marker='o', zorder=5, alpha=0.9, edgecolors='black', linewidths=1)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'전체 경로 클러스터 시각화 (k={k}, 샘플 수={num_samples})', fontsize=14)
    if k <= 10:
        ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    save_path = os.path.join(save_dir, f'planning_vocabulary_k{k}_all_trajectories_clustered.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"전체 경로 클러스터 시각화 저장: {save_path}")
    plt.close()
    
    print(f"\n모든 작업 완료! 결과는 {save_dir}에 저장되었습니다.")
    print(f"\nPlanning Vocabulary 요약:")
    print(f"  어휘집 크기 (k): {k}")
    print(f"  Vocabulary shape: {vocabulary.shape}")
    print(f"  각 vocabulary 항목은 (160,) 차원의 궤적 벡터")
    print(f"  사용: vocabulary.reshape(k, 80, 2)로 (80, 2) 형태로 변환 가능")
    
    return vocabulary, cluster_labels, sample_indices


def main():
    parser = argparse.ArgumentParser(description='Planning Vocabulary 생성')
    parser.add_argument('--data_path', type=str,
                       default=DEFAULT_DATA_PATH,
                       help='.npz 파일 경로 (환경 변수 DATA_PATH로 설정 가능)')
    parser.add_argument('--k', type=int, default=100, help='어휘집 크기 (클러스터 수)')
    parser.add_argument('--num_samples', type=int, default=1000, help='샘플링할 궤적 수')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--save_dir', type=str, default='./planning_vocabulary', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    create_planning_vocabulary(
        data_path=args.data_path,
        k=args.k,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
