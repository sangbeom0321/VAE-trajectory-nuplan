"""
Trajectory 데이터 시각화 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

# 환경 변수에서 기본 경로 가져오기
DEFAULT_DATA_PATH = os.getenv('DATA_PATH', os.path.expanduser('~/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz'))


def visualize_trajectories(data_path, num_samples=100000, max_display=None, save_dir='./trajectory_visualizations'):
    """
    Trajectory 데이터 시각화
    
    Args:
        data_path: .npz 파일 경로
        num_samples: 시각화할 샘플 수 (최대)
        max_display: 실제로 그릴 샘플 수 (None이면 전체 샘플 모두 그림)
        save_dir: 저장 디렉토리
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
    
    # 샘플 수 제한
    num_samples = min(num_samples, len(trajectories))
    trajectories = trajectories[:num_samples]
    
    # (N, 160) -> (N, 80, 2)로 reshape
    trajectories_xy = trajectories.reshape(-1, 80, 2)
    
    # 시작점 (x_0, y_0)은 항상 (0, 0)이어야 함 (로컬 좌표계)
    trajectories_xy[:, 0, 0] = 0.0  # x_0 = 0
    trajectories_xy[:, 0, 1] = 0.0  # y_0 = 0
    
    print(f"시각화할 샘플 수: {len(trajectories_xy)}")
    
    # max_display가 None이면 전체 샘플 모두 그림
    if max_display is None:
        display_count = len(trajectories_xy)
        indices = np.arange(len(trajectories_xy))
        print(f"실제로 그릴 샘플 수: {display_count} (전체)")
    else:
        display_count = min(max_display, len(trajectories_xy))
        indices = np.linspace(0, len(trajectories_xy) - 1, display_count, dtype=int)
        print(f"실제로 그릴 샘플 수: {display_count}")
    
    print(f"시작점 강제 설정: 모든 경로의 시작점을 (0, 0)으로 설정")
    
    # 통계 정보
    print("\n경로 통계:")
    all_x = trajectories_xy[:, :, 0].flatten()
    all_y = trajectories_xy[:, :, 1].flatten()
    print(f"  X 범위: [{np.min(all_x):.2f}, {np.max(all_x):.2f}]")
    print(f"  Y 범위: [{np.min(all_y):.2f}, {np.max(all_y):.2f}]")
    print(f"  X 평균: {np.mean(all_x):.2f}, 표준편차: {np.std(all_x):.2f}")
    print(f"  Y 평균: {np.mean(all_y):.2f}, 표준편차: {np.std(all_y):.2f}")
    
    print(f"\n시각화 생성 중...")
    
    # 전체 경로 오버레이 (선으로만 표시)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for idx in tqdm(indices, desc="Drawing trajectories"):
        traj = trajectories_xy[idx]
        # 선으로만 그리기 (점 없음)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.2, linewidth=1.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Trajectory Visualization ({display_count} samples)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    save_path = os.path.join(save_dir, 'all_trajectories_overlay.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"전체 경로 오버레이 저장: {save_path}")
    plt.close()
    
    print(f"\n모든 시각화 완료! 결과는 {save_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='Trajectory 시각화')
    parser.add_argument('--data_path', type=str, 
                       default=DEFAULT_DATA_PATH,
                       help='.npz 파일 경로 (환경 변수 DATA_PATH로 설정 가능)')
    parser.add_argument('--num_samples', type=int, default=100000, help='시각화할 샘플 수')
    parser.add_argument('--max_display', type=int, default=5000, help='실제로 그릴 샘플 수 (오버레이용, 기본값: 5000)')
    parser.add_argument('--save_dir', type=str, default='./trajectory_visualizations', help='저장 디렉토리')
    
    args = parser.parse_args()
    
    visualize_trajectories(
        data_path=args.data_path,
        num_samples=args.num_samples,
        max_display=args.max_display,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
