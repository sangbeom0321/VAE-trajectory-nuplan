"""
Trajectory 데이터를 README 기준에 따라 분류하여 시각화하는 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from collections import Counter

# 환경 변수에서 기본 경로 가져오기
DEFAULT_DATA_PATH = os.getenv('DATA_PATH', os.path.expanduser('~/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz'))


def classify_trajectory_readme(trajectory_xy):
    """
    README 기준에 따른 궤적 분류
    
    분류 기준:
    - Stop: 경로 길이 < 2.0m
    - Straight: -10° ≤ 각도 ≤ 10°
    - Straight(sharp): 직진 + 평균 곡률 > 0.15 rad 또는 최대 곡률 > 0.3 rad
    - Straight(slow): 직진 + 평균 속도 < 5 m/s
    - Left Turn: 각도 > 10°
    - Left Turn(Slow): 좌회전 + 평균 속도 < 5 m/s
    - Right Turn: 각도 < -10°
    - Right Turn(Slow): 우회전 + 평균 속도 < 5 m/s
    
    Args:
        trajectory_xy: (80, 2) - [x, y] coordinate array
        
    Returns:
        label: 분류 라벨 문자열
    """
    # Start and end points
    start = trajectory_xy[0]
    end = trajectory_xy[-1]
    
    # Total distance traveled (직선 거리)
    total_distance = np.linalg.norm(end - start)
    
    # Stop: 경로 길이 < 2.0m
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
    else:  # angle_deg < -10.0
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
    else:
        avg_curvature = 0
        max_curvature = 0
    
    # 곡률 분류: 급커브 vs 완만한
    is_sharp = avg_curvature > 0.15 or max_curvature > 0.3  # 라디안 기준
    
    # 속도 분류: 평균 속도 계산
    # 8초 경로, 80 타임스텝 = 10Hz, 따라서 0.5 m/step = 5 m/s
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds[speeds > 1e-6]) if np.any(speeds > 1e-6) else 0
    is_slow = avg_speed < 0.5  # 5 m/s 미만
    
    # 조합된 라벨 생성 (README 기준)
    if direction == 'straight':
        if is_sharp:
            return 'straight_sharp'
        elif is_slow:
            return 'straight_slow'
        else:
            return 'straight'
    elif direction == 'left':
        if is_slow:
            return 'left_slow'
        else:
            return 'left'
    else:  # direction == 'right'
        if is_slow:
            return 'right_slow'
        else:
            return 'right'


def get_color_for_label(label):
    """
    라벨에 따른 색상 반환 (이미지 참고)
    
    Returns:
        color: matplotlib 색상 또는 (edgecolor, facecolor) 튜플
    """
    color_map = {
        'stop': 'r',  # 빨간색
        'straight': 'orange',  # 주황색
        'straight_sharp': ('orange', 'darkorange'),  # 주황색 테두리 + 어두운 주황색
        'straight_slow': ('orange', 'yellow'),  # 주황색 테두리 + 노란색
        'left': 'b',  # 파란색
        'left_slow': ('b', 'lightblue'),  # 파란색 테두리 + 옅은 파란색
        'right': 'g',  # 녹색
        'right_slow': ('g', 'lightgreen'),  # 녹색 테두리 + 옅은 녹색
    }
    return color_map.get(label, 'gray')


def visualize_trajectories_classified(data_path, num_samples=100000, max_display=5000, save_dir='./trajectory_visualizations'):
    """
    Trajectory 데이터를 분류하여 시각화
    
    Args:
        data_path: .npz 파일 경로
        num_samples: 시각화할 샘플 수 (최대)
        max_display: 실제로 그릴 샘플 수
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
    
    # 샘플 수 제한 (None이면 전체 사용)
    if num_samples is not None:
        num_samples = min(num_samples, len(trajectories))
        trajectories = trajectories[:num_samples]
    else:
        num_samples = len(trajectories)
    
    # (N, 160) -> (N, 80, 2)로 reshape
    trajectories_xy = trajectories.reshape(-1, 80, 2)
    
    # 시작점 (x_0, y_0)은 항상 (0, 0)이어야 함 (로컬 좌표계)
    trajectories_xy[:, 0, 0] = 0.0  # x_0 = 0
    trajectories_xy[:, 0, 1] = 0.0  # y_0 = 0
    
    print(f"시각화할 샘플 수: {len(trajectories_xy)}")
    
    # 분류 수행
    print("\n궤적 분류 중...")
    labels = []
    for traj in tqdm(trajectories_xy, desc="Classifying"):
        label = classify_trajectory_readme(traj)
        labels.append(label)
    
    labels = np.array(labels)
    
    # 분류 통계 출력
    print("\n분류 통계:")
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        percentage = count / len(labels) * 100
        print(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
    
    # max_display 샘플 선택 (None이면 전체 사용)
    unique_labels = np.unique(labels)
    
    if max_display is not None:
        display_count = min(max_display, len(trajectories_xy))
        # 각 클래스별로 샘플 선택
        selected_indices = []
        samples_per_class = display_count // len(unique_labels)
        remaining = display_count % len(unique_labels)
        
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                n_samples = samples_per_class + (1 if i < remaining else 0)
                n_samples = min(n_samples, len(label_indices))
                selected = np.random.choice(label_indices, size=n_samples, replace=False)
                selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
    else:
        # 전체 샘플 사용
        selected_indices = np.arange(len(trajectories_xy))
        display_count = len(trajectories_xy)
    
    print(f"\n실제로 그릴 샘플 수: {len(selected_indices)}")
    
    # 통계 정보
    print("\n경로 통계:")
    all_x = trajectories_xy[:, :, 0].flatten()
    all_y = trajectories_xy[:, :, 1].flatten()
    print(f"  X 범위: [{np.min(all_x):.2f}, {np.max(all_x):.2f}]")
    print(f"  Y 범위: [{np.min(all_y):.2f}, {np.max(all_y):.2f}]")
    print(f"  X 평균: {np.mean(all_x):.2f}, 표준편차: {np.std(all_x):.2f}")
    print(f"  Y 평균: {np.mean(all_y):.2f}, 표준편차: {np.std(all_y):.2f}")
    
    print(f"\n시각화 생성 중...")
    
    # 전체 경로 오버레이 (색상별로 구분)
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 각 라벨별로 그리기
    for label in unique_labels:
        label_indices = selected_indices[labels[selected_indices] == label]
        color = get_color_for_label(label)
        
        if isinstance(color, tuple):
            # 테두리와 채우기 색상이 다른 경우
            edgecolor, facecolor = color
            for idx in tqdm(label_indices, desc=f"Drawing {label}", leave=False):
                traj = trajectories_xy[idx]
                ax.plot(traj[:, 0], traj[:, 1], color=edgecolor, alpha=0.3, linewidth=1.5)
        else:
            # 단색인 경우
            for idx in tqdm(label_indices, desc=f"Drawing {label}", leave=False):
                traj = trajectories_xy[idx]
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3, linewidth=1.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Trajectory Visualization by Classification ({len(selected_indices)} samples)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 범례 추가
    legend_elements = []
    for label in sorted(unique_labels):
        color = get_color_for_label(label)
        if isinstance(color, tuple):
            edgecolor, facecolor = color
            legend_elements.append(plt.Line2D([0], [0], color=edgecolor, lw=2, 
                                            label=label.replace('_', ' ').title()))
        else:
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                            label=label.replace('_', ' ').title()))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    save_path = os.path.join(save_dir, 'trajectories_classified.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"분류별 경로 오버레이 저장: {save_path}")
    plt.close()
    
    # 클래스별 개별 시각화
    print("\n클래스별 개별 시각화 생성 중...")
    for label in unique_labels:
        label_indices = selected_indices[labels[selected_indices] == label]
        if len(label_indices) == 0:
            continue
        
        # 전체 샘플 사용
        display_indices = label_indices
        
        fig, ax = plt.subplots(figsize=(12, 12))
        color = get_color_for_label(label)
        
        if isinstance(color, tuple):
            edgecolor, facecolor = color
            for idx in display_indices:
                traj = trajectories_xy[idx]
                ax.plot(traj[:, 0], traj[:, 1], color=edgecolor, alpha=0.4, linewidth=1.5)
        else:
            for idx in display_indices:
                traj = trajectories_xy[idx]
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.4, linewidth=1.5)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{label.replace("_", " ").title()} Trajectories ({len(display_indices)} samples)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        save_path = os.path.join(save_dir, f'trajectories_{label}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  {label} 시각화 저장: {save_path}")
        plt.close()
    
    print(f"\n모든 시각화 완료! 결과는 {save_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='Trajectory 분류 시각화')
    parser.add_argument('--data_path', type=str, 
                       default=DEFAULT_DATA_PATH,
                       help='.npz 파일 경로 (환경 변수 DATA_PATH로 설정 가능)')
    parser.add_argument('--num_samples', type=int, default=None, help='시각화할 샘플 수 (None이면 전체)')
    parser.add_argument('--max_display', type=int, default=None, help='실제로 그릴 샘플 수 (None이면 전체)')
    parser.add_argument('--save_dir', type=str, default='./trajectory_visualizations', help='저장 디렉토리')
    
    args = parser.parse_args()
    
    visualize_trajectories_classified(
        data_path=args.data_path,
        num_samples=args.num_samples,
        max_display=args.max_display,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
