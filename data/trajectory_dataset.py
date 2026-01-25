"""
8초 경로 데이터셋 로더
160차원 벡터 (80 타임스텝 * 2차원) 처리
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from typing import Dict


class TrajectoryDataset(Dataset):
    """
    8초 경로 데이터셋
    입력: 160차원 벡터 [x_0, y_0, x_1, y_1, ..., x_79, y_79]
    """
    
    def __init__(self, data_path: str, norm_params_path: str = None, normalize: bool = True, max_samples: int = None):
        """
        Args:
            data_path: .npz 파일 경로 (trajectories 키 포함)
            norm_params_path: 정규화 파라미터 JSON 파일 경로 (선택사항)
            normalize: 정규화 적용 여부 (파일명에 '_normalized'가 있으면 자동으로 False로 설정)
            max_samples: 사용할 최대 샘플 수 (None이면 전체 데이터 사용)
        """
        self.data_path = data_path
        
        # 파일명에 '_normalized'가 있으면 이미 정규화된 파일이므로 normalize=False로 설정
        if '_normalized' in os.path.basename(data_path):
            if normalize:
                print(f"경고: 정규화된 파일({os.path.basename(data_path)})을 사용합니다. normalize를 False로 설정합니다.")
            self.normalize = False
        else:
            self.normalize = normalize
        
        # 데이터 로드
        data = np.load(data_path)
        trajectories_raw = data['trajectories'].astype(np.float32)
        
        # 데이터 shape 확인 및 변환
        # 240차원인 경우 (80, 3) -> (80, 2)로 변환 (x, y만 사용)
        # 160차원인 경우 그대로 사용
        if trajectories_raw.shape[1] == 240:
            # (N, 240) -> (N, 80, 3) -> (N, 80, 2) -> (N, 160)
            trajectories_reshaped = trajectories_raw.reshape(-1, 80, 3)
            trajectories_xy = trajectories_reshaped[:, :, :2]  # x, y만 추출
            self.trajectories = trajectories_xy.reshape(-1, 160).astype(np.float32)
            print(f"경고: 240차원 데이터를 160차원으로 변환 (heading 제거)")
        elif trajectories_raw.shape[1] == 160:
            self.trajectories = trajectories_raw
        else:
            raise ValueError(f"예상하지 못한 데이터 shape: {trajectories_raw.shape}. 예상: (N, 160) 또는 (N, 240)")
        
        # 시작점이 (0, 0)인지 확인하고 강제 설정 (로컬 좌표계)
        # 모든 trajectory의 시작점 (x_0, y_0)을 (0, 0)으로 설정
        self.trajectories[:, 0] = 0.0  # x_0 = 0
        self.trajectories[:, 1] = 0.0  # y_0 = 0
        
        # 샘플 수 제한
        original_count = len(self.trajectories)
        if max_samples is not None and max_samples > 0:
            max_samples = min(max_samples, len(self.trajectories))
            self.trajectories = self.trajectories[:max_samples]
            print(f"샘플 수 제한: {max_samples}개 사용 (전체 {original_count}개 중)")
        
        print(f"데이터셋 로드 완료: {len(self.trajectories)}개 샘플")
        print(f"  원본 Shape: {trajectories_raw.shape}")
        print(f"  최종 Shape: {self.trajectories.shape}")
        print(f"  정규화 적용: {self.normalize}")
        
        # 정규화 파라미터 로드 또는 계산 (정규화를 적용할 때만)
        self.norm_params = None
        if self.normalize:
            if norm_params_path and os.path.exists(norm_params_path):
                # 정규화 파라미터 파일이 있으면 로드
                with open(norm_params_path, 'r') as f:
                    self.norm_params = json.load(f)
                print(f"정규화 파라미터 로드: {norm_params_path}")
            else:
                # 정규화 파라미터 파일이 없으면 데이터셋에서 계산
                print("정규화 파라미터 파일이 없습니다. 데이터셋에서 계산 중...")
                self.norm_params = self._compute_normalization_params()
                print("정규화 파라미터 계산 완료")
                
                # 계산된 파라미터 저장 (선택사항)
                if norm_params_path:
                    save_dir = os.path.dirname(norm_params_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    with open(norm_params_path, 'w') as f:
                        json.dump(self.norm_params, f, indent=2)
                    print(f"정규화 파라미터 저장: {norm_params_path}")
    
    def _compute_normalization_params(self) -> dict:
        """
        데이터셋에서 정규화 파라미터 계산
        
        Returns:
            norm_params: 정규화 파라미터 딕셔너리
        """
        # 전체 데이터의 최대/최소값 계산
        traj_min = np.min(self.trajectories, axis=0)  # (160,)
        traj_max = np.max(self.trajectories, axis=0)  # (160,)
        
        # 범위 계산 (0으로 나누기 방지)
        traj_range = traj_max - traj_min
        traj_range = np.where(traj_range < 1e-6, 1.0, traj_range)
        
        norm_params = {
            'min': traj_min.tolist(),
            'max': traj_max.tolist(),
            'range': traj_range.tolist()
        }
        
        return norm_params
    
    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        정규화 적용
        
        Args:
            trajectory: (160,) - 원본 경로 벡터
            
        Returns:
            normalized: (160,) - 정규화된 경로 벡터
        """
        if not self.normalize or self.norm_params is None:
            return trajectory
        
        traj_min = np.array(self.norm_params['min'], dtype=np.float32)
        traj_max = np.array(self.norm_params['max'], dtype=np.float32)
        traj_range = traj_max - traj_min
        traj_range = np.where(traj_range < 1e-6, 1.0, traj_range)
        
        # [-1, 1] 범위로 정규화
        normalized = (trajectory - traj_min) / traj_range * 2.0 - 1.0
        
        return normalized
    
    def _denormalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        정규화 해제 (역변환)
        시작점 (x_0, y_0)은 원본에서 (0, 0)이었으므로 역정규화 후에도 (0, 0)으로 강제 설정
        
        Args:
            trajectory: (160,) - 정규화된 경로 벡터
            
        Returns:
            denormalized: (160,) - 원본 스케일 경로 벡터
        """
        if not self.normalize or self.norm_params is None:
            return trajectory
        
        traj_min = np.array(self.norm_params['min'], dtype=np.float32)
        traj_max = np.array(self.norm_params['max'], dtype=np.float32)
        traj_range = traj_max - traj_min
        traj_range = np.where(traj_range < 1e-6, 1.0, traj_range)
        
        # 역변환: (x + 1) / 2 * range + min
        denormalized = (trajectory + 1.0) / 2.0 * traj_range + traj_min
        
        # 시작점 (x_0, y_0)은 원본에서 (0, 0)이었으므로 역정규화 후에도 (0, 0)으로 강제 설정
        # 정규화 과정에서 시작점을 강제로 0으로 설정했기 때문에, 역정규화만으로는 원래 값이 복원되지 않음
        denormalized[0] = 0.0  # x_0 = 0
        denormalized[1] = 0.0  # y_0 = 0
        
        return denormalized
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Returns:
            trajectory: (160,) - 정규화된 경로 벡터
        """
        trajectory = self.trajectories[idx].copy()
        
        # 정규화 적용
        if self.normalize:
            trajectory = self._normalize_trajectory(trajectory)
        
        # 시작점 (x_0, y_0)은 항상 (0, 0)이어야 함 (로컬 좌표계)
        # 정규화된 데이터에서도 시작점을 (0, 0)으로 강제 설정
        trajectory[0] = 0.0  # x_0 = 0
        trajectory[1] = 0.0  # y_0 = 0
        
        return {
            'trajectory': torch.FloatTensor(trajectory)  # (160,)
        }
    
    def get_trajectory_as_xy(self, idx: int, denormalize: bool = True) -> np.ndarray:
        """
        경로를 (80, 2) 형태로 반환
        
        Args:
            idx: 샘플 인덱스
            denormalize: 정규화 해제 여부
            
        Returns:
            trajectory_xy: (80, 2) - [x, y] 좌표 배열
        """
        trajectory = self.trajectories[idx].copy()
        
        if denormalize and self.normalize:
            trajectory = self._denormalize_trajectory(trajectory)
        elif not self.normalize:
            # 정규화되지 않은 데이터도 시작점이 (0, 0)인지 확인하고 강제 설정
            trajectory[0] = 0.0  # x_0 = 0
            trajectory[1] = 0.0  # y_0 = 0
        
        # (160,) -> (80, 2)
        trajectory_xy = trajectory.reshape(80, 2)
        
        return trajectory_xy


def collate_fn(batch):
    """
    Batch collate function
    
    Returns:
        trajectories: (batch, 160) - 배치화된 경로 벡터
    """
    trajectories = torch.stack([item['trajectory'] for item in batch])
    
    return {
        'trajectories': trajectories  # (batch, 160)
    }
