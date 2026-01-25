"""
8초 주행 경로(Ego-trajectory) 추출 스크립트
- nuPlan DB에서 8초 분량의 에고 경로 10만 개를 로컬 좌표계로 추출
- 샘플링 비율: 10Hz (0.1초 간격)
- 시퀀스 길이: 8.0초 (총 80개 타임스텝)
- 입력 차원: [x_0, y_0, x_1, y_1, ..., x_79, y_79] 총 160차원 단일 벡터
"""

import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple
import signal
import atexit
import time
import sys
import random

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):
    """
    ScenarioFilter를 위한 파라미터 생성 함수
    data_process.py와 동일한 함수
    """
    scenario_types = None
    scenario_tokens = scenario_tokens  # List of scenario tokens to include
    log_names = log_names  # Filter scenarios by log names
    map_names = None  # Filter scenarios by map names

    num_scenarios_per_type = num_scenarios_per_type  # Number of scenarios per type
    limit_total_scenarios = limit_total_scenarios  # Limit total scenarios (float = fraction, int = num)
    timestamp_threshold_s = None  # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None  # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True  # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False  # Whether to remove scenarios where the mission goal is invalid
    shuffle = shuffle  # Whether to shuffle the scenarios

    ego_start_speed_threshold = None  # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None  # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None  # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


class TrajectoryExtractor:
    """
    8초 경로 추출기
    """
    
    def __init__(self, sampling_rate: int = 10, sequence_length: float = 8.0):
        """
        Args:
            sampling_rate: 샘플링 비율 (Hz)
            sequence_length: 시퀀스 길이 (초)
        """
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.num_samples = int(sampling_rate * sequence_length)  # 80개
        
    def extract_trajectory(self, scenario) -> np.ndarray:
        """
        시나리오에서 8초 경로 추출 및 로컬 좌표계 변환
        data_process.py의 get_ego_future_array_from_scenario와 동일한 방식 사용
        
        Args:
            scenario: nuPlan 시나리오 객체
            
        Returns:
            trajectory: (160,) - [x_0, y_0, x_1, y_1, ..., x_79, y_79]
        """
        # 현재 ego state
        current_ego_state = scenario.initial_ego_state
        
        # 8초 미래 경로 추출
        future_trajectory_absolute_states = scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=self.num_samples,
            time_horizon=self.sequence_length
        )
        
        # 절대 좌표에서 상대 좌표로 변환 (로컬 좌표계)
        # data_process.py의 get_ego_future_array_from_scenario와 동일한 방식
        future_trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_ego_state.rear_axle,
            [state.rear_axle for state in future_trajectory_absolute_states]
        )
        
        # (num_samples, 2) 형태: [x, y]
        # 길이 확인 및 조정
        if len(future_trajectory_relative_poses) < self.num_samples:
            # 부족한 경우 마지막 포인트로 패딩
            last_point = future_trajectory_relative_poses[-1] if len(future_trajectory_relative_poses) > 0 else np.array([0.0, 0.0])
            padding = np.tile(last_point, (self.num_samples - len(future_trajectory_relative_poses), 1))
            future_trajectory_relative_poses = np.concatenate([future_trajectory_relative_poses, padding], axis=0)
        elif len(future_trajectory_relative_poses) > self.num_samples:
            # 초과하는 경우 자르기
            future_trajectory_relative_poses = future_trajectory_relative_poses[:self.num_samples]
        
        # convert_absolute_to_relative_poses는 시작점을 (0, 0)으로 변환하지만,
        # 수치 오차로 인해 정확히 (0, 0)이 아닐 수 있으므로 강제로 (0, 0)으로 설정
        start_point = future_trajectory_relative_poses[0]
        if np.abs(start_point[0]) > 1e-6 or np.abs(start_point[1]) > 1e-6:
            # 시작점을 (0, 0)으로 강제 설정
            future_trajectory_relative_poses = future_trajectory_relative_poses - start_point
        
        # Flatten: (80, 2) -> (160,)
        trajectory_flat = future_trajectory_relative_poses.flatten().astype(np.float32)
        
        return trajectory_flat
    
    def normalize_trajectories(self, trajectories: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        전체 데이터셋의 Max/Min을 계산하여 [-1, 1] 범위로 정규화
        시작점 (x_0, y_0)은 항상 (0, 0)이므로 정규화 후에도 (0, 0)으로 유지
        
        Args:
            trajectories: (N, 160) - N개의 경로
            
        Returns:
            normalized: (N, 160) - 정규화된 경로
            norm_params: 정규화 파라미터 딕셔너리
        """
        # 전체 데이터의 최대/최소값 계산
        traj_min = np.min(trajectories, axis=0)  # (160,)
        traj_max = np.max(trajectories, axis=0)  # (160,)
        
        # 범위 계산 (0으로 나누기 방지)
        traj_range = traj_max - traj_min
        traj_range = np.where(traj_range < 1e-6, 1.0, traj_range)
        
        # [-1, 1] 범위로 정규화: (x - min) / range * 2 - 1
        normalized = (trajectories - traj_min) / traj_range * 2.0 - 1.0
        
        # 시작점 (x_0, y_0)은 원본에서 (0, 0)이므로 정규화 후에도 (0, 0)으로 강제 설정
        # normalized[:, 0] = x_0, normalized[:, 1] = y_0
        # 시작점이 (0, 0)이면: (0 - min) / range * 2 - 1 = -min/range * 2 - 1
        # 이를 0으로 만들려면: normalized[:, 0] = 0, normalized[:, 1] = 0
        normalized[:, 0] = 0.0  # x_0 = 0
        normalized[:, 1] = 0.0  # y_0 = 0
        
        # 정규화 파라미터 저장
        norm_params = {
            'min': traj_min.tolist(),
            'max': traj_max.tolist(),
            'range': traj_range.tolist()
        }
        
        return normalized, norm_params


# 전역 변수 (중단 시 저장용)
_extracted_trajectories = []
_extract_start_time = None
_save_path = None


def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    print("\n\n중단 신호 수신. 현재까지 추출된 데이터 저장 중...")
    finalize_and_save()
    exit(0)


def finalize_and_save():
    """최종 저장"""
    global _extracted_trajectories, _save_path
    
    if len(_extracted_trajectories) > 0 and _save_path:
        trajectories_array = np.array(_extracted_trajectories, dtype=np.float32)
        
        # 중간 저장 파일명
        intermediate_path = _save_path.replace('.npz', '_intermediate.npz')
        np.savez(intermediate_path, trajectories=trajectories_array)
        print(f"\n중간 결과 저장 완료: {intermediate_path} ({len(_extracted_trajectories)}개 샘플)")


def main():
    global _extracted_trajectories, _extract_start_time, _save_path
    
    parser = argparse.ArgumentParser(description='8초 경로 추출')
    parser.add_argument('--data_path', type=str, required=True, help='nuPlan 원본 데이터 경로')
    parser.add_argument('--map_path', type=str, required=True, help='nuPlan 맵 데이터 경로')
    parser.add_argument('--save_path', type=str, required=True, help='저장 경로 (.npz 파일)')
    parser.add_argument('--num_samples', type=int, default=100000, help='추출할 샘플 수 (기본값: 100000)')
    parser.add_argument('--log_names_file', type=str, default=None, help='log_names JSON 파일 경로 (선택사항)')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='시나리오 타입당 개수')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='시나리오 셔플 여부')
    parser.add_argument('--num_db_files', type=int, default=30, help='사용할 db 파일 개수 (랜덤 선택)')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드 (재현 가능성을 위해)')
    
    args = parser.parse_args()
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(finalize_and_save)
    
    _save_path = args.save_path
    _extract_start_time = time.time()
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    
    print("="*80)
    print("8초 경로 추출 시작")
    print("="*80)
    print(f"데이터 경로: {args.data_path}")
    print(f"맵 경로: {args.map_path}")
    print(f"저장 경로: {args.save_path}")
    print(f"목표 샘플 수: {args.num_samples}")
    print(f"샘플링 비율: 10Hz")
    print(f"시퀀스 길이: 8.0초 (80 타임스텝)")
    print(f"출력 차원: 160차원 (80 * 2)")
    print("="*80)
    
    # log_names 로드 또는 자동 추출
    if args.log_names_file and os.path.exists(args.log_names_file):
        with open(args.log_names_file, 'r', encoding='utf-8') as f:
            log_names = json.load(f)
        print(f"log_names 파일 로드: {len(log_names)}개")
    else:
        # 데이터 경로에서 .db 파일들을 찾아서 log_names 자동 생성
        print(f"log_names 파일이 제공되지 않음. 데이터 경로에서 자동 추출 중...")
        db_files_list = []
        if os.path.exists(args.data_path):
            # 직접 .db 파일들이 있는 경우 (test 데이터셋)
            db_files_list = [f.replace('.db', '') for f in os.listdir(args.data_path) 
                           if f.endswith('.db') and os.path.isfile(os.path.join(args.data_path, f))]
            
            # 하위 디렉토리에 .db 파일들이 있는 경우 (trainval 데이터셋)
            if len(db_files_list) == 0:
                for root, dirs, files in os.walk(args.data_path):
                    for file in files:
                        if file.endswith('.db'):
                            # 파일명에서 확장자 제거하여 log_name 추출
                            db_files_list.append(file.replace('.db', ''))
        
        all_log_names = sorted(list(set(db_files_list)))  # 중복 제거 및 정렬
        print(f"데이터 경로에서 총 {len(all_log_names)}개 db 파일 발견")
        
        # 지정된 개수만큼 랜덤 선택
        if len(all_log_names) > args.num_db_files:
            random.seed(args.random_seed)
            log_names = random.sample(all_log_names, args.num_db_files)
            log_names = sorted(log_names)  # 정렬하여 출력
            print(f"랜덤으로 {args.num_db_files}개 db 파일 선택 (시드: {args.random_seed})")
        else:
            log_names = all_log_names
            print(f"전체 {len(log_names)}개 db 파일 사용 (요청한 {args.num_db_files}개보다 적음)")
        
        if len(log_names) > 0:
            print(f"  선택된 db 파일 목록:")
            for i, name in enumerate(log_names[:10]):  # 처음 10개만 출력
                print(f"    {i+1}. {name}")
            if len(log_names) > 10:
                print(f"    ... 외 {len(log_names) - 10}개")
    
    # NuPlan Scenario Builder 초기화
    print("\n시나리오 빌더 초기화 중...")
    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(
        data_root=args.data_path,
        map_root=args.map_path,
        sensor_root=None,
        db_files=None,
        map_version=map_version
    )
    
    # Scenario Filter 생성
    # 목표 샘플 수에 맞춰 시나리오 수를 제한 (일부 시나리오에서 추출 실패할 수 있으므로 여유를 둠)
    # 추출 성공률을 고려하여 목표 샘플 수의 1.2배 정도 로드
    limit_scenarios = int(args.num_samples * 1.2) if args.scenarios_per_type is None else None
    
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,  # num_scenarios_per_type
            limit_scenarios,  # limit_total_scenarios (목표 샘플 수에 맞춰 제한)
            args.shuffle_scenarios,  # shuffle
            log_names=log_names  # log_names
        )
    )
    
    # 시나리오 로드
    print("시나리오 로드 중...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    
    print(f"총 {len(scenarios)}개 시나리오 로드 완료 (목표 샘플 수: {args.num_samples}개)")
    
    # Trajectory Extractor 초기화
    extractor = TrajectoryExtractor(sampling_rate=10, sequence_length=8.0)
    
    # 경로 추출
    print(f"\n경로 추출 시작 (목표: {args.num_samples}개)...")
    extracted_count = 0
    failed_count = 0
    
    trajectories_list = []
    
    for scenario in tqdm(scenarios, desc="경로 추출"):
        try:
            trajectory = extractor.extract_trajectory(scenario)
            
            # 유효성 검사: NaN이나 Inf가 없는지 확인
            if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
                failed_count += 1
                continue
            
            # 유효성 검사: 경로가 너무 짧거나 이상한 경우 제외
            # 시작점과 끝점 사이 거리가 최소한 일정 거리 이상이어야 함
            start_point = trajectory[:2]
            end_point = trajectory[-2:]
            distance = np.linalg.norm(end_point - start_point)
            
            if distance < 0.1:  # 0.1m 미만이면 제외
                failed_count += 1
                continue
            
            trajectories_list.append(trajectory)
            extracted_count += 1
            
            # 목표 샘플 수 도달 시 종료
            if extracted_count >= args.num_samples:
                break
                
        except Exception as e:
            failed_count += 1
            if failed_count % 100 == 0:
                print(f"\n경고: {failed_count}개 시나리오 처리 실패 (최근 오류: {str(e)[:100]})")
            continue
    
    # 모든 데이터 저장
    print(f"\n추출된 데이터 저장 중...")
    if len(trajectories_list) > 0:
        trajectories_array = np.array(trajectories_list, dtype=np.float32)
        np.savez(args.save_path, trajectories=trajectories_array)
        print(f"데이터 저장 완료: {args.save_path} ({len(trajectories_list)}개 샘플)")
    else:
        print("저장할 데이터가 없습니다.")
        return
    
    # 최종 데이터 로드 및 후처리
    print(f"\n최종 데이터 로드 및 후처리 중...")
    trajectories = trajectories_array
    
    print(f"추출된 총 샘플 수: {len(trajectories)}")
    print(f"실패한 시나리오 수: {failed_count}")
    print(f"원본 데이터 shape: {trajectories.shape}")
    
    # 240차원인 경우 160차원으로 변환 (heading 제거)
    if trajectories.shape[1] == 240:
        print("240차원 데이터를 160차원으로 변환 중...")
        trajectories_reshaped = trajectories.reshape(-1, 80, 3)
        trajectories_xy = trajectories_reshaped[:, :, :2]  # x, y만 추출
        trajectories = trajectories_xy.reshape(-1, 160).astype(np.float32)
        print(f"변환 후 shape: {trajectories.shape}")
        # 변환된 데이터를 다시 저장
        np.savez(args.save_path, trajectories=trajectories)
        print(f"변환된 데이터 저장: {args.save_path}")
    elif trajectories.shape[1] != 160:
        raise ValueError(f"예상하지 못한 데이터 shape: {trajectories.shape}. 예상: (N, 160) 또는 (N, 240)")
    
    # 통계 출력
    print("\n" + "="*80)
    print("추출 완료 통계")
    print("="*80)
    print(f"총 추출 샘플 수: {len(trajectories)}")
    print(f"실패한 시나리오 수: {failed_count}")
    print(f"경로 shape: {trajectories.shape}")
    print(f"경로 통계:")
    trajectories_xy = trajectories.reshape(-1, 80, 2)
    print(f"  - X 범위: [{np.min(trajectories_xy[:, :, 0]):.2f}, {np.max(trajectories_xy[:, :, 0]):.2f}]")
    print(f"  - Y 범위: [{np.min(trajectories_xy[:, :, 1]):.2f}, {np.max(trajectories_xy[:, :, 1]):.2f}]")
    print(f"  - 시작점 (x_0, y_0): 모든 경로가 (0, 0)에서 시작")
    print(f"정규화: 모델 학습 시 적용됨")
    print("="*80)
    
    elapsed_time = time.time() - _extract_start_time
    print(f"\n총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
    print(f"평균 처리 속도: {len(trajectories)/elapsed_time:.2f} 샘플/초")


if __name__ == "__main__":
    main()
