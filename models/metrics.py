"""
평가 지표: ADE, FDE
"""

import torch
import numpy as np


def compute_ade(predicted, target):
    """
    Average Displacement Error (ADE)
    전체 80개 점에 대한 평균 거리 오차
    
    Args:
        predicted: (batch, 80, 2) 또는 (batch, 160) - 예측된 경로
        target: (batch, 80, 2) 또는 (batch, 160) - 실제 경로
        
    Returns:
        ade: scalar - 평균 ADE
    """
    # Flatten된 형태인 경우 reshape
    if len(predicted.shape) == 2:
        predicted = predicted.reshape(-1, 80, 2)
        target = target.reshape(-1, 80, 2)
    
    # 각 타임스텝에서의 거리 오차 계산
    displacement_errors = torch.norm(predicted - target, dim=-1)  # (batch, 80)
    
    # 전체 타임스텝에 대한 평균
    ade = displacement_errors.mean()
    
    return ade.item()


def compute_fde(predicted, target):
    """
    Final Displacement Error (FDE)
    8초 마지막 지점(t=80)의 종점 오차
    
    Args:
        predicted: (batch, 80, 2) 또는 (batch, 160) - 예측된 경로
        target: (batch, 80, 2) 또는 (batch, 160) - 실제 경로
        
    Returns:
        fde: scalar - 평균 FDE
    """
    # Flatten된 형태인 경우 reshape
    if len(predicted.shape) == 2:
        predicted = predicted.reshape(-1, 80, 2)
        target = target.reshape(-1, 80, 2)
    
    # 마지막 지점 (t=80, 인덱스 79)
    final_predicted = predicted[:, -1, :]  # (batch, 2)
    final_target = target[:, -1, :]  # (batch, 2)
    
    # 종점 거리 오차
    fde = torch.norm(final_predicted - final_target, dim=-1).mean()
    
    return fde.item()


def compute_metrics(predicted, target):
    """
    ADE와 FDE를 모두 계산
    
    Args:
        predicted: (batch, 80, 2) 또는 (batch, 160) - 예측된 경로
        target: (batch, 80, 2) 또는 (batch, 160) - 실제 경로
        
    Returns:
        metrics: dict - {'ade': float, 'fde': float}
    """
    ade = compute_ade(predicted, target)
    fde = compute_fde(predicted, target)
    
    return {
        'ade': ade,
        'fde': fde
    }
