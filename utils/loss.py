"""
Loss 함수 정의
- L2 Loss (Reconstruction Loss) - 모든 입력 데이터에 대해
- KL Divergence Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(mu, logvar):
    """
    KL Divergence between posterior q(z|x) and prior p(z) = N(0, I)
    KL(q(z|x) || N(0, I))
    
    Args:
        mu: (batch, latent_dim) - posterior mean
        logvar: (batch, latent_dim) - posterior log variance
    Returns:
        kl_loss: scalar
    """
    # KL(q || N(0, I)) = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    kl = 0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1 - logvar,
        dim=1
    )
    
    return kl.mean()


def reconstruction_loss_mse(predicted, target, mask=None):
    """
    L2 Reconstruction Loss (MSE)
    
    Args:
        predicted: (batch, ...) - 복원된 데이터
        target: (batch, ...) - 원본 데이터
        mask: (batch, ...) - 유효한 데이터 마스크 (None이면 모두 유효)
    Returns:
        mse_loss: scalar
    """
    if mask is not None:
        # 유효한 데이터만 고려
        diff = (predicted - target) ** 2
        masked_diff = diff * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)
    else:
        # 모든 데이터에 대해 평균 loss 계산
        loss = F.mse_loss(predicted, target)
    
    return loss


def compute_loss(
    reconstructed_ego_future_trajectory, ego_future_trajectory,
    mu, logvar,
    kl_weight=0.01
):
    """
    160차원 경로 벡터에 대한 Loss 계산
    
    Args:
        reconstructed_ego_future_trajectory: (batch, 160) 또는 (batch, T_future, future_dim)
        ego_future_trajectory: (batch, 160) 또는 (batch, T_future, future_dim)
        mu: (batch, latent_dim)
        logvar: (batch, latent_dim)
        kl_weight: KL Loss 가중치
    Returns:
        total_loss: scalar
        recon_loss: scalar
        kl_loss: scalar
    """
    # 입력이 3차원인 경우 2차원으로 flatten
    if len(reconstructed_ego_future_trajectory.shape) == 3:
        batch_size = reconstructed_ego_future_trajectory.shape[0]
        reconstructed_flat = reconstructed_ego_future_trajectory.reshape(batch_size, -1)
        target_flat = ego_future_trajectory.reshape(batch_size, -1)
    else:
        reconstructed_flat = reconstructed_ego_future_trajectory
        target_flat = ego_future_trajectory
    
    # Reconstruction loss (MSE)
    recon_loss = reconstruction_loss_mse(reconstructed_flat, target_flat)
    
    # KL divergence loss
    kl_loss = kl_divergence_loss(mu, logvar)
    
    # Total loss: L = MSE Loss + β * KL Divergence Loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss
