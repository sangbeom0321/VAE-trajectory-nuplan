"""
VAE Loss 함수 정의

Loss 구성:
- Reconstruction Loss (MSE): 복원된 trajectory와 원본 trajectory 간의 차이
- KL Divergence Loss: Posterior q(z|x)와 Prior p(z) = N(0, I) 간의 KL divergence
- Total Loss: L = Reconstruction Loss + β * KL Divergence Loss

이론적 배경:
VAE의 목적은 Evidence Lower Bound (ELBO)를 최대화하는 것입니다:
  ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

Loss로 변환하면 (최소화해야 하므로):
  Loss = -ELBO = -E_q(z|x)[log p(x|z)] + KL(q(z|x) || p(z))
              = Reconstruction Loss + KL Divergence Loss

따라서 두 loss를 더하는 것이 맞습니다 (빼는 것이 아닙니다).
KL divergence는 항상 양수이며, 이를 최소화하면 posterior q(z|x)가 
prior p(z) = N(0, I)에 가까워집니다.

수식:
- Reconstruction: L_recon = MSE(x, x_recon) = ||x - x_recon||²
- KL Divergence: L_KL = KL(q(z|x) || N(0, I)) = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
- Total: L_total = L_recon + β * L_KL
"""

import torch
import torch.nn.functional as F


def kl_divergence_loss(mu, logvar, normalize_by_dim=True):
    """
    KL Divergence Loss 계산
    
    Posterior q(z|x) = N(μ, σ²)와 Prior p(z) = N(0, I) 간의 KL divergence를 계산합니다.
    
    수식:
        KL(q(z|x) || N(0, I)) = -0.5 * Σ(1 + logvar - mu² - exp(logvar))
                              = 0.5 * Σ(exp(logvar) + mu² - 1 - logvar)
                              = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
    
    참고: 표준 VAE 구현과 동일한 형태입니다:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    
    Args:
        mu: (batch, latent_dim) - Posterior의 평균 벡터
        logvar: (batch, latent_dim) - Posterior의 로그 분산 벡터 (log(σ²))
        normalize_by_dim: (bool) - True면 latent_dim으로 나누어 정규화 (기본값: True)
                          Reconstruction loss와 스케일을 맞추기 위해 사용
    
    Returns:
        kl_loss: (scalar) - 배치 평균 KL divergence loss (정규화됨)
    """
    # 표준 VAE 구현 형태: -0.5 * Σ(1 + logvar - mu² - exp(logvar))
    # 이것은 0.5 * Σ(exp(logvar) + mu² - 1 - logvar)와 수학적으로 동일합니다
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
    
    # 정규화: latent_dim으로 나누어 차원당 평균 KL로 변환
    # 이렇게 하면 Reconstruction loss와 비슷한 스케일이 됨
    if normalize_by_dim:
        latent_dim = mu.shape[1]
        kl = kl / latent_dim
    
    # 배치 평균 반환 (표준 구현과 동일)
    return kl.mean()


def reconstruction_loss_mse(predicted, target, mask=None):
    """
    Reconstruction Loss 계산 (Mean Squared Error)
    
    복원된 trajectory와 원본 trajectory 간의 평균 제곱 오차를 계산합니다.
    
    수식:
        L_recon = MSE(x, x_recon) = ||x - x_recon||²
    
    Args:
        predicted: (batch, ...) - 모델이 복원한 trajectory
        target: (batch, ...) - 원본 trajectory
        mask: (batch, ...) - 유효한 데이터 마스크 (None이면 모든 데이터 유효)
    
    Returns:
        mse_loss: (scalar) - 평균 제곱 오차
    """
    if mask is not None:
        # 마스크가 제공된 경우: 유효한 데이터만 고려
        diff = (predicted - target) ** 2
        masked_diff = diff * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)  # 안정성을 위한 작은 epsilon 추가
    else:
        # 마스크가 없는 경우: 모든 데이터에 대해 평균 loss 계산
        loss = F.mse_loss(predicted, target)
    
    return loss


def compute_loss(
    reconstructed_ego_future_trajectory,
    ego_future_trajectory,
    mu,
    logvar,
    kl_weight=1.0,
    normalize_kl_by_dim=True
):
    """
    VAE Total Loss 계산
    
    Reconstruction Loss와 KL Divergence Loss를 결합하여 전체 손실을 계산합니다.
    
    수식:
        L_total = L_recon + β * L_KL
                = MSE(x, x_recon) + β * KL(q(z|x) || N(0, I))
    
    Args:
        reconstructed_ego_future_trajectory: (batch, 160) 또는 (batch, T_future, future_dim)
            - 모델이 복원한 미래 trajectory
        ego_future_trajectory: (batch, 160) 또는 (batch, T_future, future_dim)
            - 원본 미래 trajectory (ground truth)
        mu: (batch, latent_dim) - Posterior의 평균 벡터
        logvar: (batch, latent_dim) - Posterior의 로그 분산 벡터
        kl_weight: (float) - KL Loss 가중치 (β 값, 기본값: 1.0)
        normalize_kl_by_dim: (bool) - KL loss를 latent_dim으로 정규화할지 여부 (기본값: True)
    
    Returns:
        total_loss: (scalar) - 전체 손실 (L_recon + β * L_KL)
        recon_loss: (scalar) - 재구성 손실
        kl_loss: (scalar) - KL divergence 손실 (정규화됨)
    """
    # 입력 shape 정규화: 3차원인 경우 2차원으로 flatten
    if len(reconstructed_ego_future_trajectory.shape) == 3:
        batch_size = reconstructed_ego_future_trajectory.shape[0]
        reconstructed_flat = reconstructed_ego_future_trajectory.reshape(batch_size, -1)
        target_flat = ego_future_trajectory.reshape(batch_size, -1)
    else:
        reconstructed_flat = reconstructed_ego_future_trajectory
        target_flat = ego_future_trajectory
    
    # Reconstruction Loss 계산
    recon_loss = reconstruction_loss_mse(reconstructed_flat, target_flat)
    
    # KL Divergence Loss 계산 (latent_dim으로 정규화하여 스케일 맞춤)
    kl_loss = kl_divergence_loss(mu, logvar, normalize_by_dim=normalize_kl_by_dim)
    
    # Total Loss: L = L_recon + β * L_KL
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss
