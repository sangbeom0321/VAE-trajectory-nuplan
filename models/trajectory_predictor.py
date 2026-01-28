"""
전체 Trajectory Predictor 모델 통합
Ego Future Trajectory만 사용하는 VAE 구조
"""

import torch
import torch.nn as nn
from .vae import VAEEncoder, VAEDecoder, reparameterize


class TrajectoryPredictor(nn.Module):
    """
    Ego Future Trajectory만 사용하는 VAE 기반 Trajectory Predictor
    Ego Future Trajectory를 인코딩하고 복원
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        # 입력 데이터 차원 계산
        self.future_horizon = model_cfg['future_horizon']
        self.future_dim = model_cfg['future_dim']
        
        # Ego Future Trajectory 차원: (future_horizon, future_dim)
        self.input_dim = self.future_horizon * self.future_dim
        
        # VAE Components
        latent_dim = model_cfg['latent_dim']
        
        # 정규화 여부 확인 (config에서 data.normalize 확인)
        # 정규화된 데이터면 Tanh 사용, 아니면 제거
        data_cfg = config.get('data', {})
        use_normalize = data_cfg.get('normalize', True)  # 기본값: True (정규화 사용)
        use_tanh = use_normalize  # 정규화 사용 시에만 Tanh 적용
        
        # Encoder: 160차원 경로 벡터 → 32차원 latent
        # Deep MLP 구조: 160 → 512 → 256 → 128 → 32
        self.vae_encoder = VAEEncoder(
            input_dim=self.input_dim,  # 160차원
            latent_dim=latent_dim,  # 32차원
            hidden_dims=[512, 256, 128]  # Deep MLP 구조
        )
        
        # Decoder: 32차원 latent → 160차원 경로 벡터 복원
        # Deep MLP 구조: 32 → 128 → 256 → 512 → 160
        # 정규화 사용 시: Tanh로 [-1, 1] 범위 제한
        # 정규화 없이: Tanh 제거하여 원본 스케일로 출력
        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,  # 32차원
            output_dim=self.input_dim,  # 160차원
            hidden_dims=[128, 256, 512],  # Encoder와 대칭 구조
            use_tanh=use_tanh  # 정규화 여부에 따라 Tanh 적용/제거
        )
        
        # 출력 shape 정보 저장
        self.future_shape = (self.future_horizon, self.future_dim)
    
    def _flatten_input(self, ego_future_trajectory):
        """
        Ego Future Trajectory를 하나의 벡터로 flatten
        
        Args:
            ego_future_trajectory: (batch, future_horizon, future_dim)
        Returns:
            flattened: (batch, input_dim)
        """
        batch_size = ego_future_trajectory.shape[0]
        
        # Flatten
        flattened = ego_future_trajectory.reshape(batch_size, -1)  # (batch, future_horizon * future_dim)
        
        # 차원 검증
        assert flattened.shape[1] == self.input_dim, \
            f"Flattened dimension mismatch: {flattened.shape[1]} != {self.input_dim}"
        
        return flattened
    
    def _unflatten_output(self, reconstructed):
        """
        복원된 벡터를 원래 shape으로 복원
        
        Args:
            reconstructed: (batch, input_dim)
        Returns:
            ego_future_trajectory: (batch, future_horizon, future_dim)
        """
        batch_size = reconstructed.shape[0]
        
        # Reshape to original shape
        ego_future_trajectory = reconstructed.reshape(batch_size, *self.future_shape)
        
        return ego_future_trajectory
    
    
    def forward(self, ego_future_trajectory, mode='train'):
        """
        Args:
            ego_future_trajectory: (batch, future_horizon, future_dim) - Ego Future Trajectory
            mode: 'train' or 'inference' (현재는 동일하게 처리)
        Returns:
            reconstructed_ego_future_trajectory: (batch, future_horizon, future_dim)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Flatten input
        flattened_input = self._flatten_input(ego_future_trajectory)
        
        # Encode to latent
        mu, logvar = self.vae_encoder(flattened_input)
        z = reparameterize(mu, logvar)
        
        # Decode to reconstruct
        reconstructed_flat = self.vae_decoder(z)
        
        # Unflatten to original shape
        reconstructed_ego_future = self._unflatten_output(reconstructed_flat)
        
        return {
            'reconstructed_ego_future_trajectory': reconstructed_ego_future,
            'mu': mu,
            'logvar': logvar
        }
    
    def sample(self, num_samples=1, batch_size=1, device='cuda'):
        """
        Multi-modal 예측을 위한 샘플링
        Args:
            num_samples: 샘플링할 경로 수
            batch_size: 배치 크기
            device: 디바이스
        Returns:
            samples: (batch_size, num_samples, future_horizon, future_dim)
        """
        # Prior에서 샘플링 (N(0, I))
        z = torch.randn(batch_size, self.vae_encoder.mu_head.out_features, device=device)
        
        samples = []
        for _ in range(num_samples):
            # Decode
            reconstructed_flat = self.vae_decoder(z)
            
            # Unflatten to original shape
            recon_future = self._unflatten_output(reconstructed_flat)
            
            samples.append(recon_future)
        
        # Stack samples: (batch_size, num_samples, future_horizon, future_dim)
        samples = torch.stack(samples, dim=1)
        
        return samples
