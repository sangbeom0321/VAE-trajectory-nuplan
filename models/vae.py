"""
VAE 모듈: 진짜 VAE 구조
- Encoder: 모든 입력 데이터 (ego, agents, static, map, 경로 히스토리, 미래 경로) → Latent Distribution
- Decoder: Latent z → 모든 입력 데이터 복원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    VAE Encoder
    160차원 경로 벡터를 latent distribution으로 인코딩
    Deep MLP 구조: 160 → 512 → 256 → 128 → Latent Space (32차원)
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Encoder layers with Batch Normalization
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent distribution heads
        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - flatten된 모든 입력 데이터
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        encoded = self.encoder(x)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder
    Latent z에서 160차원 경로 벡터를 복원
    Deep MLP 구조: Latent Space (32차원) → 128 → 256 → 512 → 160차원
    """
    
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256, 512]):
        super().__init__()
        
        # Decoder layers with Batch Normalization
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, no normalization)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            reconstructed: (batch, output_dim) - 복원된 모든 입력 데이터
        """
        return self.decoder(z)


def reparameterize(mu, logvar):
    """
    Reparameterization Trick
    z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
