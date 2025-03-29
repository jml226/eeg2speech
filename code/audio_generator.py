"""
오디오 생성 모듈

이 모듈은 잠재 표현에서 오디오를 생성하는 모델을 제공합니다.
AudioLDM2와 Whisper 기반 모델을 결합한 이중 경로 구조를 구현합니다.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
from tqdm import tqdm
import time
import random
import librosa
import librosa.display
import soundfile as sf
from einops import rearrange


class SinusoidalPositionalEmbedding(nn.Module):
    """
    사인 위치 임베딩
    """
    
    def __init__(self, dim: int, max_positions: int = 10000):
        """
        초기화 함수
        
        Args:
            dim: 임베딩 차원
            max_positions: 최대 위치 수
        """
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x seq_len x dim]
            
        Returns:
            위치 임베딩이 추가된 텐서 [batch_size x seq_len x dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 위치 인덱스
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        position = position.expand(-1, self.dim // 2)
        
        # 차원 인덱스
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float) * -(np.log(self.max_positions) / self.dim))
        
        # 사인 위치 임베딩 계산
        pe = torch.zeros(seq_len, self.dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 차원 추가 및 디바이스 이동
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        return x + pe


class TransformerBlock(nn.Module):
    """
    트랜스포머 블록
    """
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 mlp_ratio: int = 4, 
                 dropout: float = 0.1):
        """
        초기화 함수
        
        Args:
            dim: 특성 차원
            num_heads: 어텐션 헤드 수
            mlp_ratio: MLP 확장 비율
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # 자기 주의 레이어
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # MLP 레이어
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x seq_len x dim]
            mask: 어텐션 마스크 [batch_size x seq_len x seq_len]
            
        Returns:
            출력 텐서 [batch_size x seq_len x dim]
        """
        # 자기 주의 레이어
        attn_output, _ = self.attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=mask
        )
        x = x + attn_output
        
        # MLP 레이어
        x = x + self.mlp(self.norm2(x))
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    교차 주의 블록
    """
    
    def __init__(self, 
                 dim: int, 
                 context_dim: int, 
                 num_heads: int = 8, 
                 mlp_ratio: int = 4, 
                 dropout: float = 0.1):
        """
        초기화 함수
        
        Args:
            dim: 특성 차원
            context_dim: 컨텍스트 차원
            num_heads: 어텐션 헤드 수
            mlp_ratio: MLP 확장 비율
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # 교차 주의 레이어
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        
        # 컨텍스트 투영 (차원이 다른 경우)
        self.context_proj = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()
        
        # MLP 레이어
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x seq_len x dim]
            context: 컨텍스트 텐서 [batch_size x context_len x context_dim]
            mask: 어텐션 마스크 [batch_size x seq_len x context_len]
            
        Returns:
            출력 텐서 [batch_size x seq_len x dim]
        """
        # 컨텍스트 정규화 및 투영
        context = self.norm_context(context)
        context = self.context_proj(context)
        
        # 교차 주의 레이어
        cross_attn_output, _ = self.cross_attn(
            query=self.norm1(x),
            key=context,
            value=context,
            attn_mask=mask
        )
        x = x + cross_attn_output
        
        # MLP 레이어
        x = x + self.mlp(self.norm2(x))
        
        return x


class WhisperDecoder(nn.Module):
    """
    Whisper 기반 디코더 모델
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 n_mels: int = 80,
                 mel_time_steps: int = 250,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        초기화 함수
        
        Args:
            latent_dim: 잠재 벡터 차원
            n_mels: 멜 스펙트로그램 빈 수
            mel_time_steps: 멜 스펙트로그램 시간 스텝 수
            hidden_dim: 은닉 차원
            num_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.mel_time_steps = mel_time_steps
        self.hidden_dim = hidden_dim
        
        # 잠재 벡터 투영
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # 위치 임베딩
        self.pos_embedding = SinusoidalPositionalEmbedding(hidden_dim)
        
        # 트랜스포머 레이어
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 출력 투영 (멜 스펙트로그램)
        self.mel_proj = nn.Linear(hidden_dim, n_mels)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            z: 잠재 벡터 [batch_size x latent_dim]
            
        Returns:
            멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
        """
        batch_size = z.size(0)
        
        # 잠재 벡터 투영
        z = self.latent_proj(z)  # [batch_size x hidden_dim]
        
        # 시퀀스 확장
        z = z.unsqueeze(1).expand(-1, self.mel_time_steps, -1)  # [batch_size x mel_time_steps x hidden_dim]
        
        # 위치 임베딩 추가
        x = self.pos_embedding(z)  # [batch_size x mel_time_steps x hidden_dim]
        
        # 트랜스포머 레이어 통과
        for layer in self.layers:
            x = layer(x)
        
        # 멜 스펙트로그램 생성
        mel = self.mel_proj(x)  # [batch_size x mel_time_steps x n_mels]
        
        # 차원 변환
        mel = mel.transpose(1, 2)  # [batch_size x n_mels x mel_time_steps]
        
        return mel


class DiffusionModel(nn.Module):
    """
    확산 모델 (AudioLDM2 기반)
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 n_mels: int = 80,
                 mel_time_steps: int = 250,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 diffusion_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        """
        초기화 함수
        
        Args:
            latent_dim: 잠재 벡터 차원
            n_mels: 멜 스펙트로그램 빈 수
            mel_time_steps: 멜 스펙트로그램 시간 스텝 수
            hidden_dim: 은닉 차원
            num_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            diffusion_steps: 확산 스텝 수
            beta_start: 초기 노이즈 스케줄 베타
            beta_end: 최종 노이즈 스케줄 베타
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.mel_time_steps = mel_time_steps
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        
        # 노이즈 스케줄 설정
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 잠재 벡터 투영
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # 시간 임베딩
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 입력 투영
        self.input_proj = nn.Conv1d(n_mels, hidden_dim, kernel_size=1)
        
        # U-Net 구조
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim * 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim * 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim * 4),
                nn.GELU()
            )
        ])
        
        # 중간 블록
        self.mid_block = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.GELU()
        )
        
        # 교차 주의 블록
        self.cross_attn = CrossAttentionBlock(
            dim=hidden_dim * 4,
            context_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 업 블록
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 8, hidden_dim * 4, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim * 4),
                nn.GELU(),
                nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim * 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim * 2),
                nn.GELU(),
                nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU()
            )
        ])
        
        # 출력 투영
        self.output_proj = nn.Conv1d(hidden_dim * 2, n_mels, kernel_size=1)
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        시간 임베딩 생성
        
        Args:
            t: 시간 스텝 [batch_size]
            
        Returns:
            시간 임베딩 [batch_size x hidden_dim]
        """
        # 사인 위치 임베딩 계산
        half_dim = self.hidden_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # MLP 통과
        emb = self.time_embedding(emb)
        
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        순전파 (노이즈 예측)
        
        Args:
            x: 노이즈가 추가된 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
            t: 시간 스텝 [batch_size]
            z: 잠재 벡터 [batch_size x latent_dim]
            
        Returns:
            예측된 노이즈 [batch_size x n_mels x mel_time_steps]
        """
        # 시간 임베딩
        t_emb = self.get_time_embedding(t)  # [batch_size x hidden_dim]
        
        # 잠재 벡터 투영
        z_emb = self.latent_proj(z)  # [batch_size x hidden_dim]
        
        # 입력 투영
        h = self.input_proj(x)  # [batch_size x hidden_dim x mel_time_steps]
        
        # 시간 임베딩 추가
        h = h + t_emb.unsqueeze(-1)
        
        # 다운 샘플링 및 스킵 연결 저장
        skips = []
        for block in self.down_blocks:
            skips.append(h)
            h = block(h)
        
        # 중간 블록
        h = self.mid_block(h)
        
        # 교차 주의 적용
        # 차원 변환: [batch_size x hidden_dim*4 x mel_time_steps/4] -> [batch_size x mel_time_steps/4 x hidden_dim*4]
        h = h.permute(0, 2, 1)
        
        # 잠재 벡터를 컨텍스트로 사용
        z_context = z_emb.unsqueeze(1)  # [batch_size x 1 x hidden_dim]
        h = self.cross_attn(h, z_context)
        
        # 차원 변환: [batch_size x mel_time_steps/4 x hidden_dim*4] -> [batch_size x hidden_dim*4 x mel_time_steps/4]
        h = h.permute(0, 2, 1)
        
        # 업 샘플링 및 스킵 연결 결합
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        
        # 출력 투영
        output = self.output_proj(torch.cat([h, skips[0]], dim=1))
        
        return output
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        노이즈 추가
        
        Args:
            x: 원본 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
            t: 시간 스텝 [batch_size]
            noise: 노이즈 (없으면 생성) [batch_size x n_mels x mel_time_steps]
            
        Returns:
            노이즈가 추가된 멜 스펙트로그램, 노이즈
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def sample(self, z: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        샘플링 (DDPM)
        
        Args:
            z: 잠재 벡터 [batch_size x latent_dim]
            steps: 샘플링 스텝 수
            
        Returns:
            생성된 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
        """
        batch_size = z.size(0)
        device = z.device
        
        # 노이즈에서 시작
        x = torch.randn(batch_size, self.n_mels, self.mel_time_steps, device=device)
        
        # 역방향 확산 과정
        for i in tqdm(reversed(range(0, steps)), desc="Sampling", total=steps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 노이즈 예측
            predicted_noise = self(x, t, z)
            
            # 노이즈 제거 단계
            alpha = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0)
            
            # 계수 계산
            beta = self.betas[i]
            sqrt_recip_alpha = torch.rsqrt(alpha)
            sqrt_alpha_recip = torch.rsqrt(alpha_prev)
            
            # 평균 계산
            pred_x0 = (x - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)
            
            # 다음 x 계산
            mean = sqrt_recip_alpha * (x - beta * predicted_noise / torch.sqrt(1.0 - alpha))
            
            # 마지막 스텝이 아니면 노이즈 추가
            if i > 0:
                variance = (1.0 - alpha_prev) / (1.0 - alpha) * beta
                std = torch.sqrt(variance)
                noise = torch.randn_like(x)
                x = mean + std * noise
            else:
                x = mean
        
        return x


class MelToAudio(nn.Module):
    """
    멜 스펙트로그램을 오디오로 변환하는 모듈
    """
    
    def __init__(self, 
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 sample_rate: int = 16000):
        """
        초기화 함수
        
        Args:
            n_mels: 멜 스펙트로그램 빈 수
            n_fft: FFT 크기
            hop_length: 홉 길이
            sample_rate: 샘플링 레이트
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # 멜 필터뱅크 생성
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        # 의사 역 계산
        mel_inverse = np.linalg.pinv(mel_basis)
        self.register_buffer('mel_inverse', torch.from_numpy(mel_inverse).float())
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        순전파 (멜 스펙트로그램 -> 오디오)
        
        Args:
            mel_spec: 멜 스펙트로그램 [batch_size x n_mels x time_steps]
            
        Returns:
            오디오 [batch_size x audio_samples]
        """
        batch_size = mel_spec.size(0)
        device = mel_spec.device
        
        # 로그 멜 스펙트로그램을 선형 스케일로 변환
        mel_spec = torch.exp(mel_spec)
        
        # 의사 역을 사용하여 선형 스펙트로그램 복원
        spec = torch.matmul(self.mel_inverse, mel_spec)
        
        # 위상 재구성 (Griffin-Lim 알고리즘)
        # 여기서는 간단한 구현을 위해 랜덤 위상으로 시작
        angles = torch.rand_like(spec) * 2 * np.pi - np.pi
        spec_complex = spec * torch.exp(1j * angles)
        
        # 배치 처리를 위한 결과 저장
        audio = []
        
        # 배치의 각 항목에 대해 Griffin-Lim 알고리즘 적용
        for i in range(batch_size):
            spec_item = spec_complex[i].cpu().numpy()
            audio_item = librosa.griffinlim(
                np.abs(spec_item),
                hop_length=self.hop_length,
                win_length=self.n_fft,
                n_iter=32
            )
            audio.append(torch.from_numpy(audio_item).to(device))
        
        # 패딩을 통해 모든 오디오 샘플을 동일한 길이로 만듦
        max_len = max(a.size(0) for a in audio)
        audio_padded = torch.zeros(batch_size, max_len, device=device)
        
        for i, a in enumerate(audio):
            audio_padded[i, :a.size(0)] = a
        
        return audio_padded


class DualPathAudioGenerator(nn.Module):
    """
    이중 경로 오디오 생성기 (Whisper + AudioLDM2)
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 n_mels: int = 80,
                 mel_time_steps: int = 250,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: int = 0.1,
                 diffusion_steps: int = 1000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 sample_rate: int = 16000,
                 fusion_hidden_dim: int = 512):
        """
        초기화 함수
        
        Args:
            latent_dim: 잠재 벡터 차원
            n_mels: 멜 스펙트로그램 빈 수
            mel_time_steps: 멜 스펙트로그램 시간 스텝 수
            hidden_dim: 은닉 차원
            num_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            diffusion_steps: 확산 스텝 수
            n_fft: FFT 크기
            hop_length: 홉 길이
            sample_rate: 샘플링 레이트
            fusion_hidden_dim: 융합 은닉 차원
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.mel_time_steps = mel_time_steps
        
        # Whisper 디코더
        self.whisper_decoder = WhisperDecoder(
            latent_dim=latent_dim,
            n_mels=n_mels,
            mel_time_steps=mel_time_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 확산 모델
        self.diffusion_model = DiffusionModel(
            latent_dim=latent_dim,
            n_mels=n_mels,
            mel_time_steps=mel_time_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            diffusion_steps=diffusion_steps
        )
        
        # 멜 스펙트로그램을 오디오로 변환
        self.mel_to_audio = MelToAudio(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(n_mels * 2, fusion_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, fusion_hidden_dim),
            nn.GELU(),
            nn.Conv1d(fusion_hidden_dim, n_mels, kernel_size=3, padding=1)
        )
    
    def forward(self, z: torch.Tensor, diffusion_steps: int = 100) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            z: 잠재 벡터 [batch_size x latent_dim]
            diffusion_steps: 확산 샘플링 스텝 수
            
        Returns:
            출력 딕셔너리:
                - whisper_mel: Whisper 디코더의 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
                - diffusion_mel: 확산 모델의 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
                - fused_mel: 융합된 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
                - audio: 생성된 오디오 [batch_size x audio_samples]
        """
        # Whisper 디코더로 멜 스펙트로그램 생성
        whisper_mel = self.whisper_decoder(z)
        
        # 확산 모델로 멜 스펙트로그램 생성
        diffusion_mel = self.diffusion_model.sample(z, steps=diffusion_steps)
        
        # 멜 스펙트로그램 융합
        fused_mel = self.fusion_layer(torch.cat([whisper_mel, diffusion_mel], dim=1))
        
        # 멜 스펙트로그램을 오디오로 변환
        audio = self.mel_to_audio(fused_mel)
        
        return {
            'whisper_mel': whisper_mel,
            'diffusion_mel': diffusion_mel,
            'fused_mel': fused_mel,
            'audio': audio
        }


class AudioGeneratorTrainer:
    """
    오디오 생성기 학습 클래스
    """
    
    def __init__(self, 
                 model: DualPathAudioGenerator,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 whisper_weight: float = 1.0,
                 diffusion_weight: float = 1.0,
                 mel_loss_weight: float = 1.0,
                 audio_loss_weight: float = 0.5,
                 checkpoint_dir: str = './checkpoints'):
        """
        초기화 함수
        
        Args:
            model: 이중 경로 오디오 생성기 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 학습 디바이스
            learning_rate: 학습률
            whisper_weight: Whisper 손실 가중치
            diffusion_weight: 확산 모델 손실 가중치
            mel_loss_weight: 멜 스펙트로그램 손실 가중치
            audio_loss_weight: 오디오 손실 가중치
            checkpoint_dir: 체크포인트 디렉토리
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.whisper_weight = whisper_weight
        self.diffusion_weight = diffusion_weight
        self.mel_loss_weight = mel_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.checkpoint_dir = checkpoint_dir
        
        # 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_whisper_loss': [],
            'val_whisper_loss': [],
            'train_diffusion_loss': [],
            'val_diffusion_loss': []
        }
    
    def compute_mel_loss(self, pred_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """
        멜 스펙트로그램 손실 계산
        
        Args:
            pred_mel: 예측 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
            target_mel: 타겟 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
            
        Returns:
            멜 스펙트로그램 손실
        """
        # L1 손실과 MSE 손실 결합
        l1_loss = self.l1_loss(pred_mel, target_mel)
        mse_loss = self.mse_loss(pred_mel, target_mel)
        
        return l1_loss + mse_loss
    
    def compute_audio_loss(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        오디오 손실 계산
        
        Args:
            pred_audio: 예측 오디오 [batch_size x audio_samples]
            target_audio: 타겟 오디오 [batch_size x audio_samples]
            
        Returns:
            오디오 손실
        """
        # 길이 맞추기
        min_len = min(pred_audio.size(1), target_audio.size(1))
        pred_audio = pred_audio[:, :min_len]
        target_audio = target_audio[:, :min_len]
        
        # L1 손실
        return self.l1_loss(pred_audio, target_audio)
    
    def compute_diffusion_loss(self, model: DiffusionModel, mel: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        확산 모델 손실 계산
        
        Args:
            model: 확산 모델
            mel: 멜 스펙트로그램 [batch_size x n_mels x mel_time_steps]
            z: 잠재 벡터 [batch_size x latent_dim]
            
        Returns:
            확산 모델 손실
        """
        batch_size = mel.size(0)
        device = mel.device
        
        # 랜덤 시간 스텝 선택
        t = torch.randint(0, model.diffusion_steps, (batch_size,), device=device)
        
        # 노이즈 추가
        noisy_mel, noise = model.add_noise(mel, t)
        
        # 노이즈 예측
        predicted_noise = model(noisy_mel, t, z)
        
        # 손실 계산
        return self.mse_loss(predicted_noise, noise)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        한 에폭 학습
        
        Returns:
            손실 딕셔너리
        """
        self.model.train()
        total_loss = 0.0
        total_whisper_loss = 0.0
        total_diffusion_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # 배치 데이터
            latent = batch['latent'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            
            # Whisper 디코더 학습
            whisper_mel = self.model.whisper_decoder(latent)
            whisper_loss = self.compute_mel_loss(whisper_mel, mel_spec)
            
            # 확산 모델 학습
            diffusion_loss = self.compute_diffusion_loss(self.model.diffusion_model, mel_spec, latent)
            
            # 총 손실
            loss = self.whisper_weight * whisper_loss + self.diffusion_weight * diffusion_loss
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 손실 누적
            total_loss += loss.item()
            total_whisper_loss += whisper_loss.item()
            total_diffusion_loss += diffusion_loss.item()
        
        # 평균 손실
        avg_loss = total_loss / len(self.train_loader)
        avg_whisper_loss = total_whisper_loss / len(self.train_loader)
        avg_diffusion_loss = total_diffusion_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'whisper_loss': avg_whisper_loss,
            'diffusion_loss': avg_diffusion_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """
        검증
        
        Returns:
            손실 딕셔너리
        """
        self.model.eval()
        total_loss = 0.0
        total_whisper_loss = 0.0
        total_diffusion_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 배치 데이터
                latent = batch['latent'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                audio = batch['audio'].to(self.device)
                
                # Whisper 디코더 검증
                whisper_mel = self.model.whisper_decoder(latent)
                whisper_loss = self.compute_mel_loss(whisper_mel, mel_spec)
                
                # 확산 모델 검증
                diffusion_loss = self.compute_diffusion_loss(self.model.diffusion_model, mel_spec, latent)
                
                # 총 손실
                loss = self.whisper_weight * whisper_loss + self.diffusion_weight * diffusion_loss
                
                # 손실 누적
                total_loss += loss.item()
                total_whisper_loss += whisper_loss.item()
                total_diffusion_loss += diffusion_loss.item()
        
        # 평균 손실
        avg_loss = total_loss / len(self.val_loader)
        avg_whisper_loss = total_whisper_loss / len(self.val_loader)
        avg_diffusion_loss = total_diffusion_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'whisper_loss': avg_whisper_loss,
            'diffusion_loss': avg_diffusion_loss
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        모델 학습
        
        Args:
            num_epochs: 에폭 수
            early_stopping_patience: 조기 종료 인내심
            
        Returns:
            학습 히스토리
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 학습
            train_losses = self.train_epoch()
            
            # 검증
            val_losses = self.validate()
            
            # 히스토리 업데이트
            self.history['train_loss'].append(train_losses['loss'])
            self.history['val_loss'].append(val_losses['loss'])
            self.history['train_whisper_loss'].append(train_losses['whisper_loss'])
            self.history['val_whisper_loss'].append(val_losses['whisper_loss'])
            self.history['train_diffusion_loss'].append(train_losses['diffusion_loss'])
            self.history['val_diffusion_loss'].append(val_losses['diffusion_loss'])
            
            # 결과 출력
            print(f"Train Loss: {train_losses['loss']:.4f}, Whisper Loss: {train_losses['whisper_loss']:.4f}, Diffusion Loss: {train_losses['diffusion_loss']:.4f}")
            print(f"Val Loss: {val_losses['loss']:.4f}, Whisper Loss: {val_losses['whisper_loss']:.4f}, Diffusion Loss: {val_losses['diffusion_loss']:.4f}")
            
            # 체크포인트 저장
            if val_losses['loss'] < best_val_loss:
                best_val_loss = val_losses['loss']
                patience_counter = 0
                
                # 최상의 모델 저장
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_audio_generator.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, checkpoint_path)
                
                print(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                
                # 조기 종료
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return self.history
    
    def generate_samples(self, test_loader: DataLoader, num_samples: int = 5) -> Dict[str, np.ndarray]:
        """
        샘플 생성
        
        Args:
            test_loader: 테스트 데이터 로더
            num_samples: 생성할 샘플 수
            
        Returns:
            샘플 딕셔너리:
                - latents: 잠재 벡터
                - target_mels: 타겟 멜 스펙트로그램
                - whisper_mels: Whisper 디코더의 멜 스펙트로그램
                - diffusion_mels: 확산 모델의 멜 스펙트로그램
                - fused_mels: 융합된 멜 스펙트로그램
                - target_audios: 타겟 오디오
                - generated_audios: 생성된 오디오
        """
        self.model.eval()
        
        # 결과 저장
        latents = []
        target_mels = []
        whisper_mels = []
        diffusion_mels = []
        fused_mels = []
        target_audios = []
        generated_audios = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 배치 데이터
                latent = batch['latent'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                audio = batch['audio'].to(self.device)
                
                # 오디오 생성
                outputs = self.model(latent, diffusion_steps=50)
                
                # 결과 저장
                latents.append(latent.cpu().numpy())
                target_mels.append(mel_spec.cpu().numpy())
                whisper_mels.append(outputs['whisper_mel'].cpu().numpy())
                diffusion_mels.append(outputs['diffusion_mel'].cpu().numpy())
                fused_mels.append(outputs['fused_mel'].cpu().numpy())
                target_audios.append(audio.cpu().numpy())
                generated_audios.append(outputs['audio'].cpu().numpy())
                
                if len(np.concatenate(latents)) >= num_samples:
                    break
        
        # 데이터 준비
        latents = np.concatenate(latents)[:num_samples]
        target_mels = np.concatenate(target_mels)[:num_samples]
        whisper_mels = np.concatenate(whisper_mels)[:num_samples]
        diffusion_mels = np.concatenate(diffusion_mels)[:num_samples]
        fused_mels = np.concatenate(fused_mels)[:num_samples]
        target_audios = np.concatenate(target_audios)[:num_samples]
        generated_audios = np.concatenate(generated_audios)[:num_samples]
        
        return {
            'latents': latents,
            'target_mels': target_mels,
            'whisper_mels': whisper_mels,
            'diffusion_mels': diffusion_mels,
            'fused_mels': fused_mels,
            'target_audios': target_audios,
            'generated_audios': generated_audios
        }
    
    def visualize_spectrograms(self, samples: Dict[str, np.ndarray]) -> None:
        """
        스펙트로그램 시각화
        
        Args:
            samples: 샘플 딕셔너리
        """
        num_samples = len(samples['target_mels'])
        
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i in range(num_samples):
            # 타겟 멜 스펙트로그램
            plt.subplot(num_samples, 4, i * 4 + 1)
            plt.imshow(samples['target_mels'][i], aspect='auto', origin='lower')
            plt.title(f"Sample {i+1}: Target Mel")
            plt.colorbar()
            
            # Whisper 멜 스펙트로그램
            plt.subplot(num_samples, 4, i * 4 + 2)
            plt.imshow(samples['whisper_mels'][i], aspect='auto', origin='lower')
            plt.title(f"Sample {i+1}: Whisper Mel")
            plt.colorbar()
            
            # 확산 모델 멜 스펙트로그램
            plt.subplot(num_samples, 4, i * 4 + 3)
            plt.imshow(samples['diffusion_mels'][i], aspect='auto', origin='lower')
            plt.title(f"Sample {i+1}: Diffusion Mel")
            plt.colorbar()
            
            # 융합 멜 스펙트로그램
            plt.subplot(num_samples, 4, i * 4 + 4)
            plt.imshow(samples['fused_mels'][i], aspect='auto', origin='lower')
            plt.title(f"Sample {i+1}: Fused Mel")
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    def plot_loss_curves(self) -> None:
        """
        손실 곡선 시각화
        """
        plt.figure(figsize=(12, 8))
        
        # 총 손실
        plt.subplot(3, 1, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Whisper 손실
        plt.subplot(3, 1, 2)
        plt.plot(self.history['train_whisper_loss'], label='Train Whisper Loss')
        plt.plot(self.history['val_whisper_loss'], label='Validation Whisper Loss')
        plt.title('Whisper Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 확산 모델 손실
        plt.subplot(3, 1, 3)
        plt.plot(self.history['train_diffusion_loss'], label='Train Diffusion Loss')
        plt.plot(self.history['val_diffusion_loss'], label='Validation Diffusion Loss')
        plt.title('Diffusion Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_audio_samples(self, samples: Dict[str, np.ndarray], output_dir: str, sample_rate: int = 16000) -> List[str]:
        """
        오디오 샘플 저장
        
        Args:
            samples: 샘플 딕셔너리
            output_dir: 출력 디렉토리
            sample_rate: 샘플링 레이트
            
        Returns:
            저장된 파일 경로 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = []
        
        for i in range(len(samples['target_audios'])):
            # 타겟 오디오 저장
            target_path = os.path.join(output_dir, f"sample_{i+1}_target.wav")
            sf.write(target_path, samples['target_audios'][i], sample_rate)
            file_paths.append(target_path)
            
            # 생성된 오디오 저장
            generated_path = os.path.join(output_dir, f"sample_{i+1}_generated.wav")
            sf.write(generated_path, samples['generated_audios'][i], sample_rate)
            file_paths.append(generated_path)
        
        return file_paths


class LatentToAudioDataset(Dataset):
    """
    잠재 벡터에서 오디오로의 데이터셋
    """
    
    def __init__(self, latents: np.ndarray, mel_specs: np.ndarray, audios: np.ndarray):
        """
        초기화 함수
        
        Args:
            latents: 잠재 벡터 [num_samples x latent_dim]
            mel_specs: 멜 스펙트로그램 [num_samples x n_mels x mel_time_steps]
            audios: 오디오 [num_samples x audio_samples]
        """
        assert len(latents) == len(mel_specs) == len(audios), "All inputs must have the same number of samples"
        
        self.latents = latents
        self.mel_specs = mel_specs
        self.audios = audios
    
    def __len__(self) -> int:
        """
        데이터셋 길이 반환
        
        Returns:
            데이터셋 길이
        """
        return len(self.latents)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        인덱스로 항목 가져오기
        
        Args:
            idx: 인덱스
            
        Returns:
            항목 딕셔너리:
                - latent: 잠재 벡터
                - mel_spec: 멜 스펙트로그램
                - audio: 오디오
        """
        return {
            'latent': self.latents[idx],
            'mel_spec': self.mel_specs[idx],
            'audio': self.audios[idx]
        }


def main():
    """
    메인 함수
    """
    # 기본 디렉토리 설정
    base_dir = "/home/ubuntu/eeg2speech"
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    # 디렉토리 생성
    os.makedirs(models_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 더미 데이터 생성
    num_samples = 100
    latent_dim = 128
    n_mels = 80
    mel_time_steps = 250
    audio_samples = 16000
    
    # 랜덤 데이터 생성
    np.random.seed(42)
    latents = np.random.randn(num_samples, latent_dim)
    mel_specs = np.random.randn(num_samples, n_mels, mel_time_steps)
    audios = np.random.randn(num_samples, audio_samples)
    
    # 데이터셋 생성
    dataset = LatentToAudioDataset(latents, mel_specs, audios)
    
    # 데이터셋 분할
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 데이터 로더 생성
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화
    model = DualPathAudioGenerator(
        latent_dim=latent_dim,
        n_mels=n_mels,
        mel_time_steps=mel_time_steps,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        diffusion_steps=1000,
        n_fft=1024,
        hop_length=256,
        sample_rate=16000,
        fusion_hidden_dim=512
    )
    
    model = model.to(device)
    
    # 모델 요약
    print(model)
    
    # 학습기 초기화
    trainer = AudioGeneratorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        whisper_weight=1.0,
        diffusion_weight=1.0,
        mel_loss_weight=1.0,
        audio_loss_weight=0.5,
        checkpoint_dir=models_dir
    )
    
    # 모델 학습 (실제 학습은 주석 처리)
    # history = trainer.train(num_epochs=50, early_stopping_patience=10)
    
    # 손실 곡선 시각화
    # trainer.plot_loss_curves()
    
    # 샘플 생성 및 시각화
    # samples = trainer.generate_samples(test_loader, num_samples=5)
    # trainer.visualize_spectrograms(samples)
    
    # 오디오 샘플 저장
    # output_dir = os.path.join(base_dir, "samples")
    # file_paths = trainer.save_audio_samples(samples, output_dir)
    
    print("오디오 생성 모듈 테스트 완료")


if __name__ == "__main__":
    main()
