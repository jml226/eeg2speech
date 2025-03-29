"""
오디오-텍스트 인코더 모듈

이 모듈은 오디오와 텍스트 데이터를 잠재 표현으로 변환하는 인코더 모델을 제공합니다.
VAE 인코더와 CLAP 기반 대조적 인코더를 결합한 이중 경로 구조를 구현합니다.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
from tqdm import tqdm
import time
import random
from sklearn.manifold import TSNE


class AudioEncoder1DCNN(nn.Module):
    """
    오디오 데이터를 위한 1D CNN 인코더 기본 블록
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 audio_samples: int = 16000,
                 base_filters: int = 64,
                 n_layers: int = 4,
                 kernel_size: int = 7,
                 stride: int = 2,
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            in_channels: 입력 채널 수 (오디오 채널 수)
            audio_samples: 오디오 샘플 수
            base_filters: 기본 필터 수
            n_layers: 레이어 수
            kernel_size: 컨볼루션 커널 크기
            stride: 컨볼루션 스트라이드
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.audio_samples = audio_samples
        
        # 레이어 구성
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Conv1d(in_channels, base_filters, kernel_size, stride, padding=kernel_size//2))
        layers.append(nn.BatchNorm1d(base_filters))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))
        
        # 추가 레이어
        in_filters = base_filters
        for i in range(1, n_layers):
            out_filters = base_filters * (2 ** i)
            layers.append(nn.Conv1d(in_filters, out_filters, kernel_size, stride, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_filters = out_filters
        
        self.encoder = nn.Sequential(*layers)
        
        # 출력 특성 크기 계산
        self.output_dim = self._get_output_dim()
    
    def _get_output_dim(self) -> int:
        """
        출력 특성 크기 계산
        
        Returns:
            출력 특성 크기
        """
        # 더미 입력으로 순전파하여 출력 크기 계산
        dummy_input = torch.zeros(1, self.in_channels, self.audio_samples)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        
        return dummy_output.size(1) * dummy_output.size(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x in_channels x audio_samples]
            
        Returns:
            출력 텐서 [batch_size x output_dim]
        """
        # 인코더 통과
        x = self.encoder(x)
        
        # 평탄화
        x = x.view(x.size(0), -1)
        
        return x


class MelSpecEncoder2DCNN(nn.Module):
    """
    멜 스펙트로그램을 위한 2D CNN 인코더
    """
    
    def __init__(self, 
                 n_mels: int = 80, 
                 time_frames: int = 64,
                 base_filters: int = 32,
                 n_layers: int = 4,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            n_mels: 멜 빈 수
            time_frames: 시간 프레임 수
            base_filters: 기본 필터 수
            n_layers: 레이어 수
            kernel_size: 컨볼루션 커널 크기
            stride: 컨볼루션 스트라이드
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.time_frames = time_frames
        
        # 레이어 구성
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Conv2d(1, base_filters, kernel_size, stride, padding=kernel_size//2))
        layers.append(nn.BatchNorm2d(base_filters))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))
        
        # 추가 레이어
        in_filters = base_filters
        for i in range(1, n_layers):
            out_filters = base_filters * (2 ** i)
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_filters = out_filters
        
        self.encoder = nn.Sequential(*layers)
        
        # 출력 특성 크기 계산
        self.output_dim = self._get_output_dim()
    
    def _get_output_dim(self) -> int:
        """
        출력 특성 크기 계산
        
        Returns:
            출력 특성 크기
        """
        # 더미 입력으로 순전파하여 출력 크기 계산
        dummy_input = torch.zeros(1, 1, self.n_mels, self.time_frames)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        
        return dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x 1 x n_mels x time_frames]
            
        Returns:
            출력 텐서 [batch_size x output_dim]
        """
        # 인코더 통과
        x = self.encoder(x)
        
        # 평탄화
        x = x.view(x.size(0), -1)
        
        return x


class TextEncoder(nn.Module):
    """
    텍스트 데이터를 위한 트랜스포머 인코더
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 max_seq_length: int = 100,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        초기화 함수
        
        Args:
            vocab_size: 어휘 크기
            max_seq_length: 최대 시퀀스 길이
            embed_dim: 임베딩 차원
            num_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 위치 임베딩
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        # 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 투영
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """
        가중치 초기화
        """
        # 토큰 임베딩 초기화
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # 위치 임베딩 초기화
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # 출력 투영 초기화
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x seq_len]
            attention_mask: 어텐션 마스크 [batch_size x seq_len]
            
        Returns:
            출력 텐서 [batch_size x embed_dim]
        """
        # 시퀀스 길이 제한
        seq_len = min(x.size(1), self.max_seq_length)
        x = x[:, :seq_len]
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(x)  # [batch_size x seq_len x embed_dim]
        
        # 위치 임베딩 추가
        position_embeds = self.position_embedding[:, :seq_len, :]
        embeds = token_embeds + position_embeds
        
        # 드롭아웃
        embeds = self.dropout(embeds)
        
        # 어텐션 마스크 생성 (없는 경우)
        if attention_mask is None:
            attention_mask = (x != 0).float()  # 패딩 토큰(0)에 마스크 적용
        
        # 트랜스포머 인코더 통과
        transformer_output = self.transformer_encoder(embeds, src_key_padding_mask=~attention_mask.bool())
        
        # 글로벌 표현 (첫 번째 토큰 또는 평균)
        # 여기서는 시퀀스의 평균을 사용
        pooled_output = transformer_output.mean(dim=1)
        
        # 출력 투영
        output = self.output_projection(pooled_output)
        
        return output


class AudioVAE(nn.Module):
    """
    오디오 데이터를 위한 변분 오토인코더 (VAE)
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 audio_samples: int = 16000,
                 latent_dim: int = 128,
                 base_filters: int = 64,
                 n_layers: int = 4,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            in_channels: 입력 채널 수 (오디오 채널 수)
            audio_samples: 오디오 샘플 수
            latent_dim: 잠재 공간 차원
            base_filters: 기본 필터 수
            n_layers: CNN 레이어 수
            hidden_dim: 은닉 상태 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.audio_samples = audio_samples
        self.latent_dim = latent_dim
        
        # CNN 인코더
        self.cnn_encoder = AudioEncoder1DCNN(
            in_channels=in_channels,
            audio_samples=audio_samples,
            base_filters=base_filters,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # CNN 출력 차원
        cnn_output_dim = self.cnn_encoder.output_dim
        
        # 은닉 레이어
        self.hidden_layer = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # 잠재 변수 매핑 (평균 및 로그 분산)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 디코더 (재구성을 위한 선택적 구현)
        # 여기서는 간단한 MLP 디코더 사용
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, cnn_output_dim),
            nn.Tanh()  # 출력 범위 [-1, 1]
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인코딩 함수
        
        Args:
            x: 입력 텐서 [batch_size x in_channels x audio_samples]
            
        Returns:
            mu: 평균 [batch_size x latent_dim]
            logvar: 로그 분산 [batch_size x latent_dim]
        """
        # CNN 인코더 통과
        x = self.cnn_encoder(x)
        
        # 은닉 레이어 통과
        x = self.hidden_layer(x)
        
        # 평균 및 로그 분산 계산
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        재매개화 트릭
        
        Args:
            mu: 평균 [batch_size x latent_dim]
            logvar: 로그 분산 [batch_size x latent_dim]
            
        Returns:
            z: 샘플링된 잠재 변수 [batch_size x latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        디코딩 함수
        
        Args:
            z: 잠재 변수 [batch_size x latent_dim]
            
        Returns:
            재구성된 출력 [batch_size x cnn_output_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size x in_channels x audio_samples]
            
        Returns:
            출력 딕셔너리:
                - mu: 평균 [batch_size x latent_dim]
                - logvar: 로그 분산 [batch_size x latent_dim]
                - z: 샘플링된 잠재 변수 [batch_size x latent_dim]
                - recon: 재구성된 출력 [batch_size x cnn_output_dim]
        """
        # 인코딩
        mu, logvar = self.encode(x)
        
        # 재매개화
        z = self.reparameterize(mu, logvar)
        
        # 디코딩
        recon = self.decode(z)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'recon': recon
        }


class AudioTextCLAPEncoder(nn.Module):
    """
    오디오-텍스트 데이터를 위한 CLAP 기반 대조적 인코더
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 audio_samples: int = 16000,
                 n_mels: int = 80,
                 time_frames: int = 64,
                 vocab_size: int = 10000,
                 max_seq_length: int = 100,
                 projection_dim: int = 512,
                 audio_base_filters: int = 64,
                 mel_base_filters: int = 32,
                 n_audio_layers: int = 4,
                 n_mel_layers: int = 4,
                 text_embed_dim: int = 256,
                 text_num_heads: int = 8,
                 text_num_layers: int = 4,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            in_channels: 입력 채널 수 (오디오 채널 수)
            audio_samples: 오디오 샘플 수
            n_mels: 멜 빈 수
            time_frames: 시간 프레임 수
            vocab_size: 어휘 크기
            max_seq_length: 최대 시퀀스 길이
            projection_dim: 투영 차원 (공유 임베딩 공간)
            audio_base_filters: 오디오 CNN 기본 필터 수
            mel_base_filters: 멜 CNN 기본 필터 수
            n_audio_layers: 오디오 CNN 레이어 수
            n_mel_layers: 멜 CNN 레이어 수
            text_embed_dim: 텍스트 임베딩 차원
            text_num_heads: 텍스트 어텐션 헤드 수
            text_num_layers: 텍스트 트랜스포머 레이어 수
            hidden_dim: 은닉 상태 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.projection_dim = projection_dim
        
        # 오디오 파형 인코더
        self.audio_encoder = AudioEncoder1DCNN(
            in_channels=in_channels,
            audio_samples=audio_samples,
            base_filters=audio_base_filters,
            n_layers=n_audio_layers,
            dropout=dropout
        )
        
        # 멜 스펙트로그램 인코더
        self.mel_encoder = MelSpecEncoder2DCNN(
            n_mels=n_mels,
            time_frames=time_frames,
            base_filters=mel_base_filters,
            n_layers=n_mel_layers,
            dropout=dropout
        )
        
        # 텍스트 인코더
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            embed_dim=text_embed_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            dropout=dropout
        )
        
        # 오디오 투영 레이어
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_encoder.output_dim + self.mel_encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 텍스트 투영 레이어
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def encode_audio(self, audio: torch.Tensor, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        오디오 인코딩 함수
        
        Args:
            audio: 오디오 텐서 [batch_size x in_channels x audio_samples]
            mel_spec: 멜 스펙트로그램 텐서 [batch_size x 1 x n_mels x time_frames]
            
        Returns:
            오디오 임베딩 [batch_size x projection_dim]
        """
        # 오디오 파형 인코딩
        audio_features = self.audio_encoder(audio)
        
        # 멜 스펙트로그램 인코딩
        mel_features = self.mel_encoder(mel_spec)
        
        # 특성 결합
        combined_features = torch.cat([audio_features, mel_features], dim=1)
        
        # 투영
        audio_embedding = self.audio_projection(combined_features)
        
        # L2 정규화
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        return audio_embedding
    
    def encode_text(self, text: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        텍스트 인코딩 함수
        
        Args:
            text: 텍스트 텐서 [batch_size x seq_len]
            attention_mask: 어텐션 마스크 [batch_size x seq_len]
            
        Returns:
            텍스트 임베딩 [batch_size x projection_dim]
        """
        # 텍스트 인코딩
        text_features = self.text_encoder(text, attention_mask)
        
        # 투영
        text_embedding = self.text_projection(text_features)
        
        # L2 정규화
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        
        return text_embedding
    
    def forward(self, audio: torch.Tensor, mel_spec: torch.Tensor, 
               text: Optional[torch.Tensor] = None, 
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            audio: 오디오 텐서 [batch_size x in_channels x audio_samples]
            mel_spec: 멜 스펙트로그램 텐서 [batch_size x 1 x n_mels x time_frames]
            text: 텍스트 텐서 [batch_size x seq_len]
            attention_mask: 어텐션 마스크 [batch_size x seq_len]
            
        Returns:
            출력 딕셔너리:
                - audio_embedding: 오디오 임베딩 [batch_size x projection_dim]
                - text_embedding: 텍스트 임베딩 [batch_size x projection_dim] (텍스트가 제공된 경우)
        """
        # 오디오 인코딩
        audio_embedding = self.encode_audio(audio, mel_spec)
        
        # 텍스트 인코딩 (제공된 경우)
        text_embedding = None
        if text is not None:
            text_embedding = self.encode_text(text, attention_mask)
        
        outputs = {
            'audio_embedding': audio_embedding
        }
        
        if text_embedding is not None:
            outputs['text_embedding'] = text_embedding
        
        return outputs


class DualPathAudioTextEncoder(nn.Module):
    """
    오디오-텍스트 데이터를 위한 이중 경로 인코더 (VAE + CLAP)
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 audio_samples: int = 16000,
                 n_mels: int = 80,
                 time_frames: int = 64,
                 vocab_size: int = 10000,
                 max_seq_length: int = 100,
                 latent_dim: int = 128,
                 projection_dim: int = 512,
                 audio_base_filters: int = 64,
                 mel_base_filters: int = 32,
                 n_audio_layers: int = 4,
                 n_mel_layers: int = 4,
                 text_embed_dim: int = 256,
                 text_num_heads: int = 8,
                 text_num_layers: int = 4,
                 hidden_dim: int = 512,
                 fusion_hidden_dim: int = 512,
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            in_channels: 입력 채널 수 (오디오 채널 수)
            audio_samples: 오디오 샘플 수
            n_mels: 멜 빈 수
            time_frames: 시간 프레임 수
            vocab_size: 어휘 크기
            max_seq_length: 최대 시퀀스 길이
            latent_dim: VAE 잠재 공간 차원
            projection_dim: CLAP 투영 차원
            audio_base_filters: 오디오 CNN 기본 필터 수
            mel_base_filters: 멜 CNN 기본 필터 수
            n_audio_layers: 오디오 CNN 레이어 수
            n_mel_layers: 멜 CNN 레이어 수
            text_embed_dim: 텍스트 임베딩 차원
            text_num_heads: 텍스트 어텐션 헤드 수
            text_num_layers: 텍스트 트랜스포머 레이어 수
            hidden_dim: 은닉 상태 차원
            fusion_hidden_dim: 융합 은닉 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        
        # VAE 인코더
        self.vae_encoder = AudioVAE(
            in_channels=in_channels,
            audio_samples=audio_samples,
            latent_dim=latent_dim,
            base_filters=audio_base_filters,
            n_layers=n_audio_layers,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # CLAP 인코더
        self.clap_encoder = AudioTextCLAPEncoder(
            in_channels=in_channels,
            audio_samples=audio_samples,
            n_mels=n_mels,
            time_frames=time_frames,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            projection_dim=projection_dim,
            audio_base_filters=audio_base_filters,
            mel_base_filters=mel_base_filters,
            n_audio_layers=n_audio_layers,
            n_mel_layers=n_mel_layers,
            text_embed_dim=text_embed_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim + projection_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, latent_dim)
        )
    
    def forward(self, audio: torch.Tensor, mel_spec: torch.Tensor, 
               text: Optional[torch.Tensor] = None, 
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            audio: 오디오 텐서 [batch_size x in_channels x audio_samples]
            mel_spec: 멜 스펙트로그램 텐서 [batch_size x 1 x n_mels x time_frames]
            text: 텍스트 텐서 [batch_size x seq_len]
            attention_mask: 어텐션 마스크 [batch_size x seq_len]
            
        Returns:
            출력 딕셔너리:
                - vae_outputs: VAE 출력 딕셔너리
                - clap_outputs: CLAP 출력 딕셔너리
                - fused_latent: 융합된 잠재 표현 [batch_size x latent_dim]
        """
        # VAE 인코더 통과
        vae_outputs = self.vae_encoder(audio)
        
        # CLAP 인코더 통과
        clap_outputs = self.clap_encoder(audio, mel_spec, text, attention_mask)
        
        # 잠재 표현 융합
        combined = torch.cat([vae_outputs['z'], clap_outputs['audio_embedding']], dim=1)
        fused_latent = self.fusion_layer(combined)
        
        return {
            'vae_outputs': vae_outputs,
            'clap_outputs': clap_outputs,
            'fused_latent': fused_latent
        }


class AudioTextEncoderTrainer:
    """
    오디오-텍스트 인코더 학습 클래스
    """
    
    def __init__(self, 
                 model: DualPathAudioTextEncoder,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 vae_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 kld_weight: float = 0.01,
                 checkpoint_dir: str = './checkpoints'):
        """
        초기화 함수
        
        Args:
            model: 이중 경로 오디오-텍스트 인코더 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 학습 디바이스
            learning_rate: 학습률
            vae_weight: VAE 손실 가중치
            contrastive_weight: 대조적 손실 가중치
            kld_weight: KL 발산 가중치
            checkpoint_dir: 체크포인트 디렉토리
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.vae_weight = vae_weight
        self.contrastive_weight = contrastive_weight
        self.kld_weight = kld_weight
        self.checkpoint_dir = checkpoint_dir
        
        # 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 손실 함수
        self.contrastive_loss_fn = ContrastiveLoss()
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_vae_loss': [],
            'val_vae_loss': [],
            'train_contrastive_loss': [],
            'val_contrastive_loss': []
        }
    
    def compute_vae_loss(self, vae_outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE 손실 계산
        
        Args:
            vae_outputs: VAE 출력 딕셔너리
            target: 타겟 텐서
            
        Returns:
            total_loss: 총 손실
            recon_loss: 재구성 손실
            kld_loss: KL 발산 손실
        """
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(vae_outputs['recon'], target, reduction='mean')
        
        # KL 발산
        kld_loss = -0.5 * torch.mean(1 + vae_outputs['logvar'] - vae_outputs['mu'].pow(2) - vae_outputs['logvar'].exp())
        
        # 총 손실
        total_loss = recon_loss + self.kld_weight * kld_loss
        
        return total_loss, recon_loss, kld_loss
    
    def compute_contrastive_loss(self, clap_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        대조적 손실 계산
        
        Args:
            clap_outputs: CLAP 출력 딕셔너리
            
        Returns:
            contrastive_loss: 대조적 손실
        """
        # 텍스트 임베딩이 없는 경우 0 반환
        if 'text_embedding' not in clap_outputs:
            return torch.tensor(0.0, device=self.device)
        
        # 대조적 손실
        contrastive_loss = self.contrastive_loss_fn(
            clap_outputs['audio_embedding'],
            clap_outputs['text_embedding']
        )
        
        return contrastive_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """
        한 에폭 학습
        
        Returns:
            손실 딕셔너리
        """
        self.model.train()
        total_loss = 0.0
        total_vae_loss = 0.0
        total_contrastive_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # 배치 데이터
            audio = batch['audio'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            
            # 텍스트 데이터 (있는 경우)
            text = batch.get('text_tokens')
            if text is not None:
                text = text.to(self.device)
                attention_mask = (text != 0).float().to(self.device)  # 패딩 토큰(0)에 마스크 적용
            else:
                attention_mask = None
            
            # 오디오 데이터를 평탄화하여 VAE 재구성 타겟으로 사용
            audio_flat = audio.view(audio.size(0), -1)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(audio, mel_spec, text, attention_mask)
            
            # VAE 손실
            vae_outputs = outputs['vae_outputs']
            vae_total_loss, _, _ = self.compute_vae_loss(vae_outputs, audio_flat)
            
            # 대조적 손실
            clap_outputs = outputs['clap_outputs']
            contrastive_loss = self.compute_contrastive_loss(clap_outputs)
            
            # 총 손실
            loss = self.vae_weight * vae_total_loss + self.contrastive_weight * contrastive_loss
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 손실 누적
            total_loss += loss.item()
            total_vae_loss += vae_total_loss.item()
            total_contrastive_loss += contrastive_loss.item()
        
        # 평균 손실
        avg_loss = total_loss / len(self.train_loader)
        avg_vae_loss = total_vae_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'vae_loss': avg_vae_loss,
            'contrastive_loss': avg_contrastive_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """
        검증
        
        Returns:
            손실 딕셔너리
        """
        self.model.eval()
        total_loss = 0.0
        total_vae_loss = 0.0
        total_contrastive_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 배치 데이터
                audio = batch['audio'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                
                # 텍스트 데이터 (있는 경우)
                text = batch.get('text_tokens')
                if text is not None:
                    text = text.to(self.device)
                    attention_mask = (text != 0).float().to(self.device)  # 패딩 토큰(0)에 마스크 적용
                else:
                    attention_mask = None
                
                # 오디오 데이터를 평탄화하여 VAE 재구성 타겟으로 사용
                audio_flat = audio.view(audio.size(0), -1)
                
                # 순전파
                outputs = self.model(audio, mel_spec, text, attention_mask)
                
                # VAE 손실
                vae_outputs = outputs['vae_outputs']
                vae_total_loss, _, _ = self.compute_vae_loss(vae_outputs, audio_flat)
                
                # 대조적 손실
                clap_outputs = outputs['clap_outputs']
                contrastive_loss = self.compute_contrastive_loss(clap_outputs)
                
                # 총 손실
                loss = self.vae_weight * vae_total_loss + self.contrastive_weight * contrastive_loss
                
                # 손실 누적
                total_loss += loss.item()
                total_vae_loss += vae_total_loss.item()
                total_contrastive_loss += contrastive_loss.item()
        
        # 평균 손실
        avg_loss = total_loss / len(self.val_loader)
        avg_vae_loss = total_vae_loss / len(self.val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'vae_loss': avg_vae_loss,
            'contrastive_loss': avg_contrastive_loss
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
            self.history['train_vae_loss'].append(train_losses['vae_loss'])
            self.history['val_vae_loss'].append(val_losses['vae_loss'])
            self.history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
            self.history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
            
            # 결과 출력
            print(f"Train Loss: {train_losses['loss']:.4f}, VAE Loss: {train_losses['vae_loss']:.4f}, Contrastive Loss: {train_losses['contrastive_loss']:.4f}")
            print(f"Val Loss: {val_losses['loss']:.4f}, VAE Loss: {val_losses['vae_loss']:.4f}, Contrastive Loss: {val_losses['contrastive_loss']:.4f}")
            
            # 체크포인트 저장
            if val_losses['loss'] < best_val_loss:
                best_val_loss = val_losses['loss']
                patience_counter = 0
                
                # 최상의 모델 저장
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_audio_text_encoder.pt')
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
    
    def visualize_latent_space(self, loader: DataLoader, num_samples: int = 100) -> None:
        """
        잠재 공간 시각화
        
        Args:
            loader: 데이터 로더
            num_samples: 시각화할 샘플 수
        """
        self.model.eval()
        
        # 데이터 수집
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for batch in loader:
                # 배치 데이터
                audio = batch['audio'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                
                # 텍스트 데이터 (있는 경우)
                text = batch.get('text_tokens')
                if text is not None:
                    text = text.to(self.device)
                    attention_mask = (text != 0).float().to(self.device)
                else:
                    attention_mask = None
                
                # 레이블 (있는 경우)
                label = batch.get('label')
                if label is not None:
                    label = label.cpu().numpy()
                else:
                    label = np.zeros(audio.size(0))
                
                # 순전파
                outputs = self.model(audio, mel_spec, text, attention_mask)
                
                # 잠재 벡터 수집
                latent_vectors.append(outputs['fused_latent'].cpu().numpy())
                labels.append(label)
                
                if len(np.concatenate(latent_vectors)) >= num_samples:
                    break
        
        # 데이터 준비
        latent_vectors = np.concatenate(latent_vectors)[:num_samples]
        labels = np.concatenate(labels)[:num_samples]
        
        # t-SNE 적용
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
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
        
        # VAE 손실
        plt.subplot(3, 1, 2)
        plt.plot(self.history['train_vae_loss'], label='Train VAE Loss')
        plt.plot(self.history['val_vae_loss'], label='Validation VAE Loss')
        plt.title('VAE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 대조적 손실
        plt.subplot(3, 1, 3)
        plt.plot(self.history['train_contrastive_loss'], label='Train Contrastive Loss')
        plt.plot(self.history['val_contrastive_loss'], label='Validation Contrastive Loss')
        plt.title('Contrastive Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


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
    from data_preprocessing import prepare_dummy_data, EEGAudioDataset
    
    eeg_data, audio_data, mel_specs, text_tokens, labels = prepare_dummy_data(
        num_samples=200,
        eeg_channels=128,
        eeg_samples=512,
        audio_samples=16000,
        n_mels=80,
        mel_time=64,
        max_text_length=100,
        num_classes=4
    )
    
    # 데이터셋 생성
    dataset = EEGAudioDataset(eeg_data, audio_data, mel_specs, text_tokens, labels)
    
    # 데이터셋 분할
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 데이터 로더 생성
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화
    in_channels = 1
    audio_samples = 16000
    n_mels = 80
    time_frames = 64
    vocab_size = 10000
    max_seq_length = 100
    latent_dim = 128
    projection_dim = 512
    
    model = DualPathAudioTextEncoder(
        in_channels=in_channels,
        audio_samples=audio_samples,
        n_mels=n_mels,
        time_frames=time_frames,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        latent_dim=latent_dim,
        projection_dim=projection_dim
    )
    
    model = model.to(device)
    
    # 모델 요약
    print(model)
    
    # 학습기 초기화
    trainer = AudioTextEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        vae_weight=1.0,
        contrastive_weight=1.0,
        kld_weight=0.01,
        checkpoint_dir=models_dir
    )
    
    # 모델 학습 (실제 학습은 주석 처리)
    # history = trainer.train(num_epochs=10, early_stopping_patience=5)
    
    # 손실 곡선 시각화
    # trainer.plot_loss_curves()
    
    # 잠재 공간 시각화
    # trainer.visualize_latent_space(test_loader, num_samples=100)
    
    print("오디오-텍스트 인코더 모듈 테스트 완료")


if __name__ == "__main__":
    main()
