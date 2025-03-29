import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader

class LatentMappingMLP(nn.Module):
    """
    EEG 잠재 벡터를 오디오-텍스트 잠재 벡터로 매핑하는 MLP 기반 회귀 모델
    """
    def __init__(self, input_dim: int = 512, output_dim: int = 512, 
                 hidden_dims: List[int] = [1024, 1024, 512], 
                 dropout: float = 0.2, 
                 use_batch_norm: bool = True):
        """
        초기화 함수
        
        Args:
            input_dim: 입력 차원 (EEG 잠재 벡터 차원)
            output_dim: 출력 차원 (오디오-텍스트 잠재 벡터 차원)
            hidden_dims: 은닉층 차원 목록
            dropout: 드롭아웃 비율
            use_batch_norm: 배치 정규화 사용 여부
        """
        super(LatentMappingMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 레이어 구성
        layers = []
        
        # 입력층
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))
        
        # 은닉층
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
        
        # 출력층
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 시퀀셜 모델 생성
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (EEG 잠재 벡터)
            
        Returns:
            매핑된 잠재 벡터 (오디오-텍스트 잠재 벡터)
        """
        return self.model(x)


class LatentMappingResNet(nn.Module):
    """
    EEG 잠재 벡터를 오디오-텍스트 잠재 벡터로 매핑하는 ResNet 기반 회귀 모델
    """
    def __init__(self, input_dim: int = 512, output_dim: int = 512, 
                 hidden_dim: int = 1024, n_blocks: int = 4, 
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            input_dim: 입력 차원 (EEG 잠재 벡터 차원)
            output_dim: 출력 차원 (오디오-텍스트 잠재 벡터 차원)
            hidden_dim: 은닉층 차원
            n_blocks: 잔차 블록 수
            dropout: 드롭아웃 비율
        """
        super(LatentMappingResNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 입력 투영 레이어
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 잔차 블록
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        # 출력 투영 레이어
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (EEG 잠재 벡터)
            
        Returns:
            매핑된 잠재 벡터 (오디오-텍스트 잠재 벡터)
        """
        # 입력 투영
        x = self.input_proj(x)
        
        # 잔차 블록 통과
        for block in self.res_blocks:
            x = block(x)
        
        # 출력 투영
        x = self.output_proj(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    잔차 블록
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            hidden_dim: 은닉층 차원
            dropout: 드롭아웃 비율
        """
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서
            
        Returns:
            출력 텐서
        """
        identity = x
        out = self.block(x)
        out += identity  # 잔차 연결
        out = self.activation(out)
        return out


class AdversarialLatentMapping(nn.Module):
    """
    EEG 잠재 벡터를 오디오-텍스트 잠재 벡터로 매핑하는 적대적 회귀 모델
    """
    def __init__(self, input_dim: int = 512, output_dim: int = 512, 
                 hidden_dims: List[int] = [1024, 1024, 512], 
                 discriminator_dims: List[int] = [512, 256, 128, 1],
                 dropout: float = 0.2):
        """
        초기화 함수
        
        Args:
            input_dim: 입력 차원 (EEG 잠재 벡터 차원)
            output_dim: 출력 차원 (오디오-텍스트 잠재 벡터 차원)
            hidden_dims: 생성자 은닉층 차원 목록
            discriminator_dims: 판별자 은닉층 차원 목록
            dropout: 드롭아웃 비율
        """
        super(AdversarialLatentMapping, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 생성자 (매핑 모델)
        self.generator = LatentMappingMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # 판별자
        discriminator_layers = []
        
        # 입력층
        discriminator_layers.append(nn.Linear(output_dim, discriminator_dims[0]))
        discriminator_layers.append(nn.LeakyReLU(0.2))
        discriminator_layers.append(nn.Dropout(dropout))
        
        # 은닉층
        for i in range(len(discriminator_dims) - 1):
            discriminator_layers.append(nn.Linear(discriminator_dims[i], discriminator_dims[i+1]))
            if i < len(discriminator_dims) - 2:  # 마지막 레이어 제외
                discriminator_layers.append(nn.LeakyReLU(0.2))
                discriminator_layers.append(nn.Dropout(dropout))
        
        # 시그모이드 활성화 (이진 분류)
        discriminator_layers.append(nn.Sigmoid())
        
        # 판별자 모델 생성
        self.discriminator = nn.Sequential(*discriminator_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (EEG 잠재 벡터)
            
        Returns:
            (매핑된 잠재 벡터, 판별자 출력) 튜플
        """
        # 생성자 (매핑 모델) 통과
        mapped_latent = self.generator(x)
        
        # 판별자 통과
        disc_output = self.discriminator(mapped_latent)
        
        return mapped_latent, disc_output
    
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        생성자만 사용하여 매핑 수행
        
        Args:
            x: 입력 텐서 (EEG 잠재 벡터)
            
        Returns:
            매핑된 잠재 벡터
        """
        return self.generator(x)


class DualPathLatentMapping(nn.Module):
    """
    VAE와 CLAP 경로를 위한 이중 경로 잠재 벡터 매핑 모델
    """
    def __init__(self, vae_input_dim: int = 512, vae_output_dim: int = 512,
                clap_input_dim: int = 512, clap_output_dim: int = 512,
                hidden_dims: List[int] = [1024, 1024, 512],
                dropout: float = 0.2, mapping_type: str = 'mlp'):
        """
        초기화 함수
        
        Args:
            vae_input_dim: VAE 입력 차원 (EEG VAE 잠재 벡터 차원)
            vae_output_dim: VAE 출력 차원 (AudioLDM2 잠재 벡터 차원)
            clap_input_dim: CLAP 입력 차원 (EEG CLAP 잠재 벡터 차원)
            clap_output_dim: CLAP 출력 차원 (CLAP/Whisper 잠재 벡터 차원)
            hidden_dims: 은닉층 차원 목록
            dropout: 드롭아웃 비율
            mapping_type: 매핑 모델 유형 ('mlp', 'resnet', 'adversarial')
        """
        super(DualPathLatentMapping, self).__init__()
        
        self.vae_input_dim = vae_input_dim
        self.vae_output_dim = vae_output_dim
        self.clap_input_dim = clap_input_dim
        self.clap_output_dim = clap_output_dim
        self.mapping_type = mapping_type
        
        # VAE 경로 매핑 모델
        if mapping_type == 'mlp':
            self.vae_mapping = LatentMappingMLP(
                input_dim=vae_input_dim,
                output_dim=vae_output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            self.clap_mapping = LatentMappingMLP(
                input_dim=clap_input_dim,
                output_dim=clap_output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif mapping_type == 'resnet':
            self.vae_mapping = LatentMappingResNet(
                input_dim=vae_input_dim,
                output_dim=vae_output_dim,
                hidden_dim=hidden_dims[0],
                n_blocks=4,
                dropout=dropout
            )
            self.clap_mapping = LatentMappingResNet(
                input_dim=clap_input_dim,
                output_dim=clap_output_dim,
                hidden_dim=hidden_dims[0],
                n_blocks=4,
                dropout=dropout
            )
        elif mapping_type == 'adversarial':
            self.vae_mapping = AdversarialLatentMapping(
                input_dim=vae_input_dim,
                output_dim=vae_output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            self.clap_mapping = AdversarialLatentMapping(
                input_dim=clap_input_dim,
                output_dim=clap_output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        else:
            raise ValueError(f"지원되지 않는 매핑 유형: {mapping_type}")
    
    def forward(self, eeg_latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            eeg_latents: EEG 잠재 벡터 딕셔너리 {'vae_z': vae_z, 'clap_z': clap_z}
            
        Returns:
            매핑된 잠재 벡터 딕셔너리 {'vae_mapped': vae_mapped, 'clap_mapped': clap_mapped}
        """
        vae_z = eeg_latents['vae_z']
        clap_z = eeg_latents['clap_z']
        
        # VAE 경로 매핑
        if self.mapping_type == 'adversarial':
            vae_mapped, vae_disc = self.vae_mapping(vae_z)
            clap_mapped, clap_disc = self.clap_mapping(clap_z)
            
            return {
                'vae_mapped': vae_mapped,
                'clap_mapped': clap_mapped,
                'vae_disc': vae_disc,
                'clap_disc': clap_disc
            }
        else:
            vae_mapped = self.vae_mapping(vae_z)
            clap_mapped = self.clap_mapping(clap_z)
            
            return {
                'vae_mapped': vae_mapped,
                'clap_mapped': clap_mapped
            }
    
    def generate(self, eeg_latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        생성 모드 (적대적 모델의 경우 판별자 출력 없음)
        
        Args:
            eeg_latents: EEG 잠재 벡터 딕셔너리 {'vae_z': vae_z, 'clap_z': clap_z}
            
        Returns:
            매핑된 잠재 벡터 딕셔너리 {'vae_mapped': vae_mapped, 'clap_mapped': clap_mapped}
        """
        vae_z = eeg_latents['vae_z']
        clap_z = eeg_latents['clap_z']
        
        # VAE 경로 매핑
        if self.mapping_type == 'adversarial':
            vae_mapped = self.vae_mapping.generate(vae_z)
            clap_mapped = self.clap_mapping.generate(clap_z)
        else:
            vae_mapped = self.vae_mapping(vae_z)
            clap_mapped = self.clap_mapping(clap_z)
        
        return {
            'vae_mapped': vae_mapped,
            'clap_mapped': clap_mapped
        }


class LatentVectorDataset(Dataset):
    """
    잠재 벡터 매핑을 위한 데이터셋
    """
    def __init__(self, eeg_vae_latents: torch.Tensor, eeg_clap_latents: torch.Tensor,
                audio_vae_latents: torch.Tensor, audio_clap_latents: torch.Tensor):
        """
        초기화 함수
        
        Args:
            eeg_vae_latents: EEG VAE 잠재 벡터 (n_samples, vae_dim)
            eeg_clap_latents: EEG CLAP 잠재 벡터 (n_samples, clap_dim)
            audio_vae_latents: 오디오 VAE 잠재 벡터 (n_samples, vae_dim)
            audio_clap_latents: 오디오 CLAP 잠재 벡터 (n_samples, clap_dim)
        """
        assert len(eeg_vae_latents) == len(eeg_clap_latents) == len(audio_vae_latents) == len(audio_clap_latents), \
            "모든 잠재 벡터 배열의 길이가 같아야 합니다."
        
        self.eeg_vae_latents = eeg_vae_latents
        self.eeg_clap_latents = eeg_clap_latents
        self.audio_vae_latents = audio_vae_latents
        self.audio_clap_latents = audio_clap_latents
    
    def __len__(self) -> int:
        """
        데이터셋 길이 반환
        
        Returns:
            데이터셋 샘플 수
        """
        return len(self.eeg_vae_latents)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터셋 항목 반환
        
        Args:
            idx: 인덱스
            
        Returns:
            데이터 항목 딕셔너리
        """
        return {
            'eeg_vae_z': self.eeg_vae_latents[idx],
            'eeg_clap_z': self.eeg_clap_latents[idx],
            'audio_vae_z': self.audio_vae_latents[idx],
            'audio_clap_z': self.audio_clap_latents[idx]
        }


class LatentVectorMappingTrainer:
    """
    잠재 벡터 매핑 모델 학습을 위한 클래스
    """
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        초기화 함수
        
        Args:
            model: 잠재 벡터 매핑 모델
            device: 학습 장치 (CPU 또는 GPU)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 손실 함수 히스토리
        self.train_loss_history = []
        self.val_loss_history = []
        
        # 평가 지표 히스토리
        self.val_metrics_history = {
            'vae_mse': [],
            'clap_mse': [],
            'vae_r2': [],
            'clap_r2': [],
            'vae_cosine': [],
            'clap_cosine': []
        }
    
    def compute_mapping_loss(self, mapped_latents: Dict[str, torch.Tensor], 
                            target_latents: Dict[str, torch.Tensor],
                            l2_reg: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        매핑 손실 계산
        
        Args:
            mapped_latents: 매핑된 잠재 벡터 딕셔너리 {'vae_mapped': vae_mapped, 'clap_mapped': clap_mapped}
            target_latents: 목표 잠재 벡터 딕셔너리 {'vae_z': vae_z, 'clap_z': clap_z}
            l2_reg: L2 정규화 가중치
            
        Returns:
            손실 딕셔너리 {'vae_loss': vae_loss, 'clap_loss': clap_loss, 'total_loss': total_loss}
        """
        # VAE 경로 손실 (MSE)
        vae_loss = F.mse_loss(mapped_latents['vae_mapped'], target_latents['audio_vae_z'])
        
        # CLAP 경로 손실 (MSE + 코사인 유사도)
        clap_mse_loss = F.mse_loss(mapped_latents['clap_mapped'], target_latents['audio_clap_z'])
        
        # 코사인 유사도 손실 (1 - 코사인 유사도)
        clap_cos_loss = 1.0 - F.cosine_similarity(
            mapped_latents['clap_mapped'], 
            target_latents['audio_clap_z'], 
            dim=1
        ).mean()
        
        # CLAP 경로 손실 조합
        clap_loss = clap_mse_loss + clap_cos_loss
        
        # L2 정규화
        l2_loss = 0.0
        for param in self.model.parameters():
            l2_loss += torch.norm(param, 2)
        
        # 전체 손실
        total_loss = vae_loss + clap_loss + l2_reg * l2_loss
        
        return {
            'vae_loss': vae_loss,
            'clap_loss': clap_loss,
            'total_loss': total_loss
        }
    
    def compute_adversarial_loss(self, mapped_latents: Dict[str, torch.Tensor], 
                                target_latents: Dict[str, torch.Tensor],
                                disc_outputs: Dict[str, torch.Tensor],
                                gan_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        적대적 손실 계산
        
        Args:
            mapped_latents: 매핑된 잠재 벡터 딕셔너리 {'vae_mapped': vae_mapped, 'clap_mapped': clap_mapped}
            target_latents: 목표 잠재 벡터 딕셔너리 {'vae_z': vae_z, 'clap_z': clap_z}
            disc_outputs: 판별자 출력 딕셔너리 {'vae_disc': vae_disc, 'clap_disc': clap_disc}
            gan_weight: GAN 손실 가중치
            
        Returns:
            손실 딕셔너리 {'gen_loss': gen_loss, 'disc_loss': disc_loss}
        """
        # 매핑 손실 계산 (MSE)
        mapping_losses = self.compute_mapping_loss(mapped_latents, target_latents)
        
        # 생성자 손실 (판별자를 속이기 위한 손실)
        batch_size = mapped_latents['vae_mapped'].size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        
        vae_gen_loss = F.binary_cross_entropy(disc_outputs['vae_disc'], real_labels)
        clap_gen_loss = F.binary_cross_entropy(disc_outputs['clap_disc'], real_labels)
        
        # 전체 생성자 손실
        gen_loss = mapping_losses['total_loss'] + gan_weight * (vae_gen_loss + clap_gen_loss)
        
        # 판별자 손실
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # 실제 샘플에 대한 판별자 출력
        with torch.no_grad():
            real_vae_disc = self.model.vae_mapping.discriminator(target_latents['audio_vae_z'])
            real_clap_disc = self.model.clap_mapping.discriminator(target_latents['audio_clap_z'])
        
        # 실제 샘플에 대한 판별자 손실
        vae_disc_real_loss = F.binary_cross_entropy(real_vae_disc, real_labels)
        clap_disc_real_loss = F.binary_cross_entropy(real_clap_disc, real_labels)
        
        # 가짜 샘플에 대한 판별자 손실
        vae_disc_fake_loss = F.binary_cross_entropy(disc_outputs['vae_disc'], fake_labels)
        clap_disc_fake_loss = F.binary_cross_entropy(disc_outputs['clap_disc'], fake_labels)
        
        # 전체 판별자 손실
        disc_loss = (vae_disc_real_loss + vae_disc_fake_loss) / 2 + (clap_disc_real_loss + clap_disc_fake_loss) / 2
        
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'mapping_loss': mapping_losses['total_loss'],
            'vae_loss': mapping_losses['vae_loss'],
            'clap_loss': mapping_losses['clap_loss']
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                  optimizer: torch.optim.Optimizer,
                  disc_optimizer: Optional[torch.optim.Optimizer] = None,
                  gan_weight: float = 0.1) -> Dict[str, float]:
        """
        단일 학습 스텝
        
        Args:
            batch: 배치 데이터 딕셔너리
            optimizer: 옵티마이저 (생성자/매핑 모델용)
            disc_optimizer: 판별자 옵티마이저 (적대적 모델용)
            gan_weight: GAN 손실 가중치
            
        Returns:
            손실 딕셔너리
        """
        self.model.train()
        
        # 데이터를 장치로 이동
        eeg_vae_z = batch['eeg_vae_z'].to(self.device)
        eeg_clap_z = batch['eeg_clap_z'].to(self.device)
        audio_vae_z = batch['audio_vae_z'].to(self.device)
        audio_clap_z = batch['audio_clap_z'].to(self.device)
        
        # 입력 및 목표 딕셔너리 구성
        eeg_latents = {
            'vae_z': eeg_vae_z,
            'clap_z': eeg_clap_z
        }
        
        target_latents = {
            'audio_vae_z': audio_vae_z,
            'audio_clap_z': audio_clap_z
        }
        
        # 적대적 모델인 경우
        if hasattr(self.model, 'mapping_type') and self.model.mapping_type == 'adversarial':
            # 판별자 학습
            if disc_optimizer is not None:
                disc_optimizer.zero_grad()
                
                # 순전파
                outputs = self.model(eeg_latents)
                mapped_latents = {
                    'vae_mapped': outputs['vae_mapped'],
                    'clap_mapped': outputs['clap_mapped']
                }
                disc_outputs = {
                    'vae_disc': outputs['vae_disc'],
                    'clap_disc': outputs['clap_disc']
                }
                
                # 판별자 손실 계산
                losses = self.compute_adversarial_loss(mapped_latents, target_latents, disc_outputs, gan_weight)
                disc_loss = losses['disc_loss']
                
                # 역전파
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
            
            # 생성자 학습
            optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(eeg_latents)
            mapped_latents = {
                'vae_mapped': outputs['vae_mapped'],
                'clap_mapped': outputs['clap_mapped']
            }
            disc_outputs = {
                'vae_disc': outputs['vae_disc'],
                'clap_disc': outputs['clap_disc']
            }
            
            # 생성자 손실 계산
            losses = self.compute_adversarial_loss(mapped_latents, target_latents, disc_outputs, gan_weight)
            gen_loss = losses['gen_loss']
            
            # 역전파
            gen_loss.backward()
            optimizer.step()
            
            return {
                'gen_loss': gen_loss.item(),
                'disc_loss': losses['disc_loss'].item() if disc_optimizer is not None else 0.0,
                'mapping_loss': losses['mapping_loss'].item(),
                'vae_loss': losses['vae_loss'].item(),
                'clap_loss': losses['clap_loss'].item()
            }
        else:
            # 일반 매핑 모델
            optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(eeg_latents)
            
            # 손실 계산
            losses = self.compute_mapping_loss(outputs, target_latents)
            total_loss = losses['total_loss']
            
            # 역전파
            total_loss.backward()
            optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'vae_loss': losses['vae_loss'].item(),
                'clap_loss': losses['clap_loss'].item()
            }
    
    def validate(self, val_loader: torch.utils.data.DataLoader, 
                gan_weight: float = 0.1) -> Dict[str, float]:
        """
        검증 수행
        
        Args:
            val_loader: 검증 데이터 로더
            gan_weight: GAN 손실 가중치 (적대적 모델용)
            
        Returns:
            평균 손실 및 평가 지표 딕셔너리
        """
        self.model.eval()
        
        total_loss = 0.0
        vae_loss_sum = 0.0
        clap_loss_sum = 0.0
        gen_loss_sum = 0.0
        disc_loss_sum = 0.0
        
        # 평가 지표 계산을 위한 배열
        all_vae_mapped = []
        all_clap_mapped = []
        all_vae_target = []
        all_clap_target = []
        
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 데이터를 장치로 이동
                eeg_vae_z = batch['eeg_vae_z'].to(self.device)
                eeg_clap_z = batch['eeg_clap_z'].to(self.device)
                audio_vae_z = batch['audio_vae_z'].to(self.device)
                audio_clap_z = batch['audio_clap_z'].to(self.device)
                
                # 입력 및 목표 딕셔너리 구성
                eeg_latents = {
                    'vae_z': eeg_vae_z,
                    'clap_z': eeg_clap_z
                }
                
                target_latents = {
                    'audio_vae_z': audio_vae_z,
                    'audio_clap_z': audio_clap_z
                }
                
                # 순전파
                outputs = self.model(eeg_latents)
                
                # 적대적 모델인 경우
                if hasattr(self.model, 'mapping_type') and self.model.mapping_type == 'adversarial':
                    mapped_latents = {
                        'vae_mapped': outputs['vae_mapped'],
                        'clap_mapped': outputs['clap_mapped']
                    }
                    disc_outputs = {
                        'vae_disc': outputs['vae_disc'],
                        'clap_disc': outputs['clap_disc']
                    }
                    
                    # 손실 계산
                    losses = self.compute_adversarial_loss(mapped_latents, target_latents, disc_outputs, gan_weight)
                    
                    # 손실 누적
                    gen_loss_sum += losses['gen_loss'].item()
                    disc_loss_sum += losses['disc_loss'].item()
                    total_loss += losses['mapping_loss'].item()
                    vae_loss_sum += losses['vae_loss'].item()
                    clap_loss_sum += losses['clap_loss'].item()
                    
                    # 매핑된 잠재 벡터 저장
                    all_vae_mapped.append(outputs['vae_mapped'].cpu().numpy())
                    all_clap_mapped.append(outputs['clap_mapped'].cpu().numpy())
                else:
                    # 손실 계산
                    losses = self.compute_mapping_loss(outputs, target_latents)
                    
                    # 손실 누적
                    total_loss += losses['total_loss'].item()
                    vae_loss_sum += losses['vae_loss'].item()
                    clap_loss_sum += losses['clap_loss'].item()
                    
                    # 매핑된 잠재 벡터 저장
                    all_vae_mapped.append(outputs['vae_mapped'].cpu().numpy())
                    all_clap_mapped.append(outputs['clap_mapped'].cpu().numpy())
                
                # 목표 잠재 벡터 저장
                all_vae_target.append(audio_vae_z.cpu().numpy())
                all_clap_target.append(audio_clap_z.cpu().numpy())
                
                n_batches += 1
        
        # 평균 손실 계산
        avg_total_loss = total_loss / n_batches
        avg_vae_loss = vae_loss_sum / n_batches
        avg_clap_loss = clap_loss_sum / n_batches
        
        # 배열 연결
        all_vae_mapped = np.concatenate(all_vae_mapped)
        all_clap_mapped = np.concatenate(all_clap_mapped)
        all_vae_target = np.concatenate(all_vae_target)
        all_clap_target = np.concatenate(all_clap_target)
        
        # 평가 지표 계산
        # MSE
        vae_mse = mean_squared_error(all_vae_target, all_vae_mapped)
        clap_mse = mean_squared_error(all_clap_target, all_clap_mapped)
        
        # R² 점수
        vae_r2 = r2_score(all_vae_target, all_vae_mapped)
        clap_r2 = r2_score(all_clap_target, all_clap_mapped)
        
        # 코사인 유사도
        vae_cosine = np.mean([
            np.dot(all_vae_mapped[i], all_vae_target[i]) / 
            (np.linalg.norm(all_vae_mapped[i]) * np.linalg.norm(all_vae_target[i]))
            for i in range(len(all_vae_mapped))
        ])
        
        clap_cosine = np.mean([
            np.dot(all_clap_mapped[i], all_clap_target[i]) / 
            (np.linalg.norm(all_clap_mapped[i]) * np.linalg.norm(all_clap_target[i]))
            for i in range(len(all_clap_mapped))
        ])
        
        # 결과 딕셔너리
        results = {
            'total_loss': avg_total_loss,
            'vae_loss': avg_vae_loss,
            'clap_loss': avg_clap_loss,
            'vae_mse': vae_mse,
            'clap_mse': clap_mse,
            'vae_r2': vae_r2,
            'clap_r2': clap_r2,
            'vae_cosine': vae_cosine,
            'clap_cosine': clap_cosine
        }
        
        # 적대적 모델인 경우 추가 손실 포함
        if hasattr(self.model, 'mapping_type') and self.model.mapping_type == 'adversarial':
            results['gen_loss'] = gen_loss_sum / n_batches
            results['disc_loss'] = disc_loss_sum / n_batches
        
        return results
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             n_epochs: int = 100,
             disc_optimizer: Optional[torch.optim.Optimizer] = None,
             gan_weight: float = 0.1,
             patience: int = 10,
             checkpoint_dir: str = './checkpoints') -> None:
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            optimizer: 옵티마이저 (생성자/매핑 모델용)
            n_epochs: 에폭 수
            disc_optimizer: 판별자 옵티마이저 (적대적 모델용)
            gan_weight: GAN 손실 가중치
            patience: 조기 종료 인내심
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        # 체크포인트 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 학습
            epoch_train_loss = 0.0
            epoch_train_vae_loss = 0.0
            epoch_train_clap_loss = 0.0
            epoch_train_gen_loss = 0.0
            epoch_train_disc_loss = 0.0
            n_train_batches = 0
            
            for batch in train_loader:
                # 학습 스텝
                losses = self.train_step(batch, optimizer, disc_optimizer, gan_weight)
                
                # 손실 누적
                if 'total_loss' in losses:
                    epoch_train_loss += losses['total_loss']
                elif 'mapping_loss' in losses:
                    epoch_train_loss += losses['mapping_loss']
                
                epoch_train_vae_loss += losses['vae_loss']
                epoch_train_clap_loss += losses['clap_loss']
                
                if 'gen_loss' in losses:
                    epoch_train_gen_loss += losses['gen_loss']
                if 'disc_loss' in losses:
                    epoch_train_disc_loss += losses['disc_loss']
                
                n_train_batches += 1
            
            # 평균 학습 손실 계산
            epoch_train_loss /= n_train_batches
            epoch_train_vae_loss /= n_train_batches
            epoch_train_clap_loss /= n_train_batches
            
            if hasattr(self.model, 'mapping_type') and self.model.mapping_type == 'adversarial':
                epoch_train_gen_loss /= n_train_batches
                epoch_train_disc_loss /= n_train_batches
            
            # 검증
            val_results = self.validate(val_loader, gan_weight)
            
            # 손실 히스토리 업데이트
            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(val_results['total_loss'])
            
            # 평가 지표 히스토리 업데이트
            for metric in ['vae_mse', 'clap_mse', 'vae_r2', 'clap_r2', 'vae_cosine', 'clap_cosine']:
                self.val_metrics_history[metric].append(val_results[metric])
            
            # 로그 출력
            log_str = f"Epoch {epoch+1}/{n_epochs} - "
            log_str += f"Train Loss: {epoch_train_loss:.4f} "
            log_str += f"(VAE: {epoch_train_vae_loss:.4f}, CLAP: {epoch_train_clap_loss:.4f}) - "
            
            if hasattr(self.model, 'mapping_type') and self.model.mapping_type == 'adversarial':
                log_str += f"Gen: {epoch_train_gen_loss:.4f}, Disc: {epoch_train_disc_loss:.4f} - "
            
            log_str += f"Val Loss: {val_results['total_loss']:.4f} "
            log_str += f"(VAE: {val_results['vae_loss']:.4f}, CLAP: {val_results['clap_loss']:.4f}) - "
            log_str += f"VAE R²: {val_results['vae_r2']:.4f}, CLAP R²: {val_results['clap_r2']:.4f} - "
            log_str += f"VAE Cos: {val_results['vae_cosine']:.4f}, CLAP Cos: {val_results['clap_cosine']:.4f}"
            
            print(log_str)
            
            # 체크포인트 저장
            if val_results['total_loss'] < best_val_loss:
                best_val_loss = val_results['total_loss']
                patience_counter = 0
                
                # 최상의 모델 저장
                checkpoint_path = os.path.join(checkpoint_dir, 'best_latent_mapping.pt')
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': val_results['total_loss'],
                    'val_metrics': {k: v[-1] for k, v in self.val_metrics_history.items()}
                }
                
                if disc_optimizer is not None:
                    checkpoint['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                
                print(f"Checkpoint saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
    
    def visualize_loss(self) -> None:
        """
        손실 곡선 시각화
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def visualize_metrics(self) -> None:
        """
        평가 지표 시각화
        """
        plt.figure(figsize=(15, 10))
        
        # MSE
        plt.subplot(2, 3, 1)
        plt.plot(self.val_metrics_history['vae_mse'], label='VAE MSE')
        plt.plot(self.val_metrics_history['clap_mse'], label='CLAP MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        
        # R² 점수
        plt.subplot(2, 3, 2)
        plt.plot(self.val_metrics_history['vae_r2'], label='VAE R²')
        plt.plot(self.val_metrics_history['clap_r2'], label='CLAP R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.title('R² Score')
        plt.legend()
        plt.grid(True)
        
        # 코사인 유사도
        plt.subplot(2, 3, 3)
        plt.plot(self.val_metrics_history['vae_cosine'], label='VAE Cosine')
        plt.plot(self.val_metrics_history['clap_cosine'], label='CLAP Cosine')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_latent_mapping(self, data_loader: torch.utils.data.DataLoader, 
                               n_samples: int = 10) -> None:
        """
        잠재 벡터 매핑 시각화
        
        Args:
            data_loader: 데이터 로더
            n_samples: 시각화할 샘플 수
        """
        self.model.eval()
        
        # 샘플 데이터 수집
        eeg_vae_latents = []
        eeg_clap_latents = []
        audio_vae_latents = []
        audio_clap_latents = []
        mapped_vae_latents = []
        mapped_clap_latents = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 데이터를 장치로 이동
                eeg_vae_z = batch['eeg_vae_z'].to(self.device)
                eeg_clap_z = batch['eeg_clap_z'].to(self.device)
                audio_vae_z = batch['audio_vae_z'].to(self.device)
                audio_clap_z = batch['audio_clap_z'].to(self.device)
                
                # 입력 딕셔너리 구성
                eeg_latents = {
                    'vae_z': eeg_vae_z,
                    'clap_z': eeg_clap_z
                }
                
                # 순전파
                outputs = self.model.generate(eeg_latents)
                
                # 결과 저장
                eeg_vae_latents.append(eeg_vae_z.cpu().numpy())
                eeg_clap_latents.append(eeg_clap_z.cpu().numpy())
                audio_vae_latents.append(audio_vae_z.cpu().numpy())
                audio_clap_latents.append(audio_clap_z.cpu().numpy())
                mapped_vae_latents.append(outputs['vae_mapped'].cpu().numpy())
                mapped_clap_latents.append(outputs['clap_mapped'].cpu().numpy())
                
                # 충분한 샘플을 수집했는지 확인
                if len(np.concatenate(eeg_vae_latents)) >= n_samples:
                    break
        
        # 배열 연결 및 샘플 선택
        eeg_vae_latents = np.concatenate(eeg_vae_latents)[:n_samples]
        eeg_clap_latents = np.concatenate(eeg_clap_latents)[:n_samples]
        audio_vae_latents = np.concatenate(audio_vae_latents)[:n_samples]
        audio_clap_latents = np.concatenate(audio_clap_latents)[:n_samples]
        mapped_vae_latents = np.concatenate(mapped_vae_latents)[:n_samples]
        mapped_clap_latents = np.concatenate(mapped_clap_latents)[:n_samples]
        
        # t-SNE 적용
        from sklearn.manifold import TSNE
        
        # VAE 경로
        vae_latents = np.concatenate([eeg_vae_latents, mapped_vae_latents, audio_vae_latents])
        vae_labels = np.concatenate([
            np.zeros(n_samples),  # EEG
            np.ones(n_samples),   # Mapped
            np.ones(n_samples) * 2  # Audio
        ])
        
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples-1), random_state=42)
        vae_tsne = tsne.fit_transform(vae_latents)
        
        # CLAP 경로
        clap_latents = np.concatenate([eeg_clap_latents, mapped_clap_latents, audio_clap_latents])
        clap_labels = np.concatenate([
            np.zeros(n_samples),  # EEG
            np.ones(n_samples),   # Mapped
            np.ones(n_samples) * 2  # Audio
        ])
        
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples-1), random_state=42)
        clap_tsne = tsne.fit_transform(clap_latents)
        
        # 시각화
        plt.figure(figsize=(16, 8))
        
        # VAE 경로
        plt.subplot(1, 2, 1)
        for i, label in enumerate(['EEG', 'Mapped', 'Audio']):
            mask = vae_labels == i
            plt.scatter(
                vae_tsne[mask, 0], 
                vae_tsne[mask, 1], 
                label=label,
                alpha=0.7
            )
        
        plt.title('VAE Latent Space (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        
        # CLAP 경로
        plt.subplot(1, 2, 2)
        for i, label in enumerate(['EEG', 'Mapped', 'Audio']):
            mask = clap_labels == i
            plt.scatter(
                clap_tsne[mask, 0], 
                clap_tsne[mask, 1], 
                label=label,
                alpha=0.7
            )
        
        plt.title('CLAP Latent Space (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def create_latent_mapping_model(vae_dim: int = 512, clap_dim: int = 512, 
                              mapping_type: str = 'mlp') -> nn.Module:
    """
    잠재 벡터 매핑 모델 생성
    
    Args:
        vae_dim: VAE 잠재 벡터 차원
        clap_dim: CLAP 잠재 벡터 차원
        mapping_type: 매핑 모델 유형 ('mlp', 'resnet', 'adversarial')
        
    Returns:
        잠재 벡터 매핑 모델
    """
    if mapping_type == 'dual_path':
        model = DualPathLatentMapping(
            vae_input_dim=vae_dim,
            vae_output_dim=vae_dim,
            clap_input_dim=clap_dim,
            clap_output_dim=clap_dim,
            hidden_dims=[1024, 1024, 512],
            dropout=0.2,
            mapping_type='mlp'
        )
    elif mapping_type == 'resnet':
        model = DualPathLatentMapping(
            vae_input_dim=vae_dim,
            vae_output_dim=vae_dim,
            clap_input_dim=clap_dim,
            clap_output_dim=clap_dim,
            hidden_dims=[1024, 1024, 512],
            dropout=0.2,
            mapping_type='resnet'
        )
    elif mapping_type == 'adversarial':
        model = DualPathLatentMapping(
            vae_input_dim=vae_dim,
            vae_output_dim=vae_dim,
            clap_input_dim=clap_dim,
            clap_output_dim=clap_dim,
            hidden_dims=[1024, 1024, 512],
            dropout=0.2,
            mapping_type='adversarial'
        )
    else:
        raise ValueError(f"지원되지 않는 매핑 유형: {mapping_type}")
    
    return model


if __name__ == "__main__":
    # 테스트 코드
    # 모델 생성
    model = create_latent_mapping_model(vae_dim=512, clap_dim=512, mapping_type='dual_path')
    print(model)
    
    # 입력 텐서 생성
    batch_size = 4
    vae_dim = 512
    clap_dim = 512
    
    eeg_vae_z = torch.randn(batch_size, vae_dim)
    eeg_clap_z = torch.randn(batch_size, clap_dim)
    
    eeg_latents = {
        'vae_z': eeg_vae_z,
        'clap_z': eeg_clap_z
    }
    
    # 순전파
    outputs = model(eeg_latents)
    
    # 출력 확인
    print("VAE 매핑 크기:", outputs['vae_mapped'].shape)
    print("CLAP 매핑 크기:", outputs['clap_mapped'].shape)
