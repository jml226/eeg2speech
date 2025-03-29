import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Union, Any
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor

class EEGEncoderVAE(nn.Module):
    """
    EEG 데이터를 위한 변분 오토인코더 (VAE) 인코더
    AudioLDM2의 잠재 공간 차원과 일치하도록 설계됨
    """
    def __init__(self, n_channels: int = 128, n_times: int = 2000, 
                 latent_dim: int = 512, hidden_dims: List[int] = [64, 128, 256, 512]):
        """
        초기화 함수
        
        Args:
            n_channels: EEG 채널 수
            n_times: 시간 포인트 수
            latent_dim: 잠재 공간 차원 (AudioLDM2와 일치해야 함)
            hidden_dims: 은닉층 차원 목록
        """
        super(EEGEncoderVAE, self).__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.latent_dim = latent_dim
        
        # 1D CNN 인코더 레이어
        self.encoder_cnn = nn.ModuleList()
        
        # 첫 번째 레이어
        self.encoder_cnn.append(
            nn.Sequential(
                nn.Conv1d(n_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
        )
        
        # 나머지 레이어
        for i in range(len(hidden_dims) - 1):
            self.encoder_cnn.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # 시간 차원 계산
        self.time_dim = n_times
        for _ in range(len(hidden_dims)):
            self.time_dim = (self.time_dim + 1) // 2  # 각 레이어마다 절반으로 감소
        
        # 양방향 GRU 레이어
        self.gru = nn.GRU(
            input_size=hidden_dims[-1],
            hidden_size=hidden_dims[-1],
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 선형 투영 레이어
        self.fc_mu = nn.Linear(hidden_dims[-1] * 2, latent_dim)  # 양방향이므로 *2
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 2, latent_dim)
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """
        가중치 초기화
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EEG 데이터를 잠재 공간으로 인코딩
        
        Args:
            x: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            
        Returns:
            (평균, 로그 분산) 튜플
        """
        # CNN 인코더 통과
        for layer in self.encoder_cnn:
            x = layer(x)
        
        # 차원 변환: (batch_size, channels, time) -> (batch_size, time, channels)
        x = x.permute(0, 2, 1)
        
        # GRU 통과
        x, _ = self.gru(x)
        
        # 마지막 시간 스텝 또는 모든 시간 스텝의 평균 사용
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim*2)
        
        # 평균과 로그 분산 계산
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        재매개화 트릭
        
        Args:
            mu: 평균 텐서
            logvar: 로그 분산 텐서
            
        Returns:
            샘플링된 잠재 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            
        Returns:
            (잠재 벡터, 평균, 로그 분산) 튜플
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class EEGEncoderCLAP(nn.Module):
    """
    EEG 데이터를 위한 CLAP 기반 대조적 인코더
    CLAP의 잠재 공간 차원과 일치하도록 설계됨
    """
    def __init__(self, n_channels: int = 128, n_times: int = 2000, 
                 latent_dim: int = 512, hidden_dims: List[int] = [64, 128, 256, 512]):
        """
        초기화 함수
        
        Args:
            n_channels: EEG 채널 수
            n_times: 시간 포인트 수
            latent_dim: 잠재 공간 차원 (CLAP와 일치해야 함)
            hidden_dims: 은닉층 차원 목록
        """
        super(EEGEncoderCLAP, self).__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.latent_dim = latent_dim
        
        # 1D CNN 인코더 레이어
        self.encoder_cnn = nn.ModuleList()
        
        # 첫 번째 레이어
        self.encoder_cnn.append(
            nn.Sequential(
                nn.Conv1d(n_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
        )
        
        # 나머지 레이어
        for i in range(len(hidden_dims) - 1):
            self.encoder_cnn.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.GELU()
                )
            )
        
        # 시간 차원 계산
        self.time_dim = n_times
        for _ in range(len(hidden_dims)):
            self.time_dim = (self.time_dim + 1) // 2  # 각 레이어마다 절반으로 감소
        
        # 자기 주의 메커니즘
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dims[-1], dropout=0.1, max_len=self.time_dim)
        
        # 선형 투영 레이어
        self.fc = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 정규화 레이어
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """
        가중치 초기화
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            
        Returns:
            CLAP 호환 잠재 벡터
        """
        # CNN 인코더 통과
        for layer in self.encoder_cnn:
            x = layer(x)
        
        # 차원 변환: (batch_size, channels, time) -> (batch_size, time, channels)
        x = x.permute(0, 2, 1)
        
        # 위치 인코딩 적용
        x = self.pos_encoder(x)
        
        # 자기 주의 메커니즘 적용
        x, _ = self.self_attention(x, x, x)
        
        # 전역 평균 풀링
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        # 선형 투영
        x = self.fc(x)
        
        # 정규화
        x = self.layer_norm(x)
        
        # L2 정규화 (CLAP와 유사하게)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class PositionalEncoding(nn.Module):
    """
    트랜스포머를 위한 위치 인코딩
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        초기화 함수
        
        Args:
            d_model: 모델 차원
            dropout: 드롭아웃 비율
            max_len: 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, d_model)
            
        Returns:
            위치 인코딩이 적용된 텐서
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class DualPathEEGEncoder(nn.Module):
    """
    EEG 데이터를 위한 이중 경로 인코더
    VAE 경로와 CLAP 경로를 결합
    """
    def __init__(self, n_channels: int = 128, n_times: int = 2000, 
                 vae_latent_dim: int = 512, clap_latent_dim: int = 512,
                 hidden_dims: List[int] = [64, 128, 256, 512]):
        """
        초기화 함수
        
        Args:
            n_channels: EEG 채널 수
            n_times: 시간 포인트 수
            vae_latent_dim: VAE 잠재 공간 차원 (AudioLDM2와 일치해야 함)
            clap_latent_dim: CLAP 잠재 공간 차원 (CLAP와 일치해야 함)
            hidden_dims: 은닉층 차원 목록
        """
        super(DualPathEEGEncoder, self).__init__()
        
        # VAE 경로
        self.vae_encoder = EEGEncoderVAE(
            n_channels=n_channels,
            n_times=n_times,
            latent_dim=vae_latent_dim,
            hidden_dims=hidden_dims
        )
        
        # CLAP 경로
        self.clap_encoder = EEGEncoderCLAP(
            n_channels=n_channels,
            n_times=n_times,
            latent_dim=clap_latent_dim,
            hidden_dims=hidden_dims
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            x: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            
        Returns:
            잠재 벡터 딕셔너리 {'vae_z': vae_z, 'vae_mu': vae_mu, 'vae_logvar': vae_logvar, 'clap_z': clap_z}
        """
        # VAE 경로
        vae_z, vae_mu, vae_logvar = self.vae_encoder(x)
        
        # CLAP 경로
        clap_z = self.clap_encoder(x)
        
        return {
            'vae_z': vae_z,
            'vae_mu': vae_mu,
            'vae_logvar': vae_logvar,
            'clap_z': clap_z
        }


class EEGEncoderTrainer:
    """
    EEG 인코더 학습을 위한 클래스
    """
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        초기화 함수
        
        Args:
            model: EEG 인코더 모델
            device: 학습 장치 (CPU 또는 GPU)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 손실 함수 히스토리
        self.train_loss_history = []
        self.val_loss_history = []
    
    def compute_vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                         mu: torch.Tensor, logvar: torch.Tensor, 
                         kl_weight: float = 0.1) -> torch.Tensor:
        """
        VAE 손실 계산
        
        Args:
            recon_x: 재구성된 EEG 데이터
            x: 원본 EEG 데이터
            mu: 평균 벡터
            logvar: 로그 분산 벡터
            kl_weight: KL 발산 가중치
            
        Returns:
            VAE 손실 (재구성 손실 + KL 발산)
        """
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL 발산
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 전체 손실
        loss = recon_loss + kl_weight * kl_loss
        
        return loss
    
    def compute_contrastive_loss(self, eeg_embeds: torch.Tensor, audio_embeds: torch.Tensor, 
                                temperature: float = 0.07) -> torch.Tensor:
        """
        대조적 손실 계산 (InfoNCE)
        
        Args:
            eeg_embeds: EEG 임베딩 (batch_size, embed_dim)
            audio_embeds: 오디오 임베딩 (batch_size, embed_dim)
            temperature: 온도 파라미터
            
        Returns:
            대조적 손실
        """
        # 코사인 유사도 행렬 계산
        logits = torch.matmul(eeg_embeds, audio_embeds.t()) / temperature
        
        # 레이블 생성 (대각선 요소가 양성 쌍)
        labels = torch.arange(logits.size(0), device=self.device)
        
        # 대조적 손실 계산 (교차 엔트로피)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        loss = loss / 2  # 양방향 손실의 평균
        
        return loss
    
    def train_step(self, eeg_data: torch.Tensor, audio_embeds: torch.Tensor, 
                  optimizer: torch.optim.Optimizer, vae_decoder: Optional[nn.Module] = None,
                  contrastive_weight: float = 1.0, vae_weight: float = 1.0) -> Dict[str, float]:
        """
        단일 학습 스텝
        
        Args:
            eeg_data: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            audio_embeds: 오디오 임베딩 텐서 (batch_size, embed_dim)
            optimizer: 옵티마이저
            vae_decoder: VAE 디코더 (선택적)
            contrastive_weight: 대조적 손실 가중치
            vae_weight: VAE 손실 가중치
            
        Returns:
            손실 딕셔너리 {'total_loss': total_loss, 'vae_loss': vae_loss, 'contrastive_loss': contrastive_loss}
        """
        self.model.train()
        optimizer.zero_grad()
        
        # 데이터를 장치로 이동
        eeg_data = eeg_data.to(self.device)
        audio_embeds = audio_embeds.to(self.device)
        
        # 순전파
        outputs = self.model(eeg_data)
        vae_z, vae_mu, vae_logvar = outputs['vae_z'], outputs['vae_mu'], outputs['vae_logvar']
        clap_z = outputs['clap_z']
        
        # 손실 계산
        vae_loss = 0.0
        if vae_decoder is not None:
            # VAE 디코더가 제공된 경우 재구성 손실 계산
            recon_x = vae_decoder(vae_z)
            vae_loss = self.compute_vae_loss(recon_x, eeg_data, vae_mu, vae_logvar)
        else:
            # 디코더 없이 KL 발산만 계산
            vae_loss = -0.5 * torch.mean(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
        
        # 대조적 손실 계산
        contrastive_loss = self.compute_contrastive_loss(clap_z, audio_embeds)
        
        # 전체 손실
        total_loss = vae_weight * vae_loss + contrastive_weight * contrastive_loss
        
        # 역전파
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'vae_loss': vae_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader, 
                vae_decoder: Optional[nn.Module] = None,
                contrastive_weight: float = 1.0, vae_weight: float = 1.0) -> Dict[str, float]:
        """
        검증 수행
        
        Args:
            val_loader: 검증 데이터 로더
            vae_decoder: VAE 디코더 (선택적)
            contrastive_weight: 대조적 손실 가중치
            vae_weight: VAE 손실 가중치
            
        Returns:
            평균 손실 딕셔너리 {'total_loss': avg_total_loss, 'vae_loss': avg_vae_loss, 'contrastive_loss': avg_contrastive_loss}
        """
        self.model.eval()
        total_loss = 0.0
        vae_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 데이터 추출
                eeg_data = batch['eeg'].to(self.device)
                audio_embeds = batch['audio_embed'].to(self.device)
                
                # 순전파
                outputs = self.model(eeg_data)
                vae_z, vae_mu, vae_logvar = outputs['vae_z'], outputs['vae_mu'], outputs['vae_logvar']
                clap_z = outputs['clap_z']
                
                # 손실 계산
                batch_vae_loss = 0.0
                if vae_decoder is not None:
                    # VAE 디코더가 제공된 경우 재구성 손실 계산
                    recon_x = vae_decoder(vae_z)
                    batch_vae_loss = self.compute_vae_loss(recon_x, eeg_data, vae_mu, vae_logvar)
                else:
                    # 디코더 없이 KL 발산만 계산
                    batch_vae_loss = -0.5 * torch.mean(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
                
                # 대조적 손실 계산
                batch_contrastive_loss = self.compute_contrastive_loss(clap_z, audio_embeds)
                
                # 전체 손실
                batch_total_loss = vae_weight * batch_vae_loss + contrastive_weight * batch_contrastive_loss
                
                # 손실 누적
                total_loss += batch_total_loss.item()
                vae_loss_sum += batch_vae_loss.item()
                contrastive_loss_sum += batch_contrastive_loss.item()
                n_batches += 1
        
        # 평균 손실 계산
        avg_total_loss = total_loss / n_batches
        avg_vae_loss = vae_loss_sum / n_batches
        avg_contrastive_loss = contrastive_loss_sum / n_batches
        
        return {
            'total_loss': avg_total_loss,
            'vae_loss': avg_vae_loss,
            'contrastive_loss': avg_contrastive_loss
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             n_epochs: int = 100,
             vae_decoder: Optional[nn.Module] = None,
             contrastive_weight: float = 1.0,
             vae_weight: float = 1.0,
             patience: int = 10,
             checkpoint_dir: str = './checkpoints') -> None:
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            optimizer: 옵티마이저
            n_epochs: 에폭 수
            vae_decoder: VAE 디코더 (선택적)
            contrastive_weight: 대조적 손실 가중치
            vae_weight: VAE 손실 가중치
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
            epoch_train_contrastive_loss = 0.0
            n_train_batches = 0
            
            for batch in train_loader:
                # 데이터 추출
                eeg_data = batch['eeg']
                audio_embeds = batch['audio_embed']
                
                # 학습 스텝
                losses = self.train_step(
                    eeg_data, 
                    audio_embeds, 
                    optimizer, 
                    vae_decoder, 
                    contrastive_weight, 
                    vae_weight
                )
                
                # 손실 누적
                epoch_train_loss += losses['total_loss']
                epoch_train_vae_loss += losses['vae_loss']
                epoch_train_contrastive_loss += losses['contrastive_loss']
                n_train_batches += 1
            
            # 평균 학습 손실 계산
            epoch_train_loss /= n_train_batches
            epoch_train_vae_loss /= n_train_batches
            epoch_train_contrastive_loss /= n_train_batches
            
            # 검증
            val_losses = self.validate(
                val_loader, 
                vae_decoder, 
                contrastive_weight, 
                vae_weight
            )
            
            # 손실 히스토리 업데이트
            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(val_losses['total_loss'])
            
            # 로그 출력
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f} "
                  f"(VAE: {epoch_train_vae_loss:.4f}, Contrastive: {epoch_train_contrastive_loss:.4f}) - "
                  f"Val Loss: {val_losses['total_loss']:.4f} "
                  f"(VAE: {val_losses['vae_loss']:.4f}, Contrastive: {val_losses['contrastive_loss']:.4f})")
            
            # 체크포인트 저장
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                patience_counter = 0
                
                # 최상의 모델 저장
                checkpoint_path = os.path.join(checkpoint_dir, 'best_eeg_encoder.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': val_losses['total_loss'],
                }, checkpoint_path)
                
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
    
    def visualize_latent_space(self, data_loader: torch.utils.data.DataLoader, 
                              n_samples: int = 1000, perplexity: int = 30) -> None:
        """
        잠재 공간 시각화 (t-SNE)
        
        Args:
            data_loader: 데이터 로더
            n_samples: 시각화할 샘플 수
            perplexity: t-SNE 퍼플렉시티
        """
        from sklearn.manifold import TSNE
        
        self.model.eval()
        
        # 잠재 벡터와 레이블 수집
        vae_latents = []
        clap_latents = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 데이터 추출
                eeg_data = batch['eeg'].to(self.device)
                if 'label' in batch:
                    batch_labels = batch['label'].cpu().numpy()
                else:
                    batch_labels = np.zeros(len(eeg_data))
                
                # 순전파
                outputs = self.model(eeg_data)
                vae_z = outputs['vae_z'].cpu().numpy()
                clap_z = outputs['clap_z'].cpu().numpy()
                
                # 결과 저장
                vae_latents.append(vae_z)
                clap_latents.append(clap_z)
                labels.append(batch_labels)
                
                # 충분한 샘플을 수집했는지 확인
                if len(np.concatenate(vae_latents)) >= n_samples:
                    break
        
        # 배열 연결
        vae_latents = np.concatenate(vae_latents)[:n_samples]
        clap_latents = np.concatenate(clap_latents)[:n_samples]
        labels = np.concatenate(labels)[:n_samples]
        
        # t-SNE 적용
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        vae_tsne = tsne.fit_transform(vae_latents)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        clap_tsne = tsne.fit_transform(clap_latents)
        
        # 시각화
        plt.figure(figsize=(16, 8))
        
        # VAE 잠재 공간
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(vae_tsne[:, 0], vae_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.title('VAE Latent Space (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # CLAP 잠재 공간
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(clap_tsne[:, 0], clap_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.title('CLAP Latent Space (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.show()


def load_pretrained_clap_model() -> Tuple[Any, Any]:
    """
    사전 학습된 CLAP 모델 로드
    
    Returns:
        (CLAP 모델, CLAP 프로세서) 튜플
    """
    try:
        # CLAP 모델 및 프로세서 로드
        model_id = "laion/clap-htsat-fused"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        
        return model, processor
    except Exception as e:
        print(f"CLAP 모델 로드 중 오류 발생: {e}")
        print("예시 모델 및 프로세서 반환")
        return None, None


def extract_clap_embeddings(audio_files: List[str], clap_model: Any, clap_processor: Any) -> np.ndarray:
    """
    오디오 파일에서 CLAP 임베딩 추출
    
    Args:
        audio_files: 오디오 파일 경로 목록
        clap_model: CLAP 모델
        clap_processor: CLAP 프로세서
        
    Returns:
        CLAP 임베딩 배열 (n_samples, embed_dim)
    """
    if clap_model is None or clap_processor is None:
        print("CLAP 모델 또는 프로세서가 로드되지 않았습니다. 예시 임베딩 반환")
        return np.random.randn(len(audio_files), 512)
    
    embeddings = []
    
    for audio_file in audio_files:
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_file, sr=48000, mono=True)
            
            # CLAP 입력 형식으로 변환
            inputs = clap_processor(
                audios=audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            # 임베딩 추출
            with torch.no_grad():
                outputs = clap_model(**inputs)
                embedding = outputs.audio_embeds.cpu().numpy()
            
            embeddings.append(embedding)
        except Exception as e:
            print(f"{audio_file} 처리 중 오류 발생: {e}")
            # 오류 발생 시 랜덤 임베딩 사용
            embeddings.append(np.random.randn(1, 512))
    
    return np.vstack(embeddings)


def create_eeg_encoder(n_channels: int = 128, n_times: int = 2000,
                      vae_latent_dim: int = 512, clap_latent_dim: int = 512) -> DualPathEEGEncoder:
    """
    EEG 인코더 생성
    
    Args:
        n_channels: EEG 채널 수
        n_times: 시간 포인트 수
        vae_latent_dim: VAE 잠재 공간 차원 (AudioLDM2와 일치해야 함)
        clap_latent_dim: CLAP 잠재 공간 차원 (CLAP와 일치해야 함)
        
    Returns:
        이중 경로 EEG 인코더 모델
    """
    model = DualPathEEGEncoder(
        n_channels=n_channels,
        n_times=n_times,
        vae_latent_dim=vae_latent_dim,
        clap_latent_dim=clap_latent_dim,
        hidden_dims=[64, 128, 256, 512]
    )
    
    return model


if __name__ == "__main__":
    # 테스트 코드
    # 모델 생성
    model = create_eeg_encoder(n_channels=128, n_times=2000)
    print(model)
    
    # 입력 텐서 생성
    batch_size = 4
    n_channels = 128
    n_times = 2000
    x = torch.randn(batch_size, n_channels, n_times)
    
    # 순전파
    outputs = model(x)
    
    # 출력 확인
    print("VAE 잠재 벡터 크기:", outputs['vae_z'].shape)
    print("CLAP 잠재 벡터 크기:", outputs['clap_z'].shape)
