import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import json
import time
import datetime
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, random_split

# 프로젝트 모듈 임포트
from eeg_encoder import DualPathEEGEncoder, EEGEncoderTrainer
from latent_mapping import DualPathLatentMapping, LatentVectorMappingTrainer, LatentVectorDataset
from pretrained_models import PretrainedModelManager, PretrainedModelIntegrator

class EEG2SpeechTrainer:
    """
    EEG에서 음성으로 변환하는 전체 파이프라인 학습 및 평가를 위한 클래스
    """
    def __init__(self, eeg_encoder: nn.Module, latent_mapping: nn.Module, 
                pretrained_manager: PretrainedModelManager,
                device: torch.device = None,
                output_dir: str = "./results"):
        """
        초기화 함수
        
        Args:
            eeg_encoder: EEG 인코더 모델
            latent_mapping: 잠재 벡터 매핑 모델
            pretrained_manager: 사전 학습된 모델 관리자
            device: 모델 실행 장치 (CPU 또는 GPU)
            output_dir: 결과 저장 디렉토리
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eeg_encoder = eeg_encoder.to(self.device)
        self.latent_mapping = latent_mapping.to(self.device)
        self.pretrained_manager = pretrained_manager
        
        # 모델 통합기 생성
        self.model_integrator = PretrainedModelIntegrator(
            eeg_encoder, 
            latent_mapping, 
            pretrained_manager, 
            device
        )
        
        # 결과 저장 디렉토리 생성
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 학습 히스토리
        self.history = {
            'encoder_train_loss': [],
            'encoder_val_loss': [],
            'mapping_train_loss': [],
            'mapping_val_loss': [],
            'mapping_val_metrics': {
                'vae_mse': [],
                'clap_mse': [],
                'vae_r2': [],
                'clap_r2': [],
                'vae_cosine': [],
                'clap_cosine': []
            }
        }
    
    def prepare_data(self, eeg_dataset: Dataset, audio_files: List[str], 
                    batch_size: int = 32, val_split: float = 0.2, 
                    test_split: float = 0.1, random_seed: int = 42) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
        """
        데이터 준비
        
        Args:
            eeg_dataset: EEG 데이터셋
            audio_files: 오디오 파일 경로 목록
            batch_size: 배치 크기
            val_split: 검증 세트 비율
            test_split: 테스트 세트 비율
            random_seed: 랜덤 시드
            
        Returns:
            (데이터 로더 딕셔너리, 임베딩 딕셔너리) 튜플
        """
        print("데이터 준비 중...")
        
        # 랜덤 시드 설정
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 데이터셋 분할
        dataset_size = len(eeg_dataset)
        test_size = int(dataset_size * test_split)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            eeg_dataset, 
            [train_size, val_size, test_size]
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 오디오 임베딩 추출
        print("오디오 임베딩 추출 중...")
        
        # 오디오 파일 분할
        train_audio_files = audio_files[:train_size]
        val_audio_files = audio_files[train_size:train_size+val_size]
        test_audio_files = audio_files[train_size+val_size:train_size+val_size+test_size]
        
        # CLAP 임베딩 추출
        train_clap_embeds = self.pretrained_manager.extract_clap_embeddings(train_audio_files)
        val_clap_embeds = self.pretrained_manager.extract_clap_embeddings(val_audio_files)
        test_clap_embeds = self.pretrained_manager.extract_clap_embeddings(test_audio_files)
        
        # Whisper 임베딩 추출
        train_whisper_embeds = self.pretrained_manager.extract_whisper_embeddings(train_audio_files)
        val_whisper_embeds = self.pretrained_manager.extract_whisper_embeddings(val_audio_files)
        test_whisper_embeds = self.pretrained_manager.extract_whisper_embeddings(test_audio_files)
        
        # AudioLDM2 잠재 벡터 추출
        train_audioldm_latents = self.pretrained_manager.extract_audioldm2_latents(train_audio_files)
        val_audioldm_latents = self.pretrained_manager.extract_audioldm2_latents(val_audio_files)
        test_audioldm_latents = self.pretrained_manager.extract_audioldm2_latents(test_audio_files)
        
        # 데이터 로더 딕셔너리
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # 임베딩 딕셔너리
        embeddings = {
            'train': {
                'clap': train_clap_embeds,
                'whisper': train_whisper_embeds,
                'audioldm': train_audioldm_latents
            },
            'val': {
                'clap': val_clap_embeds,
                'whisper': val_whisper_embeds,
                'audioldm': val_audioldm_latents
            },
            'test': {
                'clap': test_clap_embeds,
                'whisper': test_whisper_embeds,
                'audioldm': test_audioldm_latents
            }
        }
        
        print("데이터 준비 완료")
        return data_loaders, embeddings
    
    def train_eeg_encoder(self, train_loader: DataLoader, val_loader: DataLoader, 
                         audio_embeds: Dict[str, np.ndarray],
                         n_epochs: int = 100, lr: float = 1e-4, 
                         patience: int = 10) -> EEGEncoderTrainer:
        """
        EEG 인코더 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            audio_embeds: 오디오 임베딩 딕셔너리
            n_epochs: 에폭 수
            lr: 학습률
            patience: 조기 종료 인내심
            
        Returns:
            EEG 인코더 트레이너
        """
        print("EEG 인코더 학습 중...")
        
        # 체크포인트 디렉토리 생성
        checkpoint_dir = os.path.join(self.output_dir, "encoder_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 트레이너 생성
        encoder_trainer = EEGEncoderTrainer(self.eeg_encoder, self.device)
        
        # 옵티마이저 생성
        optimizer = optim.Adam(self.eeg_encoder.parameters(), lr=lr)
        
        # 학습 데이터셋 준비
        train_clap_embeds = torch.tensor(audio_embeds['train']['clap'], dtype=torch.float32)
        val_clap_embeds = torch.tensor(audio_embeds['val']['clap'], dtype=torch.float32)
        
        # 학습
        encoder_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            vae_decoder=None,  # VAE 디코더 없이 학습
            contrastive_weight=1.0,
            vae_weight=0.1,
            patience=patience,
            checkpoint_dir=checkpoint_dir
        )
        
        # 학습 히스토리 저장
        self.history['encoder_train_loss'] = encoder_trainer.train_loss_history
        self.history['encoder_val_loss'] = encoder_trainer.val_loss_history
        
        print("EEG 인코더 학습 완료")
        return encoder_trainer
    
    def extract_eeg_latents(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        EEG 잠재 벡터 추출
        
        Args:
            data_loader: 데이터 로더
            
        Returns:
            EEG 잠재 벡터 딕셔너리
        """
        print("EEG 잠재 벡터 추출 중...")
        
        self.eeg_encoder.eval()
        
        eeg_vae_latents = []
        eeg_clap_latents = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="EEG 잠재 벡터 추출"):
                # EEG 데이터 추출
                eeg_data = batch['eeg'].to(self.device)
                
                # 순전파
                outputs = self.eeg_encoder(eeg_data)
                
                # 잠재 벡터 저장
                eeg_vae_latents.append(outputs['vae_z'].cpu().numpy())
                eeg_clap_latents.append(outputs['clap_z'].cpu().numpy())
        
        # 배열 연결
        eeg_vae_latents = np.concatenate(eeg_vae_latents)
        eeg_clap_latents = np.concatenate(eeg_clap_latents)
        
        return {
            'vae': eeg_vae_latents,
            'clap': eeg_clap_latents
        }
    
    def prepare_latent_datasets(self, data_loaders: Dict[str, DataLoader], 
                               audio_embeds: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, LatentVectorDataset]:
        """
        잠재 벡터 데이터셋 준비
        
        Args:
            data_loaders: 데이터 로더 딕셔너리
            audio_embeds: 오디오 임베딩 딕셔너리
            
        Returns:
            잠재 벡터 데이터셋 딕셔너리
        """
        print("잠재 벡터 데이터셋 준비 중...")
        
        latent_datasets = {}
        
        for split in ['train', 'val', 'test']:
            # EEG 잠재 벡터 추출
            eeg_latents = self.extract_eeg_latents(data_loaders[split])
            
            # 데이터셋 생성
            latent_datasets[split] = LatentVectorDataset(
                eeg_vae_latents=torch.tensor(eeg_latents['vae'], dtype=torch.float32),
                eeg_clap_latents=torch.tensor(eeg_latents['clap'], dtype=torch.float32),
                audio_vae_latents=torch.tensor(audio_embeds[split]['audioldm'], dtype=torch.float32),
                audio_clap_latents=torch.tensor(audio_embeds[split]['clap'], dtype=torch.float32)
            )
        
        print("잠재 벡터 데이터셋 준비 완료")
        return latent_datasets
    
    def train_latent_mapping(self, latent_datasets: Dict[str, LatentVectorDataset],
                            batch_size: int = 32, n_epochs: int = 100, 
                            lr: float = 1e-4, patience: int = 10,
                            mapping_type: str = 'mlp') -> LatentVectorMappingTrainer:
        """
        잠재 벡터 매핑 모델 학습
        
        Args:
            latent_datasets: 잠재 벡터 데이터셋 딕셔너리
            batch_size: 배치 크기
            n_epochs: 에폭 수
            lr: 학습률
            patience: 조기 종료 인내심
            mapping_type: 매핑 모델 유형 ('mlp', 'resnet', 'adversarial')
            
        Returns:
            잠재 벡터 매핑 트레이너
        """
        print(f"잠재 벡터 매핑 모델 학습 중 (유형: {mapping_type})...")
        
        # 체크포인트 디렉토리 생성
        checkpoint_dir = os.path.join(self.output_dir, "mapping_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 데이터 로더 생성
        train_loader = DataLoader(latent_datasets['train'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(latent_datasets['val'], batch_size=batch_size, shuffle=False)
        
        # 트레이너 생성
        mapping_trainer = LatentVectorMappingTrainer(self.latent_mapping, self.device)
        
        # 옵티마이저 생성
        optimizer = optim.Adam(self.latent_mapping.parameters(), lr=lr)
        
        # 적대적 모델인 경우 판별자 옵티마이저 생성
        disc_optimizer = None
        if mapping_type == 'adversarial':
            # VAE 매핑 판별자 파라미터
            vae_disc_params = list(self.latent_mapping.vae_mapping.discriminator.parameters())
            # CLAP 매핑 판별자 파라미터
            clap_disc_params = list(self.latent_mapping.clap_mapping.discriminator.parameters())
            # 판별자 옵티마이저
            disc_optimizer = optim.Adam(vae_disc_params + clap_disc_params, lr=lr)
        
        # 학습
        mapping_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            disc_optimizer=disc_optimizer,
            gan_weight=0.1,
            patience=patience,
            checkpoint_dir=checkpoint_dir
        )
        
        # 학습 히스토리 저장
        self.history['mapping_train_loss'] = mapping_trainer.train_loss_history
        self.history['mapping_val_loss'] = mapping_trainer.val_loss_history
        
        # 평가 지표 히스토리 저장
        for metric in ['vae_mse', 'clap_mse', 'vae_r2', 'clap_r2', 'vae_cosine', 'clap_cosine']:
            self.history['mapping_val_metrics'][metric] = mapping_trainer.val_metrics_history[metric]
        
        print("잠재 벡터 매핑 모델 학습 완료")
        return mapping_trainer
    
    def train_pipeline(self, eeg_dataset: Dataset, audio_files: List[str],
                      batch_size: int = 32, val_split: float = 0.2, 
                      test_split: float = 0.1, random_seed: int = 42,
                      encoder_epochs: int = 100, mapping_epochs: int = 100,
                      encoder_lr: float = 1e-4, mapping_lr: float = 1e-4,
                      patience: int = 10, mapping_type: str = 'mlp') -> Dict[str, Any]:
        """
        전체 파이프라인 학습
        
        Args:
            eeg_dataset: EEG 데이터셋
            audio_files: 오디오 파일 경로 목록
            batch_size: 배치 크기
            val_split: 검증 세트 비율
            test_split: 테스트 세트 비율
            random_seed: 랜덤 시드
            encoder_epochs: EEG 인코더 에폭 수
            mapping_epochs: 잠재 벡터 매핑 모델 에폭 수
            encoder_lr: EEG 인코더 학습률
            mapping_lr: 잠재 벡터 매핑 모델 학습률
            patience: 조기 종료 인내심
            mapping_type: 매핑 모델 유형 ('mlp', 'resnet', 'adversarial')
            
        Returns:
            학습 결과 딕셔너리
        """
        # 학습 시작 시간
        start_time = time.time()
        
        # 데이터 준비
        data_loaders, audio_embeds = self.prepare_data(
            eeg_dataset, 
            audio_files, 
            batch_size, 
            val_split, 
            test_split, 
            random_seed
        )
        
        # EEG 인코더 학습
        encoder_trainer = self.train_eeg_encoder(
            data_loaders['train'],
            data_loaders['val'],
            audio_embeds,
            encoder_epochs,
            encoder_lr,
            patience
        )
        
        # 잠재 벡터 데이터셋 준비
        latent_datasets = self.prepare_latent_datasets(data_loaders, audio_embeds)
        
        # 잠재 벡터 매핑 모델 학습
        mapping_trainer = self.train_latent_mapping(
            latent_datasets,
            batch_size,
            mapping_epochs,
            mapping_lr,
            patience,
            mapping_type
        )
        
        # 학습 종료 시간
        end_time = time.time()
        training_time = end_time - start_time
        
        # 학습 결과 저장
        results = {
            'training_time': training_time,
            'encoder_final_loss': encoder_trainer.val_loss_history[-1],
            'mapping_final_loss': mapping_trainer.val_loss_history[-1],
            'mapping_final_metrics': {
                metric: mapping_trainer.val_metrics_history[metric][-1]
                for metric in ['vae_mse', 'clap_mse', 'vae_r2', 'clap_r2', 'vae_cosine', 'clap_cosine']
            }
        }
        
        # 결과 저장
        self.save_results(results)
        
        return results
    
    def evaluate(self, test_dataset: Dataset, audio_files: List[str],
                output_dir: str = None, batch_size: int = 4,
                num_samples: int = 10) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            test_dataset: 테스트 데이터셋
            audio_files: 테스트 오디오 파일 경로 목록
            output_dir: 출력 디렉토리
            batch_size: 배치 크기
            num_samples: 생성할 샘플 수
            
        Returns:
            평가 결과 딕셔너리
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "evaluation")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("모델 평가 중...")
        
        # 테스트 데이터 로더 생성
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 평가 결과
        results = {
            'audio_paths': [],
            'transcriptions': [],
            'metrics': {}
        }
        
        # 샘플 생성
        for i, batch in enumerate(tqdm(test_loader, desc="샘플 생성")):
            if i >= num_samples:
                break
            
            # EEG 데이터 추출
            eeg_data = batch['eeg']
            
            # 샘플 디렉토리 생성
            sample_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 오디오 생성 및 텍스트 변환
            result = self.model_integrator.process_and_transcribe(
                eeg_data,
                sample_dir,
                prompt="speech, clear voice",
                num_inference_steps=50,
                audio_length_in_s=5.0
            )
            
            # 결과 저장
            results['audio_paths'].append(result['audio_path'])
            results['transcriptions'].append(result['transcription'])
            
            # 오디오 시각화
            if result['audio_path'] is not None:
                self.pretrained_manager.visualize_audio(
                    result['audio_path'],
                    title=f"Sample {i} - Generated Audio"
                )
        
        # 평가 지표 계산
        # 여기서는 간단한 예시로 텍스트 변환 성공률만 계산
        success_rate = sum(1 for path in results['audio_paths'] if path is not None) / len(results['audio_paths'])
        results['metrics']['success_rate'] = success_rate
        
        # 결과 저장
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump({
                'transcriptions': results['transcriptions'],
                'metrics': results['metrics']
            }, f, indent=4)
        
        print(f"모델 평가 완료. 성공률: {success_rate:.2f}")
        return results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        학습 결과 저장
        
        Args:
            results: 학습 결과 딕셔너리
        """
        # 결과 디렉토리 생성
        results_dir = os.path.join(self.output_dir, "training_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 학습 히스토리 저장
        with open(os.path.join(results_dir, "training_history.json"), 'w') as f:
            # NumPy 배열을 리스트로 변환
            history_json = {
                'encoder_train_loss': self.history['encoder_train_loss'],
                'encoder_val_loss': self.history['encoder_val_loss'],
                'mapping_train_loss': self.history['mapping_train_loss'],
                'mapping_val_loss': self.history['mapping_val_loss'],
                'mapping_val_metrics': self.history['mapping_val_metrics']
            }
            json.dump(history_json, f, indent=4)
        
        # 학습 결과 저장
        with open(os.path.join(results_dir, "training_results.json"), 'w') as f:
            # 학습 시간을 읽기 쉬운 형식으로 변환
            if 'training_time' in results:
                training_time = results['training_time']
                results['training_time_formatted'] = str(datetime.timedelta(seconds=int(training_time)))
            
            json.dump(results, f, indent=4)
        
        # 학습 곡선 시각화
        self.visualize_training_curves(os.path.join(results_dir, "training_curves.png"))
        
        print(f"학습 결과가 {results_dir}에 저장되었습니다.")
    
    def visualize_training_curves(self, output_path: str = None) -> None:
        """
        학습 곡선 시각화
        
        Args:
            output_path: 출력 이미지 파일 경로
        """
        plt.figure(figsize=(15, 10))
        
        # EEG 인코더 손실
        plt.subplot(2, 2, 1)
        plt.plot(self.history['encoder_train_loss'], label='Train Loss')
        plt.plot(self.history['encoder_val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('EEG Encoder Loss')
        plt.legend()
        plt.grid(True)
        
        # 잠재 벡터 매핑 손실
        plt.subplot(2, 2, 2)
        plt.plot(self.history['mapping_train_loss'], label='Train Loss')
        plt.plot(self.history['mapping_val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Latent Vector Mapping Loss')
        plt.legend()
        plt.grid(True)
        
        # R² 점수
        plt.subplot(2, 2, 3)
        plt.plot(self.history['mapping_val_metrics']['vae_r2'], label='VAE R²')
        plt.plot(self.history['mapping_val_metrics']['clap_r2'], label='CLAP R²')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('R² Score')
        plt.legend()
        plt.grid(True)
        
        # 코사인 유사도
        plt.subplot(2, 2, 4)
        plt.plot(self.history['mapping_val_metrics']['vae_cosine'], label='VAE Cosine')
        plt.plot(self.history['mapping_val_metrics']['clap_cosine'], label='CLAP Cosine')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_path is not None:
            plt.savefig(output_path)
        
        plt.show()


def create_eeg2speech_trainer(eeg_encoder: nn.Module, latent_mapping: nn.Module, 
                             pretrained_manager: PretrainedModelManager,
                             device: torch.device = None,
                             output_dir: str = "./results") -> EEG2SpeechTrainer:
    """
    EEG2Speech 트레이너 생성
    
    Args:
        eeg_encoder: EEG 인코더 모델
        latent_mapping: 잠재 벡터 매핑 모델
        pretrained_manager: 사전 학습된 모델 관리자
        device: 모델 실행 장치 (CPU 또는 GPU)
        output_dir: 결과 저장 디렉토리
        
    Returns:
        EEG2Speech 트레이너
    """
    trainer = EEG2SpeechTrainer(eeg_encoder, latent_mapping, pretrained_manager, device, output_dir)
    return trainer


if __name__ == "__main__":
    # 테스트 코드
    from data_preprocessing import EEGAudioDataset
    from eeg_encoder import create_eeg_encoder
    from latent_mapping import create_latent_mapping_model
    from pretrained_models import create_pretrained_model_manager
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성
    eeg_encoder = create_eeg_encoder(n_channels=128, n_times=2000)
    latent_mapping = create_latent_mapping_model(vae_dim=512, clap_dim=512, mapping_type='mlp')
    pretrained_manager = create_pretrained_model_manager(device)
    
    # 사전 학습된 모델 로드
    pretrained_manager.load_clap_model()
    pretrained_manager.load_whisper_model()
    pretrained_manager.load_audioldm2_model()
    
    # 트레이너 생성
    trainer = create_eeg2speech_trainer(eeg_encoder, latent_mapping, pretrained_manager, device)
    
    # 데이터셋 생성 (예시)
    # eeg_dataset = EEGAudioDataset(...)
    # audio_files = [...]
    
    # 학습
    # results = trainer.train_pipeline(eeg_dataset, audio_files)
    
    # 평가
    # eval_results = trainer.evaluate(eeg_dataset, audio_files)
    
    print("테스트 완료")
