import os
import torch
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import soundfile as sf
import librosa
import time
from tqdm import tqdm

# 프로젝트 모듈 임포트
from data_preprocessing import EEGProcessor, AudioProcessor, EEGAudioDataset
from eeg_encoder import DualPathEEGEncoder, create_eeg_encoder
from latent_mapping import DualPathLatentMapping, create_latent_mapping_model
from pretrained_models import PretrainedModelManager, PretrainedModelIntegrator, create_pretrained_model_manager
from eeg2speech_trainer import EEG2SpeechTrainer, create_eeg2speech_trainer

class EEG2SpeechPipeline:
    """
    EEG에서 음성으로 변환하는 end-to-end 파이프라인
    """
    def __init__(self, config_path: str = None, device: torch.device = None):
        """
        초기화 함수
        
        Args:
            config_path: 설정 파일 경로
            device: 모델 실행 장치 (CPU 또는 GPU)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 기본 설정
        self.config = {
            'eeg': {
                'n_channels': 128,
                'n_times': 2000,
                'vae_latent_dim': 512,
                'clap_latent_dim': 512,
                'hidden_dims': [64, 128, 256, 512]
            },
            'mapping': {
                'mapping_type': 'mlp',
                'hidden_dims': [1024, 1024, 512],
                'dropout': 0.2
            },
            'training': {
                'batch_size': 32,
                'val_split': 0.2,
                'test_split': 0.1,
                'random_seed': 42,
                'encoder_epochs': 100,
                'mapping_epochs': 100,
                'encoder_lr': 1e-4,
                'mapping_lr': 1e-4,
                'patience': 10
            },
            'generation': {
                'prompt': 'speech, clear voice',
                'num_inference_steps': 50,
                'audio_length_in_s': 5.0
            },
            'paths': {
                'data_dir': './data',
                'output_dir': './results',
                'model_dir': './models',
                'pretrained_cache_dir': './pretrained_models'
            }
        }
        
        # 설정 파일 로드
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # 설정 병합
                self._update_config(self.config, user_config)
        
        # 디렉토리 생성
        for dir_path in self.config['paths'].values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 모델 초기화
        self.eeg_encoder = None
        self.latent_mapping = None
        self.pretrained_manager = None
        self.model_integrator = None
        self.trainer = None
        
        print(f"EEG2Speech 파이프라인 초기화 완료. 장치: {self.device}")
    
    def _update_config(self, config: Dict, update: Dict) -> Dict:
        """
        설정 업데이트
        
        Args:
            config: 기존 설정 딕셔너리
            update: 업데이트할 설정 딕셔너리
            
        Returns:
            업데이트된 설정 딕셔너리
        """
        for key, value in update.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
        return config
    
    def initialize_models(self, load_pretrained: bool = True) -> None:
        """
        모델 초기화
        
        Args:
            load_pretrained: 사전 학습된 모델 로드 여부
        """
        print("모델 초기화 중...")
        
        # EEG 인코더 생성
        self.eeg_encoder = create_eeg_encoder(
            n_channels=self.config['eeg']['n_channels'],
            n_times=self.config['eeg']['n_times'],
            vae_latent_dim=self.config['eeg']['vae_latent_dim'],
            clap_latent_dim=self.config['eeg']['clap_latent_dim']
        ).to(self.device)
        
        # 잠재 벡터 매핑 모델 생성
        self.latent_mapping = create_latent_mapping_model(
            vae_dim=self.config['eeg']['vae_latent_dim'],
            clap_dim=self.config['eeg']['clap_latent_dim'],
            mapping_type=self.config['mapping']['mapping_type']
        ).to(self.device)
        
        # 사전 학습된 모델 관리자 생성
        self.pretrained_manager = create_pretrained_model_manager(
            device=self.device,
            cache_dir=self.config['paths']['pretrained_cache_dir']
        )
        
        # 사전 학습된 모델 로드
        if load_pretrained:
            self.pretrained_manager.load_clap_model()
            self.pretrained_manager.load_whisper_model()
            self.pretrained_manager.load_audioldm2_model()
        
        # 모델 통합기 생성
        self.model_integrator = PretrainedModelIntegrator(
            self.eeg_encoder,
            self.latent_mapping,
            self.pretrained_manager,
            self.device
        )
        
        # 트레이너 생성
        self.trainer = create_eeg2speech_trainer(
            self.eeg_encoder,
            self.latent_mapping,
            self.pretrained_manager,
            self.device,
            self.config['paths']['output_dir']
        )
        
        print("모델 초기화 완료")
    
    def load_models(self, encoder_path: str, mapping_path: str) -> None:
        """
        학습된 모델 로드
        
        Args:
            encoder_path: EEG 인코더 체크포인트 경로
            mapping_path: 잠재 벡터 매핑 모델 체크포인트 경로
        """
        print("학습된 모델 로드 중...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None:
            self.initialize_models(load_pretrained=True)
        
        # EEG 인코더 로드
        if os.path.exists(encoder_path):
            checkpoint = torch.load(encoder_path, map_location=self.device)
            self.eeg_encoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"EEG 인코더 로드 완료: {encoder_path}")
        else:
            print(f"경고: EEG 인코더 체크포인트를 찾을 수 없습니다: {encoder_path}")
        
        # 잠재 벡터 매핑 모델 로드
        if os.path.exists(mapping_path):
            checkpoint = torch.load(mapping_path, map_location=self.device)
            self.latent_mapping.load_state_dict(checkpoint['model_state_dict'])
            print(f"잠재 벡터 매핑 모델 로드 완료: {mapping_path}")
        else:
            print(f"경고: 잠재 벡터 매핑 모델 체크포인트를 찾을 수 없습니다: {mapping_path}")
        
        # 평가 모드로 설정
        self.eeg_encoder.eval()
        self.latent_mapping.eval()
        
        # 모델 통합기 업데이트
        self.model_integrator = PretrainedModelIntegrator(
            self.eeg_encoder,
            self.latent_mapping,
            self.pretrained_manager,
            self.device
        )
    
    def prepare_data(self, eeg_data_dir: str = None, audio_data_dir: str = None) -> Tuple[EEGAudioDataset, List[str]]:
        """
        데이터 준비
        
        Args:
            eeg_data_dir: EEG 데이터 디렉토리
            audio_data_dir: 오디오 데이터 디렉토리
            
        Returns:
            (EEG-오디오 데이터셋, 오디오 파일 경로 목록) 튜플
        """
        print("데이터 준비 중...")
        
        # 기본 디렉토리 사용
        if eeg_data_dir is None:
            eeg_data_dir = os.path.join(self.config['paths']['data_dir'], 'eeg')
        if audio_data_dir is None:
            audio_data_dir = os.path.join(self.config['paths']['data_dir'], 'audio')
        
        # 디렉토리 생성
        os.makedirs(eeg_data_dir, exist_ok=True)
        os.makedirs(audio_data_dir, exist_ok=True)
        
        # EEG 프로세서 생성
        eeg_processor = EEGProcessor(data_dir=eeg_data_dir)
        
        # 오디오 프로세서 생성
        audio_processor = AudioProcessor(data_dir=audio_data_dir)
        
        # EEG 데이터 로드
        eeg_data = eeg_processor.load_data()
        
        # 오디오 파일 로드
        audio_files = audio_processor.get_audio_files()
        
        # 데이터셋 생성
        dataset = EEGAudioDataset(
            eeg_data=eeg_data,
            audio_files=audio_files,
            eeg_processor=eeg_processor,
            audio_processor=audio_processor
        )
        
        print(f"데이터 준비 완료. EEG 샘플 수: {len(dataset)}, 오디오 파일 수: {len(audio_files)}")
        return dataset, audio_files
    
    def train(self, eeg_dataset: EEGAudioDataset = None, audio_files: List[str] = None) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            eeg_dataset: EEG 데이터셋
            audio_files: 오디오 파일 경로 목록
            
        Returns:
            학습 결과 딕셔너리
        """
        print("모델 학습 시작...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None or self.trainer is None:
            self.initialize_models(load_pretrained=True)
        
        # 데이터셋이 제공되지 않은 경우 데이터 준비
        if eeg_dataset is None or audio_files is None:
            eeg_dataset, audio_files = self.prepare_data()
        
        # 학습 설정
        training_config = self.config['training']
        
        # 학습 실행
        results = self.trainer.train_pipeline(
            eeg_dataset=eeg_dataset,
            audio_files=audio_files,
            batch_size=training_config['batch_size'],
            val_split=training_config['val_split'],
            test_split=training_config['test_split'],
            random_seed=training_config['random_seed'],
            encoder_epochs=training_config['encoder_epochs'],
            mapping_epochs=training_config['mapping_epochs'],
            encoder_lr=training_config['encoder_lr'],
            mapping_lr=training_config['mapping_lr'],
            patience=training_config['patience'],
            mapping_type=self.config['mapping']['mapping_type']
        )
        
        # 모델 저장
        self.save_models()
        
        print("모델 학습 완료")
        return results
    
    def save_models(self) -> Tuple[str, str]:
        """
        모델 저장
        
        Returns:
            (EEG 인코더 저장 경로, 잠재 벡터 매핑 모델 저장 경로) 튜플
        """
        print("모델 저장 중...")
        
        # 모델 디렉토리 생성
        model_dir = self.config['paths']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        # 타임스탬프
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # EEG 인코더 저장
        encoder_path = os.path.join(model_dir, f"eeg_encoder_{timestamp}.pt")
        torch.save({
            'model_state_dict': self.eeg_encoder.state_dict(),
            'config': self.config['eeg']
        }, encoder_path)
        
        # 잠재 벡터 매핑 모델 저장
        mapping_path = os.path.join(model_dir, f"latent_mapping_{timestamp}.pt")
        torch.save({
            'model_state_dict': self.latent_mapping.state_dict(),
            'config': self.config['mapping']
        }, mapping_path)
        
        print(f"모델 저장 완료. EEG 인코더: {encoder_path}, 잠재 벡터 매핑 모델: {mapping_path}")
        return encoder_path, mapping_path
    
    def evaluate(self, test_dataset: EEGAudioDataset = None, audio_files: List[str] = None,
                num_samples: int = 10) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            test_dataset: 테스트 데이터셋
            audio_files: 테스트 오디오 파일 경로 목록
            num_samples: 생성할 샘플 수
            
        Returns:
            평가 결과 딕셔너리
        """
        print("모델 평가 시작...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None or self.trainer is None:
            self.initialize_models(load_pretrained=True)
        
        # 데이터셋이 제공되지 않은 경우 데이터 준비
        if test_dataset is None or audio_files is None:
            test_dataset, audio_files = self.prepare_data()
        
        # 평가 디렉토리 생성
        eval_dir = os.path.join(self.config['paths']['output_dir'], "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 평가 실행
        results = self.trainer.evaluate(
            test_dataset=test_dataset,
            audio_files=audio_files,
            output_dir=eval_dir,
            batch_size=4,
            num_samples=num_samples
        )
        
        print("모델 평가 완료")
        return results
    
    def generate_speech_from_eeg(self, eeg_data: torch.Tensor, output_dir: str = None,
                               prompt: str = None, visualize: bool = True) -> Dict[str, str]:
        """
        EEG 데이터에서 음성 생성
        
        Args:
            eeg_data: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            output_dir: 출력 디렉토리
            prompt: 텍스트 프롬프트
            visualize: 시각화 여부
            
        Returns:
            결과 딕셔너리 {'audio_path': audio_path, 'transcription': transcription}
        """
        print("EEG 데이터에서 음성 생성 중...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None or self.model_integrator is None:
            self.initialize_models(load_pretrained=True)
        
        # 기본 출력 디렉토리 사용
        if output_dir is None:
            output_dir = os.path.join(self.config['paths']['output_dir'], "generated")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 기본 프롬프트 사용
        if prompt is None:
            prompt = self.config['generation']['prompt']
        
        # 오디오 생성 및 텍스트 변환
        result = self.model_integrator.process_and_transcribe(
            eeg_data=eeg_data,
            output_dir=output_dir,
            prompt=prompt,
            num_inference_steps=self.config['generation']['num_inference_steps'],
            audio_length_in_s=self.config['generation']['audio_length_in_s']
        )
        
        # 오디오 시각화
        if visualize and result['audio_path'] is not None:
            self.pretrained_manager.visualize_audio(
                result['audio_path'],
                title="Generated Audio from EEG"
            )
        
        print(f"음성 생성 완료. 오디오 경로: {result['audio_path']}")
        print(f"텍스트 변환 결과: {result['transcription']}")
        
        return result
    
    def batch_generate_speech(self, eeg_dataset: EEGAudioDataset, output_dir: str = None,
                            prompt: str = None, batch_size: int = 4,
                            num_samples: int = 10) -> List[Dict[str, str]]:
        """
        배치 모드로 음성 생성
        
        Args:
            eeg_dataset: EEG 데이터셋
            output_dir: 출력 디렉토리
            prompt: 텍스트 프롬프트
            batch_size: 배치 크기
            num_samples: 생성할 샘플 수
            
        Returns:
            결과 딕셔너리 목록
        """
        print("배치 모드로 음성 생성 중...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None or self.model_integrator is None:
            self.initialize_models(load_pretrained=True)
        
        # 기본 출력 디렉토리 사용
        if output_dir is None:
            output_dir = os.path.join(self.config['paths']['output_dir'], "batch_generated")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 기본 프롬프트 사용
        if prompt is None:
            prompt = self.config['generation']['prompt']
        
        # 데이터 로더 생성
        data_loader = torch.utils.data.DataLoader(
            eeg_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        results = []
        
        for i, batch in enumerate(tqdm(data_loader, desc="배치 처리")):
            if i >= (num_samples + batch_size - 1) // batch_size:
                break
            
            # EEG 데이터 추출
            eeg_data = batch['eeg']
            
            # 배치 내 각 샘플 처리
            for j in range(len(eeg_data)):
                if i * batch_size + j >= num_samples:
                    break
                
                # 출력 파일 경로
                sample_dir = os.path.join(output_dir, f"sample_{i * batch_size + j}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # 단일 샘플 처리
                result = self.model_integrator.process_and_transcribe(
                    eeg_data[j:j+1],
                    sample_dir,
                    prompt,
                    self.config['generation']['num_inference_steps'],
                    self.config['generation']['audio_length_in_s']
                )
                
                # 결과 저장
                results.append(result)
        
        # 결과 저장
        with open(os.path.join(output_dir, "batch_results.json"), 'w') as f:
            json.dump({
                'transcriptions': [r['transcription'] for r in results],
                'audio_paths': [r['audio_path'] for r in results]
            }, f, indent=4)
        
        print(f"배치 모드 음성 생성 완료. 생성된 샘플 수: {len(results)}")
        return results
    
    def demo(self, eeg_file: str = None, output_dir: str = None) -> Dict[str, str]:
        """
        데모 실행
        
        Args:
            eeg_file: EEG 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            결과 딕셔너리 {'audio_path': audio_path, 'transcription': transcription}
        """
        print("데모 실행 중...")
        
        # 모델이 초기화되지 않은 경우 초기화
        if self.eeg_encoder is None or self.latent_mapping is None or self.model_integrator is None:
            self.initialize_models(load_pretrained=True)
        
        # 기본 출력 디렉토리 사용
        if output_dir is None:
            output_dir = os.path.join(self.config['paths']['output_dir'], "demo")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # EEG 데이터 로드
        if eeg_file is not None and os.path.exists(eeg_file):
            # 파일에서 EEG 데이터 로드
            eeg_processor = EEGProcessor()
            eeg_data = eeg_processor.load_eeg_file(eeg_file)
        else:
            # 예시 EEG 데이터 생성
            print("경고: EEG 파일을 찾을 수 없습니다. 예시 EEG 데이터를 생성합니다.")
            eeg_data = torch.randn(1, self.config['eeg']['n_channels'], self.config['eeg']['n_times'])
        
        # 음성 생성
        result = self.generate_speech_from_eeg(
            eeg_data=eeg_data,
            output_dir=output_dir,
            prompt=self.config['generation']['prompt'],
            visualize=True
        )
        
        print("데모 실행 완료")
        return result


def main():
    """
    메인 함수
    """
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='EEG2Speech 파이프라인')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로')
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'evaluate', 'demo'], help='실행 모드')
    parser.add_argument('--encoder', type=str, default=None, help='EEG 인코더 체크포인트 경로')
    parser.add_argument('--mapping', type=str, default=None, help='잠재 벡터 매핑 모델 체크포인트 경로')
    parser.add_argument('--eeg_file', type=str, default=None, help='EEG 파일 경로 (데모 모드)')
    parser.add_argument('--output_dir', type=str, default=None, help='출력 디렉토리')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 인덱스')
    args = parser.parse_args()
    
    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 파이프라인 생성
    pipeline = EEG2SpeechPipeline(config_path=args.config, device=device)
    
    # 모델 초기화
    pipeline.initialize_models()
    
    # 학습된 모델 로드
    if args.encoder is not None and args.mapping is not None:
        pipeline.load_models(args.encoder, args.mapping)
    
    # 모드에 따른 실행
    if args.mode == 'train':
        # 데이터 준비
        eeg_dataset, audio_files = pipeline.prepare_data()
        
        # 학습
        pipeline.train(eeg_dataset, audio_files)
    
    elif args.mode == 'evaluate':
        # 데이터 준비
        eeg_dataset, audio_files = pipeline.prepare_data()
        
        # 평가
        pipeline.evaluate(eeg_dataset, audio_files)
    
    elif args.mode == 'demo':
        # 데모 실행
        pipeline.demo(args.eeg_file, args.output_dir)


if __name__ == "__main__":
    main()
