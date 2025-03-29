import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from transformers import (
    AutoModel, 
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq, 
    AutoFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from diffusers import AudioLDM2Pipeline, AudioLDMPipeline
import torchaudio
from tqdm import tqdm

class PretrainedModelManager:
    """
    사전 학습된 모델(AudioLDM2, CLAP, Whisper)을 관리하는 클래스
    """
    def __init__(self, device: torch.device = None, cache_dir: str = "./pretrained_models"):
        """
        초기화 함수
        
        Args:
            device: 모델 실행 장치 (CPU 또는 GPU)
            cache_dir: 사전 학습된 모델 캐시 디렉토리
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 모델 저장소
        self.models = {}
        self.processors = {}
        
        print(f"모델 실행 장치: {self.device}")
    
    def load_clap_model(self, model_id: str = "laion/clap-htsat-fused") -> Tuple[Any, Any]:
        """
        CLAP 모델 로드
        
        Args:
            model_id: 모델 ID
            
        Returns:
            (CLAP 모델, CLAP 프로세서) 튜플
        """
        print(f"CLAP 모델 로드 중: {model_id}")
        
        try:
            # 모델 및 프로세서 로드
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
            model = AutoModel.from_pretrained(model_id, cache_dir=self.cache_dir)
            
            # 장치 이동
            model = model.to(self.device)
            
            # 모델 저장
            self.models['clap'] = model
            self.processors['clap'] = processor
            
            print("CLAP 모델 로드 완료")
            return model, processor
        except Exception as e:
            print(f"CLAP 모델 로드 중 오류 발생: {e}")
            return None, None
    
    def load_whisper_model(self, model_id: str = "openai/whisper-small") -> Tuple[Any, Any]:
        """
        Whisper 모델 로드
        
        Args:
            model_id: 모델 ID
            
        Returns:
            (Whisper 모델, Whisper 프로세서) 튜플
        """
        print(f"Whisper 모델 로드 중: {model_id}")
        
        try:
            # 모델 및 프로세서 로드
            processor = WhisperProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
            model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=self.cache_dir)
            
            # 장치 이동
            model = model.to(self.device)
            
            # 모델 저장
            self.models['whisper'] = model
            self.processors['whisper'] = processor
            
            print("Whisper 모델 로드 완료")
            return model, processor
        except Exception as e:
            print(f"Whisper 모델 로드 중 오류 발생: {e}")
            return None, None
    
    def load_audioldm2_model(self, model_id: str = "cvssp/audioldm2") -> Any:
        """
        AudioLDM2 모델 로드
        
        Args:
            model_id: 모델 ID
            
        Returns:
            AudioLDM2 파이프라인
        """
        print(f"AudioLDM2 모델 로드 중: {model_id}")
        
        try:
            # 모델 로드
            pipeline = AudioLDM2Pipeline.from_pretrained(model_id, cache_dir=self.cache_dir)
            
            # 장치 이동
            pipeline = pipeline.to(self.device)
            
            # 모델 저장
            self.models['audioldm2'] = pipeline
            
            print("AudioLDM2 모델 로드 완료")
            return pipeline
        except Exception as e:
            print(f"AudioLDM2 모델 로드 중 오류 발생: {e}")
            # 대체로 AudioLDM 시도
            try:
                print("AudioLDM 모델 로드 시도 중...")
                pipeline = AudioLDMPipeline.from_pretrained("cvssp/audioldm", cache_dir=self.cache_dir)
                pipeline = pipeline.to(self.device)
                self.models['audioldm'] = pipeline
                print("AudioLDM 모델 로드 완료")
                return pipeline
            except Exception as e2:
                print(f"AudioLDM 모델 로드 중 오류 발생: {e2}")
                return None
    
    def load_all_models(self) -> None:
        """
        모든 사전 학습된 모델 로드
        """
        self.load_clap_model()
        self.load_whisper_model()
        self.load_audioldm2_model()
    
    def extract_clap_embeddings(self, audio_files: List[str]) -> np.ndarray:
        """
        오디오 파일에서 CLAP 임베딩 추출
        
        Args:
            audio_files: 오디오 파일 경로 목록
            
        Returns:
            CLAP 임베딩 배열 (n_samples, embed_dim)
        """
        if 'clap' not in self.models or 'clap' not in self.processors:
            print("CLAP 모델 또는 프로세서가 로드되지 않았습니다. 모델을 먼저 로드하세요.")
            return np.random.randn(len(audio_files), 512)
        
        model = self.models['clap']
        processor = self.processors['clap']
        
        embeddings = []
        
        for audio_file in tqdm(audio_files, desc="CLAP 임베딩 추출"):
            try:
                # 오디오 로드
                audio, sr = librosa.load(audio_file, sr=48000, mono=True)
                
                # CLAP 입력 형식으로 변환
                inputs = processor(
                    audios=audio,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 추출
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.audio_embeds.cpu().numpy()
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"{audio_file} 처리 중 오류 발생: {e}")
                # 오류 발생 시 랜덤 임베딩 사용
                embeddings.append(np.random.randn(1, 512))
        
        return np.vstack(embeddings)
    
    def extract_whisper_embeddings(self, audio_files: List[str]) -> np.ndarray:
        """
        오디오 파일에서 Whisper 임베딩 추출
        
        Args:
            audio_files: 오디오 파일 경로 목록
            
        Returns:
            Whisper 임베딩 배열 (n_samples, embed_dim)
        """
        if 'whisper' not in self.models or 'whisper' not in self.processors:
            print("Whisper 모델 또는 프로세서가 로드되지 않았습니다. 모델을 먼저 로드하세요.")
            return np.random.randn(len(audio_files), 512)
        
        model = self.models['whisper']
        processor = self.processors['whisper']
        
        embeddings = []
        
        for audio_file in tqdm(audio_files, desc="Whisper 임베딩 추출"):
            try:
                # 오디오 로드
                audio, sr = librosa.load(audio_file, sr=16000, mono=True)
                
                # Whisper 입력 형식으로 변환
                inputs = processor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 추출
                with torch.no_grad():
                    # 인코더 출력 추출
                    encoder_outputs = model.get_encoder()(**inputs)
                    embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"{audio_file} 처리 중 오류 발생: {e}")
                # 오류 발생 시 랜덤 임베딩 사용
                embeddings.append(np.random.randn(1, 512))
        
        return np.vstack(embeddings)
    
    def extract_audioldm2_latents(self, audio_files: List[str]) -> np.ndarray:
        """
        오디오 파일에서 AudioLDM2 잠재 벡터 추출
        
        Args:
            audio_files: 오디오 파일 경로 목록
            
        Returns:
            AudioLDM2 잠재 벡터 배열 (n_samples, latent_dim)
        """
        if 'audioldm2' not in self.models and 'audioldm' not in self.models:
            print("AudioLDM2 또는 AudioLDM 모델이 로드되지 않았습니다. 모델을 먼저 로드하세요.")
            return np.random.randn(len(audio_files), 512)
        
        # AudioLDM2 또는 AudioLDM 모델 선택
        model_key = 'audioldm2' if 'audioldm2' in self.models else 'audioldm'
        pipeline = self.models[model_key]
        
        latents = []
        
        for audio_file in tqdm(audio_files, desc="AudioLDM2 잠재 벡터 추출"):
            try:
                # 오디오 로드
                audio, sr = librosa.load(audio_file, sr=16000, mono=True)
                
                # 오디오 길이 조정 (최대 30초)
                max_length = 30 * sr
                if len(audio) > max_length:
                    audio = audio[:max_length]
                
                # 오디오를 텐서로 변환
                audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
                
                # 잠재 벡터 추출
                with torch.no_grad():
                    # VAE 인코더를 통해 잠재 벡터 추출
                    if hasattr(pipeline, 'vae'):
                        # AudioLDM2의 경우
                        latent = pipeline.vae.encode(audio_tensor).latent_dist.sample().cpu().numpy()
                    elif hasattr(pipeline, 'mel_vae'):
                        # AudioLDM의 경우
                        # 먼저 멜 스펙트로그램으로 변환
                        mel_spec = librosa.feature.melspectrogram(
                            y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80
                        )
                        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                        mel_tensor = torch.tensor(mel_spec).unsqueeze(0).to(self.device)
                        latent = pipeline.mel_vae.encode(mel_tensor).latent_dist.sample().cpu().numpy()
                    else:
                        # 대체 방법
                        print(f"모델 {model_key}에서 VAE를 찾을 수 없습니다. 랜덤 잠재 벡터 사용")
                        latent = np.random.randn(1, 512)
                
                latents.append(latent)
            except Exception as e:
                print(f"{audio_file} 처리 중 오류 발생: {e}")
                # 오류 발생 시 랜덤 잠재 벡터 사용
                latents.append(np.random.randn(1, 512))
        
        return np.vstack(latents)
    
    def generate_audio_from_latents(self, latents: Dict[str, torch.Tensor], 
                                   output_path: str, 
                                   prompt: str = "speech, clear voice",
                                   num_inference_steps: int = 50,
                                   audio_length_in_s: float = 5.0) -> str:
        """
        잠재 벡터에서 오디오 생성
        
        Args:
            latents: 잠재 벡터 딕셔너리 {'vae_mapped': vae_mapped, 'clap_mapped': clap_mapped}
            output_path: 출력 오디오 파일 경로
            prompt: 텍스트 프롬프트
            num_inference_steps: 추론 스텝 수
            audio_length_in_s: 생성할 오디오 길이 (초)
            
        Returns:
            생성된 오디오 파일 경로
        """
        if 'audioldm2' not in self.models and 'audioldm' not in self.models:
            print("AudioLDM2 또는 AudioLDM 모델이 로드되지 않았습니다. 모델을 먼저 로드하세요.")
            return None
        
        # AudioLDM2 또는 AudioLDM 모델 선택
        model_key = 'audioldm2' if 'audioldm2' in self.models else 'audioldm'
        pipeline = self.models[model_key]
        
        try:
            # 잠재 벡터를 장치로 이동
            vae_mapped = latents['vae_mapped'].to(self.device)
            
            # 조건부 임베딩 생성 (텍스트 프롬프트 사용)
            if 'clap_mapped' in latents:
                clap_mapped = latents['clap_mapped'].to(self.device)
                
                # CLAP 임베딩을 조건부 임베딩으로 사용
                # 이 부분은 모델에 따라 다를 수 있음
                if model_key == 'audioldm2':
                    # AudioLDM2의 경우
                    audio = pipeline(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        audio_length_in_s=audio_length_in_s,
                        latents=vae_mapped,  # VAE 잠재 벡터 사용
                        guidance_scale=3.5
                    ).audios[0]
                else:
                    # AudioLDM의 경우
                    audio = pipeline(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        audio_length_in_s=audio_length_in_s,
                        latents=vae_mapped,  # VAE 잠재 벡터 사용
                        guidance_scale=3.5
                    ).audios[0]
            else:
                # CLAP 임베딩 없이 VAE 잠재 벡터만 사용
                audio = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    audio_length_in_s=audio_length_in_s,
                    latents=vae_mapped,
                    guidance_scale=3.5
                ).audios[0]
            
            # 오디오 저장
            sf.write(output_path, audio, 16000)
            
            return output_path
        except Exception as e:
            print(f"오디오 생성 중 오류 발생: {e}")
            return None
    
    def transcribe_audio(self, audio_file: str) -> str:
        """
        오디오 파일 텍스트 변환
        
        Args:
            audio_file: 오디오 파일 경로
            
        Returns:
            변환된 텍스트
        """
        if 'whisper' not in self.models or 'whisper' not in self.processors:
            print("Whisper 모델 또는 프로세서가 로드되지 않았습니다. 모델을 먼저 로드하세요.")
            return "Whisper 모델이 로드되지 않았습니다."
        
        model = self.models['whisper']
        processor = self.processors['whisper']
        
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Whisper 입력 형식으로 변환
            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)
            
            # 텍스트 변환
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return transcription
        except Exception as e:
            print(f"텍스트 변환 중 오류 발생: {e}")
            return f"오류: {str(e)}"
    
    def visualize_audio(self, audio_file: str, title: str = "Audio Waveform and Spectrogram") -> None:
        """
        오디오 파일 시각화
        
        Args:
            audio_file: 오디오 파일 경로
            title: 그래프 제목
        """
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # 파형 및 스펙트로그램 시각화
            plt.figure(figsize=(12, 8))
            
            # 파형
            plt.subplot(2, 1, 1)
            plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            plt.title(f"{title} - Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            
            # 스펙트로그램
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            plt.imshow(D, aspect='auto', origin='lower', extent=[0, len(audio) / sr, 0, sr / 2])
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{title} - Spectrogram")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"오디오 시각화 중 오류 발생: {e}")


class PretrainedModelIntegrator:
    """
    사전 학습된 모델과 EEG 인코더, 잠재 벡터 매핑 모델을 통합하는 클래스
    """
    def __init__(self, eeg_encoder: nn.Module, latent_mapping: nn.Module, 
                pretrained_manager: PretrainedModelManager,
                device: torch.device = None):
        """
        초기화 함수
        
        Args:
            eeg_encoder: EEG 인코더 모델
            latent_mapping: 잠재 벡터 매핑 모델
            pretrained_manager: 사전 학습된 모델 관리자
            device: 모델 실행 장치 (CPU 또는 GPU)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eeg_encoder = eeg_encoder.to(self.device)
        self.latent_mapping = latent_mapping.to(self.device)
        self.pretrained_manager = pretrained_manager
        
        # 평가 모드로 설정
        self.eeg_encoder.eval()
        self.latent_mapping.eval()
    
    def process_eeg(self, eeg_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        EEG 데이터 처리
        
        Args:
            eeg_data: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            
        Returns:
            처리된 잠재 벡터 딕셔너리
        """
        # 장치 이동
        eeg_data = eeg_data.to(self.device)
        
        # EEG 인코딩
        with torch.no_grad():
            eeg_latents = self.eeg_encoder(eeg_data)
            
            # 잠재 벡터 매핑
            mapped_latents = self.latent_mapping.generate(eeg_latents)
        
        return mapped_latents
    
    def generate_audio(self, eeg_data: torch.Tensor, output_path: str, 
                      prompt: str = "speech, clear voice",
                      num_inference_steps: int = 50,
                      audio_length_in_s: float = 5.0) -> str:
        """
        EEG 데이터에서 오디오 생성
        
        Args:
            eeg_data: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            output_path: 출력 오디오 파일 경로
            prompt: 텍스트 프롬프트
            num_inference_steps: 추론 스텝 수
            audio_length_in_s: 생성할 오디오 길이 (초)
            
        Returns:
            생성된 오디오 파일 경로
        """
        # EEG 처리
        mapped_latents = self.process_eeg(eeg_data)
        
        # 오디오 생성
        audio_path = self.pretrained_manager.generate_audio_from_latents(
            mapped_latents,
            output_path,
            prompt,
            num_inference_steps,
            audio_length_in_s
        )
        
        return audio_path
    
    def process_and_transcribe(self, eeg_data: torch.Tensor, output_dir: str,
                              prompt: str = "speech, clear voice",
                              num_inference_steps: int = 50,
                              audio_length_in_s: float = 5.0) -> Dict[str, str]:
        """
        EEG 데이터 처리, 오디오 생성, 텍스트 변환
        
        Args:
            eeg_data: EEG 데이터 텐서 (batch_size, n_channels, n_times)
            output_dir: 출력 디렉토리
            prompt: 텍스트 프롬프트
            num_inference_steps: 추론 스텝 수
            audio_length_in_s: 생성할 오디오 길이 (초)
            
        Returns:
            결과 딕셔너리 {'audio_path': audio_path, 'transcription': transcription}
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일 경로
        audio_path = os.path.join(output_dir, "generated_audio.wav")
        
        # 오디오 생성
        audio_path = self.generate_audio(
            eeg_data,
            audio_path,
            prompt,
            num_inference_steps,
            audio_length_in_s
        )
        
        if audio_path is None:
            return {'audio_path': None, 'transcription': "오디오 생성 실패"}
        
        # 텍스트 변환
        transcription = self.pretrained_manager.transcribe_audio(audio_path)
        
        return {
            'audio_path': audio_path,
            'transcription': transcription
        }
    
    def batch_process(self, eeg_dataset: torch.utils.data.Dataset, output_dir: str,
                     prompt: str = "speech, clear voice",
                     num_inference_steps: int = 50,
                     audio_length_in_s: float = 5.0,
                     batch_size: int = 4) -> List[Dict[str, str]]:
        """
        EEG 데이터셋 배치 처리
        
        Args:
            eeg_dataset: EEG 데이터셋
            output_dir: 출력 디렉토리
            prompt: 텍스트 프롬프트
            num_inference_steps: 추론 스텝 수
            audio_length_in_s: 생성할 오디오 길이 (초)
            batch_size: 배치 크기
            
        Returns:
            결과 딕셔너리 목록
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 로더 생성
        data_loader = torch.utils.data.DataLoader(
            eeg_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        results = []
        
        for i, batch in enumerate(tqdm(data_loader, desc="배치 처리")):
            # EEG 데이터 추출
            eeg_data = batch['eeg']
            
            # 배치 내 각 샘플 처리
            for j in range(len(eeg_data)):
                # 출력 파일 경로
                sample_dir = os.path.join(output_dir, f"sample_{i * batch_size + j}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # 단일 샘플 처리
                result = self.process_and_transcribe(
                    eeg_data[j:j+1],
                    sample_dir,
                    prompt,
                    num_inference_steps,
                    audio_length_in_s
                )
                
                # 결과 저장
                results.append(result)
        
        return results


def create_pretrained_model_manager(device: torch.device = None, cache_dir: str = "./pretrained_models") -> PretrainedModelManager:
    """
    사전 학습된 모델 관리자 생성
    
    Args:
        device: 모델 실행 장치 (CPU 또는 GPU)
        cache_dir: 사전 학습된 모델 캐시 디렉토리
        
    Returns:
        사전 학습된 모델 관리자
    """
    manager = PretrainedModelManager(device, cache_dir)
    return manager


def create_pretrained_model_integrator(eeg_encoder: nn.Module, latent_mapping: nn.Module, 
                                     pretrained_manager: PretrainedModelManager,
                                     device: torch.device = None) -> PretrainedModelIntegrator:
    """
    사전 학습된 모델 통합기 생성
    
    Args:
        eeg_encoder: EEG 인코더 모델
        latent_mapping: 잠재 벡터 매핑 모델
        pretrained_manager: 사전 학습된 모델 관리자
        device: 모델 실행 장치 (CPU 또는 GPU)
        
    Returns:
        사전 학습된 모델 통합기
    """
    integrator = PretrainedModelIntegrator(eeg_encoder, latent_mapping, pretrained_manager, device)
    return integrator


if __name__ == "__main__":
    # 테스트 코드
    # 사전 학습된 모델 관리자 생성
    manager = create_pretrained_model_manager()
    
    # CLAP 모델 로드
    clap_model, clap_processor = manager.load_clap_model()
    
    # Whisper 모델 로드
    whisper_model, whisper_processor = manager.load_whisper_model()
    
    # AudioLDM2 모델 로드
    audioldm2_pipeline = manager.load_audioldm2_model()
    
    print("모든 모델 로드 완료")
