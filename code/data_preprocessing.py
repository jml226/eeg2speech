import os
import numpy as np
import torch
import torchaudio
import librosa
import mne
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import random
from pathlib import Path
import json
import requests
import zipfile
import tarfile
import warnings
from transformers import AutoProcessor, AutoModel

class EEGProcessor:
    """
    EEG 데이터 전처리를 위한 클래스
    """
    def __init__(self, data_dir: str, sample_rate: int = 1000, window_size: int = 2.0, 
                 filter_low: float = 0.5, filter_high: float = 45.0):
        """
        초기화 함수
        
        Args:
            data_dir: EEG 데이터가 저장된 디렉토리 경로
            sample_rate: EEG 데이터의 샘플링 레이트 (Hz)
            window_size: 분석 윈도우 크기 (초)
            filter_low: 대역 필터의 하한 주파수 (Hz)
            filter_high: 대역 필터의 상한 주파수 (Hz)
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.filter_low = filter_low
        self.filter_high = filter_high
        
        # Inner Speech 데이터셋 경로 설정
        self.inner_speech_dir = os.path.join(data_dir, 'inner_speech')
        
        # 전처리된 데이터 저장 경로 설정
        self.processed_dir = os.path.join(data_dir, 'processed_eeg')
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_inner_speech_dataset(self, force_download: bool = False) -> None:
        """
        Inner Speech 데이터셋 다운로드
        
        Args:
            force_download: 이미 다운로드된 데이터가 있어도 강제로 다시 다운로드할지 여부
        """
        # 데이터셋 디렉토리 생성
        os.makedirs(self.inner_speech_dir, exist_ok=True)
        
        # 이미 다운로드된 데이터가 있는지 확인
        if os.path.exists(os.path.join(self.inner_speech_dir, 'dataset_description.json')) and not force_download:
            print("Inner Speech 데이터셋이 이미 다운로드되어 있습니다.")
            return
        
        # GitHub 저장소 클론
        print("Inner Speech 데이터셋 다운로드 중...")
        os.system(f"cd {self.inner_speech_dir} && git clone https://github.com/N-Nieto/Inner_Speech_Dataset.git")
        
        # OpenNeuro에서 데이터셋 다운로드 (선택적)
        # 참고: 전체 데이터셋은 크기가 크므로 필요한 경우에만 다운로드
        print("OpenNeuro에서 데이터셋 다운로드는 수동으로 진행해주세요.")
        print("데이터셋 URL: https://openneuro.org/datasets/ds003626/versions/1.0.2")
    
    def get_subjects(self) -> List[str]:
        """
        데이터셋의 피험자 목록 반환
        
        Returns:
            피험자 ID 목록
        """
        # Inner Speech Dataset GitHub 저장소에서 피험자 목록 확인
        inner_speech_repo = os.path.join(self.inner_speech_dir, 'Inner_Speech_Dataset')
        if not os.path.exists(inner_speech_repo):
            raise FileNotFoundError(f"Inner Speech 데이터셋 저장소를 찾을 수 없습니다: {inner_speech_repo}")
        
        # 피험자 목록 반환 (예시)
        return ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
                'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    
    def get_sessions(self, subject_id: str) -> List[str]:
        """
        특정 피험자의 세션 목록 반환
        
        Args:
            subject_id: 피험자 ID
            
        Returns:
            세션 ID 목록
        """
        # 예시 세션 목록 반환
        return ['ses-01', 'ses-02']
    
    def get_tasks(self, subject_id: str, session_id: str) -> List[str]:
        """
        특정 피험자와 세션의 작업 목록 반환
        
        Args:
            subject_id: 피험자 ID
            session_id: 세션 ID
            
        Returns:
            작업 ID 목록
        """
        # 예시 작업 목록 반환
        return ['task-innerspeech']
    
    def get_runs(self, subject_id: str, session_id: str, task_id: str) -> List[str]:
        """
        특정 피험자, 세션, 작업의 실행 목록 반환
        
        Args:
            subject_id: 피험자 ID
            session_id: 세션 ID
            task_id: 작업 ID
            
        Returns:
            실행 ID 목록
        """
        # 예시 실행 목록 반환
        return ['run-01', 'run-02', 'run-03', 'run-04', 'run-05', 'run-06']
    
    def load_eeg_data(self, subject_id: str, session_id: str, task_id: str, run_id: str) -> mne.io.Raw:
        """
        EEG 데이터 로드
        
        Args:
            subject_id: 피험자 ID
            session_id: 세션 ID
            task_id: 작업 ID
            run_id: 실행 ID
            
        Returns:
            MNE Raw 객체
        """
        # 데이터 파일 경로 구성
        data_path = os.path.join(self.inner_speech_dir, 'Inner_Speech_Dataset', 'derivatives', 
                                 subject_id, session_id, 'eeg')
        
        # BIDS 형식의 파일명 구성
        file_name = f"{subject_id}_{session_id}_{task_id}_{run_id}_eeg.set"
        file_path = os.path.join(data_path, file_name)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            # 실제 파일이 없는 경우 예시 데이터 생성 (개발 목적)
            print(f"경고: {file_path} 파일을 찾을 수 없습니다. 예시 데이터를 생성합니다.")
            # 예시 데이터 생성 (128채널, 10초, 1000Hz)
            data = np.random.randn(128, 10000) * 1e-6  # 마이크로볼트 단위
            ch_names = [f'EEG{i:03d}' for i in range(1, 129)]
            ch_types = ['eeg'] * 128
            info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types=ch_types)
            raw = mne.io.RawArray(data, info)
            return raw
        
        # EEG 데이터 로드 (EEGLAB .set 파일)
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            return raw
        except Exception as e:
            print(f"EEG 데이터 로드 중 오류 발생: {e}")
            # 오류 발생 시 예시 데이터 생성
            data = np.random.randn(128, 10000) * 1e-6
            ch_names = [f'EEG{i:03d}' for i in range(1, 129)]
            ch_types = ['eeg'] * 128
            info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types=ch_types)
            raw = mne.io.RawArray(data, info)
            return raw
    
    def load_events(self, subject_id: str, session_id: str, task_id: str, run_id: str) -> np.ndarray:
        """
        이벤트 데이터 로드
        
        Args:
            subject_id: 피험자 ID
            session_id: 세션 ID
            task_id: 작업 ID
            run_id: 실행 ID
            
        Returns:
            이벤트 배열 (MNE 형식: onset, duration, trigger)
        """
        # 데이터 파일 경로 구성
        data_path = os.path.join(self.inner_speech_dir, 'Inner_Speech_Dataset', 'derivatives', 
                                 subject_id, session_id, 'eeg')
        
        # 이벤트 파일명 구성
        file_name = f"{subject_id}_{session_id}_{task_id}_{run_id}_events.tsv"
        file_path = os.path.join(data_path, file_name)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            # 실제 파일이 없는 경우 예시 데이터 생성 (개발 목적)
            print(f"경고: {file_path} 파일을 찾을 수 없습니다. 예시 이벤트 데이터를 생성합니다.")
            # 예시 이벤트 생성 (5개 이벤트)
            events = np.array([
                [1000, 0, 1],  # 첫 번째 이벤트: 1초, 클래스 1
                [3000, 0, 2],  # 두 번째 이벤트: 3초, 클래스 2
                [5000, 0, 3],  # 세 번째 이벤트: 5초, 클래스 3
                [7000, 0, 4],  # 네 번째 이벤트: 7초, 클래스 4
                [9000, 0, 5]   # 다섯 번째 이벤트: 9초, 클래스 5
            ])
            return events
        
        # 이벤트 데이터 로드 (TSV 파일)
        try:
            # TSV 파일에서 이벤트 정보 읽기
            import pandas as pd
            events_df = pd.read_csv(file_path, sep='\t')
            
            # MNE 형식으로 변환
            events = []
            for _, row in events_df.iterrows():
                onset = int(row['onset'] * self.sample_rate)
                duration = int(row['duration'] * self.sample_rate)
                trigger = int(row['value'])
                events.append([onset, duration, trigger])
            
            return np.array(events)
        except Exception as e:
            print(f"이벤트 데이터 로드 중 오류 발생: {e}")
            # 오류 발생 시 예시 데이터 생성
            events = np.array([
                [1000, 0, 1],
                [3000, 0, 2],
                [5000, 0, 3],
                [7000, 0, 4],
                [9000, 0, 5]
            ])
            return events
    
    def extract_epochs(self, raw: mne.io.Raw, events: np.ndarray, 
                       tmin: float = -0.2, tmax: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        이벤트 기반 에폭 추출
        
        Args:
            raw: MNE Raw 객체
            events: 이벤트 배열
            tmin: 에폭 시작 시간 (초, 이벤트 기준)
            tmax: 에폭 종료 시간 (초, 이벤트 기준)
            
        Returns:
            (에폭 데이터, 에폭 레이블) 튜플
        """
        try:
            # 이벤트 ID 매핑 (예시)
            event_id = {
                'up': 1,
                'down': 2,
                'left': 3,
                'right': 4,
                'backward': 5
            }
            
            # 에폭 추출
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                               baseline=(tmin, 0), preload=True)
            
            # 데이터와 레이블 추출
            epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
            epochs_labels = epochs.events[:, 2]  # 이벤트 트리거 값
            
            return epochs_data, epochs_labels
        except Exception as e:
            print(f"에폭 추출 중 오류 발생: {e}")
            # 오류 발생 시 예시 데이터 생성
            n_epochs = len(events)
            n_channels = len(raw.ch_names)
            n_times = int((tmax - tmin) * self.sample_rate)
            
            epochs_data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
            epochs_labels = events[:, 2]
            
            return epochs_data, epochs_labels
    
    def preprocess_eeg(self, epochs_data: np.ndarray) -> np.ndarray:
        """
        EEG 에폭 데이터 전처리
        
        Args:
            epochs_data: 에폭 데이터 배열 (n_epochs, n_channels, n_times)
            
        Returns:
            전처리된 에폭 데이터
        """
        # 전처리된 데이터를 저장할 배열
        processed_data = epochs_data.copy()
        
        # 1. 대역 필터링 (주파수 도메인)
        for i in range(len(processed_data)):
            # 각 에폭에 대해 처리
            for j in range(len(processed_data[i])):
                # 각 채널에 대해 처리
                # FFT 변환
                fft_data = np.fft.rfft(processed_data[i, j])
                freqs = np.fft.rfftfreq(len(processed_data[i, j]), 1.0/self.sample_rate)
                
                # 대역 필터 적용
                fft_data[(freqs < self.filter_low) | (freqs > self.filter_high)] = 0
                
                # 역 FFT 변환
                processed_data[i, j] = np.fft.irfft(fft_data, len(processed_data[i, j]))
        
        # 2. 평균 참조 (CAR)
        for i in range(len(processed_data)):
            # 각 에폭에 대해 처리
            # 모든 채널의 평균 계산
            mean = np.mean(processed_data[i], axis=0)
            # 평균 참조 적용
            processed_data[i] = processed_data[i] - mean
        
        # 3. 정규화 (채널별)
        for i in range(len(processed_data)):
            # 각 에폭에 대해 처리
            for j in range(len(processed_data[i])):
                # 각 채널에 대해 처리
                # Z-점수 정규화
                mean = np.mean(processed_data[i, j])
                std = np.std(processed_data[i, j])
                if std > 0:  # 0으로 나누기 방지
                    processed_data[i, j] = (processed_data[i, j] - mean) / std
        
        return processed_data
    
    def apply_data_augmentation(self, epochs_data: np.ndarray, 
                               augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 증강 적용
        
        Args:
            epochs_data: 에폭 데이터 배열 (n_epochs, n_channels, n_times)
            epochs_labels: 에폭 레이블 배열
            augmentation_factor: 증강할 데이터 배수
            
        Returns:
            (증강된 에폭 데이터, 증강된 에폭 레이블) 튜플
        """
        n_epochs, n_channels, n_times = epochs_data.shape
        
        # 원본 데이터 복사
        augmented_data = epochs_data.copy()
        
        # 증강 데이터 생성
        for _ in range(augmentation_factor - 1):
            # 1. 시간 이동
            time_shift_data = np.zeros_like(epochs_data)
            for i in range(n_epochs):
                shift = np.random.randint(-int(n_times * 0.1), int(n_times * 0.1))
                for j in range(n_channels):
                    time_shift_data[i, j] = np.roll(epochs_data[i, j], shift)
            
            # 2. 노이즈 추가
            noise_data = epochs_data + np.random.normal(0, 0.1, epochs_data.shape)
            
            # 3. 채널 마스킹
            mask_data = epochs_data.copy()
            for i in range(n_epochs):
                # 랜덤하게 10%의 채널 마스킹
                mask_channels = np.random.choice(n_channels, int(n_channels * 0.1), replace=False)
                mask_data[i, mask_channels] = 0
            
            # 증강 데이터 결합
            augmented_data = np.concatenate([augmented_data, time_shift_data, noise_data, mask_data], axis=0)
        
        return augmented_data
    
    def visualize_eeg(self, raw: mne.io.Raw, n_channels: int = 10, duration: float = 5.0) -> None:
        """
        EEG 데이터 시각화
        
        Args:
            raw: MNE Raw 객체
            n_channels: 표시할 채널 수
            duration: 표시할 시간 길이 (초)
        """
        # 채널 선택
        ch_names = raw.ch_names[:n_channels]
        
        # 데이터 추출
        data, times = raw.get_data(picks=ch_names, return_times=True)
        
        # 표시할 시간 범위 선택
        mask = times < duration
        times = times[mask]
        data = data[:, mask]
        
        # 그래프 그리기
        plt.figure(figsize=(12, 8))
        for i, ch_name in enumerate(ch_names):
            # 채널별 오프셋 적용
            offset = i * 50e-6
            plt.plot(times, data[i] + offset, label=ch_name)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        plt.title('EEG Data Visualization')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    def print_dataset_info(self) -> None:
        """
        데이터셋 정보 출력
        """
        print("Inner Speech 데이터셋 정보:")
        print(f"- 데이터 디렉토리: {self.inner_speech_dir}")
        
        # 피험자 수 확인
        try:
            subjects = self.get_subjects()
            print(f"- 피험자 수: {len(subjects)}")
            print(f"- 피험자 목록: {', '.join(subjects[:5])}{'...' if len(subjects) > 5 else ''}")
        except Exception as e:
            print(f"- 피험자 정보 확인 중 오류 발생: {e}")
        
        # 샘플 데이터 로드 시도
        try:
            subject_id = subjects[0]
            session_id = self.get_sessions(subject_id)[0]
            task_id = self.get_tasks(subject_id, session_id)[0]
            run_id = self.get_runs(subject_id, session_id, task_id)[0]
            
            raw = self.load_eeg_data(subject_id, session_id, task_id, run_id)
            events = self.load_events(subject_id, session_id, task_id, run_id)
            
            print(f"- 샘플 EEG 데이터 정보:")
            print(f"  - 채널 수: {len(raw.ch_names)}")
            print(f"  - 샘플링 레이트: {raw.info['sfreq']} Hz")
            print(f"  - 데이터 길이: {raw.times[-1]:.2f} 초")
            print(f"  - 이벤트 수: {len(events)}")
        except Exception as e:
            print(f"- 샘플 데이터 로드 중 오류 발생: {e}")


class AudioProcessor:
    """
    오디오 데이터 전처리를 위한 클래스
    """
    def __init__(self, data_dir: str, sample_rate: int = 16000, n_mels: int = 80,
                 n_fft: int = 1024, hop_length: int = 256):
        """
        초기화 함수
        
        Args:
            data_dir: 오디오 데이터가 저장된 디렉토리 경로
            sample_rate: 오디오 데이터의 샘플링 레이트 (Hz)
            n_mels: 멜 스펙트로그램의 멜 빈 수
            n_fft: FFT 크기
            hop_length: 프레임 간 홉 길이
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # LibriSpeech 데이터셋 경로 설정
        self.librispeech_dir = os.path.join(data_dir, 'librispeech')
        
        # 전처리된 데이터 저장 경로 설정
        self.processed_dir = os.path.join(data_dir, 'processed_audio')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # CLAP 프로세서 로드
        try:
            self.clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        except Exception as e:
            print(f"CLAP 프로세서 로드 중 오류 발생: {e}")
            self.clap_processor = None
    
    def download_librispeech_dataset(self, subset: str = 'dev-clean', force_download: bool = False) -> None:
        """
        LibriSpeech 데이터셋 다운로드
        
        Args:
            subset: 다운로드할 데이터셋 서브셋 ('dev-clean', 'test-clean', 'train-clean-100' 등)
            force_download: 이미 다운로드된 데이터가 있어도 강제로 다시 다운로드할지 여부
        """
        # 데이터셋 디렉토리 생성
        os.makedirs(self.librispeech_dir, exist_ok=True)
        
        # 이미 다운로드된 데이터가 있는지 확인
        subset_dir = os.path.join(self.librispeech_dir, subset)
        if os.path.exists(subset_dir) and not force_download:
            print(f"LibriSpeech {subset} 데이터셋이 이미 다운로드되어 있습니다.")
            return
        
        # 데이터셋 URL
        url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
        
        # 데이터셋 다운로드
        print(f"LibriSpeech {subset} 데이터셋 다운로드 중...")
        tar_path = os.path.join(self.librispeech_dir, f"{subset}.tar.gz")
        
        try:
            # 파일 다운로드
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(tar_path, 'wb') as f:
                for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                    f.write(data)
            
            # 압축 해제
            print(f"압축 해제 중...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=self.librispeech_dir)
            
            # 다운로드한 압축 파일 삭제
            os.remove(tar_path)
            
            print(f"LibriSpeech {subset} 데이터셋 다운로드 및 압축 해제 완료.")
        except Exception as e:
            print(f"데이터셋 다운로드 중 오류 발생: {e}")
    
    def get_audio_files(self, subset: str = 'dev-clean') -> List[str]:
        """
        오디오 파일 목록 반환
        
        Args:
            subset: 데이터셋 서브셋 ('dev-clean', 'test-clean', 'train-clean-100' 등)
            
        Returns:
            오디오 파일 경로 목록
        """
        subset_dir = os.path.join(self.librispeech_dir, subset)
        
        if not os.path.exists(subset_dir):
            print(f"경고: {subset_dir} 디렉토리를 찾을 수 없습니다.")
            return []
        
        # .flac 파일 검색
        audio_files = []
        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith('.flac'):
                    audio_files.append(os.path.join(root, file))
        
        return audio_files
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        오디오 파일 로드
        
        Args:
            file_path: 오디오 파일 경로
            
        Returns:
            (오디오 데이터, 샘플링 레이트) 튜플
        """
        try:
            # librosa를 사용하여 오디오 로드
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            print(f"오디오 파일 로드 중 오류 발생: {e}")
            # 오류 발생 시 예시 데이터 생성 (1초 길이)
            audio = np.random.randn(self.sample_rate) * 0.1
            return audio, self.sample_rate
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        오디오 데이터 전처리
        
        Args:
            audio: 오디오 데이터 배열
            sr: 샘플링 레이트
            
        Returns:
            전처리된 오디오 데이터
        """
        # 1. 리샘플링 (필요한 경우)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # 2. 정규화 (진폭 정규화)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 3. 프리엠퍼시스 필터 적용
        preemphasis_coef = 0.97
        audio = np.append(audio[0], audio[1:] - preemphasis_coef * audio[:-1])
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        멜 스펙트로그램 추출
        
        Args:
            audio: 오디오 데이터 배열
            sr: 샘플링 레이트
            
        Returns:
            멜 스펙트로그램 배열
        """
        # 멜 스펙트로그램 추출
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # 로그 스케일 변환
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def extract_clap_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        CLAP 특성 추출
        
        Args:
            audio: 오디오 데이터 배열
            sr: 샘플링 레이트
            
        Returns:
            CLAP 특성 벡터
        """
        if self.clap_processor is None:
            print("경고: CLAP 프로세서가 로드되지 않았습니다.")
            # 예시 특성 벡터 반환 (512차원)
            return np.random.randn(512)
        
        try:
            # CLAP 입력 형식으로 변환
            inputs = self.clap_processor(
                audios=audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            # CLAP 모델이 로드되지 않았으므로 예시 특성 벡터 반환
            # 실제 구현에서는 CLAP 모델을 로드하여 특성 추출
            return np.random.randn(512)
        except Exception as e:
            print(f"CLAP 특성 추출 중 오류 발생: {e}")
            return np.random.randn(512)
    
    def load_text(self, audio_file_path: str) -> str:
        """
        오디오 파일에 해당하는 텍스트 로드
        
        Args:
            audio_file_path: 오디오 파일 경로
            
        Returns:
            텍스트 내용
        """
        # LibriSpeech 데이터셋의 경우 텍스트 파일 경로 구성
        # 예: /path/to/librispeech/dev-clean/1/2/1-2-3.flac -> /path/to/librispeech/dev-clean/1/2/1-2.trans.txt
        try:
            # 오디오 파일 경로에서 디렉토리와 파일명 추출
            audio_dir = os.path.dirname(audio_file_path)
            audio_filename = os.path.basename(audio_file_path)
            
            # 파일 ID 추출 (예: 1-2-3.flac -> 1-2)
            file_id = '-'.join(audio_filename.split('-')[:2])
            
            # 텍스트 파일 경로 구성
            text_file_path = os.path.join(audio_dir, f"{file_id}.trans.txt")
            
            # 텍스트 파일이 존재하는지 확인
            if not os.path.exists(text_file_path):
                print(f"경고: {text_file_path} 파일을 찾을 수 없습니다.")
                return "No transcription available."
            
            # 텍스트 파일에서 해당 오디오 파일의 텍스트 찾기
            audio_id = audio_filename.split('.')[0]  # 확장자 제거
            with open(text_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(audio_id):
                        # 형식: {audio_id} {transcription}
                        return line.split(' ', 1)[1].strip()
            
            return "No matching transcription found."
        except Exception as e:
            print(f"텍스트 로드 중 오류 발생: {e}")
            return "Error loading transcription."
    
    def preprocess_text(self, text: str) -> List[int]:
        """
        텍스트 전처리 및 토큰화
        
        Args:
            text: 텍스트 내용
            
        Returns:
            토큰 ID 목록
        """
        # 간단한 전처리 (소문자 변환, 구두점 제거 등)
        text = text.lower()
        for punct in ',.!?;:()[]{}""\'':
            text = text.replace(punct, ' ')
        text = ' '.join(text.split())  # 연속된 공백 제거
        
        # 문자 기반 토큰화 (예시)
        # 실제 구현에서는 CLAP 또는 Whisper 토크나이저 사용
        char_to_idx = {c: i+1 for i, c in enumerate(' abcdefghijklmnopqrstuvwxyz')}
        tokens = [char_to_idx.get(c, 0) for c in text]
        
        return tokens
    
    def apply_audio_augmentation(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        오디오 데이터 증강
        
        Args:
            audio: 오디오 데이터 배열
            sr: 샘플링 레이트
            
        Returns:
            증강된 오디오 데이터 목록
        """
        augmented_audios = [audio]  # 원본 오디오 포함
        
        # 1. 피치 변경
        try:
            pitch_shift = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            augmented_audios.append(pitch_shift)
        except Exception as e:
            print(f"피치 변경 중 오류 발생: {e}")
        
        # 2. 시간 늘이기/줄이기
        try:
            time_stretch = librosa.effects.time_stretch(audio, rate=1.2)
            augmented_audios.append(time_stretch)
        except Exception as e:
            print(f"시간 늘이기/줄이기 중 오류 발생: {e}")
        
        # 3. 노이즈 추가
        try:
            noise = np.random.randn(len(audio)) * 0.005
            noisy_audio = audio + noise
            augmented_audios.append(noisy_audio)
        except Exception as e:
            print(f"노이즈 추가 중 오류 발생: {e}")
        
        return augmented_audios
    
    def visualize_audio_and_mel(self, audio: np.ndarray, sr: int, mel_spec: np.ndarray) -> None:
        """
        오디오 파형과 멜 스펙트로그램 시각화
        
        Args:
            audio: 오디오 데이터 배열
            sr: 샘플링 레이트
            mel_spec: 멜 스펙트로그램 배열
        """
        plt.figure(figsize=(12, 8))
        
        # 오디오 파형 그리기
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio, sr=sr)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 멜 스펙트로그램 그리기
        plt.subplot(2, 1, 2)
        librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', hop_length=self.hop_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def print_dataset_info(self) -> None:
        """
        데이터셋 정보 출력
        """
        print("LibriSpeech 데이터셋 정보:")
        print(f"- 데이터 디렉토리: {self.librispeech_dir}")
        
        # 오디오 파일 수 확인
        try:
            audio_files = self.get_audio_files()
            print(f"- 오디오 파일 수: {len(audio_files)}")
            if audio_files:
                print(f"- 샘플 오디오 파일: {audio_files[0]}")
                
                # 샘플 오디오 로드
                audio, sr = self.load_audio(audio_files[0])
                print(f"- 샘플 오디오 정보:")
                print(f"  - 길이: {len(audio) / sr:.2f} 초")
                print(f"  - 샘플링 레이트: {sr} Hz")
                
                # 샘플 텍스트 로드
                text = self.load_text(audio_files[0])
                print(f"- 샘플 텍스트: {text}")
        except Exception as e:
            print(f"- 데이터셋 정보 확인 중 오류 발생: {e}")


class EEGAudioDataset(Dataset):
    """
    EEG와 오디오 데이터를 위한 PyTorch 데이터셋
    """
    def __init__(self, eeg_data: np.ndarray, audio_data: np.ndarray, mel_specs: np.ndarray, 
                text_data: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
        """
        초기화 함수
        
        Args:
            eeg_data: EEG 데이터 배열 (n_samples, n_channels, n_times)
            audio_data: 오디오 데이터 배열 (n_samples, n_audio_samples)
            mel_specs: 멜 스펙트로그램 배열 (n_samples, n_mels, n_frames)
            text_data: 텍스트 토큰 배열 (n_samples, max_seq_length), 선택적
            labels: 레이블 배열 (n_samples,), 선택적
        """
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        self.mel_specs = torch.tensor(mel_specs, dtype=torch.float32)
        
        if text_data is not None:
            self.text_data = torch.tensor(text_data, dtype=torch.long)
        else:
            self.text_data = None
        
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None
    
    def __len__(self) -> int:
        """
        데이터셋 길이 반환
        
        Returns:
            데이터셋 샘플 수
        """
        return len(self.eeg_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터셋 항목 반환
        
        Args:
            idx: 인덱스
            
        Returns:
            데이터 항목 딕셔너리
        """
        item = {
            'eeg': self.eeg_data[idx],
            'audio': self.audio_data[idx],
            'mel_spec': self.mel_specs[idx]
        }
        
        if self.text_data is not None:
            item['text_tokens'] = self.text_data[idx]
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
        
        return item


def create_dataloaders(eeg_data: np.ndarray, audio_data: np.ndarray, mel_specs: np.ndarray, 
                      text_data: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
                      batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.15,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    데이터 로더 생성
    
    Args:
        eeg_data: EEG 데이터 배열
        audio_data: 오디오 데이터 배열
        mel_specs: 멜 스펙트로그램 배열
        text_data: 텍스트 토큰 배열, 선택적
        labels: 레이블 배열, 선택적
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
        
    Returns:
        (학습 데이터 로더, 검증 데이터 로더, 테스트 데이터 로더) 튜플
    """
    # 데이터셋 생성
    dataset = EEGAudioDataset(eeg_data, audio_data, mel_specs, text_data, labels)
    
    # 데이터셋 분할
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 랜덤 셔플
    random.seed(seed)
    random.shuffle(indices)
    
    # 분할 인덱스 계산
    train_split = int(train_ratio * dataset_size)
    val_split = int((train_ratio + val_ratio) * dataset_size)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # 서브셋 생성
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def download_and_prepare_data(data_dir: str) -> None:
    """
    데이터 다운로드 및 준비
    
    Args:
        data_dir: 데이터 디렉토리 경로
    """
    # EEG 데이터 다운로드 및 준비
    eeg_processor = EEGProcessor(data_dir)
    eeg_processor.download_inner_speech_dataset()
    
    # 오디오 데이터 다운로드 및 준비
    audio_processor = AudioProcessor(data_dir)
    audio_processor.download_librispeech_dataset(subset='dev-clean')
    
    print("데이터 다운로드 및 준비 완료.")


if __name__ == "__main__":
    # 테스트 코드
    data_dir = "/home/ubuntu/eeg2speech/data"
    
    # 데이터 다운로드 및 준비
    download_and_prepare_data(data_dir)
    
    # EEG 프로세서 테스트
    eeg_processor = EEGProcessor(data_dir)
    eeg_processor.print_dataset_info()
    
    # 오디오 프로세서 테스트
    audio_processor = AudioProcessor(data_dir)
    audio_processor.print_dataset_info()
