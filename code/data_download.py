"""
데이터 다운로드 및 준비 모듈

이 모듈은 Inner Speech 데이터셋과 LibriSpeech 데이터셋을 다운로드하고 준비하는 기능을 제공합니다.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
import mne
import glob
import json
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm


class DataDownloader:
    """
    데이터셋 다운로드 클래스
    """
    
    def __init__(self, base_dir: str):
        """
        초기화 함수
        
        Args:
            base_dir: 기본 디렉토리 경로
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.inner_speech_dir = os.path.join(self.data_dir, "inner_speech")
        self.audio_text_dir = os.path.join(self.data_dir, "audio_text")
        
        # 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.inner_speech_dir, exist_ok=True)
        os.makedirs(self.audio_text_dir, exist_ok=True)
    
    def download_inner_speech_dataset(self) -> str:
        """
        Inner Speech 데이터셋 다운로드
        
        Returns:
            다운로드된 데이터셋 경로
        """
        print("Inner Speech 데이터셋 다운로드 중...")
        
        # GitHub 저장소 클론
        repo_url = "https://github.com/N-Nieto/Inner_Speech_Dataset.git"
        repo_dir = os.path.join(self.inner_speech_dir, "Inner_Speech_Dataset")
        
        if os.path.exists(repo_dir):
            print(f"이미 다운로드된 저장소가 있습니다: {repo_dir}")
            # 최신 버전으로 업데이트
            try:
                subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
                print("저장소를 최신 버전으로 업데이트했습니다.")
            except subprocess.CalledProcessError:
                print("저장소 업데이트 중 오류가 발생했습니다. 기존 버전을 사용합니다.")
        else:
            try:
                subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
                print(f"저장소를 성공적으로 클론했습니다: {repo_dir}")
            except subprocess.CalledProcessError:
                print("저장소 클론 중 오류가 발생했습니다.")
                raise
        
        # OpenNeuro 데이터셋 다운로드 (선택적)
        openneuro_dir = os.path.join(self.inner_speech_dir, "openneuro")
        if not os.path.exists(openneuro_dir):
            print("OpenNeuro 데이터셋은 용량이 크므로 자동으로 다운로드하지 않습니다.")
            print("필요한 경우 다음 URL에서 수동으로 다운로드하세요: https://openneuro.org/datasets/ds003626/versions/1.0.2")
            os.makedirs(openneuro_dir, exist_ok=True)
        
        return repo_dir
    
    def download_audio_text_dataset(self) -> str:
        """
        오디오-텍스트 데이터셋 다운로드 (예: LibriSpeech)
        
        Returns:
            다운로드된 데이터셋 경로
        """
        print("오디오-텍스트 데이터셋 다운로드 중...")
        
        # LibriSpeech 데이터셋의 작은 부분집합 다운로드
        librispeech_dir = os.path.join(self.audio_text_dir, "librispeech")
        os.makedirs(librispeech_dir, exist_ok=True)
        
        # dev-clean 부분집합 다운로드 (가장 작은 부분집합)
        dev_clean_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        dev_clean_file = os.path.join(self.audio_text_dir, "dev-clean.tar.gz")
        
        if not os.path.exists(os.path.join(librispeech_dir, "dev-clean")):
            if not os.path.exists(dev_clean_file):
                print(f"dev-clean 다운로드 중... (337MB)")
                self._download_file(dev_clean_url, dev_clean_file)
            
            print("dev-clean 압축 해제 중...")
            with tarfile.open(dev_clean_file, "r:gz") as tar:
                tar.extractall(path=self.audio_text_dir)
            
            # 압축 해제 후 파일 이동
            extracted_dir = os.path.join(self.audio_text_dir, "LibriSpeech")
            if os.path.exists(extracted_dir):
                for item in os.listdir(extracted_dir):
                    shutil.move(os.path.join(extracted_dir, item), librispeech_dir)
                os.rmdir(extracted_dir)
            
            # 다운로드 파일 삭제
            if os.path.exists(dev_clean_file):
                os.remove(dev_clean_file)
        
        print(f"오디오-텍스트 데이터셋 준비 완료: {librispeech_dir}")
        return librispeech_dir
    
    def _download_file(self, url: str, dest_path: str):
        """
        파일 다운로드 헬퍼 함수
        
        Args:
            url: 다운로드 URL
            dest_path: 저장 경로
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(dest_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)
    
    def prepare_datasets(self) -> Dict[str, str]:
        """
        모든 데이터셋 준비
        
        Returns:
            데이터셋 경로 딕셔너리
        """
        inner_speech_path = self.download_inner_speech_dataset()
        audio_text_path = self.download_audio_text_dataset()
        
        return {
            "inner_speech": inner_speech_path,
            "audio_text": audio_text_path
        }


# 노트북과 호환되는 다운로더 클래스 추가
class InnerSpeechDownloader:
    """
    Inner Speech 데이터셋 다운로더 클래스
    """
    
    def __init__(self, data_dir: str):
        """
        초기화 함수
        
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download(self) -> str:
        """
        Inner Speech 데이터셋 다운로드
        
        Returns:
            다운로드된 데이터셋 경로
        """
        print("Inner Speech 데이터셋 다운로드 중...")
        
        # GitHub 저장소 클론
        repo_url = "https://github.com/N-Nieto/Inner_Speech_Dataset.git"
        repo_dir = os.path.join(self.data_dir, "Inner_Speech_Dataset")
        
        if os.path.exists(repo_dir):
            print(f"이미 다운로드된 저장소가 있습니다: {repo_dir}")
            # 최신 버전으로 업데이트
            try:
                subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
                print("저장소를 최신 버전으로 업데이트했습니다.")
            except subprocess.CalledProcessError:
                print("저장소 업데이트 중 오류가 발생했습니다. 기존 버전을 사용합니다.")
        else:
            try:
                subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
                print(f"저장소를 성공적으로 클론했습니다: {repo_dir}")
            except subprocess.CalledProcessError:
                print("저장소 클론 중 오류가 발생했습니다.")
                raise
        
        # OpenNeuro 데이터셋 다운로드 (선택적)
        openneuro_dir = os.path.join(self.data_dir, "openneuro")
        if not os.path.exists(openneuro_dir):
            print("OpenNeuro 데이터셋은 용량이 크므로 자동으로 다운로드하지 않습니다.")
            print("필요한 경우 다음 URL에서 수동으로 다운로드하세요: https://openneuro.org/datasets/ds003626/versions/1.0.2")
            os.makedirs(openneuro_dir, exist_ok=True)
        
        return repo_dir


class LibriSpeechDownloader:
    """
    LibriSpeech 데이터셋 다운로더 클래스
    """
    
    def __init__(self, data_dir: str):
        """
        초기화 함수
        
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download(self, subset: str = 'dev-clean') -> str:
        """
        LibriSpeech 데이터셋 다운로드
        
        Args:
            subset: 다운로드할 부분집합 ('dev-clean', 'test-clean', 'train-clean-100' 등)
            
        Returns:
            다운로드된 데이터셋 경로
        """
        print(f"LibriSpeech {subset} 데이터셋 다운로드 중...")
        
        # URL 및 파일 경로 설정
        url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
        tar_file = os.path.join(self.data_dir, f"{subset}.tar.gz")
        
        # 이미 다운로드된 경우 스킵
        if os.path.exists(os.path.join(self.data_dir, subset)):
            print(f"이미 다운로드된 데이터셋이 있습니다: {os.path.join(self.data_dir, subset)}")
            return os.path.join(self.data_dir, subset)
        
        # 다운로드
        if not os.path.exists(tar_file):
            print(f"{subset} 다운로드 중...")
            self._download_file(url, tar_file)
        
        # 압축 해제
        print(f"{subset} 압축 해제 중...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=self.data_dir)
        
        # 압축 해제 후 파일 이동
        extracted_dir = os.path.join(self.data_dir, "LibriSpeech")
        if os.path.exists(extracted_dir):
            for item in os.listdir(extracted_dir):
                if not os.path.exists(os.path.join(self.data_dir, item)):
                    shutil.move(os.path.join(extracted_dir, item), self.data_dir)
            if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) == 0:
                os.rmdir(extracted_dir)
        
        # 다운로드 파일 삭제
        if os.path.exists(tar_file):
            os.remove(tar_file)
        
        print(f"LibriSpeech {subset} 데이터셋 준비 완료: {os.path.join(self.data_dir, subset)}")
        return os.path.join(self.data_dir, subset)
    
    def _download_file(self, url: str, dest_path: str):
        """
        파일 다운로드 헬퍼 함수
        
        Args:
            url: 다운로드 URL
            dest_path: 저장 경로
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(dest_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)


class InnerSpeechDataExplorer:
    """
    Inner Speech 데이터셋 탐색 클래스
    """
    
    def __init__(self, dataset_path: str):
        """
        초기화 함수
        
        Args:
            dataset_path: 데이터셋 경로
        """
        self.dataset_path = dataset_path
        self.processing_dir = os.path.join(dataset_path, "Python_Processing")
        
        # 시스템 경로에 처리 디렉토리 추가
        if self.processing_dir not in sys.path:
            sys.path.append(self.processing_dir)
    
    def list_subjects(self) -> List[int]:
        """
        피험자 목록 반환
        
        Returns:
            피험자 ID 목록
        """
        try:
            from Data_extractions import get_subject_list
            return get_subject_list(self.dataset_path)
        except ImportError:
            print("Data_extractions 모듈을 가져올 수 없습니다.")
            # 대체 방법: 디렉토리 구조 기반 피험자 목록 생성
            subjects_dir = os.path.join(self.dataset_path, "Subjects_data")
            if os.path.exists(subjects_dir):
                subject_dirs = [d for d in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, d))]
                return [int(d.split('_')[1]) for d in subject_dirs if d.startswith('Subject_')]
            return []
    
    def extract_subject_data(self, subject_id: int, datatype: str = "EEG") -> Tuple[np.ndarray, np.ndarray]:
        """
        피험자 데이터 추출
        
        Args:
            subject_id: 피험자 ID
            datatype: 데이터 유형 ("EEG" 또는 "EXG")
            
        Returns:
            X: 데이터 [trials x channels x samples]
            Y: 레이블 [trials x 4] (timestamp, class, condition, session)
        """
        try:
            from Data_extractions import extract_data_from_subject
            return extract_data_from_subject(self.dataset_path, subject_id, datatype)
        except ImportError:
            print("Data_extractions 모듈을 가져올 수 없습니다.")
            # 더미 데이터 반환
            return np.random.randn(100, 128, 512), np.zeros((100, 4))
    
    def get_class_names(self) -> List[str]:
        """
        클래스 이름 목록 반환
        
        Returns:
            클래스 이름 목록
        """
        try:
            from Data_extractions import get_class_list
            return get_class_list(self.dataset_path)
        except ImportError:
            print("Data_extractions 모듈을 가져올 수 없습니다.")
            return ["Up", "Down", "Right", "Left"]
    
    def get_condition_names(self) -> List[str]:
        """
        조건 이름 목록 반환
        
        Returns:
            조건 이름 목록
        """
        try:
            from Data_extractions import get_condition_list
            return get_condition_list(self.dataset_path)
        except ImportError:
            print("Data_extractions 모듈을 가져올 수 없습니다.")
            return ["Inner", "Pronounced"]
    
    def visualize_eeg_data(self, X: np.ndarray, Y: np.ndarray, trial_idx: int = 0, channels: List[int] = None):
        """
        EEG 데이터 시각화
        
        Args:
            X: EEG 데이터 [trials x channels x samples]
            Y: 레이블 [trials x 4]
            trial_idx: 시각화할 시행 인덱스
            channels: 시각화할 채널 인덱스 목록
        """
        if channels is None:
            channels = list(range(min(5, X.shape[1])))
        
        plt.figure(figsize=(12, 8))
        time = np.arange(X.shape[2]) / 256  # 샘플링 주파수 256Hz 가정
        
        for i, ch in enumerate(channels):
            plt.subplot(len(channels), 1, i+1)
            plt.plot(time, X[trial_idx, ch, :])
            plt.ylabel(f'Channel {ch}')
            plt.xlim([0, time[-1]])
            
            if i == len(channels) - 1:
                plt.xlabel('Time (s)')
        
        # 레이블 정보 표시
        class_names = self.get_class_names()
        condition_names = self.get_condition_names()
        
        class_idx = int(Y[trial_idx, 1])
        condition_idx = int(Y[trial_idx, 2])
        
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        condition_name = condition_names[condition_idx] if condition_idx < len(condition_names) else f"Condition {condition_idx}"
        
        plt.suptitle(f'Trial {trial_idx}: {class_name} - {condition_name}')
        plt.tight_layout()
        plt.show()
    
    def explore_dataset(self):
        """
        데이터셋 탐색 및 요약 정보 출력
        """
        print(f"Inner Speech 데이터셋 탐색: {self.dataset_path}")
        
        # 피험자 목록
        subjects = self.list_subjects()
        print(f"피험자 수: {len(subjects)}")
        print(f"피험자 ID: {subjects}")
        
        # 클래스 및 조건
        class_names = self.get_class_names()
        condition_names = self.get_condition_names()
        
        print(f"클래스: {class_names}")
        print(f"조건: {condition_names}")
        
        # 데이터 샘플 추출 및 시각화
        if subjects:
            subject_id = subjects[0]
            print(f"피험자 {subject_id}의 데이터 추출 중...")
            X, Y = self.extract_subject_data(subject_id)
            
            print(f"데이터 형태: {X.shape}")
            print(f"레이블 형태: {Y.shape}")
            
            # 첫 번째 시행 시각화
            self.visualize_eeg_data(X, Y, trial_idx=0)