# EEG to Speech 변환 프로젝트

이 저장소는 EEG 데이터에서 음성을 생성하는 파이프라인을 구현합니다. Perceptogram 코드를 기반으로 하여 CLAP, Whisper, AudioLDM2와 같은 최신 오디오 모델을 활용한 EEG2Speech 파이프라인을 구축했습니다.

## 프로젝트 구조

```
eeg2speech/
├── code/                  # 소스 코드
│   ├── data_preprocessing.py     # 데이터 전처리 모듈
│   ├── eeg_encoder.py            # EEG 인코더 모델
│   ├── audio_generator.py        # 오디오 생성 모델
│   ├── eeg2speech_pipeline.py    # 전체 파이프라인 통합
│   └── pipeline_design.md        # 파이프라인 설계 문서
├── data/                  # 데이터 디렉토리
│   └── inner_speech/            # Inner Speech 데이터셋
├── models/                # 학습된 모델 저장 디렉토리
├── notebooks/             # Jupyter 노트북
│   └── eeg2speech_tutorial.ipynb # 튜토리얼 노트북
└── outputs/               # 출력 결과 디렉토리
```

## 파이프라인 개요

이 프로젝트는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **데이터 전처리**: Inner Speech 데이터셋의 EEG 데이터를 로드하고 전처리합니다.
2. **이중 경로 EEG 인코더**: VAE 인코더와 CLAP 기반 대조적 인코더를 결합하여 EEG 데이터를 잠재 표현으로 변환합니다.
3. **이중 경로 오디오 생성기**: Whisper 기반 디코더와 AudioLDM2 기반 생성기를 결합하여 잠재 표현에서 오디오를 생성합니다.
4. **학습 및 평가**: 다양한 손실 함수를 결합하여 end-to-end 파이프라인을 학습하고 평가합니다.

## 설치 방법

필요한 패키지를 설치합니다:

```bash
pip install torch torchaudio numpy matplotlib librosa scikit-learn mne tqdm
```

Inner Speech 데이터셋을 다운로드합니다:

```bash
cd data
git clone https://github.com/N-Nieto/Inner_Speech_Dataset.git
```

## 사용 방법

### 튜토리얼 노트북 실행

Jupyter 노트북을 통해 전체 파이프라인을 실행할 수 있습니다:

```bash
jupyter notebook notebooks/eeg2speech_tutorial.ipynb
```

### 코드 실행

각 모듈을 개별적으로 실행할 수도 있습니다:

```bash
# 데이터 전처리
python code/data_preprocessing.py

# EEG 인코더 테스트
python code/eeg_encoder.py

# 오디오 생성기 테스트
python code/audio_generator.py

# 전체 파이프라인 테스트
python code/eeg2speech_pipeline.py
```

## 주요 기능

### 데이터 전처리 (`data_preprocessing.py`)

- EEG 데이터 로드 및 전처리
- 시간 윈도우 선택, 필터링, 재참조, 정규화
- 데이터 증강 기법 적용
- 데이터셋 분할 (학습, 검증, 테스트)

### EEG 인코더 (`eeg_encoder.py`)

- VAE 인코더: EEG 데이터의 구조적 표현 학습
- CLAP 기반 대조적 인코더: EEG와 오디오 간의 의미적 매핑 학습
- 이중 경로 인코더: 두 인코더의 출력을 결합

### 오디오 생성기 (`audio_generator.py`)

- Whisper 기반 디코더: 멜 스펙트로그램 생성 및 오디오 변환
- AudioLDM2 기반 생성기: 확산 모델을 통한 고품질 오디오 생성
- 이중 경로 생성기: 두 모델의 출력을 결합

### 전체 파이프라인 (`eeg2speech_pipeline.py`)

- EEG 인코더와 오디오 생성기 통합
- 학습 및 평가 코드
- 결과 시각화 및 분석

## 참고 자료

- [Perceptogram 저장소](https://github.com/desa-lab/Perceptogram)
- [brain2speech 저장소](https://github.com/jml226/brain2speech)
- [Inner Speech 데이터셋](https://www.nature.com/articles/s41597-022-01147-2)

## 향후 개선 방향

1. **데이터 증강**: 더 다양한 데이터 증강 기법을 적용하여 모델의 일반화 성능을 향상시킵니다.
2. **사전 학습 모델 활용**: 사전 학습된 Whisper, CLAP, AudioLDM2 모델을 활용하여 전이 학습을 수행합니다.
3. **하이퍼파라미터 최적화**: 그리드 서치나 베이지안 최적화를 통해 최적의 하이퍼파라미터를 찾습니다.
4. **실제 음성 데이터 페어링**: EEG 데이터와 실제 음성 데이터를 페어링하여 모델을 학습합니다.
