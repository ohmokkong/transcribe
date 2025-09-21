# `trans.py` — 음성 파일 전사 및 화자 분리

## 개요

`trans.py`는 음성 파일(mp3, wav 등)을 텍스트로 전사하고, 화자를 분리하여 대화록 형태의 텍스트 파일을 생성하는 스크립트입니다.

- **입력:** 음성 파일 (`mp3`, `wav`, `m4a`, `flac`, `ogg`, `aac`)
- **출력:** 입력 파일명과 동일한 `.txt` 파일
- **화자 분리:** 2명 이상이면 `A`/`B`/`C` 라벨 부여, 1명이면 문장 단위 분리
- **제목:** 입력 파일명
- **요약/후기 없음 (핵심 기능 집중판)**

## 권장 환경

- Python 3.10 ~ 3.12 (Windows 환경에서 ctranslate2 휠 호환성)
- ffmpeg 설치 필요
  - Windows: `choco install ffmpeg` (Chocolatey가 설치되어 있어야 합니다.)
  - macOS: `brew install ffmpeg` (Homebrew가 설치되어 있어야 합니다.)
- Hugging Face 토큰 필요 (`PYANNOTE_TOKEN` 환경 변수 설정)
  - `pyannote/speaker-diarization-3.1` 모델 사용을 위해 Hugging Face의 모델 페이지에서 라이선스 동의 후, 사용자 토큰을 발급받아 환경 변수에 설정해야 합니다.

## 설치

```bash
# 가상환경 생성 (예: Python 3.12)
py -3.12 -m venv .venv

# 가상환경 활성화 (PowerShell 기준)
.venv\Scripts\Activate.ps1

# requirements 설치
pip install -r requirements.txt

# (선택) GPU 사용 시, PyTorch를 CUDA 버전에 맞게 별도 설치하는 것을 권장합니다.
# https://pytorch.org/get-started/locally/

# Tip: 다음 명령으로 requirements.txt 내용을 직접 생성할 수 있습니다.
# python trans.py --emit-reqs > requirements.txt
```

## requirements.txt 예시

```text
whisperx==3.4.2
torch>=2.1.0 # CUDA 사용 시 환경에 맞는 버전 설치 권장
torchaudio>=2.1.0
pyannote.audio>=3.1.1
numpy>=1.24
ffmpeg-python>=0.2.0
soundfile>=0.12.1
rich>=13.7.0  # 선택
```

## 사용법

```bash
python trans.py <음성파일> <언어(ko|en|auto)> [옵션]
```

### 예시

```bash
# 한국어 음성 파일을 전사합니다.
python trans.py sample.mp3 ko

# 영어 음성 파일을 전사합니다.
python trans.py meeting.wav en

# 언어를 자동으로 감지하여 전사합니다.
python trans.py interview.m4a auto

# 2명 화자 고정
python trans.py debate.mp3 ko --num-speakers 2
```

### 주요 옵션

- `--outdir <폴더>`: 출력 폴더를 지정합니다. (기본값: 입력 파일과 같은 폴더)
- `--device <장치>`: 디바이스를 선택합니다. (`auto`, `cpu`, `cuda`)
- `--model <모델>`: WhisperX 모델 크기를 선택합니다. (`tiny`, `base`, `small`, `medium`, `large-v2`)
- `--hf_token <토큰>`: Hugging Face 토큰을 직접 지정합니다. (없으면 `PYANNOTE_TOKEN` 환경변수 사용)
- `--num-speakers <수>`: 화자 수를 고정합니다. (예: `2`)
- `--min-speakers <수>`, `--max-speakers <수>`: 최소/최대 화자 수를 지정하여 분리 성능을 높입니다.
- `--emit-reqs`: `requirements.txt` 내용을 출력하고 종료합니다.
- `--self-test`: 내장된 자체 테스트를 실행합니다.

## 출력 예시

입력: `sample.mp3`
출력: `sample.txt`

```text
제목: sample
추정 화자 수: 2

대화록:
[00:00:01] A: 안녕하세요. 오늘 토론에 참여해 주셔서 감사합니다.
[00:00:05] B: 네, 반갑습니다. 초대해 주셔서 감사합니다.
...
```

## 문제 해결

- **SystemExit: 2 발생** → 인자 누락. 반드시 `<음성파일> <언어>` 지정.
- **화자 분리 실패** → `PYANNOTE_TOKEN` 환경 변수가 올바르게 설정되었는지 확인하세요. 음질이 낮은 경우, 모델이 화자를 정확히 인식하지 못할 수 있습니다. 아래 명령어로 음질을 개선해 보세요.
  - `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav` (샘플링 레이트를 16kHz, 모노 채널로 변환)
- **Python 3.13 이상에서 설치 실패** → `ctranslate2` 패키지의 휠 파일이 Python 3.12까지만 제공됩니다. Python 3.12로 가상환경을 구성하세요.

## License

MIT
