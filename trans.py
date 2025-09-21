#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trans — 음성(mp3 등) → 텍스트 전사 + 화자 분리 (dotenv 패치 적용판)

- 실행 형식:  python trans.py <음성파일> <언어(ko|en|auto)> [옵션]
  예) python trans.py sample.mp3 ko
- 제목은 입력 음성 파일명 그대로 사용
- 산출물은 텍스트 1개 파일만 생성: <파일명>.txt
- 화자 분리:
  - diarization 성공 시: A/B/C 라벨
  - 실패/미설정 시: 단일 화자 폴백(문장 단위 분리)
- PYANNOTE_TOKEN 우선순위:
  1) CLI 인자 --hf_token
  2) 환경변수( .env 로드됨 ) PYANNOTE_TOKEN
  3) 없으면 diarization 스킵(경고)
"""

from __future__ import annotations
import os
import sys
import re
import argparse
import subprocess
import tempfile
from datetime import datetime
from collections import defaultdict

# === dotenv 자동 로딩 ===
try:
    from dotenv import load_dotenv  # requirements: python-dotenv
    load_dotenv()  # .env 파일을 발견하면 환경변수로 로드
except Exception:
    # dotenv 미설치여도 치명적이지 않음(환경변수/CLI로 대체 가능)
    pass

# ====== 상수/유틸 ======
SUPPORTED_PY_MIN = (3, 10, 0)
SUPPORTED_PY_MAX = (3, 12, 99)
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100MB
DEFAULT_OUTDIR = None

def python_version_supported(cur: tuple[int, int, int]) -> bool:
    mmv = (cur[0], cur[1], cur[2] if len(cur) >= 3 else 0)
    return SUPPORTED_PY_MIN <= mmv <= SUPPORTED_PY_MAX

def log(msg: str): print(f"[INFO] {msg}")
def warn(msg: str): print(f"[경고] {msg}")
def err(msg: str): print(f"[오류] {msg}", file=sys.stderr)

def seconds_to_hhmmss(t: float) -> str:
    if t is None or t < 0: t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def seconds_to_srt_ts(t: float) -> str:
    if t is None or t < 0: t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_to_vtt_ts(t: float) -> str:
    if t is None or t < 0: t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def format_pyver_warning(cur: tuple[int, int, int]) -> str:
    return (("현재 Python 버전은 %d.%d.%d 입니다. WhisperX/ctranslate2 윈도우 휠 호환성상 "
             "Python 3.10~3.12 사용을 권장합니다.\n"
             "- Windows 권장:\n"
             "  1) Python 3.12.x 설치\n"
             "  2) 가상환경: 'py -3.12 -m venv .venv'\n"
             "  3) 활성화 후: 'pip install -r requirements.txt'\n") % cur)

def ensure_supported_python() -> bool:
    cur = sys.version_info[:3]
    if not python_version_supported(cur):
        warn(format_pyver_warning(cur))
        return False
    return True

def which(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def ensure_ffmpeg() -> bool:
    if which("ffmpeg") and which("ffprobe"):
        return True
    warn("ffmpeg/ffprobe가 감지되지 않았습니다. 설치 후 다시 시도하세요. (choco install ffmpeg)")
    return False

# ====== WhisperX / pyannote ======
def import_or_fail(module: str, friendly: str):
    try:
        return __import__(module)
    except ImportError:
        err(f"필요 모듈 미설치: {module} — {friendly}")
        raise

def select_device(arg: str) -> str:
    if arg == "cpu": return "cpu"
    if arg == "cuda": return "cuda"
    try:
        torch = import_or_fail("torch", "pip install torch")
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def load_whisperx_model(device: str, model_size: str, language: str):
    whisperx = import_or_fail("whisperx", "pip install -U whisperx")
    torch = import_or_fail("torch", "pip install torch")
    compute_type = "float16" if (device == "cuda" and torch.cuda.is_available()) else "int8"
    log(f"WhisperX 모델 로딩: size={model_size}, device={device}, compute={compute_type}, lang={language}")
    asr_model = whisperx.load_model(model_size, device, compute_type=compute_type,
                                    language=None if language == "auto" else language)
    return whisperx, asr_model

def run_whisperx(whisperx, asr_model, audio_path: str, device: str, language: str):
    log("WhisperX 전사 실행...")
    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(audio, batch_size=32 if device == "cuda" else 8)
    log("단어 정렬(align) 실행...")
    model_a, metadata = whisperx.load_align_model(language_code=result.get("language", "en"), device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    return aligned  # {'segments': [...]}

def run_diarization(audio_path: str, hf_token: str | None = None,
                    num_speakers: int | None = None,
                    min_speakers: int | None = None,
                    max_speakers: int | None = None):
    from pyannote.audio import Pipeline
    import ffmpeg

    if not hf_token:
        hf_token = os.environ.get("PYANNOTE_TOKEN")
    log("pyannote diarization 준비...")
    if not hf_token:
        raise RuntimeError("PYANNOTE_TOKEN 미설정 (CLI --hf_token 또는 .env/환경변수에 설정 필요)")

    temp_wav_path = None
    try:
        diar_input_path = audio_path
        if not audio_path.lower().endswith(".wav"):
            log("diarization을 위해 오디오를 임시 wav 파일로 변환합니다...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_path = tmp.name
            (ffmpeg.input(audio_path)
             .output(temp_wav_path, acodec='pcm_s16le', ac=1, ar='16000')
             .run(quiet=True, overwrite_output=True))
            diar_input_path = temp_wav_path

        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diar_kwargs = {}
        if isinstance(num_speakers, int) and num_speakers > 0:
            diar_kwargs["num_speakers"] = int(num_speakers)
        else:
            if isinstance(min_speakers, int) and min_speakers > 0: diar_kwargs["min_speakers"] = int(min_speakers)
            if isinstance(max_speakers, int) and max_speakers > 0: diar_kwargs["max_speakers"] = int(max_speakers)
        if diar_kwargs: log(f"diarization 힌트: {diar_kwargs}")
        diar = pipe(diar_input_path, **diar_kwargs)
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    diar_segments = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        diar_segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    speakers = sorted(list({d["speaker"] for d in diar_segments}))
    log(f"diarization 결과: segments={len(diar_segments)}, speakers={len(speakers)}")
    return diar_segments, speakers

# ====== 병합/라벨링/분할 ======
def overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def map_speakers_to_labels(speakers):
    alphabet = [chr(ord('A') + i) for i in range(26)]
    mapping = {}
    for i, spk in enumerate(speakers):
        mapping[spk] = alphabet[i] if i < len(alphabet) else f"SPK{i+1}"
    return mapping

def assign_speakers_to_whisper_segments(whisper_segments, diar_segments):
    if not diar_segments:
        for seg in whisper_segments:
            seg["speaker"] = None
        return whisper_segments, []
    for seg in whisper_segments:
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        votes = defaultdict(float)
        for d in diar_segments:
            ov = overlap(s, e, d["start"], d["end"])
            if ov > 0:
                votes[d["speaker"]] += ov
        seg["speaker"] = max(votes.items(), key=lambda x: x[1])[0] if votes else None
    used = sorted(list({seg["speaker"] for seg in whisper_segments if seg.get("speaker")}))
    return whisper_segments, used

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")
def sentence_split(text: str) -> list[str]:
    pieces = _SENT_SPLIT_RE.split((text or "").strip())
    return [p.strip() for p in pieces if p.strip()]

def explode_to_sentence_level(segments):
    """단일 화자 입력일 때 문장 단위로 쪼개고, 단어 타임스탬프로 근사 분배."""
    sentence_entries = []
    for seg in segments:
        words = seg.get("words") or []
        sent_texts = sentence_split(seg.get("text", "") or "")
        if not sent_texts:
            sentence_entries.append({
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", seg.get("start", 0.0) or 0.0)),
                "text": seg.get("text", "") or "",
                "speaker": None,
            })
            continue
        if not words:
            total_sents = len(sent_texts)
            seg_start = float(seg.get("start", 0.0) or 0.0)
            seg_end = float(seg.get("end", seg_start))
            span = max(0.0, seg_end - seg_start)
            step = span / total_sents if total_sents else 0.0
            for i, st in enumerate(sent_texts):
                st_start = seg_start + i * step
                st_end = seg_start + (i + 1) * step if i < total_sents - 1 else seg_end
                sentence_entries.append({"start": st_start, "end": st_end, "text": st, "speaker": None})
            continue
        n_words = len(words)
        base = max(1, n_words // len(sent_texts))
        cursor = 0
        for i, st in enumerate(sent_texts):
            if i == len(sent_texts) - 1:
                w_slice = words[cursor:]
            else:
                w_slice = words[cursor: cursor + base]
            if w_slice:
                start_t = w_slice[0].get("start", seg.get("start", 0.0) or 0.0)
                end_t = w_slice[-1].get("end", start_t)
            else:
                start_t = seg.get("start", 0.0) or 0.0
                end_t = seg.get("end", start_t)
            sentence_entries.append({
                "start": float(start_t) if start_t is not None else 0.0,
                "end": float(end_t) if end_t is not None else float(start_t) if start_t else 0.0,
                "text": st,
                "speaker": None,
            })
            cursor += base
    return sentence_entries

# ====== 출력(TXT) ======
def export_txt_dialogue(entries: list, outpath: str, title: str, est_speakers: int, multi_speaker: bool):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(f"제목: {title}\n")
        f.write(f"추정 화자 수: {est_speakers}\n\n")
        f.write("대화록:\n")
        for e in entries:
            t = seconds_to_hhmmss(e['start'])
            text = (e.get('text') or '').strip()
            if multi_speaker and e.get('speaker'):
                f.write(f"[{t}] {e['speaker']}: {text}\n")
            else:
                f.write(f"[{t}] {text}\n")
    log(f"TXT 저장: {outpath}")

# ====== 입력 검증 ======
def validate_input(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"파일 용량 초과(≤100MB): {size/1024/1024:.1f}MB")
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        warn(f"비표준 확장자({ext}) — ffmpeg가 지원하면 계속 진행합니다.")

# ====== requirements 도우미 ======
def requirements_helper() -> str:
    return (
        """
whisperx==3.4.2
torch>=2.1.0
torchaudio>=2.1.0
pyannote.audio>=3.1.1
numpy>=1.24
ffmpeg-python>=0.2.0
soundfile>=0.12.1
python-dotenv>=1.0.1
        """.strip()
    )

# ====== SELF TESTS ======
def run_self_test():
    tmpdir = tempfile.mkdtemp(prefix="trans_selftest_")
    print(f"[SELF-TEST] 임시 폴더: {tmpdir}")
    assert seconds_to_srt_ts(0.0) == "00:00:00,000"
    assert seconds_to_vtt_ts(1.234) == "00:00:01.234"
    sents_ko = sentence_split("안녕하세요. 반갑습니다! 테스트인가요? 네.")
    assert len(sents_ko) == 4
    segs = [{
        "start": 0.0, "end": 4.0,
        "text": "첫 문장입니다. 두 번째 문장입니다.",
        "words": [
            {"start": 0.0, "end": 1.0, "word": "첫"},
            {"start": 1.0, "end": 2.0, "word": "문장입니다."},
            {"start": 2.0, "end": 3.0, "word": "두"},
            {"start": 3.0, "end": 4.0, "word": "번째"},
        ],
    }]
    exploded = explode_to_sentence_level(segs)
    assert len(exploded) >= 1
    print("[SELF-TEST] OK")

# ====== CLI ======
def print_usage_examples(parser: argparse.ArgumentParser):
    print("\n사용법:")
    parser.print_usage()
    print("""
예시:
  python trans.py sample.mp3 ko
  python trans.py meeting.wav en
  python trans.py interview.m4a auto --num-speakers 2
옵션:
  --outdir C:\\Temp\\
  --device auto|cpu|cuda
  --model tiny|base|small|medium|large-v2
  --hf_token <HF_TOKEN>
  --num-speakers 2
  --min-speakers 1 --max-speakers 3
  --emit-reqs
  --self-test
""".rstrip())

def validate_cli_args(args, parser: argparse.ArgumentParser) -> bool:
    if not args:
        return False
    if getattr(args, 'input_path', None) and getattr(args, 'language', None):
        return True
    warn("필수 인자 누락: <음성파일> <언어(ko|en|auto)> 를 제공하세요.")
    print_usage_examples(parser)
    return False

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True, prog="trans", description="음성 전사 + 화자 분리 (dotenv 패치판)")
    p.add_argument("input_path", nargs="?", help="입력 오디오 파일 (mp3, wav, m4a, flac, ogg, aac)")
    p.add_argument("language", nargs="?", choices=["ko", "en", "auto"], help="전사 언어 (ko|en|auto)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="출력 폴더 (기본: 입력 파일과 같은 폴더)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="사용 디바이스")
    p.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v2"], help="WhisperX 모델")
    p.add_argument("--hf_token", default=None, help="Hugging Face 토큰 (pyannote)")
    p.add_argument("--num-speakers", type=int, default=None, help="화자 수 고정")
    p.add_argument("--min-speakers", type=int, default=None, help="최소 화자 수 힌트")
    p.add_argument("--max-speakers", type=int, default=None, help="최대 화자 수 힌트")
    p.add_argument("--emit-reqs", action="store_true", help="requirements.txt 내용 출력 후 종료")
    p.add_argument("--self-test", action="store_true", help="내장 테스트 실행 후 종료")
    return p

# ====== MAIN ======
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.emit_reqs:
        print(requirements_helper())
        return 0

    if args.self_test:
        run_self_test()
        return 0

    if not validate_cli_args(args, parser):
        return 0

    ensure_supported_python()
    ensure_ffmpeg()

    input_path = os.path.abspath(args.input_path)
    validate_input(input_path)
    title = os.path.splitext(os.path.basename(input_path))[0]

    device = select_device(args.device)
    whisperx, asr_model = load_whisperx_model(device=device, model_size=args.model, language=args.language)

    aligned = run_whisperx(whisperx, asr_model, input_path, device, args.language)
    whisper_segments = aligned.get("segments", []) or []

    # === 토큰 우선순위: CLI > ENV(.env) ===
    resolved_hf_token = args.hf_token or os.environ.get("PYANNOTE_TOKEN")

    # diarization 시도
    diar_segments, diar_speakers = [], []
    multi_speaker = False
    try:
        diar_segments, diar_speakers = run_diarization(
            input_path,
            hf_token=resolved_hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        whisper_segments, used_speakers = assign_speakers_to_whisper_segments(whisper_segments, diar_segments)
        diar_speakers = used_speakers or diar_speakers
        num_found_speakers = len(diar_speakers)
        multi_speaker = num_found_speakers >= 2

        # 사용자가 화자 수를 지정하지 않았는데, 2명 미만이 감지된 경우 경고
        if not multi_speaker and args.num_speakers is None:
            warn(f"화자 분리 결과, 2명 미만의 화자({num_found_speakers}명)가 감지되어 단일 화자 모드로 처리합니다.")
            warn("정확한 화자 분리가 필요하다면 --num-speakers 2 와 같이 화자 수를 직접 지정해 보세요.")

    except ImportError:
        warn("diarization 건너뜀: 'pyannote.audio' 또는 'python-dotenv' 라이브러리가 설치되지 않았습니다.")
        warn("화자 분리 기능을 사용하려면 'pip install -r requirements.txt'로 설치하세요.")
        multi_speaker = False
    except RuntimeError as e:
        warn(f"diarization 건너뜀: {e}")
        warn("Hugging Face 토큰(--hf_token)이 올바른지, pyannote.audio 모델 다운로드에 문제가 없는지 확인하세요.")
        multi_speaker = False
    except Exception as e:
        warn(f"diarization 중 예상치 못한 오류 발생, 건너뜀: {e}")
        if "'NoneType' object has no attribute 'eval'" in str(e):
            warn("이 오류는 pyannote 모델 다운로드 실패 시 발생할 수 있습니다.")
            warn("Hugging Face 웹사이트에서 pyannote/speaker-diarization-3.1 및 pyannote/segmentation-3.0 모델의 라이선스 동의를 했는지 확인하세요.")
        multi_speaker = False

    # 출력 엔트리 구성
    entries = []
    est_speakers = 1
    if multi_speaker:
        mapping = map_speakers_to_labels(diar_speakers)
        est_speakers = max(2, len(diar_speakers))
        for seg in whisper_segments:
            entries.append({
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": seg.get("text", "") or "",
                "speaker": mapping.get(seg.get("speaker")) if seg.get("speaker") else None,
            })
    else:
        entries = explode_to_sentence_level(whisper_segments)
        est_speakers = 1

    entries.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    for e in entries:
        e["text"] = re.sub(r"\s+", " ", (e.get("text") or "").strip())

    outdir = args.outdir
    if outdir is None:
        # --outdir 옵션이 없으면 입력 파일의 디렉토리를 사용
        outdir = os.path.dirname(input_path)
    if not outdir: # dirname이 비어있는 경우 (e.g. 현재 디렉토리의 파일)
        outdir = "."
    outpath = os.path.join(outdir, f"{title}.txt")
    export_txt_dialogue(entries, outpath, title=title, est_speakers=est_speakers, multi_speaker=multi_speaker)
    log("완료")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        err("중단됨 (Ctrl+C)")
        sys.exit(130)
