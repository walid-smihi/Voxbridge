from __future__ import annotations

import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyaudio


CHUNK = 1024
THRESHOLD = 300
TRAILING_SILENCE_SEC = 0.20
MAX_SEGMENT_SEC = 3.0
OVERLAP_SEC = 0.35
FORMAT = pyaudio.paInt16


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int


@dataclass
class RunOptions:
    mode: str  # transcription | traduction
    source: str  # loopback | device
    device_index: int
    model_path: Path
    use_cuda: bool


def discover_models(project_root: Path) -> list[str]:
    model_dir = project_root / "whisper.cpp" / "models"
    if not model_dir.exists():
        return []

    names = []
    for path in sorted(model_dir.glob("ggml*.bin")):
        if path.name.startswith("for-tests-"):
            continue
        names.append(path.name)
    return names


def build_whisper_cli_path(project_root: Path) -> Path:
    whisper_dir = project_root / "whisper.cpp"
    candidates = [
        whisper_dir / "build-cuda" / "bin" / "Release" / "whisper-cli.exe",
        whisper_dir / "build" / "bin" / "Release" / "whisper-cli.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def list_input_devices() -> list[DeviceInfo]:
    p = pyaudio.PyAudio()
    devices: list[DeviceInfo] = []
    try:
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            max_in = int(d.get("maxInputChannels", 0))
            if max_in <= 0:
                continue
            devices.append(
                DeviceInfo(
                    index=i,
                    name=str(d.get("name", "")),
                    max_input_channels=max_in,
                    default_sample_rate=int(d.get("defaultSampleRate", 44100)),
                )
            )
    finally:
        p.terminate()
    return devices


def build_translator() -> Callable[[str], str]:
    import argostranslate.translate

    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    to_lang = next((lang for lang in installed_languages if lang.code == "fr"), None)
    if from_lang and to_lang:
        return from_lang.get_translation(to_lang).translate
    return lambda text: text


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        project_root: Path,
        options: RunOptions,
        on_event: Callable[[str, str], None],
    ) -> None:
        super().__init__(daemon=True)
        self.project_root = project_root
        self.options = options
        self.on_event = on_event
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def emit(self, kind: str, message: str) -> None:
        self.on_event(kind, message)

    def _get_loopback_backend(self):
        try:
            import pyaudiowpatch as pa_lib  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loopback demande pyaudiowpatch: pip install pyaudiowpatch") from exc
        return pa_lib

    def _get_loopback_device(self, p, pa_lib):
        if not hasattr(pa_lib, "paWASAPI"):
            raise RuntimeError("Backend audio sans support WASAPI loopback")

        wasapi = p.get_host_api_info_by_type(pa_lib.paWASAPI)
        default_output_idx = int(wasapi["defaultOutputDevice"])
        default_output = p.get_device_info_by_index(default_output_idx)

        if default_output.get("isLoopbackDevice", False):
            return default_output

        if hasattr(p, "get_loopback_device_info_generator"):
            default_name = str(default_output.get("name", ""))
            for loop_dev in p.get_loopback_device_info_generator():
                if default_name in str(loop_dev.get("name", "")):
                    return loop_dev

        raise RuntimeError("Impossible de trouver le device loopback WASAPI")

    def run(self) -> None:
        whisper_cli = build_whisper_cli_path(self.project_root)
        if not whisper_cli.exists():
            self.emit("error", f"whisper-cli introuvable: {whisper_cli}")
            self.emit("stopped", "")
            return

        if not self.options.model_path.exists():
            self.emit("error", f"Modele introuvable: {self.options.model_path}")
            self.emit("stopped", "")
            return

        translate_fn: Callable[[str], str] | None = None
        if self.options.mode == "traduction":
            try:
                translate_fn = build_translator()
            except Exception as exc:
                self.emit("error", f"Erreur initialisation traduction: {exc}")
                self.emit("stopped", "")
                return

        pa_lib = pyaudio
        if self.options.source == "loopback":
            try:
                pa_lib = self._get_loopback_backend()
            except Exception as exc:
                self.emit("error", str(exc))
                self.emit("stopped", "")
                return

        p = pa_lib.PyAudio()
        stream = None
        channels = 2
        rate = 44100

        try:
            if self.options.source == "loopback":
                loop_dev = self._get_loopback_device(p, pa_lib)
                channels = max(1, min(2, int(loop_dev.get("maxInputChannels", 2))))
                rate = int(loop_dev.get("defaultSampleRate", 44100))
                stream = p.open(
                    format=pa_lib.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=int(loop_dev["index"]),
                    frames_per_buffer=CHUNK,
                )
                self.emit("status", f"Capture: {loop_dev.get('name', '')} (loopback)")
            else:
                info = p.get_device_info_by_index(int(self.options.device_index))
                channels = max(1, min(2, int(info.get("maxInputChannels", 1))))
                rate = int(info.get("defaultSampleRate", 44100))
                stream = p.open(
                    format=pa_lib.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=int(self.options.device_index),
                    frames_per_buffer=CHUNK,
                )
                self.emit("status", f"Capture: {info.get('name', '')} (index {self.options.device_index})")
        except Exception as exc:
            self.emit("error", f"Erreur audio: {exc}")
            if stream is not None:
                stream.close()
            p.terminate()
            self.emit("stopped", "")
            return

        log_file = self.project_root / "logs.txt"
        wav_file = self.project_root / "temp_audio.wav"
        wav_16k = self.project_root / "temp_audio_16k.wav"

        max_segment_chunks = max(1, int(rate * MAX_SEGMENT_SEC / CHUNK))
        overlap_chunks = max(0, int(rate * OVERLAP_SEC / CHUNK))
        silence_chunks = max(1, int(rate * TRAILING_SILENCE_SEC / CHUNK))

        self.emit("status", f"Whisper CLI: {whisper_cli}")
        self.emit("status", f"Streaming: max {MAX_SEGMENT_SEC:.1f}s, overlap {OVERLAP_SEC:.2f}s")
        self.emit("status", "Worker demarre")

        carry_frames: list[bytes] = []
        last_transcription = ""

        try:
            while not self.stop_event.is_set():
                frames: list[bytes] = carry_frames.copy()
                carry_frames = []

                heard_voice = False
                silence_counter = 0
                force_split = False

                while not self.stop_event.is_set():
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    amplitude = int(np.max(np.abs(audio_data))) if audio_data.size else 0
                    is_voice = amplitude > THRESHOLD

                    if is_voice:
                        heard_voice = True
                        silence_counter = 0
                    elif heard_voice:
                        silence_counter += 1

                    if heard_voice or frames:
                        frames.append(data)

                    if len(frames) >= max_segment_chunks:
                        force_split = True
                        break

                    if heard_voice and silence_counter >= silence_chunks:
                        break

                if self.stop_event.is_set():
                    break

                if not heard_voice:
                    continue

                if force_split and overlap_chunks > 0 and len(frames) > overlap_chunks:
                    carry_frames = frames[-overlap_chunks:]

                with wave.open(str(wav_file), "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(p.get_sample_size(pa_lib.paInt16))
                    wf.setframerate(rate)
                    wf.writeframes(b"".join(frames))

                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_file), "-ar", "16000", str(wav_16k)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

                whisper_command = [
                    str(whisper_cli),
                    "-m",
                    str(self.options.model_path),
                    "-f",
                    str(wav_16k),
                    "-l",
                    "en",
                    "-nt",
                ]
                if not self.options.use_cuda:
                    whisper_command.append("-ng")

                result = subprocess.run(whisper_command, capture_output=True, text=True, timeout=30)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(result.stdout + "\n" + result.stderr + "\n")

                if result.returncode != 0:
                    self.emit("error", f"Whisper error: {result.stderr.strip()}")
                    continue

                transcription = " ".join(result.stdout.split())
                if not transcription:
                    self.emit("status", "Aucune transcription obtenue.")
                    continue

                if transcription == last_transcription:
                    continue
                last_transcription = transcription

                self.emit("transcription", transcription)

                if translate_fn is not None:
                    try:
                        translation = translate_fn(transcription)
                        self.emit("translation", translation)
                    except Exception as exc:
                        self.emit("error", f"Erreur traduction: {exc}")

                time.sleep(0.05)
        except Exception as exc:
            self.emit("error", f"Worker exception: {exc}")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            p.terminate()
            self.emit("stopped", "")
