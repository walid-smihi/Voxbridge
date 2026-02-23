from __future__ import annotations

import os
import sys
import time
import wave
import subprocess
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyaudio


CHUNK = 1024
THRESHOLD = 300
DEFAULT_CHANNELS = 2
DEFAULT_RATE = 44100
DEVICE_INDEX = 2
MAX_SILENCE = 0.1


@dataclass
class CliFlags:
    minimal: bool
    loopback: bool
    list_devices: bool


def parse_flags(argv: list[str]) -> CliFlags:
    return CliFlags(
        minimal="--minimal" in argv,
        loopback="--loopback" in argv,
        list_devices="--list-devices" in argv,
    )


def select_audio_backend(loopback_mode: bool):
    pa_lib = pyaudio
    if loopback_mode:
        try:
            import pyaudiowpatch as pa_lib  # type: ignore
        except ImportError:
            print("Mode --loopback indisponible: installe pyaudiowpatch")
            print("Commande: pip install pyaudiowpatch")
            sys.exit(1)
    return pa_lib


def resolve_whisper_paths(script_dir: str) -> tuple[str, str]:
    whisper_dir = os.path.join(script_dir, "whisper.cpp")

    whisper_cli = os.path.join(whisper_dir, "build-cuda", "bin", "Release", "whisper-cli.exe")
    if not os.path.exists(whisper_cli):
        whisper_cli = os.path.join(whisper_dir, "build", "bin", "Release", "whisper-cli.exe")

    model_path = os.path.join(whisper_dir, "models", "ggml-tiny.en.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(whisper_dir, "models", "ggml-tiny.en-q5_1.bin")

    if not os.path.exists(whisper_cli):
        print("Erreur: whisper-cli.exe introuvable.")
        print("Attendu: whisper.cpp/build-cuda/bin/Release/whisper-cli.exe")
        print("ou:      whisper.cpp/build/bin/Release/whisper-cli.exe")
        sys.exit(1)

    if not os.path.exists(model_path):
        print("Erreur: aucun modele Whisper trouve.")
        print("Place un modele ici: whisper.cpp/models/")
        print("Exemple attendu: ggml-tiny.en.bin ou ggml-tiny.en-q5_1.bin")
        print("Commande utile: cd whisper.cpp/models && ./download-ggml-model.cmd tiny.en")
        sys.exit(1)

    return whisper_cli, model_path


def print_devices(p):
    print("index | maxInput | maxOutput | defaultRate | name")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        print(
            f"{i:>5} | {int(d.get('maxInputChannels', 0)):>8} | "
            f"{int(d.get('maxOutputChannels', 0)):>9} | "
            f"{int(d.get('defaultSampleRate', 0)):>11} | {d.get('name', '')}"
        )


def get_wasapi_loopback_device(p, pa_lib):
    if not hasattr(pa_lib, "paWASAPI"):
        raise RuntimeError("Backend sans support WASAPI")

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


def open_stream(p, pa_lib, fmt, loopback_mode: bool):
    if loopback_mode:
        loop_dev = get_wasapi_loopback_device(p, pa_lib)
        channels = max(1, min(2, int(loop_dev.get("maxInputChannels", 2))))
        rate = int(loop_dev.get("defaultSampleRate", DEFAULT_RATE))
        stream = p.open(
            format=fmt,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(loop_dev["index"]),
            frames_per_buffer=CHUNK,
        )
        return stream, channels, rate, f"{loop_dev.get('name', '')} (loopback)"

    stream = p.open(
        format=fmt,
        channels=DEFAULT_CHANNELS,
        rate=DEFAULT_RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )
    try:
        info = p.get_device_info_by_index(DEVICE_INDEX)
        name = str(info.get("name", ""))
    except Exception:
        name = f"index {DEVICE_INDEX}"
    return stream, DEFAULT_CHANNELS, DEFAULT_RATE, name


def build_translator_en_fr() -> Callable[[str], str]:
    import argostranslate.translate

    logging.getLogger("stanza").setLevel(logging.ERROR)
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    to_lang = next((lang for lang in installed_languages if lang.code == "fr"), None)
    if from_lang and to_lang:
        return from_lang.get_translation(to_lang).translate
    return lambda text: text


def run_cli(mode: str) -> int:
    # mode: transcription | traduction
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    flags = parse_flags(sys.argv)

    pa_lib = select_audio_backend(flags.loopback)
    fmt = pa_lib.paInt16

    p = pa_lib.PyAudio()
    stream = None

    try:
        if flags.list_devices:
            print_devices(p)
            return 0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        whisper_cli, model_path = resolve_whisper_paths(script_dir)

        translate_text = None
        if mode == "traduction":
            translate_text = build_translator_en_fr()

        try:
            stream, channels, rate, device_name = open_stream(p, pa_lib, fmt, flags.loopback)
        except Exception as e:
            print(f"Erreur ouverture audio: {e}")
            return 1

        log_file = os.path.join(script_dir, "logs.txt")

        if not flags.minimal:
            if mode == "traduction":
                print("Ecoute, transcription et traduction en cours...")
            else:
                print("Ecoute et transcription en cours...")
            print(f"Whisper CLI: {whisper_cli}")
            print(f"Capture device: {device_name}")

        while True:
            frames = []
            silence_counter = 0

            if not flags.minimal:
                print("Enregistrement en cours...")

            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.max(np.abs(audio_data))

                if amplitude > THRESHOLD:
                    frames.append(data)
                    silence_counter = 0
                else:
                    silence_counter += 1

                if silence_counter > int(rate / CHUNK) * MAX_SILENCE and frames:
                    break

            if not frames:
                continue

            wav_file = os.path.join(script_dir, "temp_audio.wav")
            with wave.open(wav_file, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(fmt))
                wf.setframerate(rate)
                wf.writeframes(b"".join(frames))

            wav_16k = os.path.join(script_dir, "temp_audio_16k.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_file, "-ar", "16000", wav_16k],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if not flags.minimal:
                print("Lancement de Whisper...")

            whisper_command = [
                whisper_cli,
                "-m",
                model_path,
                "-f",
                wav_16k,
                "-l",
                "en",
            ]

            try:
                result = subprocess.run(whisper_command, capture_output=True, text=True, timeout=20)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(result.stdout + "\n" + result.stderr + "\n")

                if result.returncode != 0:
                    if not flags.minimal:
                        print(f"Erreur Whisper: {result.stderr}")
                    continue

                transcription = result.stdout.strip()
                if not transcription:
                    if not flags.minimal:
                        print("Aucune transcription obtenue.")
                    continue

                if mode == "transcription":
                    print(transcription)
                    continue

                # mode traduction
                assert translate_text is not None
                translation = translate_text(transcription)
                if flags.minimal:
                    print(f"FR: {translation}")
                else:
                    print(f"Transcription: {transcription}\nTraduction FR: {translation}")

            except Exception as e:
                if not flags.minimal:
                    print(f"Exception: {e}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Arret utilisateur.")
        return 0
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        p.terminate()
