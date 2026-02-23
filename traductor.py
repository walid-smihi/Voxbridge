import os
import sys
import time
import wave
import subprocess
import logging

import numpy as np
import pyaudio
import argostranslate.translate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.getLogger("stanza").setLevel(logging.ERROR)

# Modes CLI
minimal_mode = "--minimal" in sys.argv
loopback_mode = "--loopback" in sys.argv
list_devices_mode = "--list-devices" in sys.argv

# Audio defaults (non-loopback)
CHUNK = 1024
THRESHOLD = 300
DEFAULT_CHANNELS = 2
DEFAULT_RATE = 44100
DEVICE_INDEX = 2
MAX_SILENCE = 0.1

# Select backend
pa_lib = pyaudio
if loopback_mode:
    try:
        import pyaudiowpatch as pa_lib  # type: ignore
    except ImportError:
        print("Mode --loopback indisponible: installe pyaudiowpatch")
        print("Commande: pip install pyaudiowpatch")
        sys.exit(1)

FORMAT = pa_lib.paInt16

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(SCRIPT_DIR, "whisper.cpp")
WHISPER_CLI = os.path.join(WHISPER_DIR, "build-cuda", "bin", "Release", "whisper-cli.exe")
if not os.path.exists(WHISPER_CLI):
    WHISPER_CLI = os.path.join(WHISPER_DIR, "build", "bin", "Release", "whisper-cli.exe")

MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-tiny.en.bin")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-tiny.en-q5_1.bin")

if not os.path.exists(WHISPER_CLI):
    print("Erreur: whisper-cli.exe introuvable.")
    print("Attendu: whisper.cpp/build-cuda/bin/Release/whisper-cli.exe")
    print("ou:      whisper.cpp/build/bin/Release/whisper-cli.exe")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print("Erreur: aucun modele Whisper trouve.")
    print("Place un modele ici: whisper.cpp/models/")
    print("Exemple attendu: ggml-tiny.en.bin ou ggml-tiny.en-q5_1.bin")
    print("Commande utile: cd whisper.cpp/models && .\download-ggml-model.cmd tiny.en")
    sys.exit(1)


def build_translator_en_fr():
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    to_lang = next((lang for lang in installed_languages if lang.code == "fr"), None)
    if from_lang and to_lang:
        return from_lang.get_translation(to_lang).translate
    return lambda text: text


def print_devices(p):
    print("index | maxInput | maxOutput | defaultRate | name")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        print(
            f"{i:>5} | {int(d.get('maxInputChannels', 0)):>8} | "
            f"{int(d.get('maxOutputChannels', 0)):>9} | "
            f"{int(d.get('defaultSampleRate', 0)):>11} | {d.get('name', '')}"
        )


def get_wasapi_loopback_device(p):
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


def open_stream(p):
    if loopback_mode:
        loop_dev = get_wasapi_loopback_device(p)
        channels = max(1, min(2, int(loop_dev.get("maxInputChannels", 2))))
        rate = int(loop_dev.get("defaultSampleRate", DEFAULT_RATE))
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(loop_dev["index"]),
            frames_per_buffer=CHUNK,
        )
        return stream, channels, rate, f"{loop_dev.get('name', '')} (loopback)"

    stream = p.open(
        format=FORMAT,
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


translate_text = build_translator_en_fr()

# Init audio
p = pa_lib.PyAudio()
if list_devices_mode:
    print_devices(p)
    p.terminate()
    sys.exit(0)

try:
    stream, channels, rate, device_name = open_stream(p)
except Exception as e:
    print(f"Erreur ouverture audio: {e}")
    p.terminate()
    sys.exit(1)

# Log file
log_file = os.path.join(SCRIPT_DIR, "logs.txt")

if not minimal_mode:
    print(f"Whisper CLI: {WHISPER_CLI}")
    print(f"Capture device: {device_name}")
    print("Ecoute, transcription et traduction en cours...")

while True:
    frames = []
    silence_counter = 0

    if not minimal_mode:
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

    if frames:
        wav_file = os.path.join(SCRIPT_DIR, "temp_audio.wav")
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        wav_16k = os.path.join(SCRIPT_DIR, "temp_audio_16k.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_file, "-ar", "16000", wav_16k],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not minimal_mode:
            print("Lancement de Whisper...")

        whisper_command = [
            WHISPER_CLI,
            "-m", MODEL_PATH,
            "-f", wav_16k,
            "-l", "en",
        ]

        try:
            result = subprocess.run(whisper_command, capture_output=True, text=True, timeout=20)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(result.stdout + "\n" + result.stderr + "\n")

            if result.returncode == 0:
                transcription = result.stdout.strip()
                if transcription:
                    translation = translate_text(transcription)
                    if minimal_mode:
                        print(f"FR: {translation}")
                    else:
                        print(f"Transcription: {transcription}\nTraduction FR: {translation}")
                elif not minimal_mode:
                    print("Aucune transcription obtenue.")
            elif not minimal_mode:
                print(f"Erreur Whisper: {result.stderr}")

        except Exception as e:
            if not minimal_mode:
                print(f"Exception: {e}")

    time.sleep(0.5)
