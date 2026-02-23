import pyaudio
import numpy as np

p = pyaudio.PyAudio()
device_index = 17  # Voicemeeter Out B1

# Seuil d'amplitude : ajuster si nécessaire
THRESHOLD = 500  # Plus grand si trop sensible, plus petit si pas assez

try:
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, input_device_index=device_index)
    print("🎙️ Écoute en cours...")

    while True:
        data = stream.read(1024, exception_on_overflow=False)
        # Convertir en tableau numpy pour analyser l'amplitude
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_data))

        if amplitude > THRESHOLD:
            print(f"🔊 Son détecté ! (Amplitude : {amplitude})")
        else:
            print(f"🤫 Silence (Amplitude : {amplitude})")

except Exception as e:
    print(f"⚠️ Erreur : {e}")
