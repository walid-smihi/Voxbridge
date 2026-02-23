# VoxBridge

VoxBridge est une application Python de transcription audio en direct (anglais) avec option de traduction vers le francais, basee sur `whisper.cpp`.

## Ce que fait le projet

- Capture audio (micro/peripherique ou loopback Windows)
- Segmentation audio automatique
- Transcription via `whisper-cli`
- Traduction EN -> FR (optionnelle)
- Interface desktop (Tkinter) + scripts CLI

## Structure du projet

- `app_gui.py` : lance l'interface desktop
- `app/` : logique coeur + config + UI
- `transcriptor.py` : mode CLI transcription
- `traductor.py` : mode CLI transcription + traduction
- `install_deps.ps1` : installe les dependances Python
- `install_whisper.ps1` : telecharge/prepare `whisper.cpp` + modele

## Important: whisper.cpp n'est pas inclus

Le dossier `whisper.cpp` n'est pas versionne dans ce repository.

Tu dois le recuperer localement depuis:

- https://github.com/ggml-org/whisper.cpp

Le plus simple est d'utiliser le script fourni dans ce projet (voir section installation).

## Prerequis

- Windows
- Python 3.10+
- `ffmpeg` disponible dans le `PATH`
- Git
- CMake

## Installation rapide

### 1) Installer les dependances Python

```powershell
.\install_deps.ps1
```

Avec support loopback Windows (capture de la sortie casque/speakers):

```powershell
.\install_deps.ps1 -WithLoopback
```

### 2) Installer whisper.cpp + un modele

CPU:

```powershell
.\install_whisper.ps1 -Model tiny.en
```

CUDA (si GPU compatible):

```powershell
.\install_whisper.ps1 -Model tiny.en -BuildCuda
```

## Emplacements attendus en local

Apres installation, tu dois avoir au minimum:

- `./whisper.cpp/`
- `./whisper.cpp/models/ggml-tiny.en.bin` (ou autre modele `.bin`)
- `./whisper.cpp/build/bin/Release/whisper-cli.exe` (CPU)
  ou `./whisper.cpp/build-cuda/bin/Release/whisper-cli.exe` (CUDA)

## Modeles Whisper (ggml)

Sources utiles:

- https://github.com/ggml-org/whisper.cpp/tree/master/models
- https://huggingface.co/ggerganov/whisper.cpp/tree/main

Modeles courants:

- `tiny` / `tiny.en`: tres rapide, precision plus faible
- `base` / `base.en`: bon compromis
- `small` / `small.en`: plus precis, plus lourd
- `medium` / `medium.en`: encore plus precis, tres lourd
- `large-*`: precision max, tres lourd/lent

Notes:

- suffixe `.en` = optimise anglais uniquement
- suffixe `-q*` = quantifie (plus leger, precision parfois legerement reduite)

## Utilisation

### Interface desktop

```powershell
python .\app_gui.py
```

Fonctions principales:

- choix du modele
- transcription ou traduction
- source audio (`loopback` ou `device`)
- choix du peripherique quand `source=device`
- option GPU (active/desactive)
- sauvegarde locale des preferences (`app_config.json`)

### CLI transcription

```powershell
python .\transcriptor.py
python .\transcriptor.py --minimal
python .\transcriptor.py --list-devices
python .\transcriptor.py --loopback
```

#### CLI transcription + traduction

```powershell
python .\traductor.py
python .\traductor.py --minimal
python .\traductor.py --list-devices
python .\traductor.py --loopback
```

## Comportement si des elements manquent

- Si `whisper-cli.exe` est absent: le script s'arrete avec un message explicite
- Si aucun modele `.bin` n'est trouve dans `whisper.cpp/models/`: le script s'arrete avec message explicite
- Dans la GUI: le bouton Start affiche une erreur et ne demarre pas le worker

## Fichiers locaux generes

- `logs.txt`
- `temp_audio.wav`
- `temp_audio_16k.wav`
- `app_config.json`

Ces fichiers sont ignores par Git via `.gitignore`.

## Licence

MIT (voir `LICENSE`).

