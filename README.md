# VoxBridge

Application Python de transcription audio en direct (anglais) avec `whisper.cpp`, avec option de traduction vers le francais.

## Contenu

- `transcriptor.py`: transcription EN en temps reel.
- `traductor.py`: transcription EN + traduction FR (Argos Translate).
- `testvbaudio.py`: test de detection de son sur un device audio.
- `app_gui.py`: lance l'interface desktop (Tkinter).
- `app/`: logique coeur + config + UI.

## Fonctionnement

1. Capture audio via `pyaudio`.
2. Sauvegarde temporaire en `.wav`.
3. Conversion en 16 kHz via `ffmpeg`.
4. Transcription avec `whisper-cli.exe`.
5. Traduction EN -> FR (uniquement `traductor.py`).

## Prerequis

- Windows
- Python 3.10+
- `ffmpeg` dans le `PATH`
- `whisper.cpp` (binaire CLI + modele)

Dependances Python:
- `pyaudio`
- `numpy`
- `argostranslate` (pour `traductor.py`)

## Installation des dependances

### Option recommandee (script automatique)

```powershell
.\install_deps.ps1
```

Avec support loopback Windows (ecoute de la sortie casque/speakers sans Voicemeeter):

```powershell
.\install_deps.ps1 -WithLoopback
```

### Option manuelle

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option loopback:

```bash
pip install -r requirements-loopback.txt
```

## Publication GitHub

### A publier

- `app_gui.py`
- dossier `app/`
- `transcriptor.py`
- `traductor.py`
- `testvbaudio.py`
- `install_deps.ps1`
- `requirements.txt`
- `requirements-loopback.txt`
- `README.md`
- `.gitignore`

### A ne pas publier

- `.venv/`
- `.vs/` et `.vscode/`
- `logs.txt`
- `temp_audio.wav`, `temp_audio_16k.wav`
- `app_config.json`
- `whisper.cpp/build/` et `whisper.cpp/build-cuda/`
- `whisper.cpp/models/*.bin` (modeles lourds)

## Fichiers volumineux a telecharger (ne pas versionner)

Ce projet ne doit pas stocker les gros binaires/modeles dans le repo.  
Il faut les telecharger localement et les placer aux emplacements suivants.

### 1) Binaire Whisper CLI

Fichiers attendus (au moins un des deux):
- `whisper.cpp/build-cuda/bin/Release/whisper-cli.exe`
- `whisper.cpp/build/bin/Release/whisper-cli.exe`

Si absent, compiler `whisper.cpp` localement (voir doc officielle):
- https://github.com/ggml-org/whisper.cpp

### 2) Modeles Whisper (ggml)

Dossier cible:
- `whisper.cpp/models/`

Placement obligatoire des modeles:
- copie les fichiers `.bin` directement dans `whisper.cpp/models/`

Les scripts utilisent par defaut:
- `whisper.cpp/models/ggml-tiny.en.bin`

Sources officielles des modeles:
- https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md
- https://huggingface.co/ggerganov/whisper.cpp/tree/main

Script officiel de download (dans `whisper.cpp/models/`):
- `download-ggml-model.cmd` (Windows)

Exemple:

```powershell
cd whisper.cpp\models
.\download-ggml-model.cmd tiny.en
```

## Types de modeles (resume)

- `tiny` / `tiny.en`: tres rapide, qualite plus faible.
- `base` / `base.en`: bon compromis vitesse/qualite.
- `small` / `small.en`: meilleure qualite, plus lourd.
- `medium` / `medium.en`: encore plus precis, tres lourd.
- `large-*`: meilleure qualite, tres lourd et plus lent.
- suffixe `-q5_0` / `-q5_1`: versions quantifiees (plus legeres, souvent un peu moins precises).
- suffixe `.en`: optimise anglais uniquement.

Pour ce projet, `tiny.en` est le meilleur point de depart.

## Configuration importante

Les scripts utilisent un index audio local par defaut (`DEVICE_INDEX = 2`) pour le mode entree micro/device.

Si tu veux capturer directement l'audio de sortie Windows (casque/haut-parleurs) sans Voicemeeter, utilise `--loopback`.

## Interface desktop (MVP)

Lancer l'interface:

```bash
python app_gui.py
```

Fonctions disponibles:
- choix du modele `ggml*.bin`
- mode `transcription` ou `traduction`
- source audio `loopback` ou `device`
- selection du peripherique quand `source=device`
- toggle GPU (`CUDA`) via option `-ng` de `whisper-cli`
- sauvegarde des preferences utilisateur dans `app_config.json`

## Utilisation

### 1) Tester le device audio

```bash
python testvbaudio.py
```

### 2) Lister les peripheriques audio

```bash
python transcriptor.py --list-devices
```

### 3) Transcription (EN)

```bash
python transcriptor.py
python transcriptor.py --minimal
python transcriptor.py --loopback
```

### 4) Transcription + traduction (EN -> FR)

```bash
python traductor.py
python traductor.py --minimal
python traductor.py --loopback
```

## Fichiers generes localement

- `temp_audio.wav`
- `temp_audio_16k.wav`
- `logs.txt`

## Etat de verification locale

- Compilation Python (`py_compile`): OK
- `ffmpeg`: OK
- Imports Python requis: OK
- `whisper-cli.exe`: OK
- `ggml-tiny.en.bin`: OK

## Licence

MIT (voir le fichier `LICENSE`).


## Comportement au lancement sans fichiers externes

- Si `whisper-cli.exe` est absent: les scripts CLI (`transcriptor.py`, `traductor.py`) s'arretent avec un message d'erreur clair.
- Si aucun modele `.bin` n'est present dans `whisper.cpp/models/`: les scripts CLI s'arretent avec un message indiquant ou placer les modeles.
- Dans la GUI (`app_gui.py`):
  - si aucun modele n'est detecte, le bouton Start affiche une erreur et ne lance pas le worker,
  - si `whisper-cli.exe` est absent, le worker s'arrete immediatement avec message d'erreur.


## Whisper.cpp Externe

Le dossier whisper.cpp n'est PAS inclus dans ce repository.
Tu dois le telecharger localement depuis:
https://github.com/ggml-org/whisper.cpp

Automatisation recommandee (depuis ce projet):
- .\install_deps.ps1
- .\install_whisper.ps1 -Model tiny.en
- option CUDA: .\install_whisper.ps1 -Model tiny.en -BuildCuda

Emplacement attendu en local:
- ./whisper.cpp/
- ./whisper.cpp/models/
- ./whisper.cpp/build/bin/Release/whisper-cli.exe (CPU)
  ou ./whisper.cpp/build-cuda/bin/Release/whisper-cli.exe (CUDA)

Si les modeles ne sont pas telecharges:
- les scripts CLI s'arretent avec un message clair,
- la GUI affiche une erreur au Start et ne lance pas le worker.

