from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppConfig:
    mode: str = "traduction"  # transcription | traduction
    source: str = "loopback"  # loopback | device
    device_index: int = 0
    model_name: str = ""
    use_cuda: bool = True
    show_transcription_with_translation: bool = False
    show_status_info: bool = True


CONFIG_FILENAME = "app_config.json"


def config_path(project_root: Path) -> Path:
    return project_root / CONFIG_FILENAME


def load_config(project_root: Path) -> AppConfig:
    path = config_path(project_root)
    if not path.exists():
        return AppConfig()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()

    base = AppConfig()
    for key, value in data.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def save_config(project_root: Path, cfg: AppConfig) -> None:
    path = config_path(project_root)
    path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=True), encoding="utf-8")
