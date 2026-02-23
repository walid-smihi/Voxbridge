from pathlib import Path

from app.ui import launch_gui


if __name__ == "__main__":
    launch_gui(Path(__file__).resolve().parent)
