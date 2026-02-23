from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import AppConfig, load_config, save_config
from .core import RunOptions, TranscriptionWorker, discover_models, list_input_devices


class TranslatorAppUI:
    def __init__(self, root: tk.Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.root.title("VoxBridge - Desktop")
        self.root.geometry("1000x760")

        self.cfg = load_config(project_root)
        self.event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: TranscriptionWorker | None = None

        self.models = discover_models(project_root)
        self.devices = list_input_devices()

        self.pending_transcription: str | None = None

        self._build_ui()
        self._load_config_to_form()
        self._refresh_dynamic_controls()

        self.root.after(120, self._poll_events)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="traduction")
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            state="readonly",
            values=["traduction", "transcription"],
            width=18,
        )
        self.mode_combo.grid(row=1, column=0, padx=(0, 12), sticky="we")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_dynamic_controls())

        ttk.Label(top, text="Source audio").grid(row=0, column=1, sticky="w")
        self.source_var = tk.StringVar(value="loopback")
        self.source_combo = ttk.Combobox(
            top,
            textvariable=self.source_var,
            state="readonly",
            values=["loopback", "device"],
            width=18,
        )
        self.source_combo.grid(row=1, column=1, padx=(0, 12), sticky="we")
        self.source_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_dynamic_controls())

        ttk.Label(top, text="Peripherique (si source=device)").grid(row=0, column=2, sticky="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=48)
        self.device_combo.grid(row=1, column=2, padx=(0, 12), sticky="we")

        ttk.Label(top, text="Modele Whisper").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=48)
        self.model_combo.grid(row=3, column=0, columnspan=3, sticky="we")
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_model_help())

        self.model_help_var = tk.StringVar(value="")
        self.model_help = ttk.Label(
            top,
            textvariable=self.model_help_var,
            justify="left",
            wraplength=920,
            foreground="#505050",
        )
        self.model_help.grid(row=4, column=0, columnspan=3, sticky="we", pady=(6, 2))

        top.columnconfigure(2, weight=1)

        opts = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        opts.pack(fill="x")

        self.cuda_var = tk.BooleanVar(value=True)
        self.cuda_check = ttk.Checkbutton(opts, text="Utiliser GPU (CUDA)", variable=self.cuda_var)
        self.cuda_check.pack(side="left")

        self.show_status_var = tk.BooleanVar(value=True)
        self.show_status_check = ttk.Checkbutton(
            opts,
            text="Afficher infos status",
            variable=self.show_status_var,
        )
        self.show_status_check.pack(side="left", padx=(16, 0))

        self.show_transcription_var = tk.BooleanVar(value=False)
        self.show_transcription_check = ttk.Checkbutton(
            opts,
            text="Afficher transcription aussi (mode traduction)",
            variable=self.show_transcription_var,
        )
        self.show_transcription_check.pack(side="left", padx=(16, 0))

        self.start_btn = ttk.Button(opts, text="Start", command=self.start_worker)
        self.start_btn.pack(side="right")
        self.stop_btn = ttk.Button(opts, text="Stop", command=self.stop_worker, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 8))

        log_wrap = ttk.Frame(self.root, padding=12)
        log_wrap.pack(fill="both", expand=True)

        self.log = ScrolledText(log_wrap, wrap="word", height=28)
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

    def _device_labels(self) -> list[str]:
        labels = []
        for d in self.devices:
            labels.append(f"{d.index} - {d.name} ({d.max_input_channels} in, {d.default_sample_rate} Hz)")
        return labels

    def _model_explanation(self, model_name: str) -> str:
        if not model_name:
            return ""

        name = model_name.lower()
        if "tiny" in name:
            tier = "Tres rapide, precision plus faible"
        elif "base" in name:
            tier = "Rapide, meilleur compromis que tiny"
        elif "small" in name:
            tier = "Plus precis, plus lourd"
        elif "medium" in name:
            tier = "Bonne precision, lourd"
        elif "large" in name:
            tier = "Meilleure precision, tres lourd et plus lent"
        else:
            tier = "Modele personnalise"

        lang = "Optimise anglais uniquement" if ".en" in name else "Multilingue"
        quant = "Quantifie (fichier plus petit, precision parfois un peu moindre)" if "-q" in name else "Non quantifie"

        model_path = self.project_root / "whisper.cpp" / "models" / model_name
        size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0
        size_info = f"Taille approx: {size_mb:.1f} MB"

        return f"{tier}. {lang}. {quant}. {size_info}."

    def _refresh_model_help(self) -> None:
        self.model_help_var.set(self._model_explanation(self.model_var.get().strip()))

    def _load_config_to_form(self) -> None:
        self.mode_var.set(self.cfg.mode)
        self.source_var.set(self.cfg.source)
        self.cuda_var.set(bool(self.cfg.use_cuda))
        self.show_status_var.set(bool(self.cfg.show_status_info))
        self.show_transcription_var.set(bool(self.cfg.show_transcription_with_translation))

        model_values = self.models if self.models else ["Aucun modele detecte"]
        self.model_combo["values"] = model_values
        if self.cfg.model_name in self.models:
            self.model_var.set(self.cfg.model_name)
        elif self.models:
            self.model_var.set(self.models[0])

        device_values = self._device_labels() if self.devices else ["Aucun device input detecte"]
        self.device_combo["values"] = device_values

        preferred_idx = self.cfg.device_index
        selected = None
        for label in device_values:
            if label.startswith(f"{preferred_idx} -"):
                selected = label
                break
        if selected is None and device_values:
            selected = device_values[0]
        if selected:
            self.device_var.set(selected)

        self._refresh_model_help()

    def _refresh_dynamic_controls(self) -> None:
        if self.source_var.get() == "device":
            self.device_combo.configure(state="readonly")
        else:
            self.device_combo.configure(state="disabled")

        if self.mode_var.get() == "traduction":
            self.show_transcription_check.configure(state="normal")
        else:
            self.show_transcription_check.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _append_status(self, text: str) -> None:
        if bool(self.show_status_var.get()):
            self._append_log(f"[status] {text}")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, msg = self.event_queue.get_nowait()

                if kind == "status":
                    self._append_status(msg)
                    continue

                if kind == "error":
                    self._append_log(f"[error] {msg}")
                    continue

                if kind == "stopped":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self._append_status("worker stopped")
                    self.pending_transcription = None
                    self.worker = None
                    continue

                mode = self.mode_var.get().strip()
                show_both = bool(self.show_transcription_var.get())

                if kind == "transcription":
                    self.pending_transcription = msg
                    if mode == "transcription":
                        self._append_log(msg)
                    continue

                if kind == "translation":
                    if mode != "traduction":
                        continue

                    if show_both and self.pending_transcription:
                        self._append_log(self.pending_transcription)
                        self._append_log(msg)
                        self._append_log("")
                    else:
                        self._append_log(msg)
                    self.pending_transcription = None
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_events)

    def _parse_selected_device_index(self) -> int:
        label = self.device_var.get().strip()
        if not label:
            return 0
        try:
            return int(label.split(" - ", 1)[0])
        except Exception:
            return 0

    def _build_run_options(self) -> RunOptions | None:
        if not self.models:
            messagebox.showerror("Modele manquant", "Aucun modele ggml*.bin detecte dans whisper.cpp/models")
            return None

        model_name = self.model_var.get().strip()
        if model_name not in self.models:
            messagebox.showerror("Modele invalide", "Selectionne un modele valide")
            return None

        mode = self.mode_var.get().strip()
        source = self.source_var.get().strip()
        device_index = self._parse_selected_device_index()

        return RunOptions(
            mode=mode,
            source=source,
            device_index=device_index,
            model_path=self.project_root / "whisper.cpp" / "models" / model_name,
            use_cuda=bool(self.cuda_var.get()),
        )

    def _save_current_config(self) -> None:
        cfg = AppConfig(
            mode=self.mode_var.get().strip(),
            source=self.source_var.get().strip(),
            device_index=self._parse_selected_device_index(),
            model_name=self.model_var.get().strip(),
            use_cuda=bool(self.cuda_var.get()),
            show_transcription_with_translation=bool(self.show_transcription_var.get()),
            show_status_info=bool(self.show_status_var.get()),
        )
        save_config(self.project_root, cfg)

    def start_worker(self) -> None:
        if self.worker is not None:
            return

        options = self._build_run_options()
        if options is None:
            return

        self._save_current_config()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.pending_transcription = None
        self._append_status("starting worker...")

        self.worker = TranscriptionWorker(
            project_root=self.project_root,
            options=options,
            on_event=lambda kind, msg: self.event_queue.put((kind, msg)),
        )
        self.worker.start()

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self._append_status("stop requested...")


def launch_gui(project_root: Path) -> None:
    root = tk.Tk()
    TranslatorAppUI(root, project_root)
    root.mainloop()

