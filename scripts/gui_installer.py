"""Simple GUI installer wizard for AgentSystem.

This script provides a Tkinter-based installer that walks users through
basic setup choices (installation directory and virtual environment) and
runs the necessary commands to get the project ready. It is intentionally
minimal so non-technical users can install dependencies without touching
the command line.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, List, Optional

# Optional import to provide a nicer text widget if available, but fall back
# gracefully if the module is not present.
try:  # pragma: no cover - this is a best-effort import.
    from tkinter import scrolledtext
except ImportError:  # pragma: no cover
    scrolledtext = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTALL_DIR = REPO_ROOT.resolve()
REQUIREMENTS_FILE = (REPO_ROOT / "AgentSystem" / "requirements.txt").resolve()


@dataclass
class InstallerTask:
    """A single shell command executed as part of the installation."""

    description: str
    command: List[str]
    cwd: Optional[Path] = None


class InstallerWizard(tk.Tk):
    """A lightweight multi-step installer wizard."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AgentSystem Installer")
        self.geometry("640x420")
        self.resizable(False, False)

        self.install_dir = tk.StringVar(value=str(DEFAULT_INSTALL_DIR))
        self.create_virtualenv = tk.BooleanVar(value=True)
        self.virtualenv_name = tk.StringVar(value="venv")

        self._frames: List[tk.Frame] = []
        self._current_step = 0

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self._build_step_frames(container)
        self._show_step(0)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_step_frames(self, container: ttk.Frame) -> None:
        self._frames = [
            self._build_welcome_frame(container),
            self._build_configuration_frame(container),
            self._build_summary_frame(container),
        ]

    def _build_welcome_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)

        title = ttk.Label(frame, text="Welcome to AgentSystem", font=("TkDefaultFont", 16, "bold"))
        title.pack(pady=(0, 12))

        description = (
            "This guided installer will help you set up the dependencies\n"
            "required to run AgentSystem. You'll be able to choose where\n"
            "to install the project and whether to create an isolated\n"
            "Python environment."
        )
        ttk.Label(frame, text=description, justify=tk.LEFT).pack(pady=(0, 24))

        ttk.Button(frame, text="Get Started", command=self._next_step).pack()
        return frame

    def _build_configuration_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)

        ttk.Label(frame, text="Installation Settings", font=("TkDefaultFont", 14, "bold")).pack(pady=(0, 8))

        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=6)
        ttk.Label(dir_frame, text="Install directory:").pack(side=tk.LEFT)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.install_dir, width=45)
        dir_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self._browse_directory).pack(side=tk.LEFT)

        venv_frame = ttk.Frame(frame)
        venv_frame.pack(fill=tk.X, pady=6)
        venv_check = ttk.Checkbutton(
            venv_frame,
            text="Create a virtual environment",
            variable=self.create_virtualenv,
            command=lambda: self._toggle_venv_name_state(name_entry),
        )
        venv_check.pack(anchor=tk.W)

        name_row = ttk.Frame(frame)
        name_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(name_row, text="Environment name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_row, textvariable=self.virtualenv_name, width=20)
        name_entry.pack(side=tk.LEFT, padx=6)

        ttk.Button(frame, text="Back", command=self._previous_step).pack(side=tk.LEFT, pady=12)
        ttk.Button(frame, text="Next", command=self._next_step).pack(side=tk.RIGHT, pady=12)

        self._toggle_venv_name_state(name_entry)
        return frame

    def _build_summary_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)

        ttk.Label(frame, text="Ready to Install", font=("TkDefaultFont", 14, "bold")).pack(pady=(0, 8))
        self.summary_label = ttk.Label(frame, text="", justify=tk.LEFT)
        self.summary_label.pack(fill=tk.X, pady=(0, 12))

        if scrolledtext:
            text_widget = scrolledtext.ScrolledText(frame, width=70, height=10, state=tk.DISABLED)
        else:  # pragma: no cover - scrolledtext always available in stdlib but keep fallback.
            text_widget = tk.Text(frame, width=70, height=10, state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        self.log_widget: tk.Text = text_widget

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=12)

        self.install_button = ttk.Button(button_row, text="Install", command=self._start_installation)
        self.install_button.pack(side=tk.RIGHT)

        ttk.Button(button_row, text="Back", command=self._previous_step).pack(side=tk.LEFT)
        return frame

    # ------------------------------------------------------------------
    # Step navigation helpers
    # ------------------------------------------------------------------
    def _show_step(self, index: int) -> None:
        current_frame = self._frames[self._current_step]
        if current_frame.winfo_manager():
            current_frame.pack_forget()
        self._current_step = index
        frame = self._frames[index]
        if frame == self._frames[2]:
            self._refresh_summary()
        frame.pack(fill=tk.BOTH, expand=True)

    def _next_step(self) -> None:
        if self._current_step < len(self._frames) - 1:
            self._show_step(self._current_step + 1)

    def _previous_step(self) -> None:
        if self._current_step > 0:
            self._show_step(self._current_step - 1)

    def _browse_directory(self) -> None:
        selection = filedialog.askdirectory(initialdir=self.install_dir.get())
        if selection:
            self.install_dir.set(selection)

    def _toggle_venv_name_state(self, entry: ttk.Entry) -> None:
        state = tk.NORMAL if self.create_virtualenv.get() else tk.DISABLED
        entry.configure(state=state)

    def _refresh_summary(self) -> None:
        install_dir = Path(self.install_dir.get()).expanduser().resolve()
        lines = [
            f"Install directory: {install_dir}",
            (
                "Virtual environment: on ("
                f"{self.virtualenv_name.get()})" if self.create_virtualenv.get() else "Virtual environment: off"
            ),
            f"Requirements file: {REQUIREMENTS_FILE}",
        ]
        if not REQUIREMENTS_FILE.exists():
            lines.append("⚠️ Could not find requirements.txt. Only virtual environment will be created.")
        self.summary_label.configure(text="\n".join(lines))

    # ------------------------------------------------------------------
    # Installation workflow
    # ------------------------------------------------------------------
    def _start_installation(self) -> None:
        self.install_button.configure(state=tk.DISABLED)
        self._append_log("Starting installation...\n")

        tasks = self._build_tasks()
        output_queue: "queue.Queue[str]" = queue.Queue()
        thread = threading.Thread(target=self._run_tasks, args=(tasks, output_queue), daemon=True)
        thread.start()
        self._monitor_queue(output_queue, thread)

    def _build_tasks(self) -> List[InstallerTask]:
        install_dir = Path(self.install_dir.get()).expanduser().resolve()
        install_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[InstallerTask] = []
        if self.create_virtualenv.get():
            venv_path = install_dir / self.virtualenv_name.get()
            tasks.append(
                InstallerTask(
                    description=f"Creating virtual environment at {venv_path}",
                    command=[sys.executable, "-m", "venv", str(venv_path)],
                    cwd=install_dir,
                )
            )
            python_executable = venv_path / ("Scripts" if sys.platform.startswith("win") else "bin") / "python"
        else:
            python_executable = Path(sys.executable)

        if REQUIREMENTS_FILE.exists():
            tasks.append(
                InstallerTask(
                    description="Installing Python dependencies",
                    command=[str(python_executable), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
                    cwd=REPO_ROOT,
                )
            )
        return tasks

    def _run_tasks(self, tasks: Iterable[InstallerTask], output_queue: "queue.Queue[str]") -> None:
        for task in tasks:
            output_queue.put(f"\n▶ {task.description}\n")
            try:
                process = subprocess.Popen(
                    task.command,
                    cwd=str(task.cwd or REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                output_queue.put(f"Error launching command: {exc}\n")
                output_queue.put("INSTALLATION_FAILED\n")
                return

            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)

            return_code = process.wait()
            if return_code != 0:
                output_queue.put(f"Command exited with status {return_code}.\n")
                output_queue.put("INSTALLATION_FAILED\n")
                return
        output_queue.put("\nInstallation completed successfully!\n")
        output_queue.put("INSTALLATION_SUCCEEDED\n")

    def _monitor_queue(self, output_queue: "queue.Queue[str]", worker: threading.Thread) -> None:
        try:
            while True:
                line = output_queue.get_nowait()
                if line.strip() == "INSTALLATION_SUCCEEDED":
                    self._append_log("\n✅ All steps finished. You can now close the installer.\n")
                    messagebox.showinfo("AgentSystem", "Installation finished successfully!")
                    self.install_button.configure(state=tk.NORMAL)
                    return
                if line.strip() == "INSTALLATION_FAILED":
                    self._append_log("\n❌ Installation failed. Please review the log above for details.\n")
                    messagebox.showerror("AgentSystem", "Installation failed. See log for details.")
                    self.install_button.configure(state=tk.NORMAL)
                    return
                self._append_log(line)
        except queue.Empty:
            pass

        if worker.is_alive():
            self.after(150, lambda: self._monitor_queue(output_queue, worker))
        else:
            # Thread finished but we did not receive a terminal signal – treat as failure.
            self.install_button.configure(state=tk.NORMAL)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)


def main() -> None:
    app = InstallerWizard()
    app.mainloop()


if __name__ == "__main__":
    main()
