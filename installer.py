"""ImageTagger Setup (PyQt5). Build: pyinstaller --clean --onefile --windowed --name ImageTaggerSetup imagetagger_setup.py"""
import os, sys, time, shutil, subprocess, urllib.request, zipfile, traceback, base64
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect, QRectF
from PyQt5.QtGui import QFont, QPainter, QColor, QLinearGradient, QBrush, QPixmap, QIcon, QImage, QPainterPath
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QFrame, QMessageBox, QGraphicsOpacityEffect)


APP_NAME = "ImageTagger"
PYTHON_URL = "https://www.python.org/ftp/python/3.10.5/python-3.10.5-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
MODEL_URL = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip"
AUTOTAGGER_URL = "https://raw.githubusercontent.com/P4rsonaYuki/DeepBooru-GUI/main/autotagger.py"
MODEL_FOLDER = "deepdanbooru-v3-20211112-sgd-e28"
APP_FILE_NAME = "autotagger.py"
ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "python"
PYTHON_EXE = RUNTIME_DIR / "python.exe"
MODEL_DIR = ROOT / "models" / MODEL_FOLDER
MODEL_ZIP_NAME = "deepdanbooru-model.zip"
STEPS = ["Portable Python", "pip package manager", "ImageTagger libraries", "DeepDanbooru model", "Extract model files"]

# User-provided launcher artwork: place launcher.png beside setup.py / the built EXE.
LAUNCH_IMAGE = ROOT / "launcher.jpg"


def hidden_process_options():
    if os.name != "nt": return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {"startupinfo": startup, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


class InstallWorker(QThread):
    progress = pyqtSignal(int, int)  # stage index, percent
    status = pyqtSignal(str)
    download_stats = pyqtSignal(str, int, int, float, object)
    log = pyqtSignal(str)
    stage = pyqtSignal(int, str)    # index, state
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str, str)

    def download(self, url, destination, label, retries=3):
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, retries + 1):
            temp = destination.with_suffix(destination.suffix + ".part")
            try:
                self.status.emit(f"Downloading {label} · attempt {attempt}/{retries}")
                request = urllib.request.Request(url, headers={"User-Agent": "ImageTaggerSetup/2.1"})
                with urllib.request.urlopen(request, timeout=45) as response, open(temp, "wb") as out:
                    total = int(response.headers.get("Content-Length", 0) or 0); received = 0; started = time.monotonic(); last_emit = 0
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk: break
                        out.write(chunk); received += len(chunk)
                        now = time.monotonic(); speed = received / max(now-started, .001)
                        eta = max(0, (total-received)/speed) if total and speed else None
                        if now-last_emit >= .15 or (total and received >= total):
                            self.download_stats.emit(label, received, total, speed, eta)
                            last_emit = now
                if total and received != total: raise IOError(f"Incomplete download ({received:,}/{total:,} bytes)")
                temp.replace(destination)
                self.download_stats.emit(label, received, total or received, 0, 0)
                self.log.emit(f"Downloaded {label} ({received/1048576:.1f} MB)")
                return
            except Exception as exc:
                last_error = exc; temp.unlink(missing_ok=True)
                self.log.emit(f"Download attempt {attempt} failed: {exc}")
                if attempt < retries:
                    self.status.emit(f"Connection interrupted. Retrying {label}…")
                    time.sleep(min(2*attempt, 5))
        raise RuntimeError(f"Could not download {label} after {retries} attempts.\n{last_error}")

    def command(self, args, description):
        self.status.emit(description + " · please wait")
        self.log.emit("$ " + " ".join(map(str, args)))
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, **hidden_process_options())
        tail = []
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    tail.append(line); self.log.emit(line)
            code = proc.wait()
        except Exception:
            proc.kill(); proc.wait(); raise
        if code:
            raise RuntimeError(f"{description} failed (exit code {code}).\n\n" + "\n".join(tail[-50:]))

    def run(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            # Portable Python
            self.stage.emit(0, "active")
            if not PYTHON_EXE.exists():
                archive = ROOT / "python-embed.zip"
                self.download(PYTHON_URL, archive, "Portable Python")
                self.status.emit("Preparing portable Python…")
                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as z: z.extractall(RUNTIME_DIR)
                archive.unlink(missing_ok=True)
            if not PYTHON_EXE.exists(): raise RuntimeError("Portable Python setup did not create python.exe.")
            pth = RUNTIME_DIR / "python310._pth"
            if pth.exists():
                lines = pth.read_text(encoding="utf-8").splitlines()
                for line in ("Lib\\site-packages", "import site"):
                    if line not in lines: lines.append(line)
                pth.write_text("\n".join(lines)+"\n", encoding="utf-8")
            self.progress.emit(0,100); self.stage.emit(0,"done")
            # pip
            self.stage.emit(1,"active")
            probe = subprocess.run([str(PYTHON_EXE), "-m", "pip", "--version"], capture_output=True, **hidden_process_options())
            if probe.returncode:
                gp = ROOT / "get-pip.py"; self.download(GET_PIP_URL, gp, "pip installer")
                self.command([str(PYTHON_EXE), str(gp), "--disable-pip-version-check"], "Installing pip"); gp.unlink(missing_ok=True)
            self.progress.emit(1,100); self.stage.emit(1,"done")
            # Libraries: explicit activity + spinner in UI while pip is working, and output streamed to log
            self.stage.emit(2,"active")
            self.status.emit("Installing libraries · this can take several minutes. Pip output is shown below.")
            self.command([str(PYTHON_EXE), "-m", "pip", "install", "--disable-pip-version-check",
                "numpy==1.23.5", "Pillow==10.4.0", "PyQt5==5.15.10", "tensorflow==2.10.1", "protobuf==3.19.6"], "Installing ImageTagger libraries")
            self.download(AUTOTAGGER_URL, ROOT / APP_FILE_NAME, "ImageTagger app")
            self.progress.emit(2,100); self.stage.emit(2,"done")
            # Model download
            self.stage.emit(3,"active")
            archive = ROOT / MODEL_ZIP_NAME; project = MODEL_DIR / "project.json"
            if not project.exists() and not archive.exists(): self.download(MODEL_URL, archive, "DeepDanbooru model")
            else: self.status.emit("Model already downloaded · checking files")
            self.progress.emit(3,100); self.stage.emit(3,"done")
            # Extract model
            self.stage.emit(4,"active")
            if not project.exists():
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as z:
                    entries = z.infolist()
                    for i, entry in enumerate(entries,1):
                        z.extract(entry, MODEL_DIR)
                        self.progress.emit(4, int(i/max(len(entries),1)*100))
                        self.status.emit(f"Extracting model files · {i:,} of {len(entries):,}")
                archive.unlink(missing_ok=True)
                nested = MODEL_DIR / MODEL_FOLDER
                if not project.exists() and (nested / "project.json").exists():
                    for child in nested.iterdir():
                        target = MODEL_DIR / child.name
                        if target.exists():
                            if target.is_dir(): shutil.rmtree(target)
                            else: target.unlink()
                        child.rename(target)
                    nested.rmdir()
            if not project.exists(): raise RuntimeError("Model extraction finished, but project.json is missing.")
            missing = [name for name in ("model-resnet_custom_v3.h5", "tags.txt") if not (MODEL_DIR/name).is_file()]
            if missing: raise RuntimeError("Model missing required file(s): " + ", ".join(missing))
            self.progress.emit(4,100); self.stage.emit(4,"done")
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class StageRow(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent); self.setObjectName("stageRow")
        layout = QHBoxLayout(self); layout.setContentsMargins(0,7,0,7); layout.setSpacing(10)
        self.icon = QLabel("○"); self.icon.setFixedWidth(18); self.icon.setObjectName("stageIcon")
        self.name = QLabel(title); self.name.setObjectName("stageName")
        self.state = QLabel("Waiting"); self.state.setObjectName("stageState")
        layout.addWidget(self.icon); layout.addWidget(self.name); layout.addStretch(); layout.addWidget(self.state)
    def set_state(self, state):
        if state == "active": self.icon.setText("●"); self.state.setText("Working")
        elif state == "done": self.icon.setText("✓"); self.state.setText("Complete")
        elif state == "error": self.icon.setText("!"); self.state.setText("Error")
        else: self.icon.setText("○"); self.state.setText("Waiting")
        self.icon.setProperty("state", state); self.state.setProperty("state", state)
        self.icon.style().unpolish(self.icon); self.icon.style().polish(self.icon)
        self.state.style().unpolish(self.state); self.state.style().polish(self.state)


class LauncherCard(QFrame):
    """Animated cyan-blue edge highlight for the primary launch card."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0
        self.glow_enabled = False
        self.setObjectName("launcherCard")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_glow)
        self.timer.start(45)
    def update_glow(self):
        if self.glow_enabled:
            self.phase = (self.phase + 3) % 360
            self.update()
    def set_ready(self, ready):
        self.glow_enabled = ready
        self.update()
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.glow_enabled:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        grad = QLinearGradient(rect.topLeft(), rect.topRight())
        grad.setColorAt(0, QColor(40, 140, 255, 90))
        grad.setColorAt((self.phase / 360.0), QColor(80, 205, 255, 235))
        grad.setColorAt(1, QColor(75, 110, 255, 120))
        from PyQt5.QtGui import QPen
        p.setPen(QPen(QBrush(grad), 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, 17, 17)
        p.end()



class ImageLaunchButton(QPushButton):
    """Rounded image button with a clear hover treatment."""
    def __init__(self, image_path, parent=None):
        super().__init__("", parent)
        self.image = QPixmap(str(image_path))  # JPG and PNG are both supported.
        self.hovered = False
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 190)
        self.setFocusPolicy(Qt.StrongFocus)

    def enterEvent(self, event):
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 15
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), float(radius), float(radius))
        painter.setClipPath(path)

        if not self.image.isNull():
            scaled = self.image.scaled(
                rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            sx = max(0, (scaled.width() - rect.width()) // 2)
            sy = max(0, (scaled.height() - rect.height()) // 2)
            painter.drawPixmap(
                rect, scaled, QRect(sx, sy, rect.width(), rect.height())
            )
        else:
            painter.fillRect(rect, QColor("#46515f"))
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
            painter.drawText(rect, Qt.AlignCenter, "Add launcher.jpg beside setup.py")

        # A subtle scrim and always-visible prompt make the image read as clickable.
        if self.isEnabled():
            painter.fillPath(path, QColor(0, 0, 0, 24 if not self.hovered else 85))
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 17, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, "CLICK TO LAUNCH" if self.hovered else "OPEN IMAGETAGGER")
        else:
            painter.fillPath(path, QColor(12, 16, 22, 115))

        painter.setClipping(False)
        painter.setPen(QColor("#ffffff") if self.isEnabled() else QColor("#68717d"))
        painter.drawPath(path)
        painter.end()

class InstallerWindow(QWidget):
    def __init__(self):
        super().__init__(); self.rows=[]; self.percent=[0]*len(STEPS); self.current_stage=0; self.worker=InstallWorker()
        self.setWindowTitle("ImageTagger Setup"); self.setFixedSize(560, 540)
        self.setStyleSheet(STYLE); self.build(); self.connect_worker(); self.worker.start()
    def closeEvent(self, event):
        for timer_name in ("spinner_timer", "launch_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        # Do not terminate ImageTagger when the setup window is closed.
        log_handle = getattr(self, "launch_log_handle", None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass
        event.accept()

    def paintEvent(self, event):
        # Subtle translucent abstract backdrop (no external image asset required).
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101318"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(67, 106, 173, 24))
        painter.drawEllipse(-110, -90, 340, 300)
        painter.setBrush(QColor(68, 185, 157, 13))
        painter.drawEllipse(self.width()-230, self.height()-190, 350, 300)
        painter.setBrush(QColor(126, 157, 218, 10))
        painter.drawEllipse(self.width()-100, -110, 220, 240)
        painter.end()

    def build(self):
        outer=QVBoxLayout(self); outer.setContentsMargins(28,24,28,24); outer.setSpacing(14
        )
       

        # Primary launcher is the focal point, visible from the start but locked until ready.
        self.launch_card=LauncherCard(); hero=QVBoxLayout(self.launch_card); hero.setContentsMargins(20,17,20,16); hero.setSpacing(8)
        self.open_btn=ImageLaunchButton(LAUNCH_IMAGE); self.open_btn.setObjectName("imageLaunchButton"); self.open_btn.setEnabled(False); self.open_btn.setFixedSize(460, 205); self.open_btn.clicked.connect(self.launch)
        self.launch_text_btn=QPushButton("LAUNCH IMAGETAGGER"); self.launch_text_btn.setObjectName("whiteLaunchButton"); self.launch_text_btn.setEnabled(False); self.launch_text_btn.setMinimumHeight(46); self.launch_text_btn.clicked.connect(self.launch)
        self.launch_hint=QLabel("Available when installation completes"); self.launch_hint.setObjectName("launchHint"); self.launch_hint.setAlignment(Qt.AlignCenter)
        hero.addWidget(self.open_btn, alignment=Qt.AlignHCenter); hero.addWidget(self.launch_text_btn); hero.addWidget(self.launch_hint)
        outer.addWidget(self.launch_card)

        self.card=QFrame(); self.card.setObjectName("card"); card=QVBoxLayout(self.card); card.setContentsMargins(18,15,18,15); card.setSpacing(9)
        self.status_title=QLabel("Setting things up"); self.status_title.setObjectName("statusTitle")
        self.status_desc=QLabel("Preparing the app and its AI model. This may take a few minutes."); self.status_desc.setObjectName("muted"); self.status_desc.setWordWrap(True)
        card.addWidget(self.status_title); card.addWidget(self.status_desc)
        progresshead=QHBoxLayout(); self.progress_label=QLabel("INSTALLATION PROGRESS"); self.progress_label.setObjectName("smallLabel"); self.percent_label=QLabel("0%"); self.percent_label.setObjectName("percent")
        progresshead.addWidget(self.progress_label); progresshead.addStretch(); progresshead.addWidget(self.percent_label); card.addLayout(progresshead)
        self.progress=QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setTextVisible(False); self.progress.setFixedHeight(8); card.addWidget(self.progress)
        self.run_activity=QProgressBar(); self.run_activity.setObjectName("runActivity"); self.run_activity.setRange(0, 0); self.run_activity.setTextVisible(False); self.run_activity.setFixedHeight(5); self.run_activity.hide(); card.addWidget(self.run_activity)
        metric=QHBoxLayout(); self.spin=QLabel("◌"); self.spin.setObjectName("spinner"); self.spin.setFixedWidth(22); self.metric_label=QLabel("Installer is running"); self.metric_label.setObjectName("muted")
        metric.addWidget(self.spin); metric.addWidget(self.metric_label); metric.addStretch(); card.addLayout(metric)
        outer.addWidget(self.card)

        # Keep installer output available internally for troubleshooting, but do not show it in the clean launcher UI.
        self.logview=QTextEdit(self); self.logview.setReadOnly(True); self.logview.hide()
        self.error_label=QLabel(""); self.error_label.setObjectName("errorText"); self.error_label.setWordWrap(True); outer.addWidget(self.error_label)
        self.spinner_frames=["◌", "◔", "◑", "◕", "◑", "◔"]; self.spinner_index=0
        self.spinner_timer=QTimer(self); self.spinner_timer.timeout.connect(self.animate_spinner); self.spinner_timer.start(160)
    def animate_spinner(self):
        if self.spin.text() in ("✓", "!"): return
        self.spinner_index=(self.spinner_index+1)%len(self.spinner_frames); self.spin.setText(self.spinner_frames[self.spinner_index])
    def toggle_log(self):
        self.logview.setVisible(not self.logview.isVisible()); self.log_toggle.setText("Hide log" if self.logview.isVisible() else "Show log")
    def connect_worker(self):
        self.worker.progress.connect(self.on_progress); self.worker.status.connect(self.on_status); self.worker.download_stats.connect(self.on_download)
        self.worker.log.connect(self.add_log); self.worker.stage.connect(self.on_stage); self.worker.finished_ok.connect(self.on_done); self.worker.failed.connect(self.on_fail)
    def update_overall_progress(self):
        overall = int(sum(self.percent) / len(self.percent))
        self.progress.setValue(overall)
        self.percent_label.setText(f"{overall}%")
    def on_progress(self, idx, val):
        self.percent[idx] = max(self.percent[idx], val)
        self.update_overall_progress()
    def on_status(self, message):
        self.metric_label.setText(message)
        self.status_desc.setText(message)
    def on_download(self, label, received, total, speed, eta):
        pct = int(received / total * 100) if total else 0
        self.percent[self.current_stage] = max(self.percent[self.current_stage], pct)
        self.update_overall_progress()
        self.metric_label.setText(f"Downloading {label} · {pct}%")
        self.status_desc.setText(f"Downloading {label}…")
    def on_stage(self, idx, state):
        if state == "active":
            self.current_stage = idx
            self.status_title.setText(STEPS[idx])
            self.status_desc.setText(f"Working on {STEPS[idx].lower()}…")
            self.metric_label.setText("Installation in progress")
    def add_log(self, line):
        self.logview.append(line)
    def on_done(self):
        self.progress.setValue(100); self.percent_label.setText("100%"); self.status_title.setText("Installation complete")
        self.status_desc.setText("ImageTagger is installed and ready to use."); self.metric_label.setText("Everything is ready."); self.spin.setText("✓")
        self.open_btn.setEnabled(True); self.launch_text_btn.setEnabled(True); self.launch_hint.setText("Click the image or the button below to launch"); self.launch_card.set_ready(True); self.open_btn.setFocus(); self.add_log("Setup completed successfully.")
    def on_fail(self, error, trace):
        self.status_title.setText("Installation needs attention")
        self.status_desc.setText("Something went wrong. The installer stopped before it could finish.")
        self.metric_label.setText("Setup stopped due to an error.")
        self.spin.setText("!")
        self.error_label.setText(error[:240])
        self.add_log("ERROR: " + error)
        self.add_log(trace)
        QMessageBox.critical(self, "ImageTagger setup failed", error)
    def launch(self):
        # Guard against repeated clicks spawning multiple background Python processes.
        if getattr(self, "launching", False):
            return
        self.launching = True
        self.open_btn.setEnabled(False)
        self.launch_text_btn.setEnabled(False)
        self.run_activity.show()
        self.launch_hint.setText("Starting ImageTagger…")
        try:
            if not (PYTHON_EXE.exists() and (ROOT/APP_FILE_NAME).exists()):
                raise FileNotFoundError("Python or autotagger.py is missing. Please run setup again.")
            launch_log = ROOT / "autotagger_launch.log"
            log_handle = open(launch_log, "a", encoding="utf-8", errors="replace")
            log_handle.write(f"\n--- Launch attempt {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_handle.flush()
            # Do not hide the application process: it is a GUI program and should create its own visible window.
            proc = subprocess.Popen([str(PYTHON_EXE), "-u", str(ROOT/APP_FILE_NAME)],
                cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
            self.launch_process = proc
            self.launch_log_handle = log_handle
            self.add_log(f"Launching ImageTagger (PID {proc.pid})")
            self.launch_hint.setText("Starting ImageTagger…")
            # Restore button after a short delay, but keep the single-launch guard while process exists.
            self.launch_timer = QTimer(self)
            self.launch_timer.setSingleShot(False)
            self.launch_timer.timeout.connect(self.check_launch)
            self.launch_timer.start(1000)
            self.check_launch()
        except Exception as exc:
            self.launching = False
            self.open_btn.setEnabled(True)
            self.launch_text_btn.setEnabled(True)
            self.run_activity.hide()
            self.launch_hint.setText("Click the image or the button below to launch")
            QMessageBox.critical(self, "Could not launch ImageTagger", str(exc))

    def check_launch(self):
        proc = getattr(self, "launch_process", None)
        if proc and proc.poll() is not None:
            self.launching = False
            if hasattr(self, "launch_timer"):
                self.launch_timer.stop()
            self.open_btn.setEnabled(True)
            self.launch_text_btn.setEnabled(True)
            self.run_activity.hide()
            self.launch_hint.setText("ImageTagger closed · click to launch again")
            if getattr(self, "launch_log_handle", None):
                try:
                    self.launch_log_handle.close()
                except Exception:
                    pass
                self.launch_log_handle = None
            log_path = ROOT / "autotagger_launch.log"
            self.add_log(f"ImageTagger exited with code {proc.returncode}. See {log_path}")
        else:
            # Process is still alive; avoid launching duplicate instances.
            self.launch_hint.setText("ImageTagger is running")
            self.open_btn.setEnabled(False)
            self.launch_text_btn.setEnabled(False)
            self.run_activity.show()

STYLE = """
QWidget { background:#101319; color:#e7eaf0; font-family:'Segoe UI'; font-size:10pt; }
QLabel { background: transparent; border: 0; }
#logoTile { background:#242b36; color:#dfe7f4; border-radius:10px; font-size:14pt; font-weight:700; qproperty-alignment:AlignCenter; }
#appTitle { font-size:17pt; font-weight:650; color:#f2f5fa; } #eyebrow,#smallLabel { color:#8793a4; font-size:8pt; font-weight:700; letter-spacing:1px; }
#launcherCard { background:transparent; border:0; border-radius:17px; }
#heroEyebrow { color:#78caff; font-size:8pt; font-weight:700; letter-spacing:2px; } #heroTitle { color:#f2f7ff; font-size:17pt; font-weight:650; } #heroDesc { color:#9daab9; font-size:9pt; }
#launchHint { color:#77899e; font-size:8pt; }
#card { background:transparent; border:0; border-radius:0; } #stageCard { background:#171b21; border:1px solid #252b34; border-radius:12px; }
#statusTitle { font-size:12pt; font-weight:600; color:#f0f3f8; } #muted { color:#a0a9b6; font-size:9pt; } #percent { color:#e5ebf5; font-weight:600; }
QProgressBar { border:0; background:#2b313a; border-radius:3px; } QProgressBar::chunk { background:#68b8f4; border-radius:3px; }
#spinner { color:#82c8f4; font-size:16pt; } #stageRow { border-bottom:1px solid #252b34; } #stageIcon { color:#697482; font-size:11pt; } #stageIcon[state=active] { color:#8fc9ff; } #stageIcon[state=done] { color:#65c79a; }
#stageName { color:#c8d0dc; font-size:9pt; } #stageState { color:#77818f; font-size:8pt; } #stageState[state=active] { color:#91c6f3; } #stageState[state=done] { color:#65c79a; }
#textButton { background:transparent; border:0; color:#9eb4d5; font-size:9pt; } #textButton:hover { color:#d2e0f7; }
#logView { background:#0c0f13; border:1px solid #242a32; border-radius:8px; color:#aab7c8; font-family:Consolas; font-size:8pt; }
#imageLaunchButton { background:transparent; border:0; padding:0; }
#whiteLaunchButton { background:#ffffff; color:#20242a; border:1px solid #ffffff; border-radius:9px; padding:10px 16px; font-size:11pt; font-weight:800; }
#whiteLaunchButton:hover:enabled { background:#e9edf2; border-color:#e9edf2; }
#whiteLaunchButton:disabled { background:#343b45; color:#8993a1; border-color:#343b45; }
#runActivity { border:0; background:#303944; border-radius:2px; }
#runActivity::chunk { background:#dce5ef; border-radius:2px; }
#errorText { color:#e38b8b; font-size:8pt; }
"""

def main():
    app=QApplication(sys.argv); app.setApplicationName("ImageTagger Setup")
    window=InstallerWindow(); window.show(); sys.exit(app.exec_())
if __name__=="__main__": main()
