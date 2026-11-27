import sys
import os
import json
import csv
import traceback
import time
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QStringListModel, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPainterPath, QColor, QLinearGradient, QIcon, QImage, QImageReader
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFrame, QProgressBar, QLineEdit, QDoubleSpinBox, QSplitter, QSizePolicy,
    QAbstractItemView, QTabWidget, QComboBox, QCheckBox,
    QInputDialog, QToolButton, QStatusBar, QGridLayout, QScrollArea, QSpinBox,
    QListWidget, QListWidgetItem, QStackedWidget
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
APP_NAME = "ImageTagger"
APP_VERSION = "4.3"
MODEL_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
GENERAL_TAGS_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/tags.txt"
CHARACTER_TAGS_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/tags-character.txt"
COPYRIGHT_TAGS_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/tags-copyright.txt"
IMAGE_SIZE = (512, 512)
DEFAULT_THRESHOLD = 0.40
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
NSFW_TAGS = {}


def load_tags(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def predict_image(model, tags, path):
    scores = model.predict(prepare_image(path), verbose=0)[0]
    return sorted(((tag, float(score)) for tag, score in zip(tags, scores)),
                  key=lambda pair: pair[1], reverse=True)


class TaggerThread(QThread):
    done = pyqtSignal(str, list)
    failed = pyqtSignal(str, str)
    status = pyqtSignal(str)

    def __init__(self, path, model, tags):
        super().__init__()
        self.path, self.model, self.tags = path, model, tags

    def run(self):
        try:
            self.status.emit("Analyzing image…")
            self.done.emit(self.path, predict_image(self.model, self.tags, self.path))
        except Exception as exc:
            self.failed.emit(self.path, str(exc))


class BatchThread(QThread):
    progress = pyqtSignal(int, int, float, float, str)
    status = pyqtSignal(str)
    done = pyqtSignal(dict, str)
    failed = pyqtSignal(str)

    def __init__(self, folder, model, tags, threshold, recursive=True, skip_existing=True, parent=None):
        super().__init__(parent)
        self.folder, self.model, self.tags = folder, model, tags
        self.threshold, self.recursive, self.skip_existing = threshold, recursive, skip_existing
        self._paused = False
        self._stopped = False
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._started_at = 0.0

    def pause(self):
        self.mutex.lock()
        self._paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._paused = False
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def stop(self):
        self.mutex.lock()
        self._stopped = True
        self._paused = False
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def flags(self):
        self.mutex.lock()
        paused, stopped = self._paused, self._stopped
        self.mutex.unlock()
        return paused, stopped

    def run(self):
        try:
            paths = []
            if self.recursive:
                for root, _, files in os.walk(self.folder):
                    paths.extend(os.path.join(root, n) for n in files
                                 if n.lower().endswith(SUPPORTED_EXTENSIONS))
            else:
                paths = [os.path.join(self.folder, n) for n in os.listdir(self.folder)
                         if os.path.isfile(os.path.join(self.folder, n))
                         and n.lower().endswith(SUPPORTED_EXTENSIONS)]
            paths.sort(key=str.lower)
            if not paths:
                self.failed.emit("No supported images were found in that folder.")
                self.done.emit({}, self.folder)
                return

            result_path = os.path.join(self.folder, "tags_results.json")
            results = {}
            if self.skip_existing and os.path.isfile(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        results = loaded
                except Exception:
                    pass

            total = len(paths)
            self._started_at = time.monotonic()
            for index, path in enumerate(paths, 1):
                paused, stopped = self.flags()
                while paused and not stopped:
                    self.mutex.lock()
                    self.wait_condition.wait(self.mutex, 500)
                    self.mutex.unlock()
                    paused, stopped = self.flags()
                if stopped:
                    break
                rel = os.path.relpath(path, self.folder)
                if rel not in results:
                    self.status.emit(f"Analyzing {index}/{total}: {os.path.basename(path)}")
                    try:
                        pred = predict_image(self.model, self.tags, path)
                        results[rel] = [[tag, score] for tag, score in pred
                                        if score >= self.threshold]
                        tmp = result_path + ".tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        os.replace(tmp, result_path)
                    except Exception as exc:
                        results[rel] = []
                        self.status.emit(f"Skipped {os.path.basename(path)}: {exc}")
                elapsed = time.monotonic() - self._started_at
                remaining = (elapsed / index * (total - index)) if index else 0.0
                self.progress.emit(index, total, elapsed, remaining, os.path.basename(path))
            self.done.emit(results, self.folder)
        except Exception as exc:
            self.failed.emit(f"Batch processing failed: {exc}\n{traceback.format_exc()}")
            self.done.emit({}, self.folder)


class DropLabel(QLabel):
    dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("dropArea")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            u.toLocalFile().lower().endswith(SUPPORTED_EXTENSIONS) for u in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                self.dropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class WorkflowDropArea(QLabel):
    """Clickable startup tile with an optional transparent image background."""
    pathDropped = pyqtSignal(str)
    clicked = pyqtSignal()

    def __init__(self, title, subtitle=None, parent=None, background_stem=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("workflowDropArea")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(300)
        self.setWordWrap(True)
        self.title = title
        self.subtitle = subtitle
        self.language = "en"
        self._hovered = False
        self.background_pixmap = QPixmap()
        if background_stem:
            script_dir = Path(__file__).resolve().parent
            for ext in ("png", "webp", "jpg", "jpeg"):
                candidate = script_dir / f"{background_stem}.{ext}"
                if candidate.is_file():
                    pixmap = QPixmap(str(candidate))
                    if not pixmap.isNull():
                        self.background_pixmap = pixmap
                        break

    def paintEvent(self, event):
        # Paint the optional image ourselves so transparent PNG artwork blends
        # naturally with the tile background and scales with the window.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()),
                            float(rect.width()), float(rect.height()), 18.0, 18.0)
        painter.setClipPath(path)
        painter.fillRect(rect, QColor("#171f2d"))

        if not self.background_pixmap.isNull():
            scaled = self.background_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        # Gentle dark wash keeps text legible without hiding transparent art.
        wash = QLinearGradient(0, 0, 0, self.height())
        wash.setColorAt(0.0, QColor(7, 12, 20, 65))
        wash.setColorAt(1.0, QColor(7, 12, 20, 155))
        painter.fillRect(rect, wash)

        painter.setPen(QColor("#f4f7fc"))
        title_font = QFont("Bahnschrift", 20, QFont.DemiBold)
        painter.setFont(title_font)
        title_rect = rect.adjusted(18, 0, -18, -18)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignVCenter, self.title)

        painter.setPen(QColor(205, 216, 232, 205))
        hint_font = QFont("Segoe UI", 10, QFont.DemiBold)
        painter.setFont(hint_font)
        hint_rect = rect.adjusted(10, 0, -10, -24)
        hint = "クリックして選択  ·  ドロップ" if self.language == "ja" else "CLICK TO BROWSE  ·  DROP HERE"
        painter.drawText(hint_rect, Qt.AlignHCenter | Qt.AlignBottom, hint)

        painter.setPen(QColor(139, 213, 202, 190) if self._hovered else QColor(52, 68, 92))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 18, 18)
        painter.end()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if any(os.path.isdir(path) or path.lower().endswith(SUPPORTED_EXTENSIONS)
                   for path in paths if path):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and (os.path.isdir(path) or path.lower().endswith(SUPPORTED_EXTENSIONS)):
                self.pathDropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class StartupChoiceButton(QPushButton):
    """Large image-backed startup action; right-click to choose its artwork."""
    customizeRequested = pyqtSignal()

    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.background_path = ""
        self.background_pixmap = QPixmap()
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(300, 245)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Click to open · Right-click to choose this button's background image")

    def set_background(self, path):
        self.background_path = path or ""
        self.background_pixmap = QPixmap(self.background_path) if self.background_path else QPixmap()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.customizeRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        clip = QPainterPath()
        clip.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), 16, 16)
        painter.setClipPath(clip)
        painter.fillRect(rect, QColor("#202a3a"))
        if not self.background_pixmap.isNull():
            scaled = self.background_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        wash = QLinearGradient(0, 0, 0, self.height())
        wash.setColorAt(0, QColor(5, 10, 18, 70))
        wash.setColorAt(1, QColor(5, 10, 18, 190))
        painter.fillRect(rect, wash)
        painter.setPen(QColor("#ffffff"))
        title_font = QFont("Bahnschrift", 23, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(16, 0, -16, -24), Qt.AlignCenter, self.title)
        painter.setPen(QColor(224, 232, 244, 225))
        painter.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        painter.drawText(rect.adjusted(12, 0, -12, -24), Qt.AlignHCenter | Qt.AlignBottom, self.subtitle)
        border = QColor("#9bded4") if self.underMouse() else QColor("#53627a")
        painter.setPen(border)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 16, 16)
        painter.end()


class ImageTagger(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tags = []
        self.character_tags = set()
        self.copyright_tags = set()
        self.single_thread = None
        self.batch_thread = None
        self.current_image = None
        self.current_results = []
        self.manual_tags = []
        self.batch_results = {}
        self.batch_folder = None
        self.batch_json_folder = None
        self.batch_tag_cache = {}
        self.batch_path_cache = {}
        self.batch_tag_counts = {}
        self.batch_tag_avg_scores = {}
        self._batch_available_tag_order = []
        self.selected_batch_tags = []  # whitelist (all required)
        self.excluded_batch_tags = []  # blacklist (any match excludes image)
        self.batch_tag_page = 0
        self.batch_tag_page_size = 40
        self.batch_gallery_page = 0
        self._thumbnail_cache = {}
        self._thumbnail_queue = deque()
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self._load_next_thumbnail_chunk)
        self.batch_visible_paths = []
        self.batch_detail_rel = None
        self.config_path = Path(__file__).resolve().parent / "config.json"
        self.recent_workspaces_path = Path(__file__).resolve().parent / "recent_workspaces.json"
        self.recent_workspaces = self.load_recent_workspaces()
        self.language = self.load_language_setting()
        self._language_combo_ready = False
        self.init_window()
        self.init_ui()
        self.apply_language()
        self.load_model()

    def load_language_setting(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return "ja" if config.get("language") == "ja" else "en"
        except Exception:
            return "en"

    def save_language_setting(self):
        try:
            config = {}
            if self.config_path.is_file():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        config = loaded
            config["language"] = self.language
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Could not save language preference: {exc}")

    def tr(self, english):
        """Translate a UI phrase; tag names and file paths are never translated."""
        return JA_TEXT.get(english, english) if self.language == "ja" else english

    def change_language(self, value):
        self.language = "ja" if value == "日本語" else "en"
        self.save_language_setting()
        self.apply_language()

    def apply_language(self):
        # Re-translate the already-built interface in place, preserving the
        # current tab, searches, selected tags, and model state.
        reverse = {v: k for k, v in JA_TEXT.items()}
        for widget in self.findChildren(QWidget):
            if isinstance(widget, StartupChoiceButton):
                # Determine the action from the button's stable identity, not
                # its translated title (which changes every time languages switch).
                is_image_button = widget is getattr(self, "start_single_area", None)
                if self.language == "ja":
                    widget.title = "＋ 画像を開く" if is_image_button else "▦ フォルダーを開く"
                    widget.subtitle = "画像を1枚選択" if is_image_button else "画像フォルダーを選択"
                else:
                    widget.title = "＋  Open image" if is_image_button else "▦  Open folder"
                    widget.subtitle = "Choose one image" if is_image_button else "Browse an image folder"
                widget.update()
            if isinstance(widget, WorkflowDropArea):
                english_title = reverse.get(widget.title, widget.title)
                widget.title = JA_TEXT.get(english_title, english_title) if self.language == "ja" else english_title
                widget.language = self.language
                widget.update()
            if isinstance(widget, (QLabel, QPushButton, QCheckBox)):
                try:
                    current = widget.text()
                    english = reverse.get(current, current)
                    translated = JA_TEXT.get(english, english) if self.language == "ja" else english
                    if translated != current:
                        widget.setText(translated)
                except (TypeError, RuntimeError):
                    pass
            if isinstance(widget, QLineEdit):
                current = widget.placeholderText()
                english = reverse.get(current, current)
                widget.setPlaceholderText(JA_TEXT.get(english, english) if self.language == "ja" else english)
        tab_labels = ["  Get started  ", "  Image workspace  ", "  Batch library  "]
        for index, english in enumerate(tab_labels):
            self.tabs.setTabText(index, JA_TEXT.get(english, english) if self.language == "ja" else english)
        # Header labels are managed separately by Qt's item model.
        self.tag_table.setHorizontalHeaderLabels([
            self.tr("Tag"), self.tr("Confidence")])
        if hasattr(self, "language_combo"):
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(1 if self.language == "ja" else 0)
            self.language_combo.blockSignals(False)
        QApplication.setFont(QFont("Yu Gothic UI" if self.language == "ja" else "Segoe UI", 10))
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")

    def init_window(self):
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1174, 840)
        self.setMinimumSize(1050, 700)

    def load_model(self):
        try:
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}\n\nPlace the model in the models folder.")
            if not os.path.isfile(GENERAL_TAGS_PATH):
                raise FileNotFoundError(f"Tag vocabulary not found:\n{GENERAL_TAGS_PATH}")
            self.status_label.setText("Loading AI model…")
            QApplication.processEvents()
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.tags = load_tags(GENERAL_TAGS_PATH)
            # Category vocabularies shipped with the model.
            self.character_tags = self._load_category_tags(CHARACTER_TAGS_PATH)
            self.copyright_tags = self._load_category_tags(COPYRIGHT_TAGS_PATH)
            self.status_label.setText(f"Ready · {len(self.tags):,} tags loaded")
            self.open_button.setEnabled(True)
            self.batch_button.setEnabled(True)
            self.start_single_area.setEnabled(True)
            self.start_batch_area.setEnabled(True)
        except Exception as exc:
            self.status_label.setText("Model unavailable")
            QMessageBox.critical(self, "Could not load model", str(exc))
            self.open_button.setEnabled(False)
            self.batch_button.setEnabled(False)
            self.start_single_area.setEnabled(False)
            self.start_batch_area.setEnabled(False)

    def init_ui(self):
        root = QVBoxLayout(self)
        # Keep the workspace edge-to-edge; the old branding header consumed
        # valuable vertical space without helping the image workflow.
        root.setContentsMargins(12, 8, 12, 10)
        root.setSpacing(8)

        self.batch_button = QPushButton("▦  Batch folder")
        self.batch_button.clicked.connect(self.choose_batch_folder)
        self.open_button = QPushButton("＋  Open image")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.choose_image)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.make_start_tab(), "  Get started  ")
        self.tabs.addTab(self.make_single_tab(), "  Image workspace  ")
        self.tabs.addTab(self.make_batch_tab(), "  Batch library  ")
        self.tabs.setCurrentIndex(0)
        root.addWidget(self.tabs, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(14, 10, 14, 10)
        top = QHBoxLayout()
        self.status_label = QLabel("Preparing…")
        self.status_label.setObjectName("status")
        top.addWidget(self.status_label, 1)
        top.addWidget(self.batch_button)
        top.addWidget(self.open_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_batch)
        self.pause_button.setEnabled(False)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_batch)
        self.resume_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop_batch)
        self.stop_button.setEnabled(False)
        top.addWidget(self.pause_button)
        top.addWidget(self.resume_button)
        top.addWidget(self.stop_button)
        fl.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        fl.addWidget(self.progress)
        root.addWidget(footer)

    def card(self, title_text, subtitle_text=None):
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 15, 16, 16)
        layout.setSpacing(10)
        header = QHBoxLayout()
        title = QLabel(title_text)
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        if subtitle_text:
            label = QLabel(subtitle_text)
            label.setObjectName("muted")
            header.addWidget(label)
        layout.addLayout(header)
        return frame, layout

    def make_start_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(26, 14, 26, 22)
        outer.setSpacing(12)

        language_row = QHBoxLayout()
        language_row.addStretch()
        language_label = QLabel("Language / 言語")
        language_row.addWidget(language_label)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "日本語"])
        self.language_combo.setFixedWidth(135)
        self.language_combo.currentTextChanged.connect(self.change_language)
        language_row.addWidget(self.language_combo)
        outer.addLayout(language_row)

        chooser_row = QHBoxLayout()
        chooser_row.setSpacing(22)
        chooser_row.addStretch(1)
        self.start_single_area = StartupChoiceButton("＋  Open image", "Choose one image")
        self.start_single_area.clicked.connect(self.choose_image)
        self.start_single_area.customizeRequested.connect(lambda: self.choose_start_background("single"))
        self.start_batch_area = StartupChoiceButton("▦  Open folder", "Browse an image folder")
        self.start_batch_area.clicked.connect(self.choose_batch_folder)
        self.start_batch_area.customizeRequested.connect(lambda: self.choose_start_background("batch"))
        self.start_single_area.setEnabled(False)
        self.start_batch_area.setEnabled(False)
        chooser_row.addWidget(self.start_single_area, 1)
        chooser_row.addWidget(self.start_batch_area, 1)
        chooser_row.addStretch(1)
        outer.addLayout(chooser_row, 1)

        hint = QLabel("Right-click either button to choose its background image")
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(hint)

        # Keep history visible, but deliberately compact so it never competes
        # with the two large image/folder launch buttons.
        history_header = QHBoxLayout()
        history_title = QLabel("Recent workspaces")
        history_title.setObjectName("sectionTitle")
        history_header.addWidget(history_title)
        history_header.addStretch(1)
        history_note = QLabel("Double-click to reopen")
        history_note.setObjectName("muted")
        history_header.addWidget(history_note)
        outer.addLayout(history_header)

        self.history_list = QListWidget()
        self.history_list.setObjectName("startupHistory")
        self.history_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_list.setMaximumHeight(116)
        self.history_list.setMinimumHeight(58)
        self.history_list.itemDoubleClicked.connect(self.open_recent_workspace)
        outer.addWidget(self.history_list, 0)
        self.load_start_backgrounds()
        self.refresh_history_list()
        return page

    def load_start_backgrounds(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            backgrounds = config.get("startup_backgrounds", {}) if isinstance(config, dict) else {}
        except Exception:
            backgrounds = {}
        self.start_single_area.set_background(backgrounds.get("single", ""))
        self.start_batch_area.set_background(backgrounds.get("batch", ""))

    def choose_start_background(self, which):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose button background image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        button = self.start_single_area if which == "single" else self.start_batch_area
        button.set_background(path)
        try:
            config = {}
            if self.config_path.is_file():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        config = loaded
            backgrounds = config.setdefault("startup_backgrounds", {})
            backgrounds[which] = path
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.warning(self, "Background not saved", str(exc))

    def load_recent_workspaces(self):
        try:
            with open(self.recent_workspaces_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def save_recent_workspace(self, path, kind):
        path = str(Path(path).expanduser().resolve())
        self.recent_workspaces = [entry for entry in self.recent_workspaces
                                  if isinstance(entry, dict) and entry.get("path") != path]
        self.recent_workspaces.insert(0, {"path": path, "kind": kind, "last_opened": time.time()})
        self.recent_workspaces = self.recent_workspaces[:15]
        try:
            with open(self.recent_workspaces_path, "w", encoding="utf-8") as f:
                json.dump(self.recent_workspaces, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Could not save workspace history: {exc}")
        self.refresh_history_list()

    def refresh_history_list(self):
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        valid_entries = []
        for entry in self.recent_workspaces:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path", "")
            kind = entry.get("kind", "folder")
            if not path or not os.path.exists(path):
                continue
            valid_entries.append(entry)
            label = Path(path).name or path
            prefix = "▦" if kind == "folder" else "{ }"
            item = QListWidgetItem(f"{prefix}   {label}")
            item.setData(Qt.UserRole, entry)
            item.setToolTip(path)
            item.setSizeHint(QSize(0, 27))
            self.history_list.addItem(item)
        if len(valid_entries) != len(self.recent_workspaces):
            self.recent_workspaces = valid_entries
            try:
                with open(self.recent_workspaces_path, "w", encoding="utf-8") as f:
                    json.dump(self.recent_workspaces, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    def open_recent_workspace(self, item):
        entry = item.data(Qt.UserRole)
        if not isinstance(entry, dict):
            return
        path = entry.get("path", "")
        kind = entry.get("kind", "folder")
        if kind == "json" or os.path.isfile(path):
            self.load_batch_json_path(path)
            return
        results_path = os.path.join(path, "tags_results.json")
        if os.path.isfile(results_path):
            self.load_batch_json_path(results_path, workspace_folder=path)
        else:
            self.choose_batch_folder(path)

    def handle_start_drop(self, path):
        if os.path.isdir(path):
            self.choose_batch_folder(path)
        elif os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTENSIONS):
            self.load_image(path)
        else:
            self.show_error("Drop a supported image file or a folder containing images.")

    def make_single_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        preview_card, pl = self.card("Preview", "Drag an image here")
        self.preview = DropLabel()
        self.preview.setText("Drop an image here\n\nor use  Open image")
        self.preview.setMinimumSize(300, 300)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.dropped.connect(self.load_image)
        pl.addWidget(self.preview, 1)
        self.image_meta = QLabel("No image selected")
        self.image_meta.setObjectName("muted")
        pl.addWidget(self.image_meta)
        split.addWidget(preview_card)

        tags_card, tl = self.card("Detected tags", "Sorted by confidence")
        controls = QHBoxLayout()
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("Search tags…")
        self.tag_search.setClearButtonEnabled(True)
        self.tag_search.textChanged.connect(self.refresh_tags)
        controls.addWidget(self.tag_search, 1)
        controls.addWidget(QLabel("Minimum confidence"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(DEFAULT_THRESHOLD)
        self.threshold.valueChanged.connect(self.refresh_tags)
        controls.addWidget(self.threshold)
        tl.addLayout(controls)

        self.tag_table = QTableWidget(0, 2)
        self.tag_table.setHorizontalHeaderLabels(["Tag", "Confidence"])
        self.tag_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tag_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tag_table.verticalHeader().hide()
        self.tag_table.setShowGrid(False)
        self.tag_table.setAlternatingRowColors(True)
        self.tag_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tag_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tag_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tag_table.setSortingEnabled(False)
        tl.addWidget(self.tag_table, 1)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("Copy tags")
        self.copy_button.clicked.connect(self.copy_tags)
        self.copy_button.setEnabled(False)
        self.add_tag_button = QPushButton("＋ Add tag")
        self.add_tag_button.clicked.connect(self.add_manual_tag)
        self.remove_tag_button = QPushButton("Remove selected")
        self.remove_tag_button.clicked.connect(self.remove_selected_tags)
        self.export_button = QPushButton("Export…")
        self.export_button.clicked.connect(self.export_single)
        for b in (self.copy_button, self.add_tag_button, self.remove_tag_button, self.export_button):
            actions.addWidget(b)
        tl.addLayout(actions)
        self.tag_summary = QLabel("0 tags")
        self.tag_summary.setObjectName("muted")
        tl.addWidget(self.tag_summary)
        split.addWidget(tags_card)
        split.setSizes([480, 800])
        layout.addWidget(split, 1)
        return tab

    def make_batch_tab(self):
        """Compact booru-style browser with a paginated thumbnail wall and tag inspector."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(5)

        sidebar = QFrame()
        sidebar.setObjectName("batchSidebar")
        sidebar.setFixedWidth(246)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 9, 10, 9)
        sl.setSpacing(6)
        # Global tag confidence is intentionally isolated in its own visual group.
        accuracy_group = QFrame()
        accuracy_group.setObjectName("globalAccuracyGroup")
        accuracy_layout = QVBoxLayout(accuracy_group)
        accuracy_layout.setContentsMargins(10, 8, 10, 8)
        accuracy_layout.setSpacing(4)
        accuracy_title = QLabel("GLOBAL TAG ACCURACY")
        accuracy_title.setObjectName("globalAccuracyTitle")
        accuracy_layout.addWidget(accuracy_title)
        app_accuracy_row = QHBoxLayout()
        app_accuracy_label = QLabel("Minimum confidence")
        app_accuracy_label.setObjectName("muted")
        app_accuracy_row.addWidget(app_accuracy_label)
        self.batch_app_accuracy = QSpinBox()
        self.batch_app_accuracy.setRange(0, 100)
        self.batch_app_accuracy.setValue(self.load_global_tag_accuracy())
        self.batch_app_accuracy.setSuffix("%")
        self.batch_app_accuracy.setFixedWidth(78)
        self.batch_app_accuracy.setToolTip("Global minimum tag confidence. Lower-confidence tags are hidden from the global tag list and filters.")
        self._batch_accuracy_timer = QTimer(self)
        self._batch_accuracy_timer.setSingleShot(True)
        self._batch_accuracy_timer.setInterval(160)
        self._batch_accuracy_timer.timeout.connect(self._apply_batch_accuracy)
        self.batch_app_accuracy.valueChanged.connect(self._schedule_batch_accuracy_update)
        app_accuracy_row.addWidget(self.batch_app_accuracy)
        app_accuracy_row.addStretch(1)
        accuracy_layout.addLayout(app_accuracy_row)
        sl.addWidget(accuracy_group)
        self.batch_search = QLineEdit()
        self.batch_search.setPlaceholderText("Search filenames…")
        self.batch_search.setClearButtonEnabled(True)
        self.batch_search.textChanged.connect(self.filter_batch)
        sl.addWidget(self.batch_search)
        self.batch_tag_search = QLineEdit()
        self.batch_tag_search.setPlaceholderText("Find a tag…")
        self.batch_tag_search.setClearButtonEnabled(True)
        self.batch_tag_search.textChanged.connect(self.refresh_batch_tag_suggestions)
        sl.addWidget(self.batch_tag_search)

        self.batch_tag_popup = QFrame(tab)
        self.batch_tag_popup.setObjectName("batchTagPopup")
        popup_outer = QVBoxLayout(self.batch_tag_popup)
        popup_outer.setContentsMargins(5, 5, 5, 5)
        self.batch_tag_scroll = QScrollArea(self.batch_tag_popup)
        self.batch_tag_scroll.setWidgetResizable(True)
        self.batch_tag_scroll.setFrameShape(QFrame.NoFrame)
        self.batch_tag_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.batch_tag_rows = QWidget()
        self.batch_tag_popup_layout = QVBoxLayout(self.batch_tag_rows)
        self.batch_tag_popup_layout.setContentsMargins(4, 4, 4, 4)
        self.batch_tag_popup_layout.setSpacing(4)
        self.batch_tag_scroll.setWidget(self.batch_tag_rows)
        popup_outer.addWidget(self.batch_tag_scroll)
        self.batch_tag_popup.hide()

        # Active filters: a real list widget guarantees one full-width row per tag.
        # Avoid the old nested-layout chip behavior that could compress filters into
        # a horizontal strip and make tag names unreadable.
        active_heading = QLabel("ACTIVE FILTERS")
        active_heading.setObjectName("activeFilterHeading")
        self.batch_active_heading = active_heading
        sl.addWidget(active_heading)
        self.batch_tag_chips = QListWidget()
        self.batch_tag_chips.setObjectName("activeBatchFilters")
        self.batch_tag_chips.setSpacing(4)
        self.batch_tag_chips.setMinimumHeight(0)
        self.batch_tag_chips.setMaximumHeight(142)
        self.batch_tag_chips.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sl.addWidget(self.batch_tag_chips)
        active_heading.hide()
        self.batch_tag_chips.hide()
        self.batch_tag_list_label = QLabel("AVAILABLE TAGS · POPULAR")
        self.batch_tag_list_label.setObjectName("sectionTitle")
        # Kept for compatibility with language/update code, but hidden to reduce clutter.
        self.batch_tag_list_label.hide()
        self.batch_available_tags = QListWidget()
        self.batch_available_tags.setObjectName("batchAvailableTags")
        self.batch_available_tags.setMinimumHeight(0)
        self.batch_available_tags.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.batch_available_tags.setSpacing(0)
        self.batch_available_tags.itemClicked.connect(self._click_available_batch_tag)
        sl.addWidget(self.batch_available_tags, 1)
        sl.setStretchFactor(self.batch_available_tags, 1)
        self.batch_vocab_label = QLabel("＋ include · − exclude")
        self.batch_vocab_label.setObjectName("muted")
        self.batch_vocab_label.setWordWrap(True)
        sl.addWidget(self.batch_vocab_label)
        # The available-tag list gets the sidebar's remaining vertical space.

        for label, handler in (
            ("Open results…", self.open_batch_json),
            ("Export CSV…", self.export_batch_csv),
            ("Save JSON…", self.export_batch_json),
            ("Export filtered images", self.export_filtered_batch),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            sl.addWidget(button)
        self.batch_meta = QLabel("Choose a folder to start")
        self.batch_meta.setObjectName("muted")
        self.batch_meta.setWordWrap(True)
        sl.addWidget(self.batch_meta)
        layout.addWidget(sidebar)

        self.batch_stack = QStackedWidget()
        gallery_page = QWidget()
        gl = QVBoxLayout(gallery_page)
        gl.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.batch_gallery_title = QLabel("Gallery")
        self.batch_gallery_title.setObjectName("sectionTitle")
        toolbar.addWidget(self.batch_gallery_title)
        toolbar.addStretch(1)
        self.batch_page_label = QLabel("Page 1 / 1")
        self.batch_page_label.setObjectName("muted")
        toolbar.addWidget(self.batch_page_label)
        self.batch_prev = QPushButton("‹")
        self.batch_prev.setFixedWidth(36)
        self.batch_prev.clicked.connect(lambda: self.change_batch_page(-1))
        self.batch_next = QPushButton("›")
        self.batch_next.setFixedWidth(36)
        self.batch_next.clicked.connect(lambda: self.change_batch_page(1))
        toolbar.addWidget(self.batch_prev)
        toolbar.addWidget(self.batch_next)
        gl.addLayout(toolbar)
        self.batch_gallery = QListWidget()
        self.batch_gallery.setObjectName("batchGallery")
        self.batch_gallery.setViewMode(QListWidget.IconMode)
        self.batch_gallery.setResizeMode(QListWidget.Adjust)
        self.batch_gallery.setMovement(QListWidget.Static)
        self.batch_gallery.setWrapping(True)
        self.batch_gallery.setWordWrap(True)
        self.batch_gallery.setIconSize(QSize(176, 176))
        self.batch_gallery.setGridSize(QSize(190, 194))
        self.batch_gallery.setSpacing(3)
        self.batch_gallery.setMinimumWidth(0)
        self.batch_gallery.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.batch_gallery.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.batch_gallery.setUniformItemSizes(True)
        self.batch_gallery.itemClicked.connect(self.open_batch_detail)
        gl.addWidget(self.batch_gallery, 1)
        self.batch_stack.addWidget(gallery_page)

        detail_page = QWidget()
        dl = QVBoxLayout(detail_page)
        dl.setContentsMargins(0, 0, 0, 0)
        detail_toolbar = QHBoxLayout()
        back = QPushButton("← Gallery")
        back.clicked.connect(lambda: self.batch_stack.setCurrentIndex(0))
        detail_toolbar.addWidget(back)
        self.batch_detail_name = QLabel("")
        self.batch_detail_name.setObjectName("sectionTitle")
        detail_toolbar.addWidget(self.batch_detail_name, 1)
        dl.addLayout(detail_toolbar)
        detail_split = QSplitter(Qt.Horizontal)
        detail_split.setChildrenCollapsible(False)
        image_frame = QFrame()
        image_frame.setObjectName("card")
        il = QVBoxLayout(image_frame)
        self.batch_preview = QLabel("Select an image")
        self.batch_preview.setObjectName("batchPreview")
        self.batch_preview.setAlignment(Qt.AlignCenter)
        self.batch_preview.setMinimumSize(360, 360)
        self.batch_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        il.addWidget(self.batch_preview, 1)
        self.batch_image_path_label = QLabel("")
        self.batch_image_path_label.setObjectName("muted")
        il.addWidget(self.batch_image_path_label)
        detail_split.addWidget(image_frame)

        tag_frame = QFrame()
        tag_frame.setObjectName("card")
        tl = QVBoxLayout(tag_frame)
        tag_header = QHBoxLayout()
        accuracy_label = QLabel("Tags min %")
        tag_header.addWidget(accuracy_label)
        self.batch_detail_accuracy = QSpinBox()
        self.batch_detail_accuracy.setRange(0, 100)
        self.batch_detail_accuracy.setValue(0)
        self.batch_detail_accuracy.setSuffix("%")
        self.batch_detail_accuracy.setFixedWidth(78)
        self.batch_detail_accuracy.setToolTip("Minimum tag confidence. 0% shows every tag.")
        self.batch_detail_accuracy.valueChanged.connect(self.refresh_batch_detail)
        tag_header.addWidget(self.batch_detail_accuracy)
        tag_header.addStretch(1)
        add = QPushButton("+ Add")
        add.clicked.connect(self.add_batch_image_tag)
        tag_header.addWidget(add)
        tl.addLayout(tag_header)
        self.batch_detail_tags = QListWidget()
        self.batch_detail_tags.setObjectName("batchDetailTags")
        self.batch_detail_tags.itemDoubleClicked.connect(self.edit_batch_image_tag)
        tl.addWidget(self.batch_detail_tags, 1)
        actions = QHBoxLayout()
        edit = QPushButton("Edit")
        edit.clicked.connect(lambda: self.edit_batch_image_tag(self.batch_detail_tags.currentItem()))
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_batch_image_tag)
        actions.addWidget(edit)
        actions.addWidget(remove)
        tl.addLayout(actions)
        hint = QLabel("Double-click a tag to rename it")
        hint.setObjectName("muted")
        tl.addWidget(hint)
        detail_split.addWidget(tag_frame)
        detail_split.setSizes([820, 280])
        dl.addWidget(detail_split, 1)
        self.batch_stack.addWidget(detail_page)
        layout.addWidget(self.batch_stack, 1)
        return tab

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        if not os.path.isfile(path):
            self.show_error("That image could not be found.")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.show_error("This image could not be opened.")
            return
        self.current_image = path
        self.manual_tags = []
        self.image_meta.setText(f"{Path(path).name}  ·  {pix.width()} × {pix.height()}")
        self.render_preview()
        self.current_results = []
        self.refresh_tags()
        self.copy_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.status_label.setText(f"Analyzing {Path(path).name}…")
        self.single_thread = TaggerThread(path, self.model, self.tags)
        self.single_thread.done.connect(self.single_done)
        self.single_thread.failed.connect(self.single_failed)
        self.single_thread.start()
        self.tabs.setCurrentIndex(1)

    def render_preview(self):
        if not self.current_image:
            return
        pix = QPixmap(self.current_image)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_preview()
        if self.batch_detail_rel and self.batch_stack.currentIndex() == 1:
            self.refresh_batch_detail()

    def single_done(self, path, results):
        self.open_button.setEnabled(True)
        if path != self.current_image:
            return
        self.current_results = results
        self.refresh_tags()
        self.copy_button.setEnabled(True)
        self.status_label.setText(f"Analysis complete · {len(results):,} candidate tags")

    def single_failed(self, path, error):
        self.open_button.setEnabled(True)
        self.status_label.setText("Analysis failed")
        self.show_error(f"Could not analyze {Path(path).name}:\n{error}")

    def visible_tag_rows(self):
        threshold = self.threshold.value()
        search = self.tag_search.text().strip().lower()
        rows = [(tag, score) for tag, score in self.current_results
                if score >= threshold and search in tag.lower()
                and tag not in self.manual_tags]
        return sorted(rows, key=lambda pair: (self.tag_category_rank(pair[0]), -pair[1], pair[0].casefold()))

    def refresh_tags(self, *_):
        results = self.visible_tag_rows()
        self.tag_table.setRowCount(0)
        for tag, score in results:
            row = self.tag_table.rowCount()
            self.tag_table.insertRow(row)
            item = QTableWidgetItem(tag)
            score_item = QTableWidgetItem(f"{score * 100:.1f}%")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            category_color = self.tag_category_color(tag)
            item.setForeground(category_color)
            score_item.setForeground(category_color)
            if tag.lower() in NSFW_TAGS:
                item.setForeground(Qt.magenta)
                score_item.setForeground(Qt.magenta)
            self.tag_table.setItem(row, 0, item)
            self.tag_table.setItem(row, 1, score_item)
        self.tag_summary.setText(f"{len(results) + len(self.manual_tags)} shown · {len(self.current_results)} predictions · {len(self.manual_tags)} manual")
        self.copy_button.setEnabled(bool(results or self.manual_tags))

    def all_export_tags(self):
        threshold = self.threshold.value()
        predicted = [(t, s) for t, s in self.current_results if s >= threshold and t not in self.manual_tags]
        predicted.sort(key=lambda pair: (self.tag_category_rank(pair[0]), -pair[1], pair[0].casefold()))
        manual = sorted(self.manual_tags, key=lambda tag: (self.tag_category_rank(tag), tag.casefold()))
        return predicted + [(t, 1.0) for t in manual]

    def copy_tags(self):
        tags = [t for t, _ in self.all_export_tags() if self.tag_search.text().strip().lower() in t.lower()]
        QApplication.clipboard().setText(", ".join(tags))
        self.status_label.setText(f"Copied {len(tags)} tags to clipboard")

    def add_manual_tag(self):
        tag, ok = QInputDialog.getText(self, "Add tag", "Tag name:")
        tag = tag.strip()
        if ok and tag and tag not in self.manual_tags:
            self.manual_tags.append(tag)
            self.refresh_tags()
            self.status_label.setText(f"Added manual tag: {tag}")

    def remove_selected_tags(self):
        selected = sorted({idx.row() for idx in self.tag_table.selectionModel().selectedRows()}, reverse=True)
        if not selected:
            QMessageBox.information(self, "Remove tags", "Select one or more rows first.")
            return
        remove = {self.tag_table.item(row, 0).text() for row in selected}
        self.manual_tags = [tag for tag in self.manual_tags if tag not in remove]
        self.current_results = [(tag, score) for tag, score in self.current_results if tag not in remove]
        self.refresh_tags()
        self.status_label.setText(f"Removed {len(remove)} tag(s) from this image")

    def export_single(self):
        if not self.current_image or not self.current_results and not self.manual_tags:
            QMessageBox.information(self, "Export", "Analyze an image first.")
            return
        path, selected = QFileDialog.getSaveFileName(self, "Export tags", "image_tags.json",
            "JSON (*.json);;Text tag list (*.txt);;CSV (*.csv)")
        if not path:
            return
        tags = self.all_export_tags()
        try:
            if path.lower().endswith(".txt") or "Text tag list" in selected:
                if not path.lower().endswith(".txt"): path += ".txt"
                Path(path).write_text(", ".join(t for t, _ in tags), encoding="utf-8")
            elif path.lower().endswith(".csv") or "CSV" in selected:
                if not path.lower().endswith(".csv"): path += ".csv"
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f); writer.writerow(["tag", "confidence", "source"])
                    writer.writerows((t, f"{s:.6f}", "manual" if t in self.manual_tags else "model") for t, s in tags)
            else:
                if not path.lower().endswith(".json"): path += ".json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"image": self.current_image, "threshold": self.threshold.value(),
                               "tags": [{"tag": t, "confidence": s, "source": "manual" if t in self.manual_tags else "model"}
                                        for t, s in tags]}, f, indent=2, ensure_ascii=False)
            self.status_label.setText(f"Exported {len(tags)} tags · {Path(path).name}")
        except Exception as exc:
            self.show_error(f"Export failed:\n{exc}")

    def choose_batch_folder(self, folder=None):
        if self.batch_thread and self.batch_thread.isRunning():
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if not folder:
            return
        if not os.path.isdir(folder):
            self.show_error("Please choose a folder containing images.")
            return
        self.batch_folder = folder
        self.save_recent_workspace(folder, "folder")
        self.progress.setValue(0)
        self.progress.setFormat("Preparing batch…")
        self.set_batch_running(True)
        self.status_label.setText("Starting batch…")
        self.batch_thread = BatchThread(folder, self.model, self.tags, self.threshold.value(), parent=self)
        self.batch_thread.progress.connect(self.update_progress)
        self.batch_thread.status.connect(self.status_label.setText)
        self.batch_thread.failed.connect(self.show_error)
        self.batch_thread.done.connect(self.batch_done)
        self.batch_thread.start()
        self.tabs.setCurrentIndex(2)

    def set_batch_running(self, running):
        self.open_button.setEnabled(not running and self.model is not None)
        self.batch_button.setEnabled(not running and self.model is not None)
        self.pause_button.setEnabled(running)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(running)

    def pause_batch(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.status_label.setText("Paused · scanning will resume when requested")
            self.progress.setFormat("Paused")

    def resume_batch(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.status_label.setText("Resuming batch…")
            self.progress.setFormat("Resuming…")

    def stop_batch(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.status_label.setText("Stopping after current image…")
            self.progress.setFormat("Stopping…")

    def update_progress(self, current, total, elapsed=0.0, remaining=0.0, filename=""):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        percent = (100.0 * current / total) if total else 100.0
        def duration(seconds):
            seconds = max(0, int(seconds))
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            return f"{hrs}h {mins}m {secs}s" if hrs else (f"{mins}m {secs}s" if mins else f"{secs}s")
        self.progress.setFormat(f"Scanning {current:,} / {total:,} · {percent:.0f}% · ETA {duration(remaining)}")
        self.status_label.setText(f"Scanning {current:,} / {total:,} · {percent:.0f}% · elapsed {duration(elapsed)} · ETA {duration(remaining)} · {filename}")

    def batch_done(self, results, folder):
        self.set_batch_running(False)
        self.batch_results = results
        self.batch_folder = folder
        self.batch_json_folder = folder
        self.populate_batch()
        self.progress.setFormat("Completed")
        self.progress.setValue(self.progress.maximum())
        self.status_label.setText(f"Batch complete · {len(results):,} image records")
        self.batch_meta.setText(f"{len(results):,} image records · {folder}")

    def populate_batch(self):
        self.batch_tag_cache = {}
        self.batch_path_cache = {}
        self.batch_tag_counts = {}
        score_totals = {}
        score_counts = {}
        basename_index = {}
        roots = []
        for root in (self.batch_folder, self.batch_json_folder):
            if root and os.path.isdir(root) and root not in roots:
                roots.append(root)
        for root in roots:
            for walk_root, _, files in os.walk(root):
                for name in files:
                    if name.lower().endswith(SUPPORTED_EXTENSIONS):
                        basename_index.setdefault(name.casefold(), []).append(os.path.join(walk_root, name))
        self._batch_basename_index = basename_index

        threshold = self.batch_app_accuracy.value() / 100.0 if hasattr(self, "batch_app_accuracy") else 0.0
        for rel, entries in sorted(self.batch_results.items(), key=lambda kv: kv[0].casefold()):
            normalized = self.normalize_entries(entries)
            self.batch_tag_cache[rel] = normalized
            score_map = {tag.casefold(): float(score) for tag, score in normalized}
            names = frozenset(tag for tag, score in score_map.items() if score >= threshold)
            self.batch_tag_cache[(rel, "names")] = names
            self.batch_tag_cache[(rel, "scores")] = score_map
            for tag, score in score_map.items():
                score_totals[tag] = score_totals.get(tag, 0.0) + score
                score_counts[tag] = score_counts.get(tag, 0) + 1
            for tag in names:
                self.batch_tag_counts[tag] = self.batch_tag_counts.get(tag, 0) + 1

        self.batch_tag_avg_scores = {
            tag: score_totals[tag] / score_counts[tag] for tag in score_totals
        }
        self._batch_available_tag_order = sorted(
            self.batch_tag_counts,
            key=lambda tag: (-self.batch_tag_counts[tag], tag.casefold())
        )
        self.selected_batch_tags = [t for t in self.selected_batch_tags if t.casefold() in self.batch_tag_counts]
        self.excluded_batch_tags = [t for t in self.excluded_batch_tags if t.casefold() in self.batch_tag_counts]
        self.render_batch_tag_chips()
        self.refresh_batch_tag_suggestions()
        self.batch_gallery_page = 0
        self.filter_batch()

    def clear_batch_preview(self, message="Select an image from the list"):
        self.batch_preview.setPixmap(QPixmap())
        self.batch_preview.setText(message)

    @staticmethod
    def normalize_entries(entries):
        output = []
        if not isinstance(entries, (list, tuple)): return output
        for entry in entries:
            if isinstance(entry, (list, tuple)) and entry:
                try: output.append((str(entry[0]), float(entry[1]) if len(entry) > 1 else 1.0))
                except (ValueError, TypeError): output.append((str(entry[0]), 1.0))
            elif isinstance(entry, str):
                output.append((entry, 1.0))
        return output

    def refresh_batch_tag_suggestions(self, *_):
        """Show live matches from the full tag vocabulary and current batch."""
        import unicodedata

        # Read the edit's current value directly on every signal. Do not use a
        # cached/previous query: that was the source of multi-character failures.
        query = unicodedata.normalize(
            "NFKC", self.batch_tag_search.text()
        ).strip().casefold()

        # Remove old suggestion widgets from the layout. Keep the QLineEdit itself
        # untouched so rebuilding the list cannot reset its cursor or focus.
        while self.batch_tag_popup_layout.count():
            item = self.batch_tag_popup_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        # The permanent tag list always stays populated, and the search narrows
        # it to workspace-local tags only.
        self._refresh_available_tag_list(query)
        if not query:
            self.batch_tag_popup.hide()
            return

        # Filtering is scoped strictly to tags present in this workspace.
        vocabulary = {str(tag).strip() for tag in self.batch_tag_counts if str(tag).strip()}
        matches = [tag for tag in vocabulary
                   if query in unicodedata.normalize("NFKC", tag).casefold()]
        matches.sort(key=lambda tag: (-self.batch_tag_counts.get(tag.casefold(), 0), tag.casefold()))
        matches = matches[:12]

        for tag in matches:
            row_widget = QWidget(self.batch_tag_rows)
            row_widget.setFocusPolicy(Qt.NoFocus)
            row_layout = QHBoxLayout(row_widget)
            row_widget.setMinimumHeight(48)
            row_widget.setObjectName("tagSuggestionRow")
            row_layout.setContentsMargins(12, 6, 12, 6)
            row_layout.setSpacing(10)

            tag_label = QLabel(tag)
            tag_label.setObjectName("suggestionTag")
            tag_label.setStyleSheet(f"color: {self.tag_category_color(tag).name()};")
            tag_label.setFocusPolicy(Qt.NoFocus)
            tag_label.setMinimumWidth(130)
            tag_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            count = self.batch_tag_counts.get(tag.casefold(), 0)
            count_label = QLabel(f"{count:,}")
            count_label.setFocusPolicy(Qt.NoFocus)
            count_label.setObjectName("suggestionCount")
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(tag_label, 1)
            row_layout.addWidget(count_label)

            include_btn = QPushButton("+")
            include_btn.setObjectName("tagIncludeAction")
            include_btn.setFixedSize(34, 32)
            include_btn.setFocusPolicy(Qt.NoFocus)
            include_btn.setToolTip(f"Include {tag}")
            include_btn.clicked.connect(lambda checked=False, value=tag: self.add_batch_tag(value, True))
            row_layout.addWidget(include_btn)

            exclude_btn = QPushButton("−")
            exclude_btn.setObjectName("tagExcludeAction")
            exclude_btn.setFixedSize(34, 32)
            exclude_btn.setFocusPolicy(Qt.NoFocus)
            exclude_btn.setToolTip(f"Exclude {tag}")
            exclude_btn.clicked.connect(lambda checked=False, value=tag: self.add_batch_tag(value, False))
            row_layout.addWidget(exclude_btn)
            self.batch_tag_popup_layout.addWidget(row_widget)

        if not matches:
            empty_label = QLabel("No matching tags")
            empty_label.setObjectName("muted")
            empty_label.setContentsMargins(10, 8, 10, 8)
            self.batch_tag_popup_layout.addWidget(empty_label)

        # Position the floating panel directly under the search field. The
        # panel is a regular child of the tab, not a window, and never receives
        # focus; the search field remains active while results are refreshed.
        tab = self.batch_tag_popup.parentWidget()
        self.batch_tag_popup.setFixedWidth(max(340, self.batch_tag_search.width()))
        self.batch_tag_popup.setFixedHeight(min(270, max(74, len(matches) * 54 + 18)))
        popup_pos = self.batch_tag_search.mapTo(tab, self.batch_tag_search.rect().bottomLeft())
        self.batch_tag_popup.move(popup_pos.x(), popup_pos.y() + 2)
        self.batch_tag_popup.raise_()
        self.batch_tag_popup.show()

    def add_batch_tag(self, tag, whitelist):
        """Add a tag to exactly one filter list and render it as a removable chip."""
        target = self.selected_batch_tags if whitelist else self.excluded_batch_tags
        other = self.excluded_batch_tags if whitelist else self.selected_batch_tags
        if tag in other:
            other.remove(tag)
        if tag not in target:
            target.append(tag)
        self.batch_tag_popup.hide()
        self.batch_tag_search.clear()
        self.render_batch_tag_chips()
        self.refresh_batch_tag_suggestions()
        # Sidebar filters are usable from image detail too. After a filter change,
        # return to the gallery so the user immediately sees the filtered results.
        if hasattr(self, "batch_stack"):
            self.batch_stack.setCurrentIndex(0)
        self.filter_batch()
        # Keep keyboard focus in the tag filter, never the filename filter.
        self.batch_tag_search.setFocus(Qt.OtherFocusReason)

    def remove_batch_tag(self, tag, whitelist):
        target = self.selected_batch_tags if whitelist else self.excluded_batch_tags
        if tag in target:
            target.remove(tag)
        self.render_batch_tag_chips()
        # Removing a sidebar filter while viewing an image also takes the user
        # back to the gallery, where the updated result set is visible.
        if hasattr(self, "batch_stack"):
            self.batch_stack.setCurrentIndex(0)
        self.filter_batch()

    def render_batch_tag_chips(self):
        """Render active filters as readable full-width list rows."""
        self.batch_tag_chips.clear()
        chips = (
            [(tag, True) for tag in self.selected_batch_tags] +
            [(tag, False) for tag in self.excluded_batch_tags]
        )
        self.batch_active_heading.setVisible(bool(chips))
        self.batch_tag_chips.setVisible(bool(chips))
        for tag, whitelist in chips:
            item = QListWidgetItem(self.batch_tag_chips)
            item.setSizeHint(QSize(0, 38))
            row = QFrame()
            row.setObjectName("activeIncludeRow" if whitelist else "activeExcludeRow")
            row.setMinimumHeight(34)
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(9, 3, 5, 3)
            row_layout.setSpacing(7)
            mode = QLabel("IN" if whitelist else "OUT")
            mode.setObjectName("activeIncludeBadge" if whitelist else "activeExcludeBadge")
            mode.setFixedWidth(30)
            mode.setAlignment(Qt.AlignCenter)
            label = QLabel(tag)
            label.setObjectName("activeFilterTag")
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            label.setToolTip("Must contain this tag" if whitelist else "Must not contain this tag")
            remove = QPushButton("×")
            remove.setObjectName("activeFilterRemove")
            remove.setFixedSize(24, 24)
            remove.setToolTip("Remove filter")
            remove.clicked.connect(
                lambda checked=False, value=tag, mode=whitelist: self.remove_batch_tag(value, mode)
            )
            row_layout.addWidget(mode)
            row_layout.addWidget(label, 1)
            row_layout.addWidget(remove)
            self.batch_tag_chips.setItemWidget(item, row)
        self.batch_vocab_label.setText(
            f"{len(self.selected_batch_tags)} included · {len(self.excluded_batch_tags)} excluded"
            if chips else "Click a tag below to include it · use search for exclusions"
        )

    def export_filtered_batch(self):
        included = {tag.casefold() for tag in self.selected_batch_tags}
        excluded = {tag.casefold() for tag in self.excluded_batch_tags}
        if not included and not excluded:
            QMessageBox.information(self, "Export filtered images", "Add at least one whitelist (+) or blacklist (−) tag first.")
            return
        filtered = {}
        min_accuracy = self.batch_app_accuracy.value() / 100.0 if hasattr(self, "batch_app_accuracy") else 0.0
        for rel, entries in self.batch_results.items():
            tag_names = {tag.casefold() for tag, score in self.normalize_entries(entries) if score >= min_accuracy}
            if included.issubset(tag_names) and not (excluded & tag_names):
                filtered[rel] = entries
        if not filtered:
            QMessageBox.information(self, "Export filtered images", "No images match all selected tags.")
            return
        destination = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not destination:
            return
        exported = {}
        skipped = []
        for rel, entries in filtered.items():
            source = self.resolve_batch_image(rel)
            if not source or not os.path.isfile(source):
                skipped.append(rel)
                continue
            target_rel = Path(rel)
            if target_rel.is_absolute():
                target_rel = Path(target_rel.name)
            target = Path(destination) / target_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source, target)
                exported[target_rel.as_posix()] = entries
            except Exception as exc:
                skipped.append(f"{rel} ({exc})")
        json_path = Path(destination) / "tags_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)
        self.status_label.setText(f"Exported {len(exported)} filtered images + JSON")
        QMessageBox.information(self, "Export complete",
            f"Copied {len(exported)} images and wrote a filtered JSON file to:\\n{destination}"
            + (f"\\n\\nSkipped {len(skipped)} missing/failed image(s)." if skipped else ""))

    def load_global_tag_accuracy(self):
        """Load persisted global tag confidence threshold (0–100)."""
        try:
            if self.config_path.is_file():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                value = config.get("global_tag_accuracy", 0) if isinstance(config, dict) else 0
                return max(0, min(100, int(value)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return 0



    def save_global_tag_accuracy(self, value):
        """Persist global tag accuracy without overwriting other app settings."""
        try:
            config = {}
            if self.config_path.is_file():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        config = loaded
            config["global_tag_accuracy"] = int(value)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, TypeError) as exc:
            self.status_label.setText(f"Could not save global tag accuracy: {exc}")



    def _schedule_batch_accuracy_update(self, *_):
        """Debounce accuracy updates so rapid spinner changes don't rebuild UI repeatedly."""
        if hasattr(self, "_batch_accuracy_timer"):
            self._batch_accuracy_timer.start()
        else:
            self._apply_batch_accuracy()

    def _apply_batch_accuracy(self, *_):
        """Apply confidence threshold to cached tag scores only; no image IO or model work."""
        threshold = self.batch_app_accuracy.value() / 100.0
        self.save_global_tag_accuracy(self.batch_app_accuracy.value())
        counts = {}
        for rel in self.batch_results:
            scores = self.batch_tag_cache.get((rel, "scores"), {})
            names = frozenset(tag for tag, score in scores.items() if score >= threshold)
            self.batch_tag_cache[(rel, "names")] = names
            for tag in names:
                counts[tag] = counts.get(tag, 0) + 1
        self.batch_tag_counts = counts
        self._batch_available_tag_order = sorted(
            counts, key=lambda tag: (-counts[tag], tag.casefold())
        )
        self.selected_batch_tags = [tag for tag in self.selected_batch_tags if tag.casefold() in counts]
        self.excluded_batch_tags = [tag for tag in self.excluded_batch_tags if tag.casefold() in counts]
        self.render_batch_tag_chips()
        self.refresh_batch_tag_suggestions()
        self._refresh_available_tag_list()
        self.filter_batch()

    def filter_batch(self, *_):
        filename = self.batch_search.text().strip().casefold()
        included = {tag.casefold() for tag in self.selected_batch_tags}
        excluded = {tag.casefold() for tag in self.excluded_batch_tags}
        visible = []
        for rel in self.batch_results:
            if filename and filename not in rel.casefold():
                continue
            names = self.batch_tag_cache.get((rel, "names"), ())
            if included.issubset(names) and not (excluded & names):
                visible.append(rel)
        visible.sort(key=str.casefold)
        self.batch_visible_paths = visible
        self.batch_gallery_page = min(self.batch_gallery_page, max(0, (len(visible)-1)//60))
        self.batch_meta.setText(f"{len(visible):,} / {len(self.batch_results):,} images · {self.batch_folder or 'No folder'}")
        self.render_batch_gallery()

    def change_batch_page(self, delta):
        pages = max(1, (len(self.batch_visible_paths) + 59) // 60)
        self.batch_gallery_page = max(0, min(pages - 1, self.batch_gallery_page + delta))
        self.render_batch_gallery()

    def render_batch_gallery(self):
        if not hasattr(self, "batch_gallery"):
            return
        self.batch_gallery.clear()
        total = len(self.batch_visible_paths)
        pages = max(1, (total + 59) // 60)
        self.batch_gallery_page = max(0, min(pages - 1, self.batch_gallery_page))
        offset = self.batch_gallery_page * 60
        paths = self.batch_visible_paths[offset:offset + 60]
        self._thumbnail_queue = deque()
        for rel in paths:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, rel)
            item.setText("")
            item.setToolTip(rel)
            self.batch_gallery.addItem(item)
            item.setSizeHint(QSize(186, 190))
            path = self.resolve_batch_image(rel)
            if path and os.path.isfile(path):
                cache_key = (path, 176, 176)
                cached = self._thumbnail_cache.get(cache_key)
                if cached is not None:
                    item.setData(Qt.DecorationRole, cached)
                else:
                    self._thumbnail_queue.append((item, path, cache_key))
        self.batch_gallery_title.setText(f"Gallery · {total:,} images")
        self.batch_page_label.setText(f"Page {self.batch_gallery_page + 1} / {pages}")
        self.batch_prev.setEnabled(self.batch_gallery_page > 0)
        self.batch_next.setEnabled(self.batch_gallery_page < pages - 1)
        if self._thumbnail_queue:
            self._thumbnail_timer.start(1)

    def _load_next_thumbnail_chunk(self):
        # Small chunks let the GUI repaint and respond between thumbnail decodes.
        for _ in range(min(6, len(self._thumbnail_queue))):
            item, path, cache_key = self._thumbnail_queue.popleft()
            try:
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                source_size = reader.size()
                target = QSize(176, 176)
                if source_size.isValid() and source_size.width() > 0 and source_size.height() > 0:
                    decode_size = source_size.scaled(target, Qt.KeepAspectRatio)
                    reader.setScaledSize(decode_size)
                image = reader.read()
                if not image.isNull():
                    # Decode close to thumbnail resolution; compose on a square canvas
                    # so portrait/landscape artwork fills the gallery consistently.
                    image = image.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    canvas = QImage(target, QImage.Format_ARGB32)
                    canvas.fill(QColor(16, 16, 16, 255))
                    painter = QPainter(canvas)
                    x = (target.width() - image.width()) // 2
                    y = (target.height() - image.height()) // 2
                    painter.drawImage(x, y, image)
                    painter.end()
                    pixmap = QPixmap.fromImage(canvas)
                    self._thumbnail_cache[cache_key] = pixmap
                    if len(self._thumbnail_cache) > 300:
                        self._thumbnail_cache.pop(next(iter(self._thumbnail_cache)))
                    if item and self.batch_gallery.row(item) >= 0:
                        item.setData(Qt.DecorationRole, pixmap)
            except Exception:
                pass
        if self._thumbnail_queue:
            self._thumbnail_timer.start(1)

    @staticmethod
    def _load_category_tags(path):
        """Load a category vocabulary file, returning normalized tag names."""
        try:
            return {tag.strip().casefold() for tag in load_tags(path) if tag.strip()}
        except (OSError, UnicodeError):
            return set()

    def tag_category(self, tag):
        key = str(tag).strip().casefold()
        if key in self.character_tags:
            return "character"
        if key in self.copyright_tags:
            return "copyright"
        return "general"


    def tag_category_rank(self, tag):
        # Characters first, series/copyright second, then ordinary tags.
        return {"character": 0, "copyright": 1, "general": 2}[self.tag_category(tag)]


    def tag_category_color(self, tag):
        return {
            "character": QColor("#65c9ff"),
            "copyright": QColor("#c89aff"),
            "general": QColor("#d5dedb"),
        }[self.tag_category(tag)]



    def _refresh_available_tag_list(self, query=""):
        if not hasattr(self, "batch_available_tags"):
            return
        query = (query or "").strip().casefold()
        # Counts and mean scores are pre-aggregated once per workspace/threshold.
        # Never scan every image's score map while the user types.
        ordered = getattr(self, "_batch_available_tag_order", [])
        if query:
            tags = [tag for tag in ordered if query in tag.casefold()]
        else:
            tags = ordered
        self.batch_available_tags.setUpdatesEnabled(False)
        self.batch_available_tags.clear()
        for tag in tags:
            count = self.batch_tag_counts.get(tag, 0)
            avg_score = self.batch_tag_avg_scores.get(tag, 0.0)
            item = QListWidgetItem(f"{tag}    {avg_score * 100:.1f}%  ·  {count:,}")
            item.setData(Qt.UserRole, tag)
            category = self.tag_category(tag)
            category_surfaces = {
                "character": QColor("#24364a"),
                "copyright": QColor("#352b46"),
                "general": QColor("#263b35"),
            }
            item.setForeground(self.tag_category_color(tag))
            item.setBackground(category_surfaces.get(category, QColor("#22252c")))
            item.setToolTip(f"{category.title()} tag · average confidence {avg_score * 100:.1f}% · click to include")
            self.batch_available_tags.addItem(item)
        self.batch_available_tags.setUpdatesEnabled(True)

    def _click_available_batch_tag(self, item):
        tag = item.data(Qt.UserRole)
        if tag:
            self.add_batch_tag(tag, True)

    def open_batch_detail(self, item):
        rel = item.data(Qt.UserRole)
        if rel:
            self.batch_detail_rel = rel
            self.batch_stack.setCurrentIndex(1)
            self.refresh_batch_detail()

    def refresh_batch_detail(self):
        rel = self.batch_detail_rel
        if not rel:
            return
        path = self.resolve_batch_image(rel)
        self.batch_detail_name.setText(Path(rel).name)
        self.batch_image_path_label.setText(rel)
        pix = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        self.batch_preview.setPixmap(pix.scaled(self.batch_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pix.isNull() else QPixmap())
        self.batch_preview.setText("" if not pix.isNull() else "Image file not found")
        self.batch_detail_tags.clear()
        entries = self.normalize_entries(self.batch_results.get(rel, []))
        min_accuracy = self.batch_detail_accuracy.value() / 100.0
        entries = [(tag, score) for tag, score in entries if score >= min_accuracy]
        # Image-detail view is category grouped; gallery/popularity lists remain untouched.
        category_order = [("character", "CHARACTERS"), ("copyright", "ANIME / SERIES"), ("general", "GENERAL TAGS")]
        for category, heading in category_order:
            grouped = [(tag, score) for tag, score in entries if self.tag_category(tag) == category]
            if not grouped:
                continue
            header = QListWidgetItem(heading)
            header.setData(Qt.UserRole, None)
            header.setFlags(Qt.NoItemFlags)
            header.setForeground(QColor("#93a4b8"))
            self.batch_detail_tags.addItem(header)
            # Keep within-category confidence order, but omit noisy percentages in this view.
            grouped.sort(key=lambda pair: (-pair[1], pair[0].casefold()))
            for tag, score in grouped:
                row = QListWidgetItem()
                row.setData(Qt.UserRole, tag)
                row.setToolTip(f"{tag} · {score * 100:.1f}% confidence · {category}")
                row.setSizeHint(QSize(0, 42))
                self.batch_detail_tags.addItem(row)
                pill = QFrame(self.batch_detail_tags)
                pill.setObjectName("detailTagPill")
                pill.setMinimumHeight(36)
                pill_color = self.tag_category_color(tag)
                pill.setStyleSheet(
                    f"QFrame#detailTagPill {{ background: rgba({pill_color.red()}, {pill_color.green()}, {pill_color.blue()}, 14); "
                    f"border: 1px solid rgba({pill_color.red()}, {pill_color.green()}, {pill_color.blue()}, 65); border-radius: 10px; }}"
                )
                pill_layout = QHBoxLayout(pill)
                pill_layout.setContentsMargins(11, 5, 10, 5)
                pill_layout.setSpacing(10)
                tag_label = QLabel(tag)
                tag_label.setMinimumHeight(20)
                tag_label.setWordWrap(False)
                tag_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                tag_label.setStyleSheet(f"color: {pill_color.name()}; background: transparent; border: 0; font-size: 12px;")
                percent_label = QLabel(f"{score * 100:.1f}%")
                percent_label.setMinimumWidth(44)
                percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                percent_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
                percent_label.setStyleSheet("color: #8995a8; background: transparent; border: 0; font-size: 10px;")
                pill_layout.addWidget(tag_label, 1)
                pill_layout.addWidget(percent_label, 0)
                self.batch_detail_tags.setItemWidget(row, pill)

    def save_batch_tag_edits(self):
        folder = self.batch_json_folder or self.batch_folder
        if folder and os.path.isdir(folder):
            target = Path(folder) / "tags_results.json"
            tmp = target.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(self.batch_results, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, target)
            except Exception as exc:
                self.show_error(f"Could not autosave tag edits: {exc}")
        self.populate_batch()
        self.refresh_batch_detail()

    def add_batch_image_tag(self):
        if not self.batch_detail_rel:
            return
        tag, ok = QInputDialog.getText(self, "Add tag", "Tag name:")
        tag = tag.strip()
        if not ok or not tag:
            return
        entries = self.normalize_entries(self.batch_results.get(self.batch_detail_rel, []))
        if any(t.casefold() == tag.casefold() for t, _ in entries):
            return
        confidence, score_ok = QInputDialog.getDouble(
            self, "Tag confidence", "Confidence (%): ", 100.0, 0.0, 100.0, 1
        )
        if not score_ok:
            return
        entries.append((tag, confidence / 100.0))
        self.batch_results[self.batch_detail_rel] = [[t, score] for t, score in entries]
        self.save_batch_tag_edits()

    def edit_batch_image_tag(self, item):
        if not item or not self.batch_detail_rel:
            return
        old = item.data(Qt.UserRole)
        new, ok = QInputDialog.getText(self, "Edit tag", "Tag name:", text=old)
        new = new.strip()
        if not ok or not new:
            return
        entries = self.normalize_entries(self.batch_results.get(self.batch_detail_rel, []))
        if any(t.casefold() == new.casefold() and t != old for t, _ in entries):
            QMessageBox.information(self, "Edit tag", "That tag already exists.")
            return
        current_score = next((score for t, score in entries if t == old), 1.0)
        confidence, score_ok = QInputDialog.getDouble(
            self, "Edit tag confidence", "Confidence (%):", current_score * 100.0, 0.0, 100.0, 1
        )
        if not score_ok:
            return
        self.batch_results[self.batch_detail_rel] = [
            [new if t == old else t, confidence / 100.0 if t == old else score]
            for t, score in entries
        ]
        self.save_batch_tag_edits()

    def remove_batch_image_tag(self):
        item = self.batch_detail_tags.currentItem()
        if not item or not self.batch_detail_rel:
            return
        tag = item.data(Qt.UserRole)
        entries = [(t, score) for t, score in self.normalize_entries(self.batch_results.get(self.batch_detail_rel, [])) if t != tag]
        self.batch_results[self.batch_detail_rel] = [[t, score] for t, score in entries]
        self.save_batch_tag_edits()

    def resolve_batch_image(self, rel):
        # First honor an existing absolute path exactly as stored.
        candidate = Path(rel).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate)
        roots = [self.batch_folder, self.batch_json_folder]
        # Resolve relative path against known folders; Path handles nested subfolders.
        for root in roots:
            if root:
                path = Path(root) / rel
                if path.is_file():
                    return str(path.resolve())
        # A result may store paths relative to a previous batch root. Match filename
        # against the one-time recursive index, preferring an unambiguous candidate.
        matches = getattr(self, "_batch_basename_index", {}).get(Path(rel).name.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Prefer path suffix match (e.g. images/subfolder/file.png).
            suffix = Path(rel).as_posix().casefold().lstrip("./")
            for path in matches:
                if Path(path).as_posix().casefold().endswith(suffix):
                    return path
        return None


    def open_batch_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open batch results", "", "JSON (*.json)")
        if path:
            self.load_batch_json_path(path)

    def load_batch_json_path(self, path, workspace_folder=None):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object keyed by image path.")
            self.batch_results = data
            folder = workspace_folder or os.path.dirname(path)
            self.batch_folder = folder
            self.batch_json_folder = os.path.dirname(path)
            self.save_recent_workspace(path if Path(path).name.lower() != "tags_results.json" else folder,
                                       "json" if Path(path).name.lower() != "tags_results.json" else "folder")
            self.populate_batch()
            self.batch_stack.setCurrentIndex(0)
            self.tabs.setCurrentIndex(2)
            self.status_label.setText(f"Loaded {len(data):,} image records")
            self.batch_meta.setText(f"{len(data):,} image records · {folder}")
        except Exception as exc:
            self.show_error(f"Could not open results:\n{exc}")

    def export_batch_json(self):
        if not self.batch_results:
            QMessageBox.information(self, "Export", "No batch results are loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save batch JSON", "tags_results.json", "JSON (*.json)")
        if not path: return
        if not path.lower().endswith(".json"): path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.batch_results, f, indent=2, ensure_ascii=False)
            self.status_label.setText(f"Saved {Path(path).name}")
        except Exception as exc: self.show_error(f"Could not save JSON:\n{exc}")

    def export_batch_csv(self):
        if not self.batch_results:
            QMessageBox.information(self, "Export", "No batch results are loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export batch CSV", "tags_results.csv", "CSV (*.csv)")
        if not path: return
        if not path.lower().endswith(".csv"): path += ".csv"
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "tag", "confidence"])
                for image, entries in sorted(self.batch_results.items()):
                    for tag, score in self.normalize_entries(entries):
                        writer.writerow([image, tag, score])
            self.status_label.setText(f"Exported batch CSV · {Path(path).name}")
        except Exception as exc: self.show_error(f"Could not export CSV:\n{exc}")

    def show_error(self, message):
        QMessageBox.warning(self, APP_NAME, str(message))

    def closeEvent(self, event):
        # Do not call methods on stale Qt wrappers. BatchThread used to be
        # deleteLater()'d while self.batch_thread still referenced it, causing
        # "wrapped C/C++ object ... has been deleted" on shutdown.
        thread = self.batch_thread
        try:
            running = bool(thread and thread.isRunning())
        except RuntimeError:
            running = False

        if running:
            answer = QMessageBox.question(
                self, "Batch in progress",
                "A batch is still running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            try:
                thread.stop()
                # Wait for the current image prediction to finish before Qt
                # destroys the model or the QThread wrapper.
                thread.wait()
            except RuntimeError:
                pass

        single = self.single_thread
        try:
            if single and single.isRunning():
                single.wait()
        except RuntimeError:
            pass
        event.accept()


JA_TEXT = {
    "  Get started  ": "  はじめる  ",
    "  Image workspace  ": "  画像タグ付け  ",
    "  Batch library  ": "  一括処理  ",
    "Language / 言語": "表示言語",
    "Private, local AI tagging · single images and batch folders": "画像はPC内で処理。1枚ずつでもフォルダまとめてもOK",
    "▦  Batch folder": "▦  フォルダを一括処理",
    "＋  Open image": "＋  画像を開く",
    "Preparing…": "準備中…",
    "Pause": "一時停止", "Resume": "再開", "Stop": "停止",
    "SINGLE IMAGE": "画像を1枚ずつ", "BATCH FOLDER": "フォルダをまとめて",
    "Preview": "プレビュー", "Drag an image here": "ここに画像をドロップ",
    "Drop an image here\n\nor use  Open image": "画像をここにドロップ\n\nまたは「画像を開く」を選択",
    "No image selected": "画像が選択されていません",
    "Detected tags": "検出されたタグ", "Sorted by confidence": "確信度が高い順",
    "Search tags…": "タグを検索…", "Minimum confidence": "最低確信度",
    "Tag": "タグ", "Confidence": "確信度", "Copy tags": "タグをコピー",
    "＋ Add tag": "＋ タグを追加", "Remove selected": "選択したタグを削除",
    "Export…": "書き出し…", "0 tags": "0件のタグ",
    "Batch library": "一括処理ライブラリ", "Search by filename or tag": "ファイル名・タグで検索",
    "Filter filenames…": "ファイル名で絞り込み…",
    "Type a tag to include (+) or exclude (−)…": "タグを入力して追加（＋）・除外（−）…",
    "Type to find tags; + includes and − excludes": "タグを検索。＋で追加、−で除外",
    "Image": "画像", "Tags": "タグ", "Open results…": "結果を開く…",
    "Export CSV…": "CSVを書き出す…", "Save JSON…": "JSONを保存…",
    "Export filtered images + JSON": "絞り込んだ画像＋JSONを書き出す",
    "Choose a folder to start a batch": "フォルダを選ぶと一括処理を開始します",
    "Image preview": "画像プレビュー", "Select an image from the list": "一覧から画像を選んでください",
    "Tags for selected image will appear here.": "選択した画像のタグがここに表示されます。",
    "Loading AI model…": "AIモデルを読み込み中…", "Model unavailable": "モデルを読み込めません",
    "Ready · {count} tags loaded": "準備完了 · タグを{count}件読み込みました",
    "No matching tags": "一致するタグはありません",
    "Could not load model": "モデルを読み込めませんでした",
    "Drop a supported image file or a folder containing images.": "対応している画像ファイル、または画像が入ったフォルダをドロップしてください。",
    "Add tag": "タグを追加", "Tag name:": "タグ名：", "Remove tags": "タグを削除",
    "Select one or more rows first.": "先にタグを1つ以上選択してください。",
    "Analyze an image first.": "先に画像を解析してください。", "Export": "書き出し",
    "Choose image folder": "画像フォルダを選択", "Please choose a folder containing images.": "画像が入ったフォルダを選択してください。",
    "Paused": "一時停止中", "Completed": "完了", "Starting batch…": "一括処理を開始中…",
    "No supported images were found in that folder.": "このフォルダに対応している画像が見つかりませんでした。",
    "Batch in progress": "一括処理中", "A batch is still running. Stop it and exit?": "一括処理がまだ実行中です。停止して終了しますか？",
}


STYLE = """
/* Polished product UI: layered graphite, clean typography, restrained accent colors. */
QWidget { background:#17191e; color:#e4e7ed; font-family:'Segoe UI',sans-serif; font-size:13px; }
QLabel#appTitle { font-size:26px; font-weight:700; color:#f1f3f7; }
QLabel#muted, QLabel#suggestionCount { color:#9299a8; font-size:12px; background:transparent; border:0; }
QWidget#tagSuggestionRow { background:#22252d; border:1px solid #343944; border-radius:9px; }
QLabel#suggestionTag { font-family:'Segoe UI',sans-serif; font-size:14px; font-weight:600; padding:4px 7px; background:transparent; border:0; }
QLabel#sectionTitle { background:transparent; border:0; font-size:13px; font-weight:650; color:#c6cbd5; }
QLabel#status { color:#a5adbb; font-weight:600; background:transparent; border:0; }
QFrame#batchSidebar { background:#1d2026; border:0; border-right:1px solid #30343c; border-radius:0; }
QFrame#globalAccuracyGroup { background:#222832; border:1px solid #364252; border-left:3px solid #6e9fce; border-radius:10px; }
QLabel#globalAccuracyTitle { color:#91b8df; font-size:10px; font-weight:800; letter-spacing:1px; padding:1px 2px 3px; background:transparent; border:0; }
QListWidget#batchGallery { background:#17191e; border:0; border-radius:0; padding:3px; }
QListWidget#batchGallery::item { background:#22252c; border:1px solid #30343c; border-radius:7px; padding:3px; margin:3px; color:#aeb5c1; }
QListWidget#batchGallery::item:selected { border:1px solid #668fc2; background:#293447; }
QLabel#activeFilterHeading { background:transparent; border:0; color:#aab4c5; font-size:10px; font-weight:800; letter-spacing:1px; padding:3px 1px 1px; }
QListWidget#activeBatchFilters { background:transparent; border:0; border-radius:0; padding:2px; outline:0; }
QListWidget#activeBatchFilters::item { border:0; background:transparent; padding:0; margin:0; }
QFrame#activeIncludeRow { background:#1d302b; border:1px solid #315248; border-left:3px solid #55b895; border-radius:7px; }
QFrame#activeExcludeRow { background:#30232a; border:1px solid #523743; border-left:3px solid #d07b91; border-radius:7px; }
QLabel#activeIncludeBadge { background:#254a3e; color:#87d4b2; border-radius:4px; font-size:9px; font-weight:900; padding:3px; }
QLabel#activeExcludeBadge { background:#4a2d39; color:#e2a0b2; border-radius:4px; font-size:9px; font-weight:900; padding:3px; }
QLabel#activeFilterTag { color:#d4dae5; font-size:12px; font-weight:600; padding:3px 1px; background:transparent; border:0; }
QPushButton#activeFilterRemove { background:transparent; color:#a3b8b2; border:1px solid #3b514c; border-radius:5px; font-size:15px; padding:0; }
QPushButton#activeFilterRemove:hover { background:#293d37; color:#c6e4d8; border-color:#56806f; }
QScrollArea#selectedTagScroll { background:transparent; border:0; }
QScrollArea#selectedTagScroll QWidget#qt_scrollarea_viewport { background:transparent; }
QListWidget#batchAvailableTags { background:transparent; border:0; border-radius:0; color:#cbd2dd; outline:0; padding:2px; }
QListWidget#batchAvailableTags::item { background:#22252c; border:1px solid #30343c; border-radius:7px; padding:8px 10px; margin:3px 2px; min-height:24px; }
QListWidget#batchAvailableTags::item:hover { background:#292d36; border-color:#4a5362; }
QListWidget#batchAvailableTags::item:selected { background:#293448; border:1px solid #5c82b4; }
QListWidget#batchDetailTags { background:#22252c; border:1px solid #343944; border-radius:10px; padding:5px; }
QListWidget#batchDetailTags::item { border:0; border-radius:0; background:transparent; padding:0; margin:0; min-height:20px; }
QListWidget#batchDetailTags::item:selected,QListWidget#batchDetailTags::item:hover { background:transparent; border:0; }
QListWidget#startupHistory { background:#22252c; border:1px solid #343944; border-radius:9px; padding:3px; }
QListWidget#startupHistory::item { padding:5px 8px; margin:1px; border-radius:5px; color:#c6ccd7; }
QListWidget#startupHistory::item:hover { background:#2c3038; }
QListWidget#startupHistory::item:selected { background:#293448; border:1px solid #4b668c; }
QFrame#card, QFrame#footer { background:#202329; border:1px solid #30343c; border-radius:12px; }
QFrame#tagChipInclude { background:#1d302b; border:1px solid #315248; border-radius:7px; }
QFrame#tagChipExclude { background:#30232a; border:1px solid #523743; border-radius:7px; }
QLabel#dropArea, QLabel#batchPreview { background:#22252c; border:1px dashed #414752; border-radius:11px; color:#969eac; padding:12px; }
QPushButton { background:#292c33; border:1px solid #3a3f49; border-radius:7px; padding:8px 11px; font-weight:600; color:#d1d6e0; }
QPushButton:hover { background:#333740; border-color:#505866; }
QPushButton#tagIncludeAction { background:#203b32; color:#74d0a7; border:1px solid #35624f; border-radius:7px; padding:0; font-size:23px; font-weight:700; }
QPushButton#tagIncludeAction:hover { background:#2b4d3f; border-color:#5b9a7e; color:#b0e6ce; }
QPushButton#tagExcludeAction { background:#3a2630; color:#d78da3; border:1px solid #64404e; border-radius:7px; padding:0; font-size:23px; font-weight:700; }
QPushButton#tagExcludeAction:hover { background:#4a303d; border-color:#8b5b71; color:#efb6c9; }
QPushButton:disabled { color:#666d78; background:#24262b; border-color:#30343a; }
QPushButton#primary { background:#567eae; color:#f4f7fb; border:0; }
QPushButton#primary:hover { background:#6590c2; }
QPushButton#exportFiltered { background:#3d755f; color:#e6f5ee; border:1px solid #4d8b70; font-weight:750; padding:10px 14px; }
QPushButton#exportFiltered:hover { background:#4b896e; border-color:#66a587; }
QPushButton#exportFiltered:disabled { background:#24262b; color:#666d78; border-color:#30343a; }
QPushButton#danger { color:#d58b9e; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox { background:#22252c; color:#e0e4eb; border:1px solid #363b45; border-radius:7px; padding:7px 9px; selection-background-color:#526f9b; selection-color:#f7f8fb; }
QFrame#batchTagPopup { background:#22252c; border:1px solid #3a3f49; border-radius:9px; }
QFrame#batchTagPopup QWidget { background:transparent; }
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus { border-color:#6489b5; }
QTabWidget::pane { border:0; background:transparent; }
QTabBar::tab { background:transparent; color:#8f97a5; padding:10px 16px; margin-right:4px; border-bottom:2px solid transparent; font-weight:600; }
QTabBar::tab:selected { color:#c6d8f0; border-bottom:2px solid #7196c5; }
QTabBar::tab:hover { color:#e0e5ee; }
QTableWidget { background:#22252c; alternate-background-color:#202329; border:1px solid #363b45; border-radius:8px; gridline-color:transparent; selection-background-color:#2d394c; selection-color:#e3e9f2; outline:0; }
QTableWidget::item { padding:7px; border:0; }
QHeaderView::section { background:#272a31; color:#969eac; padding:9px; border:0; border-bottom:1px solid #363b45; font-weight:700; }
QProgressBar { background:#2b2e35; border:0; border-radius:4px; height:7px; text-align:center; color:transparent; }
QProgressBar::chunk { background:#688fbd; border-radius:4px; }
QSplitter::handle { background:#292c33; width:5px; }
QScrollBar:vertical { background:transparent; width:9px; }
QScrollBar::handle:vertical { background:#454b56; border-radius:4px; min-height:28px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QMessageBox { background:#22252c; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = ImageTagger()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
