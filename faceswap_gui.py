import sys
import cv2
import numpy as np
from faceswap_engine import FaceSwapEngine

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QFont

 # your existing engine file


class ModernPanel(QFrame):
    """A sleek dark card-like widget."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #222;
                border-radius: 14px;
                border: 1px solid #444;
            }
        """)


class FaceSwapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultra FaceSwap — Modern Edition")
        self.setStyleSheet("background-color: #111; color: white;")
        self.resize(1200, 700)

        # -------------------------
        # ENGINE
        # -------------------------
        self.engine = FaceSwapEngine(scale_for_speed=0.6, use_seamless_clone=False)
        self.cap = cv2.VideoCapture(0)

        # -------------------------
        # UI Layout
        # -------------------------
        main_layout = QHBoxLayout(self)

        # Left Panel (Controls)
        left_panel = ModernPanel()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(18)

        title = QLabel("🎭 Ultra Face Swap")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        left_layout.addWidget(title)

        # Target list
        self.target_list = QListWidget()
        self.target_list.setStyleSheet("""
            QListWidget {
                background-color: #191919;
                border-radius: 10px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background: #333;
            }
        """)
        self.load_target_list()

        self.target_list.currentRowChanged.connect(self.select_target)
        left_layout.addWidget(self.target_list)

        # Buttons
        btn_add = QPushButton("Add Face From Camera")
        btn_toggle = QPushButton("Toggle Swap")
        btn_reload = QPushButton("Reload Targets")
        btn_quit = QPushButton("Quit")

        for b in [btn_add, btn_toggle, btn_reload, btn_quit]:
            b.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #444;
                }
            """)

        btn_add.clicked.connect(self.add_face)
        btn_toggle.clicked.connect(self.toggle_swap)
        btn_reload.clicked.connect(self.reload_targets)
        btn_quit.clicked.connect(self.close)

        left_layout.addWidget(btn_add)
        left_layout.addWidget(btn_toggle)
        left_layout.addWidget(btn_reload)
        left_layout.addStretch()
        left_layout.addWidget(btn_quit)

        # Right Panel (Camera Display)
        right_panel = ModernPanel()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: #000; border-radius: 12px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(self.video_label)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 25)
        main_layout.addWidget(right_panel, 75)

        # -------------------------
        # Timer for camera feed
        # -------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)

    # ==========================
    # Methods
    # ==========================

    def load_target_list(self):
        self.target_list.clear()
        for t in self.engine.targets:
            item = QListWidgetItem(f"Target #{t.idx+1}")
            self.target_list.addItem(item)

        if self.engine.current_target_idx is not None:
            self.target_list.setCurrentRow(self.engine.current_target_idx)

    def select_target(self, idx):
        if idx >= 0:
            self.engine.current_target_idx = idx

    @pyqtSlot()
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        output = self.engine.process_frame(frame)

        # Convert to Qt image
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def add_face(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        ok = self.engine.add_target_from_frame(frame)
        if ok:
            self.load_target_list()

    def toggle_swap(self):
        self.engine.swap_enabled = not self.engine.swap_enabled

    def reload_targets(self):
        self.engine.load_targets_from_folder()
        self.load_target_list()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = FaceSwapGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

