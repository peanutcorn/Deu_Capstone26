import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox,
    QSizePolicy, QStatusBar, QComboBox, QCheckBox, QSpinBox,
    QMessageBox, QFrame, QRadioButton, QButtonGroup, QLineEdit
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont, QIcon

from core.video_thread import VideoThread


class VideoDisplay(QLabel):
    """영상 출력 라벨 — 종횡비 유지하며 스케일."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)
        self._pixmap = None

    def set_frame(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()

    def _rescale(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV 이상 현상 감지 시스템")
        self.resize(1280, 780)
        self._thread: VideoThread | None = None
        self._build_ui()
        self._apply_stylesheet()

    # ------------------------------------------------------------------ UI 구성
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_video_panel(), stretch=4)
        root.addWidget(self._build_control_panel(), stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("대기 중 — 파일 또는 웹캠을 선택하세요.")

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.video_display = VideoDisplay()
        self.video_display.setText("영상을 불러오세요")
        self.video_display.setFont(QFont("Malgun Gothic", 16))
        self.video_display.setStyleSheet(
            "background-color: #1a1a2e; color: #8888aa; border-radius: 6px;"
        )
        layout.addWidget(self.video_display)

        # 하단 재생 컨트롤
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self.btn_play = QPushButton("▶  재생")
        self.btn_play.setFixedHeight(36)
        self.btn_play.clicked.connect(self._on_play_pause)
        self.btn_play.setEnabled(False)

        self.btn_stop = QPushButton("■  정지")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        layout.addLayout(ctrl)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_stats_group())
        layout.addStretch()
        return panel

    def _build_source_group(self) -> QGroupBox:
        grp = QGroupBox("영상 소스")
        layout = QVBoxLayout(grp)

        self.btn_open_file = QPushButton("파일 열기...")
        self.btn_open_file.setFixedHeight(34)
        self.btn_open_file.clicked.connect(self._on_open_file)

        self.btn_webcam = QPushButton("웹캠 시작")
        self.btn_webcam.setFixedHeight(34)
        self.btn_webcam.clicked.connect(self._on_webcam)

        self.lbl_source = QLabel("소스: (없음)")
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setStyleSheet("color: #aaaaaa; font-size: 11px;")

        layout.addWidget(self.btn_open_file)
        layout.addWidget(self.btn_webcam)
        layout.addWidget(self.lbl_source)
        return grp

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("모델 설정")
        layout = QVBoxLayout(grp)

        # --- 모델 유형 선택 ---
        self._model_type_grp = QButtonGroup(self)
        self.radio_yolo = QRadioButton("YOLO 사전학습 모델")
        self.radio_custom = QRadioButton("커스텀 학습 모델 (.pt)")
        self.radio_yolo.setChecked(True)
        self._model_type_grp.addButton(self.radio_yolo, 0)
        self._model_type_grp.addButton(self.radio_custom, 1)
        self.radio_yolo.toggled.connect(self._on_model_type_toggled)
        layout.addWidget(self.radio_yolo)
        layout.addWidget(self.radio_custom)

        # YOLO 사전학습 콤보 (일반 + 포즈)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "── 일반 감지 ──",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "── 포즈 감지 ──",
            "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
        ])
        # 구분선 항목 비활성화
        for i in (0, 6):
            self.combo_model.model().item(i).setEnabled(False)
        self.combo_model.setCurrentIndex(1)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.combo_model)

        # 커스텀 모델 파일 선택
        custom_row = QHBoxLayout()
        self.edit_custom_model = QLineEdit()
        self.edit_custom_model.setPlaceholderText("best.pt 경로...")
        self.edit_custom_model.setEnabled(False)
        self.btn_browse_model = QPushButton("찾기")
        self.btn_browse_model.setFixedWidth(46)
        self.btn_browse_model.setEnabled(False)
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        custom_row.addWidget(self.edit_custom_model)
        custom_row.addWidget(self.btn_browse_model)
        layout.addLayout(custom_row)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2e2e4e;")
        layout.addWidget(line)

        lbl_conf = QLabel("신뢰도 임계값")
        layout.addWidget(lbl_conf)

        conf_row = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        self.lbl_conf_val = QLabel("0.50")
        self.lbl_conf_val.setFixedWidth(36)
        conf_row.addWidget(self.slider_conf)
        conf_row.addWidget(self.lbl_conf_val)
        layout.addLayout(conf_row)

        layout.addWidget(QLabel("DeepSORT 최대 소실 프레임"))
        self.spin_max_age = QSpinBox()
        self.spin_max_age.setRange(1, 120)
        self.spin_max_age.setValue(30)
        layout.addWidget(self.spin_max_age)

        return grp

    def _build_display_group(self) -> QGroupBox:
        grp = QGroupBox("표시 옵션")
        layout = QVBoxLayout(grp)
        self.chk_bbox = QCheckBox("바운딩 박스")
        self.chk_bbox.setChecked(True)
        self.chk_track_id = QCheckBox("트랙 ID")
        self.chk_track_id.setChecked(True)
        self.chk_keypoints = QCheckBox("관절 키포인트 (포즈 모델)")
        self.chk_keypoints.setChecked(True)
        layout.addWidget(self.chk_bbox)
        layout.addWidget(self.chk_track_id)
        layout.addWidget(self.chk_keypoints)
        return grp

    def _build_stats_group(self) -> QGroupBox:
        grp = QGroupBox("실시간 통계")
        layout = QVBoxLayout(grp)

        self.lbl_count = QLabel("감지된 인원: 0명")
        self.lbl_count.setFont(QFont("Malgun Gothic", 13, QFont.Bold))
        self.lbl_count.setStyleSheet("color: #00d4aa;")

        self.lbl_fps = QLabel("FPS: —")
        self.lbl_fps.setFont(QFont("Malgun Gothic", 11))
        self.lbl_fps.setStyleSheet("color: #aaaaaa;")

        layout.addWidget(self.lbl_count)
        layout.addWidget(self.lbl_fps)
        return grp

    # ------------------------------------------------------------------ 이벤트
    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "동영상 파일 선택", "",
            "동영상 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;모든 파일 (*)"
        )
        if path:
            self._stop_thread()
            self.lbl_source.setText(f"파일: {os.path.basename(path)}")
            self._start_thread(path)

    def _on_webcam(self):
        self._stop_thread()
        self.lbl_source.setText("소스: 웹캠 (0)")
        self._start_thread(0)

    def _on_play_pause(self):
        if self._thread is None:
            return
        if self._thread._paused:
            self._thread.resume()
            self.btn_play.setText("⏸  일시정지")
        else:
            self._thread.pause()
            self.btn_play.setText("▶  재생")

    def _on_stop(self):
        self._stop_thread()
        self.video_display.setText("영상을 불러오세요")
        self.video_display._pixmap = None
        self.lbl_count.setText("감지된 인원: 0명")
        self.lbl_fps.setText("FPS: —")
        self.status_bar.showMessage("정지됨.")

    def _on_model_type_toggled(self, yolo_selected: bool):
        self.combo_model.setEnabled(yolo_selected)
        self.edit_custom_model.setEnabled(not yolo_selected)
        self.btn_browse_model.setEnabled(not yolo_selected)

    def _on_model_changed(self, index: int):
        text = self.combo_model.itemText(index)
        is_pose = "pose" in text
        self.chk_keypoints.setEnabled(is_pose)
        if not is_pose:
            self.chk_keypoints.setChecked(False)
        else:
            self.chk_keypoints.setChecked(True)

    def _on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "학습 모델 파일 선택", "",
            "PyTorch 모델 (*.pt *.pth);;모든 파일 (*)"
        )
        if path:
            self.edit_custom_model.setText(path)

    def _on_conf_changed(self, value: int):
        conf = value / 100.0
        self.lbl_conf_val.setText(f"{conf:.2f}")
        if self._thread:
            self._thread.set_conf(conf)

    def _on_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.video_display.set_frame(pixmap)

    def _on_stats(self, count: int, fps: float):
        self.lbl_count.setText(f"감지된 인원: {count}명")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "오류", msg)
        self._stop_thread()

    def _on_thread_finished(self):
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_play.setText("▶  재생")
        self.status_bar.showMessage("재생 완료.")

    # ------------------------------------------------------------------ 스레드
    def _resolve_model_path(self) -> str | None:
        if self.radio_yolo.isChecked():
            return self.combo_model.currentText()
        path = self.edit_custom_model.text().strip()
        if not path:
            QMessageBox.warning(self, "모델 미선택", "커스텀 모델 파일 경로를 입력하세요.")
            return None
        if not os.path.isfile(path):
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{path}")
            return None
        return path

    def _start_thread(self, source):
        model = self._resolve_model_path()
        if model is None:
            return
        conf = self.slider_conf.value() / 100.0
        max_age = self.spin_max_age.value()

        self._thread = VideoThread(source, model, conf, max_age)
        self._thread.show_bbox = self.chk_bbox.isChecked()
        self._thread.show_track_id = self.chk_track_id.isChecked()
        self._thread.show_keypoints = self.chk_keypoints.isChecked()
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.stats_updated.connect(self._on_stats)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.finished_signal.connect(self._on_thread_finished)

        # 체크박스 상태를 스레드에 실시간 반영
        # bool 단순 속성 쓰기는 GIL이 원자성을 보장하므로 별도 락 불필요
        self.chk_bbox.stateChanged.connect(
            lambda s: setattr(self._thread, 'show_bbox', bool(s))
        )
        self.chk_track_id.stateChanged.connect(
            lambda s: setattr(self._thread, 'show_track_id', bool(s))
        )
        self.chk_keypoints.stateChanged.connect(
            lambda s: setattr(self._thread, 'show_keypoints', bool(s))
        )

        self._thread.start()
        self.btn_play.setEnabled(True)
        self.btn_play.setText("⏸  일시정지")
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("처리 중...")

    def _stop_thread(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        self._thread = None

    def closeEvent(self, event):
        self._stop_thread()
        event.accept()

    # ------------------------------------------------------------------ 스타일
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #12121e;
                color: #e0e0f0;
                font-family: 'Malgun Gothic', sans-serif;
            }
            QGroupBox {
                border: 1px solid #2e2e4e;
                border-radius: 6px;
                margin-top: 10px;
                padding: 6px;
                font-weight: bold;
                color: #aaaacc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2a2a4a;
                color: #e0e0f0;
                border: 1px solid #3a3a6a;
                border-radius: 5px;
                padding: 4px 10px;
            }
            QPushButton:hover { background-color: #3a3a6a; }
            QPushButton:pressed { background-color: #1a1a3a; }
            QPushButton:disabled { color: #555566; border-color: #2a2a4a; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2e2e4e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d4aa;
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #00d4aa; border-radius: 2px; }
            QComboBox {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QComboBox::drop-down { border: none; }
            QSpinBox {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QCheckBox { color: #c0c0e0; }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px;
                border: 1px solid #3a3a6a; background: #1e1e3e; }
            QCheckBox::indicator:checked { background: #00d4aa; border-color: #00d4aa; }
            QRadioButton { color: #c0c0e0; }
            QRadioButton::indicator { width: 13px; height: 13px; border-radius: 7px;
                border: 1px solid #3a3a6a; background: #1e1e3e; }
            QRadioButton::indicator:checked { background: #00d4aa; border-color: #00d4aa; }
            QRadioButton:disabled { color: #555566; }
            QLineEdit {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QLineEdit:disabled { color: #555566; border-color: #2a2a4a; }
            QLineEdit:focus { border-color: #00d4aa; }
            QStatusBar { background-color: #0e0e1e; color: #8888aa; }
            QLabel { color: #c0c0e0; }
        """)
