"""
메인 윈도우 (Main Window)

Ego Vehicle Mask Generator의 메인 GUI 윈도우입니다.
마스크 생성, 뷰어, 에디터 기능을 통합합니다.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QMenuBar,
    QMenu,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
    QSplitter,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
)

from .mask_viewer import MaskViewer, ComparisonViewer
from .mask_editor import MaskEditor
from ..config_loader import Config, CameraConfig, load_config, get_default_config
from ..data_loader import DataLoader
from ..mask_generator import MaskGenerator, MaskResult


class MaskGenerationWorker(QThread):
    """마스크 생성 워커 스레드"""

    progress = Signal(int, str)  # progress, message
    finished = Signal(object)  # MaskResult
    error = Signal(str)

    def __init__(
        self,
        source_path: str,
        camera_config: Optional[CameraConfig] = None,
        num_samples: int = 100,
        variance_threshold: float = 50.0,
        parent=None,
    ):
        super().__init__(parent)
        self.source_path = source_path
        self.camera_config = camera_config
        self.num_samples = num_samples
        self.variance_threshold = variance_threshold
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(10, "데이터 로딩 중...")

            with DataLoader(self.source_path) as loader:
                self.progress.emit(30, f"프레임 샘플링 중 ({self.num_samples}개)...")
                frames = loader.sample_frames(self.num_samples)

                if self._is_cancelled:
                    return

                self.progress.emit(50, "마스크 생성 중...")
                generator = MaskGenerator(
                    variance_threshold=self.variance_threshold,
                )

                result = generator.generate_from_frames(
                    frames,
                    camera_config=self.camera_config,
                )

                if self._is_cancelled:
                    return

                self.progress.emit(100, "완료!")
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class ParameterPanel(QWidget):
    """파라미터 조정 패널"""

    parameters_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 샘플 수
        samples_group = QGroupBox("프레임 샘플 수")
        samples_layout = QHBoxLayout(samples_group)
        self._samples_spinbox = QSpinBox()
        self._samples_spinbox.setRange(10, 500)
        self._samples_spinbox.setValue(100)
        self._samples_spinbox.setSingleStep(10)
        samples_layout.addWidget(self._samples_spinbox)
        layout.addWidget(samples_group)

        # 분산 임계값
        threshold_group = QGroupBox("분산 임계값")
        threshold_layout = QVBoxLayout(threshold_group)

        threshold_h_layout = QHBoxLayout()
        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(1, 200)
        self._threshold_slider.setValue(50)
        threshold_h_layout.addWidget(self._threshold_slider)

        self._threshold_spinbox = QDoubleSpinBox()
        self._threshold_spinbox.setRange(1.0, 200.0)
        self._threshold_spinbox.setValue(50.0)
        self._threshold_spinbox.setFixedWidth(70)
        threshold_h_layout.addWidget(self._threshold_spinbox)
        threshold_layout.addLayout(threshold_h_layout)

        threshold_desc = QLabel("낮을수록 정적 영역 판정이 엄격해집니다")
        threshold_desc.setStyleSheet("color: #888; font-size: 11px;")
        threshold_layout.addWidget(threshold_desc)
        layout.addWidget(threshold_group)

        # 연결
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_spinbox.setValue(v)
        )
        self._threshold_spinbox.valueChanged.connect(
            lambda v: self._threshold_slider.setValue(int(v))
        )

        layout.addStretch()

        # 스타일
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #e94560;
                border: 1px solid #16213e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
        """)

    @property
    def num_samples(self) -> int:
        return self._samples_spinbox.value()

    @property
    def variance_threshold(self) -> float:
        return self._threshold_spinbox.value()


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self, config: Optional[Config] = None):
        super().__init__()

        self._config = config or get_default_config()
        self._current_source: Optional[str] = None
        self._current_camera: Optional[str] = None
        self._frames: list[np.ndarray] = []
        self._mask_result: Optional[MaskResult] = None
        self._worker: Optional[MaskGenerationWorker] = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        self.setWindowTitle("Ego Vehicle Mask Generator")
        self.resize(1400, 900)

    def _setup_ui(self):
        """UI 구성"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 좌측 패널 (카메라 선택, 파라미터)
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet("background-color: #0a0a15;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)

        # 소스 선택
        source_group = QGroupBox("데이터 소스")
        source_layout = QVBoxLayout(source_group)

        source_btn_layout = QHBoxLayout()
        self._open_video_btn = QPushButton("영상 열기")
        self._open_video_btn.clicked.connect(self._open_video)
        source_btn_layout.addWidget(self._open_video_btn)

        self._open_folder_btn = QPushButton("폴더 열기")
        self._open_folder_btn.clicked.connect(self._open_folder)
        source_btn_layout.addWidget(self._open_folder_btn)
        source_layout.addLayout(source_btn_layout)

        self._source_label = QLabel("선택된 소스 없음")
        self._source_label.setWordWrap(True)
        self._source_label.setStyleSheet("color: #888; padding: 5px;")
        source_layout.addWidget(self._source_label)
        left_layout.addWidget(source_group)

        # 카메라 선택
        camera_group = QGroupBox("카메라 설정")
        camera_layout = QVBoxLayout(camera_group)

        self._camera_combo = QComboBox()
        self._camera_combo.addItem("(자동 감지)", None)
        for camera_id in self._config.get_camera_ids():
            self._camera_combo.addItem(camera_id, camera_id)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_layout.addWidget(self._camera_combo)
        left_layout.addWidget(camera_group)

        # 파라미터 패널
        self._param_panel = ParameterPanel()
        left_layout.addWidget(self._param_panel)

        # 생성 버튼
        self._generate_btn = QPushButton("마스크 생성")
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._generate_mask)
        self._generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:pressed {
                background-color: #c73e54;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        """)
        left_layout.addWidget(self._generate_btn)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 메인 컨텐츠 영역
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # 탭 위젯
        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #16213e;
                background-color: #0f0f23;
            }
            QTabBar::tab {
                background-color: #1a1a2e;
                color: #888;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #0f0f23;
                color: #e94560;
            }
            QTabBar::tab:hover:!selected {
                background-color: #16213e;
            }
        """)

        # 뷰어 탭
        self._viewer = MaskViewer()
        self._tab_widget.addTab(self._viewer, "뷰어")

        # 에디터 탭
        self._editor = MaskEditor()
        self._editor.mask_saved.connect(self._on_mask_saved)
        self._tab_widget.addTab(self._editor, "에디터")

        # 비교 탭
        self._comparison = ComparisonViewer()
        self._tab_widget.addTab(self._comparison, "비교")

        content_layout.addWidget(self._tab_widget)
        main_layout.addWidget(content_widget, 1)

        # 전체 스타일
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f23;
            }
            QWidget {
                background-color: #0f0f23;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                color: #e94560;
                border: 1px solid #16213e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 4px;
                padding: 8px 12px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #16213e;
            }
            QComboBox {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 3px;
                padding: 8px;
                color: #e0e0e0;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                selection-background-color: #e94560;
            }
        """)

    def _setup_menu(self):
        """메뉴 구성"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #0a0a15;
                color: #e0e0e0;
                padding: 5px;
            }
            QMenuBar::item:selected {
                background-color: #16213e;
            }
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
            }
            QMenu::item {
                padding: 8px 25px;
            }
            QMenu::item:selected {
                background-color: #e94560;
            }
        """)

        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")

        open_video_action = QAction("영상 열기...", self)
        open_video_action.setShortcut(QKeySequence.Open)
        open_video_action.triggered.connect(self._open_video)
        file_menu.addAction(open_video_action)

        open_folder_action = QAction("이미지 폴더 열기...", self)
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        save_mask_action = QAction("마스크 저장...", self)
        save_mask_action.setShortcut(QKeySequence.Save)
        save_mask_action.triggered.connect(self._save_mask)
        file_menu.addAction(save_mask_action)

        file_menu.addSeparator()

        exit_action = QAction("종료(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 설정 메뉴
        settings_menu = menubar.addMenu("설정(&S)")

        load_config_action = QAction("설정 파일 로드...", self)
        load_config_action.triggered.connect(self._load_config)
        settings_menu.addAction(load_config_action)

        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")

        about_action = QAction("정보...", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """툴바 구성"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #0a0a15;
                border: none;
                padding: 5px;
                spacing: 5px;
            }
            QToolButton {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
            }
            QToolButton:hover {
                background-color: #16213e;
            }
        """)
        self.addToolBar(toolbar)

        toolbar.addAction("영상 열기", self._open_video)
        toolbar.addAction("폴더 열기", self._open_folder)
        toolbar.addSeparator()
        toolbar.addAction("마스크 생성", self._generate_mask)
        toolbar.addAction("저장", self._save_mask)

    def _setup_statusbar(self):
        """상태바 구성"""
        statusbar = QStatusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #0a0a15;
                color: #888;
                border-top: 1px solid #16213e;
            }
        """)
        self.setStatusBar(statusbar)
        statusbar.showMessage("준비됨")

    @Slot()
    def _open_video(self):
        """영상 파일 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )

        if file_path:
            self._load_source(file_path)

    @Slot()
    def _open_folder(self):
        """이미지 폴더 열기"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "이미지 폴더 선택",
        )

        if folder_path:
            self._load_source(folder_path)

    def _load_source(self, source_path: str):
        """데이터 소스 로드"""
        try:
            with DataLoader(source_path) as loader:
                self._frames = loader.sample_frames(
                    min(50, loader.frame_count)
                )

            self._current_source = source_path
            source_name = Path(source_path).name
            self._source_label.setText(f"✓ {source_name}")
            self._source_label.setStyleSheet("color: #4ade80; padding: 5px;")

            self._generate_btn.setEnabled(True)
            self._viewer.set_frames(self._frames)

            self.statusBar().showMessage(
                f"로드 완료: {len(self._frames)}개 프레임"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"데이터 로드 실패:\n{str(e)}",
            )

    @Slot(int)
    def _on_camera_changed(self, index: int):
        """카메라 선택 변경"""
        camera_id = self._camera_combo.currentData()
        self._current_camera = camera_id

    @Slot()
    def _generate_mask(self):
        """마스크 생성"""
        if not self._current_source:
            return

        # 카메라 설정
        camera_config = None
        if self._current_camera:
            camera_config = self._config.get_camera(self._current_camera)

        # 진행 다이얼로그
        progress = QProgressDialog("마스크 생성 중...", "취소", 0, 100, self)
        progress.setWindowTitle("처리 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setStyleSheet("""
            QProgressDialog {
                background-color: #0f0f23;
                color: #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #16213e;
                border-radius: 5px;
                background-color: #1a1a2e;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #e94560;
                border-radius: 4px;
            }
        """)

        # 워커 생성
        self._worker = MaskGenerationWorker(
            self._current_source,
            camera_config=camera_config,
            num_samples=self._param_panel.num_samples,
            variance_threshold=self._param_panel.variance_threshold,
        )

        self._worker.progress.connect(
            lambda p, m: (progress.setValue(p), progress.setLabelText(m))
        )
        self._worker.finished.connect(self._on_generation_finished)
        self._worker.error.connect(
            lambda e: QMessageBox.critical(self, "오류", f"마스크 생성 실패:\n{e}")
        )

        progress.canceled.connect(self._worker.cancel)

        self._worker.start()

    @Slot(object)
    def _on_generation_finished(self, result: MaskResult):
        """마스크 생성 완료"""
        self._mask_result = result

        # 뷰어 업데이트
        self._viewer.set_mask(result.final_mask)

        # 에디터 업데이트
        if self._frames:
            self._editor.set_data(self._frames[0], result.final_mask)

        # 비교 뷰어 업데이트
        if self._frames:
            self._comparison.set_data(self._frames[0], result.final_mask)

        self.statusBar().showMessage("마스크 생성 완료")

    @Slot(np.ndarray)
    def _on_mask_saved(self, mask: np.ndarray):
        """에디터에서 마스크 저장됨"""
        if self._mask_result:
            self._mask_result.final_mask = mask
            self._viewer.set_mask(mask)
            self.statusBar().showMessage("마스크가 업데이트되었습니다")

    @Slot()
    def _save_mask(self):
        """마스크 저장"""
        if self._mask_result is None:
            QMessageBox.warning(
                self,
                "경고",
                "저장할 마스크가 없습니다.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "마스크 저장",
            "ego_mask.png",
            "PNG Files (*.png);;All Files (*)",
        )

        if file_path:
            cv2.imwrite(file_path, self._mask_result.final_mask)
            self.statusBar().showMessage(f"저장됨: {file_path}")

    @Slot()
    def _load_config(self):
        """설정 파일 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "설정 파일 선택",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )

        if file_path:
            try:
                self._config = load_config(file_path)

                # 카메라 콤보박스 업데이트
                self._camera_combo.clear()
                self._camera_combo.addItem("(자동 감지)", None)
                for camera_id in self._config.get_camera_ids():
                    self._camera_combo.addItem(camera_id, camera_id)

                self.statusBar().showMessage(f"설정 로드됨: {file_path}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "오류",
                    f"설정 로드 실패:\n{str(e)}",
                )

    @Slot()
    def _show_about(self):
        """정보 다이얼로그"""
        QMessageBox.about(
            self,
            "Ego Vehicle Mask Generator",
            """
            <h3>Ego Vehicle Mask Generator</h3>
            <p>버전 0.1.0</p>
            <p>자율주행 영상 데이터에서 자차 영역을 자동 검출하여
            마스크를 생성하는 Traditional CV 기반 소프트웨어입니다.</p>
            <p><b>기능:</b></p>
            <ul>
                <li>시간적 분산 기반 정적 영역 검출</li>
                <li>비네팅 검출</li>
                <li>마스크 수동 편집</li>
            </ul>
            """,
        )

