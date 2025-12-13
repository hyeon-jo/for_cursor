"""
마스크 뷰어 (Mask Viewer)

마스크를 원본 영상 위에 오버레이하여 표시하고
프레임 탐색 기능을 제공합니다.
"""

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QPushButton,
)


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """NumPy 배열을 QImage로 변환"""
    if array.ndim == 2:
        # Grayscale
        h, w = array.shape
        return QImage(array.data, w, h, w, QImage.Format_Grayscale8)
    elif array.ndim == 3:
        h, w, ch = array.shape
        if ch == 3:
            # BGR to RGB
            rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        elif ch == 4:
            # BGRA to RGBA
            rgba = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
            return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    raise ValueError(f"지원하지 않는 배열 형식: {array.shape}")


def create_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """프레임에 마스크 오버레이 생성"""
    overlay = frame.copy()

    # 마스크 영역에 색상 적용
    mask_bool = mask > 127
    overlay[mask_bool] = (
        np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)
    ).astype(np.uint8)

    return overlay


class ImageCanvas(QLabel):
    """이미지 표시용 캔버스"""

    mouse_pressed = Signal(int, int)  # x, y
    mouse_moved = Signal(int, int)
    mouse_released = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a2e; border: 1px solid #16213e;")

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def set_image(self, image: np.ndarray) -> None:
        """이미지 설정"""
        qimage = numpy_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)

        # 위젯 크기에 맞게 스케일 조정
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self._to_image_coords(event.position())
            if pos:
                self.mouse_pressed.emit(int(pos[0]), int(pos[1]))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self._to_image_coords(event.position())
        if pos:
            self.mouse_moved.emit(int(pos[0]), int(pos[1]))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self._to_image_coords(event.position())
            if pos:
                self.mouse_released.emit(int(pos[0]), int(pos[1]))
        super().mouseReleaseEvent(event)

    def _to_image_coords(self, pos):
        """위젯 좌표를 이미지 좌표로 변환"""
        pixmap = self.pixmap()
        if pixmap is None:
            return None

        # 이미지가 중앙에 위치한 경우 오프셋 계산
        offset_x = (self.width() - pixmap.width()) / 2
        offset_y = (self.height() - pixmap.height()) / 2

        x = pos.x() - offset_x
        y = pos.y() - offset_y

        if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
            return (x, y)
        return None


class MaskViewer(QWidget):
    """마스크 뷰어 위젯"""

    frame_changed = Signal(int)  # 프레임 인덱스
    mask_updated = Signal(np.ndarray)  # 업데이트된 마스크

    def __init__(self, parent=None):
        super().__init__(parent)

        self._frames: list[np.ndarray] = []
        self._mask: Optional[np.ndarray] = None
        self._current_frame_idx = 0
        self._overlay_enabled = True
        self._overlay_alpha = 0.5
        self._overlay_color = (255, 0, 0)  # Red

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 캔버스
        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas, 1)

        # 컨트롤 패널
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        # 프레임 네비게이션
        nav_group = QGroupBox("프레임 탐색")
        nav_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #e94560;
                border: 1px solid #16213e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        nav_layout = QHBoxLayout(nav_group)

        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        nav_layout.addWidget(self._frame_slider, 1)

        self._frame_spinbox = QSpinBox()
        self._frame_spinbox.setMinimum(0)
        self._frame_spinbox.setMaximum(0)
        self._frame_spinbox.valueChanged.connect(self._on_frame_changed)
        self._frame_spinbox.setFixedWidth(80)
        nav_layout.addWidget(self._frame_spinbox)

        self._frame_label = QLabel("/ 0")
        self._frame_label.setFixedWidth(60)
        nav_layout.addWidget(self._frame_label)

        control_layout.addWidget(nav_group, 1)

        # 오버레이 설정
        overlay_group = QGroupBox("오버레이 설정")
        overlay_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #e94560;
                border: 1px solid #16213e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        overlay_layout = QHBoxLayout(overlay_group)

        self._overlay_checkbox = QCheckBox("마스크 표시")
        self._overlay_checkbox.setChecked(True)
        self._overlay_checkbox.stateChanged.connect(self._on_overlay_toggled)
        overlay_layout.addWidget(self._overlay_checkbox)

        overlay_layout.addWidget(QLabel("투명도:"))
        self._alpha_slider = QSlider(Qt.Horizontal)
        self._alpha_slider.setMinimum(0)
        self._alpha_slider.setMaximum(100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.setFixedWidth(100)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        overlay_layout.addWidget(self._alpha_slider)

        overlay_layout.addWidget(QLabel("색상:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["빨강", "초록", "파랑", "노랑", "청록"])
        self._color_combo.currentIndexChanged.connect(self._on_color_changed)
        self._color_combo.setFixedWidth(80)
        overlay_layout.addWidget(self._color_combo)

        control_layout.addWidget(overlay_group)

        layout.addLayout(control_layout)

        # 스타일 적용
        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f23;
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                border: 1px solid #16213e;
                height: 8px;
                background: #1a1a2e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                border: 1px solid #e94560;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSpinBox, QComboBox {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)

    def set_frames(self, frames: list[np.ndarray]) -> None:
        """프레임 리스트 설정"""
        self._frames = frames
        max_idx = len(frames) - 1 if frames else 0

        self._frame_slider.setMaximum(max_idx)
        self._frame_spinbox.setMaximum(max_idx)
        self._frame_label.setText(f"/ {len(frames)}")

        if frames:
            self._current_frame_idx = 0
            self._update_display()

    def set_mask(self, mask: np.ndarray) -> None:
        """마스크 설정"""
        self._mask = mask
        self._update_display()

    def get_mask(self) -> Optional[np.ndarray]:
        """현재 마스크 반환"""
        return self._mask

    def get_current_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 반환"""
        if 0 <= self._current_frame_idx < len(self._frames):
            return self._frames[self._current_frame_idx]
        return None

    def _update_display(self) -> None:
        """디스플레이 업데이트"""
        if not self._frames:
            return

        frame = self._frames[self._current_frame_idx].copy()

        if self._overlay_enabled and self._mask is not None:
            # 마스크 크기 조정 (필요한 경우)
            if frame.shape[:2] != self._mask.shape[:2]:
                mask_resized = cv2.resize(
                    self._mask,
                    (frame.shape[1], frame.shape[0]),
                )
            else:
                mask_resized = self._mask

            frame = create_overlay(
                frame,
                mask_resized,
                self._overlay_color,
                self._overlay_alpha,
            )

        self._canvas.set_image(frame)

    @Slot(int)
    def _on_frame_changed(self, value: int) -> None:
        """프레임 변경 이벤트"""
        if value != self._current_frame_idx:
            self._current_frame_idx = value

            # 슬라이더와 스핀박스 동기화
            self._frame_slider.blockSignals(True)
            self._frame_spinbox.blockSignals(True)
            self._frame_slider.setValue(value)
            self._frame_spinbox.setValue(value)
            self._frame_slider.blockSignals(False)
            self._frame_spinbox.blockSignals(False)

            self._update_display()
            self.frame_changed.emit(value)

    @Slot(int)
    def _on_overlay_toggled(self, state: int) -> None:
        """오버레이 토글"""
        self._overlay_enabled = state == Qt.Checked
        self._update_display()

    @Slot(int)
    def _on_alpha_changed(self, value: int) -> None:
        """투명도 변경"""
        self._overlay_alpha = value / 100.0
        self._update_display()

    @Slot(int)
    def _on_color_changed(self, index: int) -> None:
        """오버레이 색상 변경"""
        colors = [
            (255, 0, 0),    # 빨강 (BGR)
            (0, 255, 0),    # 초록
            (0, 0, 255),    # 파랑
            (0, 255, 255),  # 노랑
            (255, 255, 0),  # 청록
        ]
        if 0 <= index < len(colors):
            self._overlay_color = colors[index]
            self._update_display()


class ComparisonViewer(QWidget):
    """마스크 비교 뷰어 (Before/After)"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        layout = QHBoxLayout(self)
        layout.setSpacing(10)

        # 원본 이미지
        original_group = QGroupBox("원본")
        original_layout = QVBoxLayout(original_group)
        self._original_canvas = ImageCanvas()
        original_layout.addWidget(self._original_canvas)
        layout.addWidget(original_group)

        # 마스크 적용 이미지
        masked_group = QGroupBox("마스크 적용")
        masked_layout = QVBoxLayout(masked_group)
        self._masked_canvas = ImageCanvas()
        masked_layout.addWidget(self._masked_canvas)
        layout.addWidget(masked_group)

        # 스타일
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #e94560;
                border: 1px solid #16213e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def set_data(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        fill_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """데이터 설정"""
        self._frame = frame
        self._mask = mask

        # 원본 표시
        self._original_canvas.set_image(frame)

        # 마스크 적용 이미지
        masked = frame.copy()
        mask_bool = mask > 127
        masked[mask_bool] = fill_color
        self._masked_canvas.set_image(masked)

