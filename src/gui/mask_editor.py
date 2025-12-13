"""
마스크 에디터 (Mask Editor)

마스크를 수동으로 편집(브러시/지우개)할 수 있는 기능을 제공합니다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QPoint
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QCursor,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QButtonGroup,
    QRadioButton,
    QSizePolicy,
)


class EditTool(Enum):
    """편집 도구"""
    BRUSH = "brush"
    ERASER = "eraser"


@dataclass
class EditHistory:
    """편집 히스토리 항목"""
    mask: np.ndarray
    description: str


class MaskEditorCanvas(QLabel):
    """마스크 편집용 캔버스"""

    mask_edited = Signal(np.ndarray)  # 편집된 마스크

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1a1a2e; border: 1px solid #16213e;")

        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._display_image: Optional[np.ndarray] = None

        self._tool = EditTool.BRUSH
        self._brush_size = 20
        self._is_drawing = False
        self._last_point: Optional[QPoint] = None

        # 스케일 정보
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        # 브러시 커서
        self._update_cursor()

    def set_frame(self, frame: np.ndarray) -> None:
        """프레임 설정"""
        self._frame = frame.copy()
        self._update_display()

    def set_mask(self, mask: np.ndarray) -> None:
        """마스크 설정"""
        self._mask = mask.copy()
        self._update_display()

    def get_mask(self) -> Optional[np.ndarray]:
        """현재 마스크 반환"""
        return self._mask.copy() if self._mask is not None else None

    def set_tool(self, tool: EditTool) -> None:
        """도구 설정"""
        self._tool = tool
        self._update_cursor()

    def set_brush_size(self, size: int) -> None:
        """브러시 크기 설정"""
        self._brush_size = max(1, min(200, size))
        self._update_cursor()

    def _update_cursor(self) -> None:
        """커서 업데이트"""
        # 원형 커서 생성
        size = max(self._brush_size, 4)
        cursor_pixmap = QPixmap(size + 2, size + 2)
        cursor_pixmap.fill(Qt.transparent)

        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._tool == EditTool.BRUSH:
            pen = QPen(QColor(255, 0, 0, 200), 2)
        else:
            pen = QPen(QColor(0, 255, 0, 200), 2)

        painter.setPen(pen)
        painter.drawEllipse(1, 1, size, size)
        painter.end()

        cursor = QCursor(cursor_pixmap, size // 2, size // 2)
        self.setCursor(cursor)

    def _update_display(self) -> None:
        """디스플레이 업데이트"""
        if self._frame is None:
            return

        display = self._frame.copy()

        if self._mask is not None:
            # 마스크 오버레이
            overlay_color = (255, 0, 0)  # Red
            alpha = 0.4

            mask_bool = self._mask > 127
            display[mask_bool] = (
                np.array(overlay_color) * alpha +
                display[mask_bool] * (1 - alpha)
            ).astype(np.uint8)

        self._display_image = display
        self._show_image(display)

    def _show_image(self, image: np.ndarray) -> None:
        """이미지 표시"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # 위젯 크기에 맞게 스케일
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # 스케일 계산
        self._scale = scaled.width() / w
        self._offset_x = (self.width() - scaled.width()) // 2
        self._offset_y = (self.height() - scaled.height()) // 2

        self.setPixmap(scaled)

    def _widget_to_image_coords(self, pos: QPoint) -> Optional[tuple[int, int]]:
        """위젯 좌표를 이미지 좌표로 변환"""
        if self._frame is None:
            return None

        x = int((pos.x() - self._offset_x) / self._scale)
        y = int((pos.y() - self._offset_y) / self._scale)

        h, w = self._frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    def _draw_at(self, x: int, y: int) -> None:
        """지정 위치에 그리기"""
        if self._mask is None:
            return

        value = 255 if self._tool == EditTool.BRUSH else 0
        scaled_brush_size = int(self._brush_size / self._scale)

        cv2.circle(
            self._mask,
            (x, y),
            scaled_brush_size // 2,
            value,
            -1,  # 채우기
        )

    def _draw_line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """두 점 사이 선 그리기"""
        if self._mask is None:
            return

        value = 255 if self._tool == EditTool.BRUSH else 0
        scaled_brush_size = int(self._brush_size / self._scale)

        cv2.line(
            self._mask,
            (x1, y1),
            (x2, y2),
            value,
            scaled_brush_size,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """마우스 클릭"""
        if event.button() == Qt.LeftButton:
            coords = self._widget_to_image_coords(event.position().toPoint())
            if coords:
                self._is_drawing = True
                self._last_point = coords
                self._draw_at(*coords)
                self._update_display()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """마우스 이동"""
        if self._is_drawing:
            coords = self._widget_to_image_coords(event.position().toPoint())
            if coords and self._last_point:
                self._draw_line(
                    self._last_point[0], self._last_point[1],
                    coords[0], coords[1],
                )
                self._last_point = coords
                self._update_display()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """마우스 릴리스"""
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._last_point = None
            if self._mask is not None:
                self.mask_edited.emit(self._mask.copy())

        super().mouseReleaseEvent(event)


class MaskEditor(QWidget):
    """마스크 에디터 위젯"""

    mask_saved = Signal(np.ndarray)  # 저장된 마스크

    def __init__(self, parent=None):
        super().__init__(parent)

        self._history: list[EditHistory] = []
        self._history_index = -1
        self._max_history = 50

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 캔버스
        self._canvas = MaskEditorCanvas()
        self._canvas.mask_edited.connect(self._on_mask_edited)
        layout.addWidget(self._canvas, 1)

        # 도구 패널
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(15)

        # 도구 선택
        tool_group = QGroupBox("도구")
        tool_group.setStyleSheet("""
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
        tool_layout = QHBoxLayout(tool_group)

        self._tool_button_group = QButtonGroup(self)

        self._brush_radio = QRadioButton("브러시")
        self._brush_radio.setChecked(True)
        self._brush_radio.toggled.connect(
            lambda checked: self._on_tool_changed(EditTool.BRUSH) if checked else None
        )
        self._tool_button_group.addButton(self._brush_radio)
        tool_layout.addWidget(self._brush_radio)

        self._eraser_radio = QRadioButton("지우개")
        self._eraser_radio.toggled.connect(
            lambda checked: self._on_tool_changed(EditTool.ERASER) if checked else None
        )
        self._tool_button_group.addButton(self._eraser_radio)
        tool_layout.addWidget(self._eraser_radio)

        tools_layout.addWidget(tool_group)

        # 브러시 크기
        size_group = QGroupBox("브러시 크기")
        size_group.setStyleSheet("""
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
        size_layout = QHBoxLayout(size_group)

        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setMinimum(1)
        self._size_slider.setMaximum(100)
        self._size_slider.setValue(20)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self._size_slider, 1)

        self._size_spinbox = QSpinBox()
        self._size_spinbox.setMinimum(1)
        self._size_spinbox.setMaximum(100)
        self._size_spinbox.setValue(20)
        self._size_spinbox.valueChanged.connect(self._on_size_changed)
        self._size_spinbox.setFixedWidth(60)
        size_layout.addWidget(self._size_spinbox)

        tools_layout.addWidget(size_group)

        # 실행 취소/다시 실행
        history_group = QGroupBox("편집")
        history_group.setStyleSheet("""
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
        history_layout = QHBoxLayout(history_group)

        self._undo_btn = QPushButton("실행 취소")
        self._undo_btn.clicked.connect(self._undo)
        self._undo_btn.setEnabled(False)
        history_layout.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("다시 실행")
        self._redo_btn.clicked.connect(self._redo)
        self._redo_btn.setEnabled(False)
        history_layout.addWidget(self._redo_btn)

        self._reset_btn = QPushButton("초기화")
        self._reset_btn.clicked.connect(self._reset)
        history_layout.addWidget(self._reset_btn)

        tools_layout.addWidget(history_group)

        # 저장
        save_group = QGroupBox("저장")
        save_group.setStyleSheet("""
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
        save_layout = QHBoxLayout(save_group)

        self._save_btn = QPushButton("마스크 저장")
        self._save_btn.clicked.connect(self._save_mask)
        self._save_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:pressed {
                background-color: #c73e54;
            }
        """)
        save_layout.addWidget(self._save_btn)

        tools_layout.addWidget(save_group)

        layout.addLayout(tools_layout)

        # 스타일
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
            QSpinBox {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
            QRadioButton {
                spacing: 5px;
            }
            QPushButton {
                background-color: #1a1a2e;
                border: 1px solid #16213e;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #16213e;
            }
            QPushButton:disabled {
                background-color: #0a0a15;
                color: #555;
            }
        """)

    def set_data(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """프레임과 마스크 설정"""
        self._canvas.set_frame(frame)
        self._canvas.set_mask(mask)

        # 히스토리 초기화
        self._history = [EditHistory(mask.copy(), "초기 상태")]
        self._history_index = 0
        self._update_history_buttons()

    def get_mask(self) -> Optional[np.ndarray]:
        """현재 마스크 반환"""
        return self._canvas.get_mask()

    @Slot(EditTool)
    def _on_tool_changed(self, tool: EditTool) -> None:
        """도구 변경"""
        self._canvas.set_tool(tool)

    @Slot(int)
    def _on_size_changed(self, value: int) -> None:
        """브러시 크기 변경"""
        # 슬라이더와 스핀박스 동기화
        self._size_slider.blockSignals(True)
        self._size_spinbox.blockSignals(True)
        self._size_slider.setValue(value)
        self._size_spinbox.setValue(value)
        self._size_slider.blockSignals(False)
        self._size_spinbox.blockSignals(False)

        self._canvas.set_brush_size(value)

    @Slot(np.ndarray)
    def _on_mask_edited(self, mask: np.ndarray) -> None:
        """마스크 편집됨"""
        # 현재 위치 이후 히스토리 삭제
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        # 새 히스토리 추가
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        else:
            self._history_index += 1

        self._history.append(EditHistory(mask.copy(), "편집"))
        self._history_index = len(self._history) - 1
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        """히스토리 버튼 상태 업데이트"""
        self._undo_btn.setEnabled(self._history_index > 0)
        self._redo_btn.setEnabled(self._history_index < len(self._history) - 1)

    @Slot()
    def _undo(self) -> None:
        """실행 취소"""
        if self._history_index > 0:
            self._history_index -= 1
            self._canvas.set_mask(self._history[self._history_index].mask.copy())
            self._update_history_buttons()

    @Slot()
    def _redo(self) -> None:
        """다시 실행"""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._canvas.set_mask(self._history[self._history_index].mask.copy())
            self._update_history_buttons()

    @Slot()
    def _reset(self) -> None:
        """초기화"""
        if self._history:
            self._canvas.set_mask(self._history[0].mask.copy())
            self._history_index = 0

            # 리셋 이후 히스토리 정리
            self._history = [self._history[0]]
            self._update_history_buttons()

    @Slot()
    def _save_mask(self) -> None:
        """마스크 저장"""
        mask = self._canvas.get_mask()
        if mask is not None:
            self.mask_saved.emit(mask)

