"""
데이터 로더

영상 파일 및 이미지 시퀀스를 로드하고 프레임을 샘플링합니다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


@dataclass
class FrameInfo:
    """프레임 정보"""
    index: int
    timestamp_ms: float
    frame: np.ndarray


class VideoLoader:
    """비디오 파일 로더"""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, video_path: str | Path):
        """
        Args:
            video_path: 비디오 파일 경로
        """
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> bool:
        """비디오 파일 열기"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {self.video_path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return True

    def close(self) -> None:
        """비디오 파일 닫기"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoLoader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def duration_sec(self) -> float:
        """영상 길이 (초)"""
        if self._fps > 0:
            return self._frame_count / self._fps
        return 0.0

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """특정 프레임 읽기"""
        if self._cap is None:
            raise RuntimeError("비디오가 열려있지 않습니다")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()

        if not ret:
            return None
        return frame

    def sample_frames(
        self,
        num_samples: int,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
    ) -> list[np.ndarray]:
        """
        균등 간격으로 프레임 샘플링

        Args:
            num_samples: 샘플링할 프레임 수
            start_ratio: 시작 위치 비율 (0.0 ~ 1.0)
            end_ratio: 끝 위치 비율 (0.0 ~ 1.0)

        Returns:
            샘플링된 프레임 리스트
        """
        if self._cap is None:
            raise RuntimeError("비디오가 열려있지 않습니다")

        start_frame = int(self._frame_count * start_ratio)
        end_frame = int(self._frame_count * end_ratio)
        total_range = end_frame - start_frame

        if total_range <= 0 or num_samples <= 0:
            return []

        # 실제 샘플 수 조정
        actual_samples = min(num_samples, total_range)
        step = total_range / actual_samples

        frames = []
        for i in range(actual_samples):
            frame_idx = int(start_frame + i * step)
            frame = self.read_frame(frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def iterate_frames(
        self,
        step: int = 1,
    ) -> Generator[FrameInfo, None, None]:
        """
        프레임 순회 제너레이터

        Args:
            step: 프레임 간격

        Yields:
            FrameInfo 객체
        """
        if self._cap is None:
            raise RuntimeError("비디오가 열려있지 않습니다")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while frame_idx < self._frame_count:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()

            if not ret:
                break

            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            yield FrameInfo(
                index=frame_idx,
                timestamp_ms=timestamp_ms,
                frame=frame,
            )

            frame_idx += step


class ImageSequenceLoader:
    """이미지 시퀀스 로더"""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, directory_path: str | Path):
        """
        Args:
            directory_path: 이미지 시퀀스가 있는 디렉토리 경로
        """
        self.directory_path = Path(directory_path)
        self._image_paths: list[Path] = []
        self._width: int = 0
        self._height: int = 0

    def scan(self) -> int:
        """디렉토리 스캔하여 이미지 파일 목록 생성"""
        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"디렉토리를 찾을 수 없습니다: {self.directory_path}"
            )

        self._image_paths = sorted([
            p for p in self.directory_path.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])

        # 첫 번째 이미지로 해상도 확인
        if self._image_paths:
            first_image = cv2.imread(str(self._image_paths[0]))
            if first_image is not None:
                self._height, self._width = first_image.shape[:2]

        return len(self._image_paths)

    @property
    def frame_count(self) -> int:
        return len(self._image_paths)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """특정 인덱스의 이미지 읽기"""
        if index < 0 or index >= len(self._image_paths):
            return None

        frame = cv2.imread(str(self._image_paths[index]))
        return frame

    def sample_frames(self, num_samples: int) -> list[np.ndarray]:
        """균등 간격으로 이미지 샘플링"""
        if not self._image_paths or num_samples <= 0:
            return []

        actual_samples = min(num_samples, len(self._image_paths))
        step = len(self._image_paths) / actual_samples

        frames = []
        for i in range(actual_samples):
            idx = int(i * step)
            frame = self.read_frame(idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def iterate_frames(
        self,
        step: int = 1,
    ) -> Generator[FrameInfo, None, None]:
        """프레임 순회 제너레이터"""
        for i in range(0, len(self._image_paths), step):
            frame = self.read_frame(i)
            if frame is not None:
                yield FrameInfo(
                    index=i,
                    timestamp_ms=0.0,  # 이미지 시퀀스는 타임스탬프 없음
                    frame=frame,
                )


class DataLoader:
    """통합 데이터 로더"""

    def __init__(self, source_path: str | Path):
        """
        Args:
            source_path: 비디오 파일 또는 이미지 디렉토리 경로
        """
        self.source_path = Path(source_path)
        self._loader: Optional[VideoLoader | ImageSequenceLoader] = None

    def _detect_source_type(self) -> str:
        """소스 타입 자동 감지"""
        if self.source_path.is_dir():
            return "image_sequence"
        elif self.source_path.suffix.lower() in VideoLoader.SUPPORTED_EXTENSIONS:
            return "video"
        else:
            raise ValueError(
                f"지원하지 않는 소스 형식입니다: {self.source_path}"
            )

    def open(self) -> bool:
        """소스 열기"""
        source_type = self._detect_source_type()

        if source_type == "video":
            self._loader = VideoLoader(self.source_path)
            return self._loader.open()
        else:
            self._loader = ImageSequenceLoader(self.source_path)
            self._loader.scan()
            return True

    def close(self) -> None:
        """소스 닫기"""
        if isinstance(self._loader, VideoLoader):
            self._loader.close()
        self._loader = None

    def __enter__(self) -> "DataLoader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        if self._loader is None:
            return 0
        return self._loader.frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        if self._loader is None:
            return (0, 0)
        return self._loader.resolution

    def sample_frames(self, num_samples: int) -> list[np.ndarray]:
        """프레임 샘플링"""
        if self._loader is None:
            raise RuntimeError("데이터 소스가 열려있지 않습니다")

        if isinstance(self._loader, VideoLoader):
            return self._loader.sample_frames(num_samples)
        else:
            return self._loader.sample_frames(num_samples)

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """특정 프레임 읽기"""
        if self._loader is None:
            raise RuntimeError("데이터 소스가 열려있지 않습니다")
        return self._loader.read_frame(index)

    def iterate_frames(
        self,
        step: int = 1,
    ) -> Generator[FrameInfo, None, None]:
        """프레임 순회"""
        if self._loader is None:
            raise RuntimeError("데이터 소스가 열려있지 않습니다")
        yield from self._loader.iterate_frames(step)

