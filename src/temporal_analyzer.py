"""
시간적 분석기 (Temporal Analyzer)

여러 프레임에서 픽셀별 시간적 분산을 계산하여
정적 영역(자차)을 검출하는 핵심 알고리즘을 제공합니다.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class AnalysisResult:
    """분석 결과"""
    variance_map: np.ndarray  # 픽셀별 분산 맵 (H, W)
    mean_frame: np.ndarray  # 평균 프레임 (H, W, 3) or (H, W)
    static_mask: np.ndarray  # 정적 영역 마스크 (H, W), binary
    confidence_map: np.ndarray  # 신뢰도 맵 (H, W), 0.0 ~ 1.0


class TemporalAnalyzer:
    """시간적 분산 기반 정적 영역 검출기"""

    def __init__(
        self,
        variance_threshold: float = 50.0,
        min_static_ratio: float = 0.01,
        use_lab_colorspace: bool = True,
    ):
        """
        Args:
            variance_threshold: 분산 임계값. 낮을수록 정적 영역 판정이 엄격
            min_static_ratio: 최소 정적 영역 비율 (전체 이미지 대비)
            use_lab_colorspace: LAB 색공간 사용 여부 (조명 변화에 강건)
        """
        self.variance_threshold = variance_threshold
        self.min_static_ratio = min_static_ratio
        self.use_lab_colorspace = use_lab_colorspace

    def compute_variance_map(
        self,
        frames: list[np.ndarray],
        use_grayscale: bool = False,
    ) -> np.ndarray:
        """
        프레임들의 픽셀별 분산 맵 계산

        Args:
            frames: 프레임 리스트 (N개의 BGR 이미지)
            use_grayscale: 그레이스케일 변환 후 계산 여부

        Returns:
            분산 맵 (H, W)
        """
        if not frames:
            raise ValueError("프레임 리스트가 비어있습니다")

        if use_grayscale:
            # 그레이스케일 변환
            gray_frames = [
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                for f in frames
            ]
            stacked = np.stack(gray_frames, axis=0)  # (N, H, W)
            variance = np.var(stacked, axis=0)  # (H, W)
        elif self.use_lab_colorspace:
            # LAB 색공간의 L 채널 사용 (조명 변화에 더 강건)
            lab_frames = [
                cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                for f in frames
            ]
            stacked = np.stack(lab_frames, axis=0)
            variance = np.var(stacked, axis=0)
        else:
            # BGR 채널 평균 분산
            float_frames = [f.astype(np.float32) for f in frames]
            stacked = np.stack(float_frames, axis=0)  # (N, H, W, 3)
            variance_per_channel = np.var(stacked, axis=0)  # (H, W, 3)
            variance = np.mean(variance_per_channel, axis=2)  # (H, W)

        return variance

    def compute_mean_frame(self, frames: list[np.ndarray]) -> np.ndarray:
        """프레임들의 평균 이미지 계산"""
        if not frames:
            raise ValueError("프레임 리스트가 비어있습니다")

        float_frames = [f.astype(np.float32) for f in frames]
        mean_frame = np.mean(float_frames, axis=0).astype(np.uint8)
        return mean_frame

    def compute_static_mask(
        self,
        variance_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        분산 맵에서 정적 영역 마스크 생성

        Args:
            variance_map: 픽셀별 분산 맵
            threshold: 분산 임계값 (None이면 인스턴스 기본값 사용)

        Returns:
            이진 마스크 (정적 영역 = 255, 동적 영역 = 0)
        """
        if threshold is None:
            threshold = self.variance_threshold

        # 낮은 분산 = 정적 영역
        static_mask = (variance_map < threshold).astype(np.uint8) * 255

        return static_mask

    def compute_confidence_map(
        self,
        variance_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        분산 맵에서 신뢰도 맵 계산

        낮은 분산일수록 높은 신뢰도 (정적 영역일 가능성 높음)

        Args:
            variance_map: 픽셀별 분산 맵
            threshold: 기준 임계값

        Returns:
            신뢰도 맵 (0.0 ~ 1.0)
        """
        if threshold is None:
            threshold = self.variance_threshold

        # 분산을 신뢰도로 변환 (분산이 낮을수록 신뢰도 높음)
        # sigmoid-like 변환으로 부드러운 신뢰도 계산
        confidence = 1.0 / (1.0 + np.exp((variance_map - threshold) / (threshold / 3)))
        return confidence.astype(np.float32)

    def analyze(
        self,
        frames: list[np.ndarray],
        threshold: Optional[float] = None,
    ) -> AnalysisResult:
        """
        프레임 시퀀스 분석

        Args:
            frames: 프레임 리스트
            threshold: 분산 임계값 (None이면 인스턴스 기본값 사용)

        Returns:
            AnalysisResult 객체
        """
        variance_map = self.compute_variance_map(frames)
        mean_frame = self.compute_mean_frame(frames)
        static_mask = self.compute_static_mask(variance_map, threshold)
        confidence_map = self.compute_confidence_map(variance_map, threshold)

        return AnalysisResult(
            variance_map=variance_map,
            mean_frame=mean_frame,
            static_mask=static_mask,
            confidence_map=confidence_map,
        )


class AdaptiveTemporalAnalyzer(TemporalAnalyzer):
    """적응형 시간적 분석기

    이미지 영역별로 다른 임계값을 적용하여
    더 정확한 정적 영역 검출을 수행합니다.
    """

    def __init__(
        self,
        base_threshold: float = 50.0,
        block_size: int = 64,
        percentile: float = 10.0,
        **kwargs,
    ):
        """
        Args:
            base_threshold: 기본 분산 임계값
            block_size: 적응형 임계값 계산을 위한 블록 크기
            percentile: 블록 내 분산의 백분위수 기준
        """
        super().__init__(variance_threshold=base_threshold, **kwargs)
        self.block_size = block_size
        self.percentile = percentile

    def compute_adaptive_threshold(
        self,
        variance_map: np.ndarray,
    ) -> np.ndarray:
        """
        적응형 임계값 맵 계산

        각 블록의 분산 분포를 기반으로 지역적 임계값 계산

        Args:
            variance_map: 분산 맵

        Returns:
            적응형 임계값 맵
        """
        h, w = variance_map.shape

        # 블록별 통계 계산
        block_h = h // self.block_size + 1
        block_w = w // self.block_size + 1

        threshold_map = np.zeros_like(variance_map)

        for i in range(block_h):
            for j in range(block_w):
                y1 = i * self.block_size
                y2 = min((i + 1) * self.block_size, h)
                x1 = j * self.block_size
                x2 = min((j + 1) * self.block_size, w)

                block = variance_map[y1:y2, x1:x2]
                local_threshold = np.percentile(block, self.percentile)

                # 기본 임계값과 지역 임계값의 조합
                combined_threshold = min(
                    self.variance_threshold,
                    local_threshold * 2.0,
                )
                threshold_map[y1:y2, x1:x2] = combined_threshold

        # 블록 경계 부드럽게 처리
        threshold_map = ndimage.gaussian_filter(
            threshold_map,
            sigma=self.block_size / 4,
        )

        return threshold_map

    def compute_static_mask(
        self,
        variance_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """적응형 임계값을 사용한 정적 마스크 생성"""
        adaptive_threshold = self.compute_adaptive_threshold(variance_map)

        # 적응형 임계값 적용
        static_mask = (variance_map < adaptive_threshold).astype(np.uint8) * 255

        return static_mask


class MultiScaleTemporalAnalyzer(TemporalAnalyzer):
    """다중 스케일 시간적 분석기

    여러 해상도에서 분석을 수행하여
    더 강건한 정적 영역 검출을 수행합니다.
    """

    def __init__(
        self,
        scales: list[float] = None,
        **kwargs,
    ):
        """
        Args:
            scales: 분석 스케일 리스트 (1.0 = 원본 해상도)
        """
        super().__init__(**kwargs)
        self.scales = scales or [1.0, 0.5, 0.25]

    def _resize_frames(
        self,
        frames: list[np.ndarray],
        scale: float,
    ) -> list[np.ndarray]:
        """프레임 리사이즈"""
        if scale == 1.0:
            return frames

        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            resized.append(cv2.resize(frame, (new_w, new_h)))

        return resized

    def analyze(
        self,
        frames: list[np.ndarray],
        threshold: Optional[float] = None,
    ) -> AnalysisResult:
        """다중 스케일 분석"""
        if not frames:
            raise ValueError("프레임 리스트가 비어있습니다")

        h, w = frames[0].shape[:2]
        combined_confidence = np.zeros((h, w), dtype=np.float32)

        for scale in self.scales:
            scaled_frames = self._resize_frames(frames, scale)
            variance_map = self.compute_variance_map(scaled_frames)
            confidence = self.compute_confidence_map(variance_map, threshold)

            # 원본 해상도로 업스케일
            if scale != 1.0:
                confidence = cv2.resize(confidence, (w, h))

            combined_confidence += confidence

        # 평균 신뢰도
        combined_confidence /= len(self.scales)

        # 최종 결과 생성
        variance_map = self.compute_variance_map(frames)
        mean_frame = self.compute_mean_frame(frames)

        # 신뢰도 기반 마스크 생성
        static_mask = (combined_confidence > 0.5).astype(np.uint8) * 255

        return AnalysisResult(
            variance_map=variance_map,
            mean_frame=mean_frame,
            static_mask=static_mask,
            confidence_map=combined_confidence,
        )

