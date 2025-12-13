"""
비네팅 검출기 (Vignetting Detector)

광각 카메라의 비네팅(주변부 어두움) 영역을 자동으로 검출합니다.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class VignettingResult:
    """비네팅 검출 결과"""
    vignetting_mask: np.ndarray  # 비네팅 영역 마스크 (H, W), binary
    brightness_map: np.ndarray  # 평균 밝기 맵 (H, W)
    radial_profile: np.ndarray  # 중심으로부터 거리별 밝기 프로파일
    detected_radius: float  # 검출된 유효 영역 반경 비율


class VignettingDetector:
    """비네팅 검출기"""

    def __init__(
        self,
        brightness_threshold: float = 0.7,
        min_brightness: float = 0.3,
        use_radial_model: bool = True,
        smooth_sigma: float = 5.0,
    ):
        """
        Args:
            brightness_threshold: 중심 대비 밝기 비율 임계값
            min_brightness: 최소 밝기 임계값 (0.0 ~ 1.0)
            use_radial_model: 방사형 비네팅 모델 사용 여부
            smooth_sigma: 마스크 경계 부드럽게 처리할 sigma 값
        """
        self.brightness_threshold = brightness_threshold
        self.min_brightness = min_brightness
        self.use_radial_model = use_radial_model
        self.smooth_sigma = smooth_sigma

    def compute_brightness_map(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray:
        """
        프레임들의 평균 밝기 맵 계산

        Args:
            frames: 프레임 리스트 (BGR 이미지)

        Returns:
            평균 밝기 맵 (H, W), 0.0 ~ 1.0
        """
        if not frames:
            raise ValueError("프레임 리스트가 비어있습니다")

        # LAB 색공간의 L 채널 사용
        brightness_maps = []
        for frame in frames:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32) / 255.0
            brightness_maps.append(l_channel)

        # 평균 밝기
        mean_brightness = np.mean(brightness_maps, axis=0)

        return mean_brightness

    def compute_radial_profile(
        self,
        brightness_map: np.ndarray,
        num_bins: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        중심으로부터 거리별 밝기 프로파일 계산

        Args:
            brightness_map: 밝기 맵
            num_bins: 거리 구간 수

        Returns:
            (거리 배열, 밝기 배열) 튜플
        """
        h, w = brightness_map.shape
        center_y, center_x = h / 2, w / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)

        # 각 픽셀의 중심으로부터 거리 계산
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distances = np.sqrt(
            (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        )
        normalized_distances = distances / max_radius

        # 거리 구간별 평균 밝기 계산
        bin_edges = np.linspace(0, 1, num_bins + 1)
        radii = (bin_edges[:-1] + bin_edges[1:]) / 2
        brightness_profile = np.zeros(num_bins)

        for i in range(num_bins):
            mask = (normalized_distances >= bin_edges[i]) & \
                   (normalized_distances < bin_edges[i + 1])
            if np.any(mask):
                brightness_profile[i] = np.mean(brightness_map[mask])

        return radii, brightness_profile

    def detect_vignetting_radial(
        self,
        brightness_map: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        방사형 모델 기반 비네팅 검출

        Args:
            brightness_map: 밝기 맵

        Returns:
            (비네팅 마스크, 유효 반경 비율) 튜플
        """
        h, w = brightness_map.shape
        center_y, center_x = h / 2, w / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)

        # 방사형 프로파일 계산
        radii, profile = self.compute_radial_profile(brightness_map)

        # 중심부 밝기 (최대값)
        center_brightness = np.max(profile[:10])  # 중심 10% 영역

        # 임계값 이하로 떨어지는 반경 찾기
        threshold_value = center_brightness * self.brightness_threshold
        valid_mask = profile >= threshold_value

        if np.any(valid_mask):
            # 유효 영역의 마지막 인덱스
            valid_indices = np.where(valid_mask)[0]
            detected_radius = radii[valid_indices[-1]]
        else:
            detected_radius = 0.5  # 기본값

        # 거리 맵 생성
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distances = np.sqrt(
            (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        )
        normalized_distances = distances / max_radius

        # 비네팅 마스크 생성 (유효 영역 외부 = 비네팅)
        vignetting_mask = (normalized_distances > detected_radius).astype(np.uint8) * 255

        return vignetting_mask, detected_radius

    def detect_vignetting_threshold(
        self,
        brightness_map: np.ndarray,
    ) -> np.ndarray:
        """
        임계값 기반 비네팅 검출

        Args:
            brightness_map: 밝기 맵

        Returns:
            비네팅 마스크
        """
        # 중심 영역 밝기 계산
        h, w = brightness_map.shape
        center_region = brightness_map[
            h // 3 : 2 * h // 3,
            w // 3 : 2 * w // 3,
        ]
        center_brightness = np.mean(center_region)

        # 임계값 계산
        threshold = center_brightness * self.brightness_threshold

        # 어두운 영역 = 비네팅
        vignetting_mask = (brightness_map < threshold).astype(np.uint8) * 255

        # 추가로 절대 밝기가 낮은 영역도 포함
        absolute_dark = (brightness_map < self.min_brightness).astype(np.uint8) * 255
        vignetting_mask = cv2.bitwise_or(vignetting_mask, absolute_dark)

        return vignetting_mask

    def refine_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        마스크 정제

        - 노이즈 제거
        - 경계 부드럽게 처리
        - 작은 홀 채우기

        Args:
            mask: 입력 마스크

        Returns:
            정제된 마스크
        """
        # 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 가우시안 블러로 경계 부드럽게
        if self.smooth_sigma > 0:
            mask_float = mask.astype(np.float32) / 255.0
            mask_float = ndimage.gaussian_filter(mask_float, sigma=self.smooth_sigma)
            mask = (mask_float > 0.5).astype(np.uint8) * 255

        return mask

    def detect(
        self,
        frames: list[np.ndarray],
    ) -> VignettingResult:
        """
        비네팅 검출 수행

        Args:
            frames: 프레임 리스트

        Returns:
            VignettingResult 객체
        """
        brightness_map = self.compute_brightness_map(frames)
        radii, profile = self.compute_radial_profile(brightness_map)

        if self.use_radial_model:
            vignetting_mask, detected_radius = self.detect_vignetting_radial(
                brightness_map
            )
        else:
            vignetting_mask = self.detect_vignetting_threshold(brightness_map)
            detected_radius = 1.0  # 방사형 모델 미사용시

        # 마스크 정제
        vignetting_mask = self.refine_mask(vignetting_mask)

        return VignettingResult(
            vignetting_mask=vignetting_mask,
            brightness_map=(brightness_map * 255).astype(np.uint8),
            radial_profile=profile,
            detected_radius=detected_radius,
        )


class EdgeVignettingDetector:
    """에지 기반 비네팅 검출기

    이미지 가장자리 영역의 비네팅을 검출합니다.
    (원형이 아닌 사각형 가장자리 비네팅에 적합)
    """

    def __init__(
        self,
        edge_margin: int = 100,
        brightness_threshold: float = 0.6,
        gradient_threshold: float = 0.1,
    ):
        """
        Args:
            edge_margin: 에지 검사 영역 마진 (픽셀)
            brightness_threshold: 밝기 임계값
            gradient_threshold: 밝기 그래디언트 임계값
        """
        self.edge_margin = edge_margin
        self.brightness_threshold = brightness_threshold
        self.gradient_threshold = gradient_threshold

    def detect_edge_vignetting(
        self,
        brightness_map: np.ndarray,
    ) -> np.ndarray:
        """
        에지 영역 비네팅 검출

        Args:
            brightness_map: 밝기 맵 (0.0 ~ 1.0)

        Returns:
            비네팅 마스크
        """
        h, w = brightness_map.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # 중심 영역 밝기
        center_region = brightness_map[
            h // 3 : 2 * h // 3,
            w // 3 : 2 * w // 3,
        ]
        center_brightness = np.mean(center_region)
        threshold = center_brightness * self.brightness_threshold

        # 각 에지 영역 검사
        # 상단
        top_region = brightness_map[: self.edge_margin, :]
        top_dark = (top_region < threshold).astype(np.uint8) * 255
        mask[: self.edge_margin, :] = top_dark

        # 하단
        bottom_region = brightness_map[-self.edge_margin :, :]
        bottom_dark = (bottom_region < threshold).astype(np.uint8) * 255
        mask[-self.edge_margin :, :] = bottom_dark

        # 좌측
        left_region = brightness_map[:, : self.edge_margin]
        left_dark = (left_region < threshold).astype(np.uint8) * 255
        mask[:, : self.edge_margin] = np.maximum(
            mask[:, : self.edge_margin], left_dark
        )

        # 우측
        right_region = brightness_map[:, -self.edge_margin :]
        right_dark = (right_region < threshold).astype(np.uint8) * 255
        mask[:, -self.edge_margin :] = np.maximum(
            mask[:, -self.edge_margin :], right_dark
        )

        # 형태학적 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def detect(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray:
        """
        비네팅 검출

        Args:
            frames: 프레임 리스트

        Returns:
            비네팅 마스크
        """
        # 평균 밝기 맵 계산
        brightness_maps = []
        for frame in frames:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32) / 255.0
            brightness_maps.append(l_channel)

        mean_brightness = np.mean(brightness_maps, axis=0)

        return self.detect_edge_vignetting(mean_brightness)

