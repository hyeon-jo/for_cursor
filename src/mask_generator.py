"""
마스크 생성기 (Mask Generator)

시간적 분석 결과와 비네팅 검출 결과를 통합하여
최종 자차 마스크를 생성하고 형태학적 정제를 수행합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

from .config_loader import CameraConfig, GlobalConfig
from .temporal_analyzer import AnalysisResult, TemporalAnalyzer
from .vignetting_detector import VignettingDetector, VignettingResult


@dataclass
class MaskResult:
    """마스크 생성 결과"""
    final_mask: np.ndarray  # 최종 마스크 (H, W), binary
    static_mask: np.ndarray  # 정적 영역 마스크
    vignetting_mask: Optional[np.ndarray]  # 비네팅 마스크 (있는 경우)
    confidence_map: np.ndarray  # 신뢰도 맵
    metadata: dict = field(default_factory=dict)


class MaskRefiner:
    """마스크 정제기"""

    def __init__(
        self,
        kernel_size: int = 5,
        iterations: int = 2,
        fill_holes: bool = True,
        smooth_boundary: bool = True,
        min_area: int = 1000,
    ):
        """
        Args:
            kernel_size: 형태학적 연산 커널 크기
            iterations: 형태학적 연산 반복 횟수
            fill_holes: 내부 홀 채우기 여부
            smooth_boundary: 경계 부드럽게 처리 여부
            min_area: 최소 영역 크기 (픽셀 수)
        """
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.fill_holes = fill_holes
        self.smooth_boundary = smooth_boundary
        self.min_area = min_area

    def apply_morphology(
        self,
        mask: np.ndarray,
        operation: str = "close",
    ) -> np.ndarray:
        """
        형태학적 연산 적용

        Args:
            mask: 입력 마스크
            operation: 연산 종류 ("open", "close", "dilate", "erode")

        Returns:
            처리된 마스크
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size),
        )

        if operation == "open":
            return cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=self.iterations
            )
        elif operation == "close":
            return cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=self.iterations
            )
        elif operation == "dilate":
            return cv2.dilate(mask, kernel, iterations=self.iterations)
        elif operation == "erode":
            return cv2.erode(mask, kernel, iterations=self.iterations)
        else:
            return mask

    def fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """마스크 내부 홀 채우기"""
        # 컨투어 찾기
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 외부 컨투어만 채우기
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)

        return filled

    def remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """작은 영역 제거"""
        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # 큰 영역만 유지
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):  # 0은 배경
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                filtered[labels == i] = 255

        return filtered

    def smooth_boundary_gaussian(
        self,
        mask: np.ndarray,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """가우시안 블러로 경계 부드럽게"""
        mask_float = mask.astype(np.float32) / 255.0
        smoothed = ndimage.gaussian_filter(mask_float, sigma=sigma)
        return (smoothed > 0.5).astype(np.uint8) * 255

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """
        마스크 정제 파이프라인

        Args:
            mask: 입력 마스크

        Returns:
            정제된 마스크
        """
        # 1. Closing: 작은 홀 채우고 영역 연결
        refined = self.apply_morphology(mask, "close")

        # 2. Opening: 작은 노이즈 제거
        refined = self.apply_morphology(refined, "open")

        # 3. 내부 홀 채우기
        if self.fill_holes:
            refined = self.fill_mask_holes(refined)

        # 4. 작은 영역 제거
        refined = self.remove_small_regions(refined)

        # 5. 경계 부드럽게
        if self.smooth_boundary:
            refined = self.smooth_boundary_gaussian(refined)

        return refined


class MaskGenerator:
    """마스크 생성기"""

    def __init__(
        self,
        global_config: Optional[GlobalConfig] = None,
        variance_threshold: Optional[float] = None,
        morphology_kernel_size: Optional[int] = None,
    ):
        """
        Args:
            global_config: 전역 설정
            variance_threshold: 분산 임계값 (설정 파일 값 오버라이드)
            morphology_kernel_size: 형태학적 커널 크기 (설정 파일 값 오버라이드)
        """
        self.global_config = global_config or GlobalConfig()

        self.variance_threshold = (
            variance_threshold
            if variance_threshold is not None
            else self.global_config.variance_threshold
        )
        self.morphology_kernel_size = (
            morphology_kernel_size
            if morphology_kernel_size is not None
            else self.global_config.morphology_kernel_size
        )

        # 분석기 초기화
        self.temporal_analyzer = TemporalAnalyzer(
            variance_threshold=self.variance_threshold,
        )
        self.vignetting_detector = VignettingDetector()
        self.refiner = MaskRefiner(
            kernel_size=self.morphology_kernel_size,
        )

    def generate_from_frames(
        self,
        frames: list[np.ndarray],
        camera_config: Optional[CameraConfig] = None,
        include_vignetting: bool = True,
    ) -> MaskResult:
        """
        프레임 리스트에서 마스크 생성

        Args:
            frames: 프레임 리스트
            camera_config: 카메라 설정 (힌트 ROI 등)
            include_vignetting: 비네팅 마스크 포함 여부

        Returns:
            MaskResult 객체
        """
        # 1. 시간적 분석
        analysis_result = self.temporal_analyzer.analyze(
            frames,
            threshold=self.variance_threshold,
        )

        # 2. 정적 마스크 정제
        static_mask = self.refiner.refine(analysis_result.static_mask)

        # 3. 비네팅 검출 (광각 카메라인 경우)
        vignetting_mask = None
        has_vignetting = (
            camera_config is not None and camera_config.has_vignetting
        ) or include_vignetting

        if has_vignetting:
            vignetting_result = self.vignetting_detector.detect(frames)
            vignetting_mask = vignetting_result.vignetting_mask

        # 4. 마스크 통합
        final_mask = static_mask.copy()
        if vignetting_mask is not None:
            final_mask = cv2.bitwise_or(final_mask, vignetting_mask)

        # 5. ROI 힌트 적용 (있는 경우)
        if camera_config is not None:
            final_mask = self._apply_roi_hints(
                final_mask,
                analysis_result.confidence_map,
                camera_config,
            )

        # 6. 최종 정제
        final_mask = self.refiner.refine(final_mask)

        return MaskResult(
            final_mask=final_mask,
            static_mask=static_mask,
            vignetting_mask=vignetting_mask,
            confidence_map=analysis_result.confidence_map,
            metadata={
                "variance_threshold": self.variance_threshold,
                "frame_count": len(frames),
                "resolution": frames[0].shape[:2] if frames else (0, 0),
            },
        )

    def _apply_roi_hints(
        self,
        mask: np.ndarray,
        confidence_map: np.ndarray,
        camera_config: CameraConfig,
    ) -> np.ndarray:
        """
        ROI 힌트를 활용하여 마스크 보정

        힌트 영역 내에서 신뢰도가 높은 픽셀을 마스크에 추가

        Args:
            mask: 현재 마스크
            confidence_map: 신뢰도 맵
            camera_config: 카메라 설정

        Returns:
            보정된 마스크
        """
        result = mask.copy()

        for roi_hint in camera_config.expected_ego_regions:
            x, y, w, h = roi_hint.to_tuple()

            # ROI 영역 내 신뢰도 확인
            roi_confidence = confidence_map[y : y + h, x : x + w]

            # 신뢰도가 높은 영역을 마스크에 추가
            high_confidence = (roi_confidence > 0.3).astype(np.uint8) * 255

            # 기존 마스크와 합치기
            result[y : y + h, x : x + w] = cv2.bitwise_or(
                result[y : y + h, x : x + w],
                high_confidence,
            )

        return result

    def generate_from_video(
        self,
        video_path: str | Path,
        camera_config: Optional[CameraConfig] = None,
        num_samples: Optional[int] = None,
    ) -> MaskResult:
        """
        비디오 파일에서 마스크 생성

        Args:
            video_path: 비디오 파일 경로
            camera_config: 카메라 설정
            num_samples: 샘플링할 프레임 수

        Returns:
            MaskResult 객체
        """
        from .data_loader import DataLoader

        num_samples = num_samples or self.global_config.frame_sample_count

        with DataLoader(video_path) as loader:
            frames = loader.sample_frames(num_samples)

        include_vignetting = (
            camera_config is not None and camera_config.has_vignetting
        )

        return self.generate_from_frames(
            frames,
            camera_config=camera_config,
            include_vignetting=include_vignetting,
        )


class BatchMaskGenerator:
    """배치 마스크 생성기"""

    def __init__(
        self,
        global_config: Optional[GlobalConfig] = None,
        output_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            global_config: 전역 설정
            output_dir: 출력 디렉토리
        """
        self.global_config = global_config or GlobalConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.mask_generator = MaskGenerator(global_config=self.global_config)

    def process_camera(
        self,
        video_path: str | Path,
        camera_config: CameraConfig,
        save_intermediate: bool = False,
    ) -> tuple[MaskResult, Path]:
        """
        단일 카메라 처리

        Args:
            video_path: 비디오 파일 경로
            camera_config: 카메라 설정
            save_intermediate: 중간 결과 저장 여부

        Returns:
            (MaskResult, 저장된 마스크 경로) 튜플
        """
        # 마스크 생성
        result = self.mask_generator.generate_from_video(
            video_path,
            camera_config=camera_config,
        )

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 최종 마스크 저장
        mask_filename = f"{camera_config.camera_id}_mask.png"
        mask_path = self.output_dir / mask_filename
        cv2.imwrite(str(mask_path), result.final_mask)

        # 중간 결과 저장
        if save_intermediate:
            # 정적 마스크
            static_path = self.output_dir / f"{camera_config.camera_id}_static.png"
            cv2.imwrite(str(static_path), result.static_mask)

            # 비네팅 마스크
            if result.vignetting_mask is not None:
                vignetting_path = (
                    self.output_dir / f"{camera_config.camera_id}_vignetting.png"
                )
                cv2.imwrite(str(vignetting_path), result.vignetting_mask)

            # 신뢰도 맵
            confidence_path = (
                self.output_dir / f"{camera_config.camera_id}_confidence.png"
            )
            confidence_vis = (result.confidence_map * 255).astype(np.uint8)
            cv2.imwrite(str(confidence_path), confidence_vis)

        return result, mask_path

    def process_batch(
        self,
        video_paths: dict[str, str | Path],
        camera_configs: dict[str, CameraConfig],
        save_intermediate: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, tuple[MaskResult, Path]]:
        """
        여러 카메라 배치 처리

        Args:
            video_paths: {camera_id: video_path} 딕셔너리
            camera_configs: {camera_id: CameraConfig} 딕셔너리
            save_intermediate: 중간 결과 저장 여부
            progress_callback: 진행 상황 콜백 함수

        Returns:
            {camera_id: (MaskResult, mask_path)} 딕셔너리
        """
        results = {}
        total = len(video_paths)

        for idx, (camera_id, video_path) in enumerate(video_paths.items()):
            if camera_id not in camera_configs:
                print(f"경고: {camera_id}에 대한 설정이 없습니다. 건너뜁니다.")
                continue

            camera_config = camera_configs[camera_id]

            try:
                result, mask_path = self.process_camera(
                    video_path,
                    camera_config,
                    save_intermediate=save_intermediate,
                )
                results[camera_id] = (result, mask_path)

                if progress_callback:
                    progress_callback(idx + 1, total, camera_id)

            except Exception as e:
                print(f"오류: {camera_id} 처리 중 에러 발생: {e}")

        return results


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """마스크를 파일로 저장"""
    cv2.imwrite(str(path), mask)


def load_mask(path: str | Path) -> np.ndarray:
    """파일에서 마스크 로드"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다: {path}")
    return mask


def apply_mask_to_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    fill_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """프레임에 마스크 적용"""
    result = frame.copy()
    mask_bool = mask > 127
    result[mask_bool] = fill_value
    return result

