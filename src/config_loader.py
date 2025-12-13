"""
카메라 설정 YAML 로더

카메라별 설정(타입, 해상도, ROI 힌트 등)을 로드하고 관리합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ROIHint:
    """ROI 힌트 정보"""
    name: str
    description: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ROIHint":
        """딕셔너리에서 ROIHint 객체 생성"""
        roi = data.get("hint_roi", [0, 0, 0, 0])
        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            x=roi[0] if len(roi) > 0 else 0,
            y=roi[1] if len(roi) > 1 else 0,
            width=roi[2] if len(roi) > 2 else 0,
            height=roi[3] if len(roi) > 3 else 0,
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        """(x, y, width, height) 튜플로 반환"""
        return (self.x, self.y, self.width, self.height)


@dataclass
class VignettingConfig:
    """비네팅 설정"""
    threshold: float = 0.7  # 중심 대비 밝기 비율 임계값
    margin: int = 50  # 비네팅 마진 (픽셀)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "VignettingConfig":
        """딕셔너리에서 VignettingConfig 객체 생성"""
        if data is None:
            return cls()
        return cls(
            threshold=data.get("threshold", 0.7),
            margin=data.get("margin", 50),
        )


@dataclass
class CameraConfig:
    """개별 카메라 설정"""
    camera_id: str
    camera_type: str  # "narrow" or "wide"
    resolution: tuple[int, int] = (1920, 1080)
    has_vignetting: bool = False
    vignetting_config: VignettingConfig = field(default_factory=VignettingConfig)
    expected_ego_regions: list[ROIHint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, camera_id: str, data: dict) -> "CameraConfig":
        """딕셔너리에서 CameraConfig 객체 생성"""
        resolution = data.get("resolution", [1920, 1080])
        regions = [
            ROIHint.from_dict(r) for r in data.get("expected_ego_regions", [])
        ]
        return cls(
            camera_id=camera_id,
            camera_type=data.get("type", "narrow"),
            resolution=(resolution[0], resolution[1]),
            has_vignetting=data.get("has_vignetting", False),
            vignetting_config=VignettingConfig.from_dict(
                data.get("vignetting_config")
            ),
            expected_ego_regions=regions,
        )


@dataclass
class GlobalConfig:
    """전역 설정"""
    default_resolution: tuple[int, int] = (1920, 1080)
    frame_sample_count: int = 100
    variance_threshold: float = 50.0
    morphology_kernel_size: int = 5

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "GlobalConfig":
        """딕셔너리에서 GlobalConfig 객체 생성"""
        if data is None:
            return cls()
        resolution = data.get("default_resolution", [1920, 1080])
        return cls(
            default_resolution=(resolution[0], resolution[1]),
            frame_sample_count=data.get("frame_sample_count", 100),
            variance_threshold=data.get("variance_threshold", 50.0),
            morphology_kernel_size=data.get("morphology_kernel_size", 5),
        )


@dataclass
class Config:
    """전체 설정"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """카메라 ID로 설정 조회"""
        return self.cameras.get(camera_id)

    def get_camera_ids(self) -> list[str]:
        """모든 카메라 ID 목록 반환"""
        return list(self.cameras.keys())

    def get_wide_cameras(self) -> list[CameraConfig]:
        """광각 카메라 목록 반환"""
        return [c for c in self.cameras.values() if c.camera_type == "wide"]

    def get_narrow_cameras(self) -> list[CameraConfig]:
        """협각 카메라 목록 반환"""
        return [c for c in self.cameras.values() if c.camera_type == "narrow"]


class ConfigLoader:
    """설정 파일 로더"""

    def __init__(self, config_path: Optional[str | Path] = None):
        """
        Args:
            config_path: 설정 파일 경로. None이면 기본 설정 사용
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[Config] = None

    def load(self) -> Config:
        """설정 파일 로드"""
        if self._config is not None:
            return self._config

        if self.config_path is None or not self.config_path.exists():
            # 기본 설정 반환
            self._config = Config()
            return self._config

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        global_config = GlobalConfig.from_dict(data.get("global"))
        cameras = {}

        for camera_id, camera_data in data.get("cameras", {}).items():
            cameras[camera_id] = CameraConfig.from_dict(camera_id, camera_data)

        self._config = Config(global_config=global_config, cameras=cameras)
        return self._config

    def reload(self) -> Config:
        """설정 파일 다시 로드"""
        self._config = None
        return self.load()


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """설정 파일 로드 헬퍼 함수"""
    loader = ConfigLoader(config_path)
    return loader.load()


# 기본 설정 경로
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "cameras.yaml"


def get_default_config() -> Config:
    """기본 설정 파일 로드"""
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    return Config()

