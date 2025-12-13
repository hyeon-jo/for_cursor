"""
Ego Vehicle Mask Generator - 메인 진입점

CLI 및 GUI 모드를 지원하는 메인 실행 파일입니다.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2


def create_parser() -> argparse.ArgumentParser:
    """커맨드 라인 파서 생성"""
    parser = argparse.ArgumentParser(
        prog="ego-vehicle-mask",
        description="자율주행 영상 데이터에서 자차 영역 마스크를 자동 생성합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GUI 모드 실행
  python -m src.main --gui

  # 단일 영상 처리
  python -m src.main --input video.mp4 --output ./output

  # 특정 카메라 설정으로 처리
  python -m src.main --input video.mp4 --camera front_narrow --output ./output

  # 배치 처리
  python -m src.main --batch --input-dir ./videos --config ./config/cameras.yaml
        """,
    )

    # 모드 선택
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui",
        action="store_true",
        help="GUI 모드로 실행",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="배치 처리 모드로 실행",
    )

    # 입력 옵션
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="입력 영상 또는 이미지 디렉토리 경로",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="배치 처리용 입력 디렉토리 (--batch 모드에서 사용)",
    )

    # 출력 옵션
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="출력 디렉토리 (기본: ./output)",
    )

    # 설정 옵션
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="카메라 설정 파일 경로 (YAML)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="카메라 ID (설정 파일에 정의된 이름)",
    )

    # 알고리즘 옵션
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=50.0,
        help="분산 임계값 (기본: 50.0, 낮을수록 엄격)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="분석할 프레임 수 (기본: 100)",
    )
    parser.add_argument(
        "--no-vignetting",
        action="store_true",
        help="비네팅 검출 비활성화",
    )

    # 기타 옵션
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="중간 결과물 저장 (정적 마스크, 신뢰도 맵 등)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력",
    )

    return parser


def run_gui(config_path: Optional[str] = None) -> int:
    """GUI 모드 실행"""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt

        from .gui.main_window import MainWindow
        from .config_loader import load_config, get_default_config

        # High DPI 지원
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        app = QApplication(sys.argv)
        app.setApplicationName("Ego Vehicle Mask Generator")
        app.setOrganizationName("EgoMask")

        # 다크 테마 스타일
        app.setStyle("Fusion")

        # 설정 로드
        if config_path:
            config = load_config(config_path)
        else:
            config = get_default_config()

        window = MainWindow(config)
        window.show()

        return app.exec()

    except ImportError as e:
        print(f"오류: GUI 모듈을 로드할 수 없습니다: {e}")
        print("PySide6가 설치되어 있는지 확인하세요: pip install PySide6")
        return 1


def run_single(
    input_path: str,
    output_dir: str,
    camera_id: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: float = 50.0,
    num_samples: int = 100,
    include_vignetting: bool = True,
    save_intermediate: bool = False,
    verbose: bool = False,
) -> int:
    """단일 영상 처리"""
    from tqdm import tqdm

    from .config_loader import load_config, get_default_config, CameraConfig
    from .data_loader import DataLoader
    from .mask_generator import MaskGenerator

    # 설정 로드
    if config_path:
        config = load_config(config_path)
    else:
        config = get_default_config()

    # 카메라 설정
    camera_config = None
    if camera_id:
        camera_config = config.get_camera(camera_id)
        if camera_config is None:
            print(f"경고: 카메라 '{camera_id}'에 대한 설정을 찾을 수 없습니다.")

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"입력: {input_path}")
        print(f"출력: {output_dir}")
        print(f"분산 임계값: {threshold}")
        print(f"샘플 수: {num_samples}")

    try:
        # 데이터 로드
        print("데이터 로딩 중...")
        with DataLoader(input_path) as loader:
            if verbose:
                print(f"프레임 수: {loader.frame_count}")
                print(f"해상도: {loader.resolution}")

            print(f"프레임 샘플링 중 ({num_samples}개)...")
            frames = loader.sample_frames(num_samples)

        # 마스크 생성
        print("마스크 생성 중...")
        generator = MaskGenerator(
            variance_threshold=threshold,
        )

        result = generator.generate_from_frames(
            frames,
            camera_config=camera_config,
            include_vignetting=include_vignetting,
        )

        # 결과 저장
        input_name = Path(input_path).stem
        mask_filename = f"{input_name}_mask.png"
        mask_path = output_path / mask_filename

        cv2.imwrite(str(mask_path), result.final_mask)
        print(f"마스크 저장됨: {mask_path}")

        # 중간 결과 저장
        if save_intermediate:
            # 정적 마스크
            static_path = output_path / f"{input_name}_static.png"
            cv2.imwrite(str(static_path), result.static_mask)

            # 비네팅 마스크
            if result.vignetting_mask is not None:
                vignetting_path = output_path / f"{input_name}_vignetting.png"
                cv2.imwrite(str(vignetting_path), result.vignetting_mask)

            # 신뢰도 맵
            confidence_path = output_path / f"{input_name}_confidence.png"
            confidence_vis = (result.confidence_map * 255).astype("uint8")
            cv2.imwrite(str(confidence_path), confidence_vis)

            print("중간 결과물 저장됨")

        print("완료!")
        return 0

    except Exception as e:
        print(f"오류: {e}")
        return 1


def run_batch(
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    threshold: float = 50.0,
    num_samples: int = 100,
    include_vignetting: bool = True,
    save_intermediate: bool = False,
    verbose: bool = False,
) -> int:
    """배치 처리"""
    from tqdm import tqdm

    from .config_loader import load_config, get_default_config
    from .data_loader import VideoLoader
    from .mask_generator import BatchMaskGenerator

    # 설정 로드
    if config_path:
        config = load_config(config_path)
    else:
        config = get_default_config()

    # 입력 디렉토리 스캔
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"오류: 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return 1

    video_files = list(input_path.glob("*.mp4")) + \
                  list(input_path.glob("*.avi")) + \
                  list(input_path.glob("*.mov"))

    if not video_files:
        print(f"오류: 입력 디렉토리에 영상 파일이 없습니다: {input_dir}")
        return 1

    print(f"발견된 영상 파일: {len(video_files)}개")

    # 배치 처리
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_generator = BatchMaskGenerator(
        global_config=config.global_config,
        output_dir=output_path,
    )

    success_count = 0
    fail_count = 0

    for video_file in tqdm(video_files, desc="처리 중"):
        try:
            # 파일명에서 카메라 ID 추측 (선택적)
            camera_id = None
            for cid in config.get_camera_ids():
                if cid in video_file.stem:
                    camera_id = cid
                    break

            camera_config = config.get_camera(camera_id) if camera_id else None

            if verbose:
                print(f"\n처리 중: {video_file.name}")
                if camera_id:
                    print(f"  카메라: {camera_id}")

            result, mask_path = batch_generator.process_camera(
                video_file,
                camera_config or config.cameras.get(
                    list(config.cameras.keys())[0]
                ) if config.cameras else None,
                save_intermediate=save_intermediate,
            )

            success_count += 1

        except Exception as e:
            print(f"오류 ({video_file.name}): {e}")
            fail_count += 1

    print(f"\n완료: 성공 {success_count}개, 실패 {fail_count}개")
    return 0 if fail_count == 0 else 1


def main() -> int:
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()

    # GUI 모드
    if args.gui:
        return run_gui(args.config)

    # 배치 모드
    if args.batch:
        if not args.input_dir:
            print("오류: --batch 모드에서는 --input-dir이 필요합니다.")
            return 1

        return run_batch(
            input_dir=args.input_dir,
            output_dir=args.output,
            config_path=args.config,
            threshold=args.threshold,
            num_samples=args.samples,
            include_vignetting=not args.no_vignetting,
            save_intermediate=args.save_intermediate,
            verbose=args.verbose,
        )

    # 단일 처리 모드
    if args.input:
        return run_single(
            input_path=args.input,
            output_dir=args.output,
            camera_id=args.camera,
            config_path=args.config,
            threshold=args.threshold,
            num_samples=args.samples,
            include_vignetting=not args.no_vignetting,
            save_intermediate=args.save_intermediate,
            verbose=args.verbose,
        )

    # 인자 없이 실행 시 GUI 모드
    if len(sys.argv) == 1:
        return run_gui(args.config)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

