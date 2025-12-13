# Ego Vehicle Mask Generator

자율주행 영상 데이터에서 자차(ego-vehicle) 영역을 Traditional CV 기법으로 자동 검출하여 마스크를 생성하는 소프트웨어입니다.

## 주요 기능

- **시간적 분산 기반 자차 영역 검출**: 여러 프레임에서 항상 고정된 영역을 자차로 판단
- **비네팅 검출**: 광각 카메라의 비네팅 영역 자동 검출
- **GUI 뷰어/에디터**: 마스크 확인 및 수동 편집 기능
- **CLI 지원**: 배치 처리를 위한 커맨드 라인 인터페이스

## 지원하는 자차 영역

| 카메라 타입 | 검출 영역 |
|------------|----------|
| 전방 협각 | 본넷, 전방 라이다 센서 |
| 전방 광각 | 본넷 앞쪽, 비네팅 영역 |
| 측방 협각 | 천장 라이다 슬롯, 사이드 미러, 차량 옆구리 |
| 측방 광각 | 차량 옆구리, 비네팅 영역 |
| 후방 광각 | 하단 범퍼, 번호판, 비네팅 영역 |

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### GUI 모드

```bash
python -m src.main --gui
```

### CLI 모드

```bash
# 단일 카메라 처리
python -m src.main --input /path/to/video.mp4 --camera front_narrow --output ./output

# 배치 처리 (여러 카메라)
python -m src.main --batch --input-dir /path/to/videos --config ./config/cameras.yaml
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--input` | 입력 영상 또는 이미지 디렉토리 경로 |
| `--camera` | 카메라 ID (config에 정의된 이름) |
| `--config` | 카메라 설정 파일 경로 |
| `--output` | 출력 디렉토리 |
| `--gui` | GUI 모드 실행 |
| `--threshold` | 분산 임계값 (기본: 50.0) |
| `--samples` | 분석할 프레임 수 (기본: 100) |

## 설정 파일

`config/cameras.yaml` 파일에서 각 카메라별 설정을 정의합니다:

```yaml
cameras:
  front_narrow:
    type: narrow
    resolution: [1920, 1080]
    has_vignetting: false
    expected_ego_regions:
      - name: hood
        hint_roi: [0, 850, 1920, 230]
```

## 알고리즘 개요

1. **프레임 샘플링**: 영상에서 N개 프레임 균등 추출
2. **분산 계산**: 픽셀별 temporal variance 계산
3. **임계값 적용**: 낮은 분산 영역을 자차 후보로 선정
4. **형태학적 처리**: Opening/Closing으로 노이즈 제거
5. **경계 정제**: 컨투어 기반 경계 정밀화
6. **비네팅 병합**: 광각 카메라의 비네팅 영역 추가

## 라이선스

MIT License

