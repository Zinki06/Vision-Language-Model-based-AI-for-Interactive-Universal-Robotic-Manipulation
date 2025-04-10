"""
유틸리티 패키지 - 여러 모듈에서 사용되는 공통 함수들을 제공합니다.

이 패키지는 기존 utils.py 모듈에서 분리된 내용을 포함하고 있으며,
기능별로 하위 모듈로 구성되어 있습니다.
"""

# 지오메트리/공간 계산 기능
from .geometry import calculate_iou, calculate_box_from_points, calculate_angle_2d

# 데이터 처리 기능
from .data import safe_get, validate_detection_format, DetectionCoordinator, diagnose_data_structure, DetectionType

# 로깅 관련 기능
from .logging import ThrottledStreamHandler, LogFilter, setup_logger, InputModeManager, input_with_suppressed_logs, suppress_logs

# 파일 입출력 기능
from .io import save_json, load_json

# 객체 관리 기능
from .objects import DetectedObject

# 설정 관리 기능
from .config import SystemConfig 