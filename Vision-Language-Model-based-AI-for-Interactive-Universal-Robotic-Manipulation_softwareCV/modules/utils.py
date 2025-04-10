"""
데이터 처리, 검증 및 표준화를 위한 유틸리티 모듈

이 모듈은 여러 모듈 간의 데이터 교환을 표준화하고,
안전한 데이터 접근 및 예외 처리를 위한 유틸리티 함수를 제공합니다.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, TypedDict
import os
import sys
from contextlib import contextmanager
import json

# 타입 정의
class DetectionType(TypedDict):
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

class DetectedObject:
    """객체 감지 결과를 표준화된 형식으로 관리하는 클래스"""
    
    def __init__(self, id: int, class_name: str, confidence: float, box: List[float], **kwargs):
        """
        객체 초기화
        
        Args:
            id: 객체 ID
            class_name: 객체 클래스 이름
            confidence: 감지 신뢰도
            box: 바운딩 박스 [x1, y1, x2, y2]
            **kwargs: 추가 속성
        """
        self.id = id
        self.class_name = class_name
        self.confidence = confidence
        self.box = box
        
        # 추가 속성 처리
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], id: Optional[int] = None) -> 'DetectedObject':
        """딕셔너리에서 객체 생성"""
        # 필수 키가 없는 경우 대체값 사용
        object_id = id if id is not None else data.get("id", 0)
        class_name = safe_get(data, "class_name") or safe_get(data, "class", "unknown")
        confidence = float(safe_get(data, "confidence", 0.0))
        box = safe_get(data, "box", [0, 0, 0, 0])
        
        # 이미 처리한 필수 키를 제외한 나머지 키를 kwargs로 전달
        kwargs = {k: v for k, v in data.items() 
                  if k not in ["id", "class_name", "class", "confidence", "box"]}
        
        return cls(object_id, class_name, confidence, box, **kwargs)

class ThrottledStreamHandler(logging.StreamHandler):
    """
    속도 제한된 스트림 핸들러
    
    지정된 시간 간격으로만 로그를 콘솔에 출력하고,
    사용자 입력 중에는 로그 출력을 억제합니다.
    """
    
    def __init__(self, stream=None, throttle_interval=3.0):
        """
        핸들러 초기화
        
        Args:
            stream: 출력 스트림
            throttle_interval: 로그 출력 간격(초)
        """
        super().__init__(stream)
        self.throttle_interval = throttle_interval
        self.last_emit_time = {}  # 로거별 마지막 출력 시간
        self.lock = threading.RLock()
        self.input_mode = False  # 사용자 입력 모드 플래그
        
        # 로깅 레벨별 최소 표시 간격 (초)
        self.level_intervals = {
            logging.DEBUG: 10.0,      # 디버그: 10초마다
            logging.INFO: 3.0,        # 정보: 3초마다
            logging.WARNING: 2.0,     # 경고: 2초마다
            logging.ERROR: 0.0,       # 오류: 항상 표시
            logging.CRITICAL: 0.0     # 치명적: 항상 표시
        }
        
        # 중요 로거/메시지는 항상 표시 (정규식 패턴)
        self.important_loggers = [
            "MainApp",
            "Camera"
        ]
        
        self.important_messages = [
            "시작",
            "완료",
            "초기화",
            "오류",
            "실패"
        ]
    
    def set_input_mode(self, active):
        """
        사용자 입력 모드 설정
        
        Args:
            active: 입력 모드 활성화 여부
        """
        with self.lock:
            self.input_mode = active
    
    def is_important(self, record):
        """
        중요 로그 메시지인지 확인
        
        Args:
            record: 로그 레코드
            
        Returns:
            중요 메시지 여부
        """
        # 오류/치명적 로그는 항상 중요
        if record.levelno >= logging.ERROR:
            return True
            
        # 중요 로거 확인
        for logger_name in self.important_loggers:
            if logger_name in record.name:
                return True
                
        # 중요 메시지 패턴 확인
        for pattern in self.important_messages:
            if pattern in record.message:
                return True
                
        return False
    
    def emit(self, record):
        """
        로그 레코드 출력
        
        Args:
            record: 로그 레코드
        """
        # 사용자 입력 모드이면서 중요하지 않은 메시지는 억제
        with self.lock:
            if self.input_mode and not self.is_important(record):
                return
            
            # 레벨별 최소 출력 간격 적용
            current_time = time.time()
            logger_key = record.name + str(record.levelno)
            
            # 마지막 출력 시간 확인
            last_time = self.last_emit_time.get(logger_key, 0)
            
            # 레벨별 간격 또는 기본 간격 가져오기
            interval = self.level_intervals.get(record.levelno, self.throttle_interval)
            
            # 중요 메시지는 간격 제한 없이 출력
            if not self.is_important(record) and current_time - last_time < interval:
                return
                
            # 출력 시간 업데이트
            self.last_emit_time[logger_key] = current_time
            
            # 실제 출력 처리
            super().emit(record)

def safe_get(dictionary: Dict[str, Any], *keys, default: Any = None) -> Any:
    """딕셔너리에서 안전하게 값 가져오기
    
    Args:
        dictionary: 검색할 딕셔너리
        *keys: 검색할 여러 키 (순서대로 검색)
        default: 키를 찾지 못했을 때 반환할 기본값
        
    Returns:
        찾은 값 또는 기본값
    """
    if not isinstance(dictionary, dict):
        return default
    
    # 단일 키 처리
    if len(keys) == 1:
        return dictionary.get(keys[0], default)
    
    # 여러 키 순서대로 검색
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    
    # 모든 키를 찾지 못한 경우 기본값 반환
    return default

def validate_detection_format(detection: Dict[str, Any], logger=None) -> DetectionType:
    """감지 결과 형식 검증 및 변환"""
    try:
        # 필수 키 확인 및 타입 변환
        result = {
            "class_name": str(safe_get(detection, "class_name") or safe_get(detection, "class", "")),
            "confidence": float(safe_get(detection, "confidence", 0)),
            "box": [float(v) for v in safe_get(detection, "box", [0, 0, 0, 0])]
        }
        return result
    except Exception as e:
        # 기본값 반환하며 로깅
        if logger:
            logger.error(f"객체 검증 오류: {e}")
        else:
            logging.error(f"객체 검증 오류: {e}")
        return {"class_name": "unknown", "confidence": 0.0, "box": [0, 0, 0, 0]}

class DetectionCoordinator:
    """객체 감지 결과 조정 및 표준화"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def standardize_detections(self, detections: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """다양한 형식의 감지 결과를 표준 형식으로 변환"""
        if not detections or "objects" not in detections:
            return {"objects": []}
            
        standard_objects = []
        for idx, obj in enumerate(detections["objects"]):
            try:
                # 키 이름 불일치 방지를 위한 표준화
                standard_obj = {
                    "id": idx,
                    "class_name": safe_get(obj, "class_name") or safe_get(obj, "class", "unknown"),
                    "confidence": float(safe_get(obj, "confidence", 0)),
                    "box": safe_get(obj, "box", [0, 0, 0, 0])
                }
                # 추가 필드 복사
                for key, value in obj.items():
                    if key not in ["id", "class_name", "class", "confidence", "box"]:
                        standard_obj[key] = value
                        
                standard_objects.append(standard_obj)
            except Exception as e:
                self.logger.error(f"객체 표준화 오류: {e}")
                
        return {"objects": standard_objects}

class SystemConfig:
    """시스템 설정 중앙화"""
    
    DEFAULT_CONFIG = {
        "detection_keys": {
            "class": "class_name",  # 표준 키 이름
            "confidence": "confidence",
            "box": "box"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    def __init__(self, custom_config=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            self._update_config(custom_config)
    
    def _update_config(self, custom_config):
        """설정 업데이트"""
        for section, values in custom_config.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
            else:
                self.config[section] = values
    
    def get_key_name(self, module, key):
        """모듈별 키 이름 조회"""
        module_config = self.config.get(module, {})
        return module_config.get(key, self.config["detection_keys"].get(key))

def diagnose_data_structure(data, depth=0, max_depth=3, path="", logger=None):
    """재귀적으로 데이터 구조 분석 및 로깅"""
    log = logger or logging.getLogger()
    
    if depth > max_depth:
        return
        
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            log.debug(f"검사 경로: {current_path}, 타입: {type(value)}")
            diagnose_data_structure(value, depth+1, max_depth, current_path, log)
    elif isinstance(data, list) and data and depth < max_depth:
        log.debug(f"목록 검사: {path}[0]/{len(data)}개 항목")
        if data:  # 비어있지 않은 리스트만 첫 요소 검사
            diagnose_data_structure(data[0], depth+1, max_depth, f"{path}[0]", log)

# 로그 억제 상태를 관리하는 클래스
class InputModeManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InputModeManager, cls).__new__(cls)
            cls._instance.is_input_mode = False
            cls._instance.last_log_time = 0
            cls._instance.log_interval = 3.0  # 로그 표시 간격 (초)
        return cls._instance
    
    def start_input_mode(self):
        """사용자 입력 모드 시작"""
        self.is_input_mode = True
    
    def end_input_mode(self):
        """사용자 입력 모드 종료"""
        self.is_input_mode = False
    
    def should_display_log(self, level=logging.INFO):
        """현재 로그를 표시해야 하는지 결정
        
        Args:
            level (int): 로그 레벨
            
        Returns:
            bool: 로그를 표시해야 하면 True, 아니면 False
        """
        # 오류 및 경고는 항상 표시
        if level >= logging.WARNING:
            return True
            
        # 입력 모드 중에는 로그 표시 안함
        if self.is_input_mode:
            return False
            
        # 로그 간격 확인
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
            
        return False

# 로그 필터 클래스
class LogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.manager = InputModeManager()
    
    def filter(self, record):
        # 파일 로그는 항상 기록
        if getattr(record, 'filename_only', False):
            return True
            
        # 콘솔 로그는 표시 여부 결정
        return self.manager.should_display_log(record.levelno)

def setup_logger(name, log_file=None, level=logging.INFO):
    """로거 설정
    
    Args:
        name (str): 로거 이름
        log_file (str, optional): 로그 파일 경로
        level (int, optional): 로그 레벨
        
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 핸들러가 이미 있으면 초기화하지 않음
    if logger.handlers:
        return logger
    
    # 로그 포맷 설정
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # 콘솔 출력 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(LogFilter())
    logger.addHandler(console_handler)
    
    # 파일 출력 설정
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def input_with_suppressed_logs(prompt=""):
    """로그 출력을 억제하며 사용자 입력 받기
    
    Args:
        prompt (str): 입력 프롬프트
        
    Returns:
        str: 사용자 입력
    """
    # 입력 모드 시작
    manager = InputModeManager()
    manager.start_input_mode()
    
    try:
        # 기존 내용 지우기 (선택 사항)
        sys.stdout.write("\033[K")  # 현재 줄 지우기
        sys.stdout.write(prompt)
        sys.stdout.flush()
        
        # 입력 받기
        user_input = input()
        return user_input
    finally:
        # 입력 모드 종료
        manager.end_input_mode()

@contextmanager
def suppress_logs():
    """로그 출력을 일시적으로 억제하는 컨텍스트 매니저"""
    manager = InputModeManager()
    manager.start_input_mode()
    try:
        yield
    finally:
        manager.end_input_mode()

def save_json(data, file_path):
    """JSON 데이터를 파일로 저장
    
    Args:
        data (dict): 저장할 데이터
        file_path (str): 저장 경로
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path):
    """JSON 파일 로드
    
    Args:
        file_path (str): 파일 경로
        
    Returns:
        dict: 로드된 데이터
    """
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f) 