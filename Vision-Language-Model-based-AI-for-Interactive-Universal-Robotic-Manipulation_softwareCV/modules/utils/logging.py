"""
로깅 관련 유틸리티 모듈

이 모듈은 로그 출력, 사용자 입력 모드 관리 등과 관련된 함수 및 클래스를 제공합니다.
"""

import logging
import os
import sys
import time
import threading
from contextlib import contextmanager

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