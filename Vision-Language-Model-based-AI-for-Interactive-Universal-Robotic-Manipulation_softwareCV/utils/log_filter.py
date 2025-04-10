"""
로그 중복 제거 필터 모듈

같은 유형의 로그 메시지가 반복적으로 표시되는 것을 방지하는 필터
"""

import logging
import time
from typing import Dict, Optional, Tuple

class DuplicateFilter(logging.Filter):
    """
    같은 로그 메시지가 반복해서 표시되는 것을 방지하는 필터
    일정 시간 내에 동일한 오류 메시지는 한 번만 로깅합니다.
    """
    
    def __init__(self, cooldown_seconds: int = 5):
        """
        Args:
            cooldown_seconds: 같은 메시지가 다시 출력되기까지의 최소 간격(초)
        """
        super().__init__()
        self.cooldown_seconds = cooldown_seconds
        self.last_logs: Dict[str, Tuple[int, float]] = {}  # 메시지: (반복횟수, 마지막 시간)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        로그 레코드를 필터링합니다.
        
        Args:
            record: 로그 레코드
            
        Returns:
            True: 로그 출력, False: 로그 무시
        """
        # 오류 로그가 아니면 항상 출력
        if record.levelno < logging.ERROR:
            return True
            
        # 메시지 해시 생성 (모듈과 메시지 내용 기반)
        msg_key = f"{record.module}:{record.getMessage()}"
        
        # 현재 시간
        current_time = time.time()
        
        # 이전에 같은 메시지가 있었는지 확인
        if msg_key in self.last_logs:
            count, last_time = self.last_logs[msg_key]
            
            # 쿨다운 시간이 지났는지 확인
            if current_time - last_time < self.cooldown_seconds:
                # 반복 횟수 증가
                self.last_logs[msg_key] = (count + 1, last_time)
                
                # 로그 내용에 처리된 메시지 수 포함 (마지막 메시지만)
                if current_time - last_time > self.cooldown_seconds - 0.1:
                    if count > 0:
                        record.msg += f" (반복 횟수: {count})"
                    return True
                
                # 쿨다운 중이면 출력하지 않음
                return False
            else:
                # 쿨다운 시간이 지났으면 초기화하고 출력
                self.last_logs[msg_key] = (0, current_time)
                return True
        else:
            # 처음 보는 메시지는 기록하고 출력
            self.last_logs[msg_key] = (0, current_time)
            return True

def setup_deduplication():
    """
    중복 제거 필터를 설정합니다.
    """
    # 루트 로거에 적용
    root_logger = logging.getLogger()
    
    # 기존 DuplicateFilter 제거 (중복 적용 방지)
    for handler in root_logger.handlers:
        for filter in handler.filters:
            if isinstance(filter, DuplicateFilter):
                handler.removeFilter(filter)
    
    # 새 필터 추가
    duplicate_filter = DuplicateFilter()
    for handler in root_logger.handlers:
        handler.addFilter(duplicate_filter)
        
    # LLM2PF6 로거에도 적용
    app_logger = logging.getLogger("LLM2PF6")
    for handler in app_logger.handlers:
        handler.addFilter(duplicate_filter)
        
    return duplicate_filter 