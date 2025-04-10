"""
성능 측정을 위한 프로파일러 유틸리티
"""

import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)

def timeit(func: Callable) -> Callable:
    """함수 실행 시간을 측정하는 데코레이터
    
    Args:
        func: 측정할 함수
        
    Returns:
        래핑된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"Function {func.__name__} took {elapsed_ms:.2f} ms to run")
        
        return result
    return wrapper

class FPSCounter:
    """FPS 계산기 클래스"""
    
    def __init__(self, window_size: int = 30):
        """초기화
        
        Args:
            window_size: 평균 계산에 사용할 프레임 수
        """
        self.frame_times = []
        self.window_size = window_size
        self.last_fps = 0
        
    def update(self) -> float:
        """새 프레임 처리 후 호출하여 FPS 업데이트
        
        Returns:
            계산된 FPS 값
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # 윈도우 크기 유지
        if len(self.frame_times) > self.window_size:
            self.frame_times = self.frame_times[-self.window_size:]
        
        # 최소 2개 이상의 프레임이 있어야 계산 가능
        if len(self.frame_times) < 2:
            return 0
            
        # FPS 계산
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed > 0:
            self.last_fps = (len(self.frame_times) - 1) / elapsed
            
        return self.last_fps
        
    def get_fps(self) -> float:
        """현재 FPS 값 반환"""
        return self.last_fps
