"""
효율적인 프레임 타이밍 및 스킵 관리 유틸리티

일정한 프레임 레이트 유지와 시스템 부하에 따른 
동적 프레임 스킵을 관리하는 기능을 제공합니다.
"""

import time
import logging
from typing import Dict, Any, List, Deque
from collections import deque

# 로거 설정
logger = logging.getLogger(__name__)

class FrameController:
    """프레임 타이밍 및 스킵 관리 클래스
    
    일정한 프레임 레이트를 유지하고 시스템 부하에 따라 
    동적으로 프레임 스킵을 조정하는 기능을 제공합니다.
    
    Attributes:
        target_fps (float): 목표 FPS
        adaptive_skip (bool): 적응형 프레임 스킵 활성화 여부
        max_skip_frames (int): 최대 스킵 프레임 수
        min_skip_frames (int): 최소 스킵 프레임 수
        current_skip_frames (int): 현재 스킵 프레임 수
        skip_counter (int): 현재 스킵 카운터
        processing_times (Deque[float]): 최근 처리 시간 기록 (최대 10개)
        frame_times (Deque[float]): 최근 프레임 처리 시간 (FPS 계산용)
        next_frame_time (float): 다음 프레임 처리 예정 시간
        wait_key_ms (int): 권장 waitKey 값 (ms)
    """
    
    def __init__(self, 
                 target_fps: float = 30.0, 
                 adaptive_skip: bool = True,
                 max_skip_frames: int = 4, 
                 min_skip_frames: int = 0,
                 initial_skip_frames: int = 0,
                 wait_key_ms: int = 33):
        """FrameController 초기화
        
        Args:
            target_fps (float): 목표 FPS (기본값: 30.0)
            adaptive_skip (bool): 적응형 프레임 스킵 활성화 여부 (기본값: True)
            max_skip_frames (int): 최대 스킵 프레임 수 (기본값: 4)
            min_skip_frames (int): 최소 스킵 프레임 수 (기본값: 0)
            initial_skip_frames (int): 초기 스킵 프레임 수 (기본값: 0)
            wait_key_ms (int): 기본 waitKey 값 (ms) (기본값: 33)
        """
        self.target_fps = target_fps
        self.adaptive_skip = adaptive_skip
        self.max_skip_frames = max_skip_frames
        self.min_skip_frames = min_skip_frames
        self.current_skip_frames = initial_skip_frames
        self.skip_counter = 0
        
        # 시간 및 성능 측정용 변수
        self.processing_times = deque(maxlen=10)  # 최근 10개 프레임 처리 시간
        self.frame_times = deque(maxlen=30)       # 최근 30개 프레임 시간 (fps 계산)
        self.frame_interval = 1.0 / target_fps    # 프레임 간 목표 시간 간격
        self.next_frame_time = time.time()        # 다음 프레임 처리 예정 시간
        self.wait_key_ms = wait_key_ms            # 기본 waitKey 값 (깜빡임 방지)
        
        # 통계
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_fps = 0.0
        
        logger.debug(f"FrameController 초기화: target_fps={target_fps}, "
                   f"adaptive_skip={adaptive_skip}, max_skip_frames={max_skip_frames}")
    
    def wait_for_next_frame(self) -> float:
        """다음 프레임 시간까지 대기
        
        목표 FPS를 유지하기 위해 필요한 시간만큼 대기합니다.
        
        Returns:
            float: 실제 대기 시간 (초)
        """
        current_time = time.time()
        wait_time = max(0, self.next_frame_time - current_time)
        
        # 대기 시간이 있으면 대기
        if wait_time > 0:
            time.sleep(wait_time)
        
        # 다음 프레임 시간 계산 (드리프트 방지)
        elapsed = time.time() - current_time
        self.next_frame_time = max(self.next_frame_time + self.frame_interval, time.time())
        
        return elapsed
    
    def should_process_frame(self) -> bool:
        """현재 프레임 처리 여부 결정
        
        스킵 로직에 따라 현재 프레임을 처리할지 여부를 결정합니다.
        
        Returns:
            bool: 프레임 처리 여부
        """
        self.total_frames += 1
        
        # 스킵 프레임이 0이면 항상 처리
        if self.current_skip_frames == 0:
            return True
            
        # 스킵 카운터 증가 및 처리 여부 결정
        self.skip_counter = (self.skip_counter + 1) % (self.current_skip_frames + 1)
        if self.skip_counter == 0:
            return True
        
        # 스킵된 프레임 카운트
        self.dropped_frames += 1
        return False
    
    def update_fps(self) -> float:
        """FPS 계산 및 업데이트
        
        최근 프레임 시간을 기준으로 현재 FPS를 계산합니다.
        
        Returns:
            float: 현재 FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # 최소 2개 이상의 프레임 시간이 있어야 계산 가능
        if len(self.frame_times) < 2:
            return 0.0
            
        # 가장 오래된 프레임과 현재 프레임 사이의 시간으로 FPS 계산
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff > 0:
            self.last_fps = (len(self.frame_times) - 1) / time_diff
            
        return self.last_fps
    
    def update_skip_frames(self, processing_time: float) -> None:
        """적응형 프레임 스킵 업데이트
        
        시스템 부하에 따라 동적으로 스킵 프레임 수를 조정합니다.
        
        Args:
            processing_time (float): 프레임 처리에 소요된 시간 (초)
        """
        if not self.adaptive_skip:
            return
            
        # 최근 처리 시간 기록
        self.processing_times.append(processing_time)
        
        # 충분한 데이터가 쌓이면 스킵 프레임 조정
        if len(self.processing_times) >= 3:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            target_time = self.frame_interval
            
            # 처리 시간이 목표 시간의 80% 이상이면 스킵 증가
            if avg_time > target_time * 0.8:
                # 처리 부하가 큰 경우 스킵 프레임 증가
                new_skip = min(self.max_skip_frames, 
                              self.current_skip_frames + 1)
                
                if new_skip != self.current_skip_frames:
                    self.current_skip_frames = new_skip
                    logger.debug(f"부하 증가로 스킵 프레임 증가: {self.current_skip_frames}")
                    
            # 처리 시간이 목표 시간의 50% 미만이고, 현재 스킵 중이면 스킵 감소
            elif avg_time < target_time * 0.5 and self.current_skip_frames > self.min_skip_frames:
                # 여유가 있는 경우 스킵 프레임 감소
                new_skip = max(self.min_skip_frames, 
                              self.current_skip_frames - 1)
                              
                if new_skip != self.current_skip_frames:
                    self.current_skip_frames = new_skip
                    logger.debug(f"여유 있어 스킵 프레임 감소: {self.current_skip_frames}")
    
    def get_wait_key_ms(self) -> int:
        """권장 waitKey 값 반환
        
        화면 깜빡임을 방지하기 위한 적정 waitKey 값을 반환합니다.
        
        Returns:
            int: 권장 waitKey 값 (ms)
        """
        return self.wait_key_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보 딕셔너리
        """
        return {
            'target_fps': self.target_fps,
            'current_fps': self.last_fps,
            'current_skip_frames': self.current_skip_frames,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'avg_processing_time_ms': (sum(self.processing_times) / max(1, len(self.processing_times))) * 1000 
                                     if self.processing_times else 0,
            'frame_interval_ms': self.frame_interval * 1000
        }
