"""
기본 프레임 프로세서 모듈

프레임 처리를 위한 기본 기능을 제공합니다.
"""

import time
import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple, Any, Set
from collections import deque
import numpy as np

from .frame_processor_interface import FrameProcessorInterface

class BaseFrameProcessor(FrameProcessorInterface):
    """기본 프레임 프로세서
    
    프레임 처리를 위한 기본 기능을 구현합니다.
    
    Attributes:
        max_queue_size (int): 최대 프레임 큐 크기
        frame_queue (deque): 프레임 큐
        processing_thread (threading.Thread): 프레임 처리 스레드
        running (bool): 처리 스레드 실행 상태
        callbacks (Set[Callable]): 프레임 처리 콜백 함수들
        stats (Dict[str, Any]): 통계 정보
        lock (threading.Lock): 스레드 동기화를 위한 락
        logger (logging.Logger): 로거
    """
    
    def __init__(self, max_queue_size: int = 100):
        """초기화
        
        Args:
            max_queue_size: 최대 프레임 큐 크기
        """
        self.max_queue_size = max_queue_size
        self.frame_queue = deque(maxlen=max_queue_size)
        self.processing_thread = None
        self.running = False
        self.callbacks: Set[Callable] = set()
        self.stats = {
            'processed_frames': 0,
            'dropped_frames': 0,
            'processing_time': 0.0,
            'avg_processing_time': 0.0,
            'last_frame_time': 0.0
        }
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, float, Dict[str, Any]], None]) -> None:
        """프레임 처리 콜백 추가
        
        Args:
            callback: 프레임 처리 콜백 함수
        """
        with self.lock:
            self.callbacks.add(callback)
        self.logger.debug(f"콜백 추가됨: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}")
    
    def remove_frame_callback(self, callback: Callable) -> bool:
        """프레임 처리 콜백 제거
        
        Args:
            callback: 제거할 콜백 함수
            
        Returns:
            bool: 제거 성공 여부
        """
        with self.lock:
            try:
                self.callbacks.remove(callback)
                self.logger.debug(f"콜백 제거됨: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}")
                return True
            except KeyError:
                self.logger.warning(f"콜백을 찾을 수 없음: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}")
                return False
    
    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """프레임 큐에 프레임 추가
        
        Args:
            frame: 추가할 프레임
            timestamp: 프레임 타임스탬프 (None이면 현재 시간)
            metadata: 프레임 메타데이터 (None이면 빈 딕셔너리)
            
        Returns:
            bool: 추가 성공 여부
        """
        if not isinstance(frame, np.ndarray):
            self.logger.error(f"유효하지 않은 프레임 형식: {type(frame)}")
            return False
            
        # 필요한 경우 타임스탬프 및 메타데이터 생성
        if timestamp is None:
            timestamp = time.time()
        if metadata is None:
            metadata = {}
            
        # 프레임 큐가 가득 찬 경우 드롭된 프레임 계산
        if len(self.frame_queue) >= self.max_queue_size:
            with self.lock:
                self.stats['dropped_frames'] += 1
                
        # 프레임 추가
        self.frame_queue.append({
            'frame': frame.copy(),
            'timestamp': timestamp,
            'metadata': metadata
        })
        
        return True
        
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """최신 프레임 가져오기
        
        Returns:
            Optional[Dict[str, Any]]: 최신 프레임 정보 (프레임, 타임스탬프, 메타데이터 등)
        """
        try:
            return self.frame_queue[-1] if self.frame_queue else None
        except Exception as e:
            self.logger.error(f"최신 프레임 가져오기 오류: {e}")
            return None
    
    def _processing_loop(self):
        """프레임 처리 루프"""
        while self.running:
            frames_processed = self.process_frames(10)  # 한 번에 최대 10개 프레임 처리
            if frames_processed == 0:
                time.sleep(0.01)  # 처리할 프레임이 없는 경우 대기
    
    def process_frames(self, max_frames: Optional[int] = None) -> int:
        """프레임 처리
        
        캡처된 프레임들을 처리합니다.
        
        Args:
            max_frames: 처리할 최대 프레임 수 (None이면 제한 없음)
            
        Returns:
            int: 처리된 프레임 수
        """
        processed = 0
        
        if not self.callbacks:
            # 콜백이 없는 경우 프레임 큐 비우기
            processed = len(self.frame_queue)
            self.frame_queue.clear()
            return processed
            
        max_to_process = len(self.frame_queue) if max_frames is None else min(max_frames, len(self.frame_queue))
        
        for _ in range(max_to_process):
            try:
                frame_data = self.frame_queue.popleft()
                
                start_time = time.time()
                
                # 모든 콜백 호출
                with self.lock:  # 콜백 컬렉션에 대한 안전한 접근
                    callbacks = self.callbacks.copy()
                
                for callback in callbacks:
                    try:
                        callback(frame_data['frame'], frame_data['timestamp'], frame_data['metadata'])
                    except Exception as e:
                        self.logger.error(f"콜백 실행 오류: {e}")
                
                processing_time = time.time() - start_time
                
                # 통계 업데이트
                with self.lock:
                    self.stats['processed_frames'] += 1
                    self.stats['processing_time'] += processing_time
                    self.stats['last_frame_time'] = frame_data['timestamp']
                    self.stats['avg_processing_time'] = (
                        self.stats['processing_time'] / self.stats['processed_frames']
                    )
                
                processed += 1
                
            except IndexError:
                break  # 큐가 비었음
            except Exception as e:
                self.logger.error(f"프레임 처리 오류: {e}")
        
        return processed
    
    def start(self) -> bool:
        """프레임 처리 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        if self.running:
            self.logger.warning("프레임 프로세서가 이미 실행 중입니다.")
            return False
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("프레임 프로세서 시작됨")
        return True
    
    def stop(self, timeout: float = 2.0) -> bool:
        """프레임 처리 중지
        
        Args:
            timeout: 종료 대기 시간(초)
            
        Returns:
            bool: 중지 성공 여부
        """
        if not self.running:
            return True
            
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout)
            if self.processing_thread.is_alive():
                self.logger.warning(f"프레임 프로세서 스레드가 {timeout}초 내에 종료되지 않았습니다.")
                return False
        
        self.logger.info("프레임 프로세서 중지됨")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보 딕셔너리
        """
        with self.lock:
            return self.stats.copy()
    
    def clear_stats(self) -> None:
        """통계 정보 초기화"""
        with self.lock:
            self.stats = {
                'processed_frames': 0,
                'dropped_frames': 0,
                'processing_time': 0.0,
                'avg_processing_time': 0.0,
                'last_frame_time': 0.0
            }
    
    def __del__(self):
        """소멸자"""
        self.stop(timeout=1.0) 