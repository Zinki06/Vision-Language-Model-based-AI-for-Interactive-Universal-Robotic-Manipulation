"""
프레임 프로세서 인터페이스 모듈

프레임 처리에 필요한 공통 인터페이스를 정의합니다.
"""

import abc
from typing import Dict, Any, Optional, Callable, Set, List, Tuple
import numpy as np

class FrameProcessorInterface(abc.ABC):
    """프레임 프로세서 인터페이스
    
    카메라에서 얻은 프레임을 처리하는 클래스들이 구현해야 하는 공통 인터페이스입니다.
    """
    
    @abc.abstractmethod
    def add_frame_callback(self, callback: Callable[[np.ndarray, Dict[str, Any]], None]) -> None:
        """프레임 처리 콜백 함수 추가
        
        Args:
            callback: 프레임 처리 콜백 함수
                     첫 번째 인자는 프레임 데이터(np.ndarray), 
                     두 번째 인자는 메타데이터(Dict)입니다.
        """
        pass
    
    @abc.abstractmethod
    def remove_frame_callback(self, callback: Callable[[np.ndarray, Dict[str, Any]], None]) -> bool:
        """프레임 처리 콜백 함수 제거
        
        Args:
            callback: 제거할 콜백 함수
            
        Returns:
            bool: 제거 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def add_frame(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """처리할 프레임 추가
        
        Args:
            frame: 처리할 프레임 데이터
            metadata: 프레임 관련 메타데이터 (타임스탬프 등)
            
        Returns:
            bool: 추가 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """가장 최근에 처리된 프레임 가져오기
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (프레임 데이터, 메타데이터) 또는 None
        """
        pass
    
    @abc.abstractmethod
    def process_frames(self, num_frames: int = 1) -> int:
        """지정된 수의 프레임 처리
        
        Args:
            num_frames: 처리할 프레임 수
            
        Returns:
            int: 실제로 처리된 프레임 수
        """
        pass
    
    @abc.abstractmethod
    def start(self) -> bool:
        """프레임 처리 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def stop(self, timeout: Optional[float] = None) -> bool:
        """프레임 처리 중지
        
        Args:
            timeout: 종료 대기 시간(초), None이면 기본값 사용
            
        Returns:
            bool: 중지 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 정보 가져오기
        
        Returns:
            Dict[str, Any]: 통계 정보 (처리된 프레임 수, 초당 처리 프레임 등)
        """
        pass
    
    @abc.abstractmethod
    def clear_stats(self) -> None:
        """통계 정보 초기화"""
        pass 