"""
카메라 인터페이스 모듈

다양한 카메라 구현체에 대한 공통 인터페이스를 정의합니다.
"""

import abc
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np

class CameraInterface(abc.ABC):
    """카메라 인터페이스
    
    다양한 카메라 구현체가 구현해야 하는 공통 인터페이스를 정의합니다.
    """
    
    @abc.abstractmethod
    def open(self) -> bool:
        """카메라 장치 열기
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def close(self) -> bool:
        """카메라 장치 닫기
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def is_open(self) -> bool:
        """카메라 장치가 열려있는지 확인
        
        Returns:
            bool: 열림 상태
        """
        pass
    
    @abc.abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """단일 프레임 가져오기
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (성공 여부, 프레임 데이터)
        """
        pass
        
    @abc.abstractmethod
    def start_capture(self) -> bool:
        """프레임 캡처 시작
        
        Returns:
            bool: 성공 여부
        """
        pass
        
    @abc.abstractmethod
    def stop_capture(self) -> bool:
        """프레임 캡처 중지
        
        Returns:
            bool: 성공 여부
        """
        pass
        
    @abc.abstractmethod
    def is_capturing(self) -> bool:
        """현재 프레임을 캡처하고 있는지 확인
        
        Returns:
            bool: 캡처 중 여부
        """
        pass
        
    @abc.abstractmethod
    def get_camera_properties(self) -> Dict[str, Any]:
        """카메라 속성 가져오기
        
        Returns:
            Dict[str, Any]: 카메라 속성 정보 (해상도, FPS 등)
        """
        pass
        
    @abc.abstractmethod
    def set_camera_property(self, property_name: str, value: Any) -> bool:
        """카메라 속성 설정
        
        Args:
            property_name: 설정할 속성 이름
            value: 설정할 값
            
        Returns:
            bool: 성공 여부
        """
        pass
        
    @abc.abstractmethod
    def get_available_cameras(self) -> List[Dict[str, Any]]:
        """사용 가능한 카메라 목록 가져오기
        
        Returns:
            List[Dict[str, Any]]: 사용 가능한 카메라 목록 (카메라 ID, 이름 등의 정보 포함)
        """
        pass
        
    @abc.abstractmethod
    def get_camera_info(self) -> Dict[str, Any]:
        """현재 카메라 정보 가져오기
        
        Returns:
            Dict[str, Any]: 현재 카메라 상세 정보
        """
        pass 