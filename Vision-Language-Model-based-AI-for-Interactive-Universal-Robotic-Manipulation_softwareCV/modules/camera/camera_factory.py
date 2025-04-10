"""
카메라 팩토리 모듈

다양한 카메라 구현체를 생성하고 관리하는 팩토리 클래스입니다.
"""

from typing import Dict, List, Optional, Any, Type
import importlib

from modules.camera.base_camera import BaseCamera
from modules.camera.webcam_camera import WebcamCamera


class CameraFactory:
    """카메라 팩토리 클래스
    
    다양한 카메라 구현체를 생성하고 관리하는 클래스입니다.
    """
    
    # 지원하는 카메라 유형과 해당 클래스
    _CAMERA_TYPES: Dict[str, Type[BaseCamera]] = {
        "webcam": WebcamCamera,
    }
    
    @classmethod
    def get_camera_types(cls) -> List[str]:
        """지원하는 카메라 유형 목록 반환
        
        Returns:
            List[str]: 지원하는 카메라 유형 목록
        """
        return list(cls._CAMERA_TYPES.keys())
    
    @classmethod
    def register_camera_type(cls, camera_type: str, camera_class: Type[BaseCamera]) -> None:
        """새로운 카메라 유형 등록
        
        Args:
            camera_type: 카메라 유형 식별자
            camera_class: 카메라 클래스 (BaseCamera 상속)
        """
        cls._CAMERA_TYPES[camera_type] = camera_class
    
    @classmethod
    def create_camera(cls, camera_type: str, camera_id: str, name: str = None, **kwargs) -> Optional[BaseCamera]:
        """카메라 인스턴스 생성
        
        Args:
            camera_type: 카메라 유형 (webcam, ip_camera 등)
            camera_id: 카메라 식별자
            name: 카메라 이름 (선택 사항)
            **kwargs: 카메라 특정 매개변수
            
        Returns:
            Optional[BaseCamera]: 생성된 카메라 객체 또는 None
        """
        if camera_type not in cls._CAMERA_TYPES:
            print(f"지원하지 않는 카메라 유형: {camera_type}")
            return None
            
        camera_class = cls._CAMERA_TYPES[camera_type]
        
        try:
            return camera_class(camera_id=camera_id, name=name, **kwargs)
        except Exception as e:
            print(f"카메라 생성 오류 ({camera_type}): {e}")
            return None
    
    @classmethod
    def discover_cameras(cls) -> List[Dict[str, Any]]:
        """사용 가능한 모든 카메라 검색
        
        모든 등록된 카메라 유형에 대해 사용 가능한 카메라를 검색합니다.
        
        Returns:
            List[Dict[str, Any]]: 사용 가능한 카메라 정보 목록
        """
        available_cameras = []
        
        # 웹캠 카메라 검색
        if "webcam" in cls._CAMERA_TYPES:
            camera_class = cls._CAMERA_TYPES["webcam"]
            if hasattr(camera_class, "get_available_cameras"):
                webcams = camera_class.get_available_cameras()
                available_cameras.extend(webcams)
        
        # 다른 카메라 유형도 검색 (확장 가능)
        # ...
        
        return available_cameras 