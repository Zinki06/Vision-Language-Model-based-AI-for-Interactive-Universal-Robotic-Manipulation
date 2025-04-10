import logging
import time
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np

# 로거 설정
logger = logging.getLogger(__name__)

class WebcamCapture:
    """웹캠에서 영상을 캡처하는 클래스.
    
    OpenCV VideoCapture를 사용하여 웹캠에서 영상을 캡처하고, 
    해상도, FPS 등의 다양한 설정을 제공합니다.
    
    Attributes:
        camera_id (int): 웹캠 장치 ID
        width (int): 캡처 영상 너비
        height (int): 캡처 영상 높이
        fps (int): 초당 프레임 수
        cap (cv2.VideoCapture): OpenCV 비디오 캡처 객체
        is_initialized (bool): 카메라 초기화 여부
    """
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, 
                 fps: int = 30, auto_init: bool = True):
        """WebcamCapture 클래스 초기화
        
        Args:
            camera_id (int): 웹캠 장치 ID (기본값: 0)
            width (int): 캡처 영상 너비 (기본값: 1280)
            height (int): 캡처 영상 높이 (기본값: 720)
            fps (int): 초당 프레임 수 (기본값: 30)
            auto_init (bool): 자동 초기화 여부 (기본값: True)
            
        Raises:
            RuntimeError: 카메라 초기화 실패 시 발생
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_initialized = False
        
        if auto_init:
            self.initialize()
    
    def initialize(self) -> None:
        """웹캠 초기화 및 설정
        
        카메라 연결을 설정하고 해상도, FPS 등의 파라미터를 설정합니다.
        
        Raises:
            RuntimeError: 카메라 초기화 실패 시 발생
        """
        logger.info(f"웹캠(ID: {self.camera_id}) 초기화 중...")
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"카메라 ID {self.camera_id}를 열 수 없습니다.")
            
            # 카메라 속성 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 설정된 실제 값 확인
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 카메라 속성 로깅
            logger.info(f"설정된 해상도: {actual_width}x{actual_height}, FPS: {actual_fps}")
            logger.info("웹캠 초기화 성공")
            
            # 카메라 안정화를 위해 몇 프레임 버리기
            self._stabilize_camera()
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"웹캠 초기화 중 오류 발생: {e}")
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.is_initialized = False
            raise RuntimeError(f"웹캠 초기화 실패: {e}")
    
    def _stabilize_camera(self, num_frames: int = 10, delay: float = 0.1) -> None:
        """카메라 안정화
        
        카메라가 안정화될 때까지 일정 수의 프레임을 캡처하고 버립니다.
        
        Args:
            num_frames (int): 버릴 프레임 수 (기본값: 10)
            delay (float): 프레임 간 지연 시간(초) (기본값: 0.1)
        """
        logger.debug(f"카메라 안정화 중 ({num_frames} 프레임)...")
        for _ in range(num_frames):
            self.cap.read()
            time.sleep(delay)
        logger.debug("카메라 안정화 완료")
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 캡처
        
        현재 프레임을 캡처합니다.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (성공 여부, 프레임 이미지)
            
        Raises:
            RuntimeError: 카메라가 초기화되지 않은 경우 발생
        """
        if not self.is_initialized or self.cap is None:
            raise RuntimeError("카메라가 초기화되지 않았습니다. initialize() 메서드를 먼저 호출하세요.")
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("프레임 캡처 실패")
            return False, None
        
        return True, frame
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """카메라 속성 정보 반환
        
        Returns:
            Dict[str, Any]: 카메라 속성 정보 딕셔너리
            
        Raises:
            RuntimeError: 카메라가 초기화되지 않은 경우 발생
        """
        if not self.is_initialized or self.cap is None:
            raise RuntimeError("카메라가 초기화되지 않았습니다.")
        
        properties = {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'hue': self.cap.get(cv2.CAP_PROP_HUE),
            'gain': self.cap.get(cv2.CAP_PROP_GAIN),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        
        return properties
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """카메라 속성 설정
        
        Args:
            property_id (int): 설정할 속성 ID (cv2.CAP_PROP_* 상수)
            value (float): 설정할 값
            
        Returns:
            bool: 설정 성공 여부
            
        Raises:
            RuntimeError: 카메라가 초기화되지 않은 경우 발생
        """
        if not self.is_initialized or self.cap is None:
            raise RuntimeError("카메라가 초기화되지 않았습니다.")
        
        result = self.cap.set(property_id, value)
        
        if result:
            actual_value = self.cap.get(property_id)
            logger.debug(f"카메라 속성 {property_id} 설정: {value} (실제 값: {actual_value})")
        else:
            logger.warning(f"카메라 속성 {property_id} 설정 실패: {value}")
        
        return result
    
    def release(self) -> None:
        """리소스 해제
        
        카메라 리소스를 해제합니다.
        """
        logger.info("웹캠 리소스 해제 중...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.is_initialized = False
        logger.info("웹캠 리소스 해제 완료")
    
    def __del__(self) -> None:
        """소멸자
        
        객체가 소멸될 때 리소스를 자동으로 해제합니다.
        """
        self.release()
