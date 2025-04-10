"""
웹캠 카메라 모듈

OpenCV를 사용하여 웹캠 카메라에 접근하고 제어하는 클래스입니다.
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, Any, Tuple, List

from modules.camera.base_camera import BaseCamera


class WebcamCamera(BaseCamera):
    """웹캠 카메라 클래스
    
    OpenCV를 사용하여 웹캠에 접근하고 프레임을 캡처합니다.
    """
    
    def __init__(self, camera_id: str, name: str = "", index: int = 0, **kwargs):
        """웹캠 카메라 초기화
        
        Args:
            camera_id: 카메라 고유 ID
            name: 카메라 이름 (기본값: "")
            index: OpenCV 카메라 인덱스 (기본값: 0)
            **kwargs: 추가 매개변수
        """
        super().__init__(camera_id=camera_id, name=name or f"Webcam {index}")
        
        self._index = index
        self._capture = None
        self._frame_width = 0
        self._frame_height = 0
        self._fps = 0
        
        # 추가 매개변수 처리
        self._params = {
            "resolution": kwargs.get("resolution", (640, 480)),  # 기본 해상도
            "flip_horizontal": kwargs.get("flip_horizontal", False),
            "flip_vertical": kwargs.get("flip_vertical", False)
        }
    
    @property
    def index(self) -> int:
        """카메라 인덱스
        
        Returns:
            int: OpenCV 카메라 인덱스
        """
        return self._index
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """현재 해상도
        
        Returns:
            Tuple[int, int]: (너비, 높이)
        """
        return (self._frame_width, self._frame_height)
    
    @property
    def fps(self) -> float:
        """현재 FPS
        
        Returns:
            float: 초당 프레임 수
        """
        return self._fps
    
    def open(self) -> bool:
        """카메라 열기
        
        Returns:
            bool: 성공 여부
        """
        if self._is_open:
            return True
            
        try:
            self._capture = cv2.VideoCapture(self._index)
            
            # 해상도 설정
            width, height = self._params["resolution"]
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 상태 확인
            if not self._capture.isOpened():
                print(f"카메라를 열 수 없음: {self.camera_id} (인덱스: {self._index})")
                self.close()
                return False
                
            # 카메라 속성 저장
            self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self._capture.get(cv2.CAP_PROP_FPS)
            
            self._is_open = True
            return True
            
        except Exception as e:
            print(f"카메라 열기 오류: {str(e)}")
            self.close()
            return False
    
    def close(self) -> bool:
        """카메라 닫기
        
        Returns:
            bool: 성공 여부
        """
        if not self._is_open:
            return True
            
        try:
            if self._capture:
                self._capture.release()
                self._capture = None
                
            self._is_open = False
            return True
            
        except Exception as e:
            print(f"카메라 닫기 오류: {str(e)}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """프레임 캡처
        
        Returns:
            Optional[np.ndarray]: 캡처된 프레임 또는 None
        """
        if not self._is_open or not self._capture:
            return None
            
        try:
            success, frame = self._capture.read()
            
            if not success or frame is None:
                return None
                
            # 이미지 플립 처리
            if self._params["flip_horizontal"] or self._params["flip_vertical"]:
                flip_code = 0
                if self._params["flip_horizontal"] and self._params["flip_vertical"]:
                    flip_code = -1
                elif self._params["flip_horizontal"]:
                    flip_code = 1
                elif self._params["flip_vertical"]:
                    flip_code = 0
                    
                frame = cv2.flip(frame, flip_code)
                
            return frame
            
        except Exception as e:
            print(f"프레임 캡처 오류: {str(e)}")
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """카메라 정보 반환
        
        Returns:
            Dict[str, Any]: 카메라 정보
        """
        info = super().get_camera_info()
        info.update({
            "type": "webcam",
            "index": self._index,
            "resolution": self.resolution,
            "fps": self.fps,
            "flip_horizontal": self._params["flip_horizontal"],
            "flip_vertical": self._params["flip_vertical"]
        })
        return info
    
    def set_parameter(self, key: str, value: Any) -> bool:
        """카메라 매개변수 설정
        
        Args:
            key: 매개변수 키
            value: 매개변수 값
            
        Returns:
            bool: 성공 여부
        """
        if key == "resolution" and isinstance(value, tuple) and len(value) == 2:
            self._params["resolution"] = value
            
            # 카메라가 열려있는 경우 해상도 업데이트
            if self._is_open and self._capture:
                width, height = value
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # 실제 적용된 해상도 업데이트
                self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
            return True
            
        elif key == "flip_horizontal" and isinstance(value, bool):
            self._params["flip_horizontal"] = value
            return True
            
        elif key == "flip_vertical" and isinstance(value, bool):
            self._params["flip_vertical"] = value
            return True
            
        return super().set_parameter(key, value)
    
    @staticmethod
    def discover_cameras() -> list:
        """사용 가능한 웹캠 카메라 검색
        
        Returns:
            list: 검색된 카메라 정보 목록
        """
        available_cameras = []
        
        # 몇 개의 카메라를 검색할지 설정 (일반적으로 0~9 범위가 적절)
        max_cameras_to_check = 10
        
        for index in range(max_cameras_to_check):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # 카메라 정보 얻기
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                camera_info = {
                    "type": "webcam",
                    "id": f"webcam_{index}",
                    "name": f"Webcam {index}",
                    "index": index,
                    "resolution": (width, height),
                    "fps": fps
                }
                
                available_cameras.append(camera_info)
                cap.release()
                
        return available_cameras
        
    def _capture_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """프레임 캡처 내부 구현
        
        BaseCamera 추상 메서드 구현
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (프레임, 메타데이터) 또는 None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
            
        # 메타데이터 생성
        metadata = {
            "timestamp": time.time(),
            "camera_id": self.camera_id,
            "resolution": self.resolution,
            "fps": self.fps
        }
        
        return frame, metadata 