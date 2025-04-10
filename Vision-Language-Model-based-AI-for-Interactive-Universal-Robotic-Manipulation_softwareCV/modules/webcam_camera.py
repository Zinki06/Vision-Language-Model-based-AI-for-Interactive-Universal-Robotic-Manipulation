"""
웹캠 카메라 모듈

OpenCV를 사용하여 웹캠 카메라에 접근하고 제어하는 클래스입니다.
"""

import cv2
import numpy as np
import platform
import logging
import time
from typing import Dict, Optional, Any, Tuple, List

from modules.camera.base_camera import BaseCamera

logger = logging.getLogger(__name__)

class WebcamCamera(BaseCamera):
    """웹캠 카메라 클래스
    
    OpenCV를 사용하여 웹캠에 접근하고 프레임을 캡처합니다.
    """
    
    # OpenCV 백엔드 매핑
    _BACKEND_MAP = {
        'default': None,
        'avfoundation': cv2.CAP_AVFOUNDATION,
        'v4l': cv2.CAP_V4L2,
        'dshow': cv2.CAP_DSHOW,  # Windows DirectShow
        'msmf': cv2.CAP_MSMF      # Windows Media Foundation
    }
    
    def __init__(self, camera_id: str, name: str = "", index: int = 0, **kwargs):
        """웹캠 카메라 초기화
        
        Args:
            camera_id: 카메라 고유 ID
            name: 카메라 이름 (기본값: "")
            index: OpenCV 카메라 인덱스 (기본값: 0)
            **kwargs: 추가 매개변수
                - resolution (Tuple[int, int]): 해상도 (너비, 높이)
                - fps (int): 초당 프레임 수
                - flip_horizontal (bool): 수평 대칭 여부
                - flip_vertical (bool): 수직 대칭 여부
                - backend (str): 사용할 백엔드 (default, avfoundation, v4l, dshow, msmf)
        """
        super().__init__(camera_id=camera_id, name=name or f"Webcam {index}")
        
        self._index = index
        self._capture = None
        self._frame_width = 0
        self._frame_height = 0
        self._fps = 0
        
        # 백엔드 설정
        self._backend = kwargs.get('backend', 'default')
        
        # 추가 매개변수 처리
        self._params = {
            "resolution": kwargs.get("resolution", (640, 480)),  # 기본 해상도
            "fps": kwargs.get("fps", 30),  # 기본 FPS
            "flip_horizontal": kwargs.get("flip_horizontal", False),
            "flip_vertical": kwargs.get("flip_vertical", False),
            "retry_count": kwargs.get("retry_count", 3)  # 연결 재시도 횟수
        }
        
        # 백엔드 호환성 검사
        self._validate_backend()
    
    def _validate_backend(self):
        """현재 플랫폼에서 지정된 백엔드의 호환성 검사"""
        system = platform.system()
        
        # AVFoundation은 macOS에서만 사용 가능
        if self._backend == 'avfoundation' and system != 'Darwin':
            logger.warning(f"AVFoundation 백엔드는 macOS에서만 지원됩니다. 현재 시스템: {system}. 기본 백엔드로 전환합니다.")
            self._backend = 'default'
            
        # V4L은 리눅스에서만 사용 가능
        elif self._backend == 'v4l' and system != 'Linux':
            logger.warning(f"V4L 백엔드는 Linux에서만 지원됩니다. 현재 시스템: {system}. 기본 백엔드로 전환합니다.")
            self._backend = 'default'
            
        # DirectShow 및 Media Foundation은 Windows에서만 사용 가능
        elif self._backend in ['dshow', 'msmf'] and system != 'Windows':
            logger.warning(f"{self._backend} 백엔드는 Windows에서만 지원됩니다. 현재 시스템: {system}. 기본 백엔드로 전환합니다.")
            self._backend = 'default'
    
    @property
    def index(self) -> int:
        """카메라 인덱스
        
        Returns:
            int: OpenCV 카메라 인덱스
        """
        return self._index
    
    @property
    def backend(self) -> str:
        """사용 중인 백엔드
        
        Returns:
            str: 백엔드 이름
        """
        return self._backend
    
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
            
        retry_count = self._params["retry_count"]
        success = False
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"카메라 열기 시도 ({attempt+1}/{retry_count}): 인덱스 {self._index}, 백엔드 {self._backend}")
                
                # 백엔드에 따른 VideoCapture 생성
                backend_value = self._BACKEND_MAP.get(self._backend)
                if backend_value is not None:
                    self._capture = cv2.VideoCapture(self._index, backend_value)
                    logger.debug(f"{self._backend} 백엔드로 카메라 열기 시도")
                else:
                    self._capture = cv2.VideoCapture(self._index)
                    logger.debug(f"기본 백엔드로 카메라 열기 시도")
                
                # 해상도 설정
                width, height = self._params["resolution"]
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # FPS 설정
                if "fps" in self._params:
                    self._capture.set(cv2.CAP_PROP_FPS, self._params["fps"])
                
                # 상태 확인
                if not self._capture.isOpened():
                    logger.warning(f"카메라를 열 수 없음: {self.camera_id} (인덱스: {self._index}, 백엔드: {self._backend})")
                    if attempt < retry_count - 1:
                        continue  # 재시도
                    else:
                        self.close()
                        return False
                
                # 카메라 속성 저장
                self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._fps = self._capture.get(cv2.CAP_PROP_FPS)
                
                # 성공 로그
                logger.info(f"카메라 열기 성공: {self.camera_id} (인덱스: {self._index}, 백엔드: {self._backend})")
                logger.info(f"해상도: {self._frame_width}x{self._frame_height}, FPS: {self._fps}")
                
                self._is_open = True
                success = True
                break  # 성공 시 루프 종료
                
            except Exception as e:
                logger.error(f"카메라 열기 시도 중 오류 ({attempt+1}/{retry_count}): {str(e)}")
                if self._capture:
                    self._capture.release()
                    self._capture = None
                
                if attempt < retry_count - 1:
                    logger.debug(f"재시도 중...")
                else:
                    logger.error(f"최대 재시도 횟수에 도달했습니다. 카메라 열기 실패: {self.camera_id}")
        
        if not success:
            self.close()
        
        return success
    
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
            logger.debug(f"카메라 닫기 성공: {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"카메라 닫기 오류: {str(e)}")
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
            logger.error(f"프레임 캡처 오류: {str(e)}")
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
            "backend": self._backend,
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
            
        elif key == "fps" and isinstance(value, (int, float)):
            self._params["fps"] = value
            
            # 카메라가 열려있는 경우 FPS 업데이트
            if self._is_open and self._capture:
                self._capture.set(cv2.CAP_PROP_FPS, value)
                self._fps = self._capture.get(cv2.CAP_PROP_FPS)
                
            return True
            
        elif key == "flip_horizontal" and isinstance(value, bool):
            self._params["flip_horizontal"] = value
            return True
            
        elif key == "flip_vertical" and isinstance(value, bool):
            self._params["flip_vertical"] = value
            return True
            
        elif key == "backend" and isinstance(value, str):
            if value in self._BACKEND_MAP:
                # 백엔드 변경 시 카메라 재시작이 필요
                was_open = self._is_open
                if was_open:
                    self.close()
                
                self._backend = value
                self._validate_backend()  # 백엔드 호환성 검사
                
                # 카메라가 열려 있었으면 다시 열기
                if was_open:
                    return self.open()
                return True
            else:
                logger.warning(f"지원되지 않는 백엔드: {value}")
                return False
            
        return super().set_parameter(key, value)
    
    @classmethod
    def get_available_cameras(cls) -> List[Dict[str, Any]]:
        """사용 가능한 웹캠 카메라 검색
        
        Returns:
            List[Dict[str, Any]]: 검색된 카메라 정보 목록
        """
        available_cameras = []
        system = platform.system()
        
        # 백엔드 목록 결정
        backends = ['default']
        if system == 'Darwin':
            backends.append('avfoundation')
        elif system == 'Linux':
            backends.append('v4l')
        elif system == 'Windows':
            backends.extend(['dshow', 'msmf'])
        
        # 몇 개의 카메라를 검색할지 설정 (일반적으로 0~9 범위가 적절)
        max_cameras_to_check = 10
        
        for backend in backends:
            backend_value = cls._BACKEND_MAP.get(backend)
            
            for index in range(max_cameras_to_check):
                cap = None
                try:
                    # 백엔드에 따른 VideoCapture 생성
                    if backend_value is not None:
                        cap = cv2.VideoCapture(index, backend_value)
                    else:
                        cap = cv2.VideoCapture(index)
                    
                    if cap.isOpened():
                        # 카메라 정보 얻기
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_id = f"{backend}_{index}" if backend != 'default' else f"webcam_{index}"
                        camera_info = {
                            "id": camera_id,
                            "name": f"카메라 {index}" + (f" ({backend})" if backend != 'default' else ""),
                            "type": "webcam",
                            "index": index,
                            "backend": backend,
                            "resolution": (width, height),
                            "fps": fps
                        }
                        
                        available_cameras.append(camera_info)
                        logger.debug(f"카메라 발견: 인덱스 {index}, 백엔드 {backend}, 해상도 {width}x{height}")
                except Exception as e:
                    logger.debug(f"카메라 검색 중 오류 (인덱스 {index}, 백엔드 {backend}): {e}")
                finally:
                    if cap is not None:
                        cap.release()
        
        logger.info(f"카메라 검색 완료: {len(available_cameras)}개 발견됨")
        return available_cameras
    
    def _capture_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """프레임 캡처 내부 구현
        
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