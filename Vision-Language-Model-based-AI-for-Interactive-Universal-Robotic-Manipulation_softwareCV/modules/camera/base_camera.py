"""
기본 카메라 클래스 모듈

모든 카메라 구현체의 기본이 되는 클래스를 정의합니다.
"""

import abc
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Set, Callable

import numpy as np

from modules.camera.camera_interface import CameraInterface
from modules.camera.frame_processor_interface import FrameProcessorInterface

class BaseCamera(CameraInterface):
    """기본 카메라 클래스
    
    모든 카메라 구현체의 기본이 되는 추상 클래스입니다.
    공통 기능을 구현하고 카메라별 특화 기능은 하위 클래스에서 구현합니다.
    """
    
    def __init__(self, camera_id: str, name: str = None):
        """초기화
        
        Args:
            camera_id: 카메라 식별자
            name: 카메라 이름 (없으면 camera_id 사용)
        """
        self._camera_id = camera_id
        self._name = name if name else camera_id
        self._is_open = False
        self._is_capturing = False
        self._capture_thread = None
        self._frame_queue = queue.Queue(maxsize=10)  # 최대 10개 프레임 버퍼
        self._frame_processors: Set[FrameProcessorInterface] = set()
        self._lock = threading.RLock()
        self._properties = {}
        self._stop_event = threading.Event()
        
        # 기본 속성 설정
        self._properties = {
            "width": 640,
            "height": 480,
            "fps": 30,
            "format": "RGB",
        }
    
    @property
    def camera_id(self) -> str:
        """카메라 ID 반환"""
        return self._camera_id
    
    @property
    def name(self) -> str:
        """카메라 이름 반환"""
        return self._name
    
    def is_open(self) -> bool:
        """카메라 열림 상태 확인
        
        Returns:
            bool: 카메라가 열려있으면 True, 아니면 False
        """
        return self._is_open
    
    def is_capturing(self) -> bool:
        """캡처 상태 확인
        
        Returns:
            bool: 캡처 중이면 True, 아니면 False
        """
        return self._is_capturing
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """카메라 속성 정보 가져오기
        
        Returns:
            Dict[str, Any]: 카메라 속성 정보 (해상도, FPS 등)
        """
        with self._lock:
            return self._properties.copy()
    
    def set_camera_property(self, property_name: str, value: Any) -> bool:
        """카메라 속성 설정
        
        Args:
            property_name: 속성 이름
            value: 설정할 값
            
        Returns:
            bool: 설정 성공 여부
        """
        with self._lock:
            self._properties[property_name] = value
            return True
    
    def get_camera_info(self) -> Dict[str, Any]:
        """카메라 정보 가져오기
        
        Returns:
            Dict[str, Any]: 카메라 정보 (ID, 이름, 상태 등)
        """
        info = {
            "id": self._camera_id,
            "name": self._name,
            "is_open": self._is_open,
            "is_capturing": self._is_capturing,
            "properties": self.get_camera_properties()
        }
        return info
    
    def add_frame_processor(self, processor: FrameProcessorInterface) -> None:
        """프레임 프로세서 추가
        
        Args:
            processor: 추가할 프레임 프로세서
        """
        with self._lock:
            self._frame_processors.add(processor)
    
    def remove_frame_processor(self, processor: FrameProcessorInterface) -> bool:
        """프레임 프로세서 제거
        
        Args:
            processor: 제거할 프레임 프로세서
            
        Returns:
            bool: 제거 성공 여부
        """
        with self._lock:
            if processor in self._frame_processors:
                self._frame_processors.remove(processor)
                return True
            return False
    
    def _process_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> None:
        """프레임 처리
        
        모든 등록된 프레임 프로세서에 프레임을 전달합니다.
        
        Args:
            frame: 처리할 프레임
            metadata: 프레임 메타데이터
        """
        with self._lock:
            processors = list(self._frame_processors)
        
        for processor in processors:
            try:
                processor.add_frame(frame, metadata)
            except Exception as e:
                print(f"프레임 처리 오류: {e}")
    
    def _capture_loop(self) -> None:
        """캡처 루프
        
        카메라에서 프레임을 지속적으로 캡처하는 스레드 함수
        """
        self._stop_event.clear()
        
        while not self._stop_event.is_set() and self._is_open:
            try:
                frame_data = self._capture_frame()
                if frame_data is not None:
                    frame, metadata = frame_data
                    
                    # 프레임 큐에 추가
                    try:
                        self._frame_queue.put((frame, metadata), block=False)
                    except queue.Full:
                        # 큐가 가득 차면 가장 오래된 프레임 제거하고 추가
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put((frame, metadata), block=False)
                        except:
                            pass
                    
                    # 프레임 처리
                    self._process_frame(frame, metadata)
            except Exception as e:
                print(f"캡처 오류: {e}")
                time.sleep(0.1)  # 오류 발생 시 잠시 대기
    
    def start_capture(self) -> bool:
        """캡처 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        with self._lock:
            if self._is_capturing or not self._is_open:
                return False
            
            self._is_capturing = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            return True
    
    def stop_capture(self) -> bool:
        """캡처 정지
        
        Returns:
            bool: 정지 성공 여부
        """
        with self._lock:
            if not self._is_capturing:
                return False
            
            self._stop_event.set()
            
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            
            self._is_capturing = False
            return True
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """프레임 가져오기
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (프레임, 메타데이터) 또는 None
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def clear_frames(self) -> None:
        """프레임 큐 비우기"""
        with self._lock:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
    
    @abc.abstractmethod
    def open(self) -> bool:
        """카메라 열기
        
        Returns:
            bool: 열기 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def close(self) -> bool:
        """카메라 닫기
        
        Returns:
            bool: 닫기 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def _capture_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """프레임 캡처
        
        구현 클래스에서 실제 프레임 캡처 구현
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (프레임, 메타데이터) 또는 None
        """
        pass
    
    @staticmethod
    def get_available_cameras() -> List[Dict[str, Any]]:
        """사용 가능한 카메라 목록 반환
        
        이 메서드는 각 카메라 구현 클래스에서 구현해야 합니다.
        
        Returns:
            List[Dict[str, Any]]: 사용 가능한 카메라 정보 목록
        """
        return [] 