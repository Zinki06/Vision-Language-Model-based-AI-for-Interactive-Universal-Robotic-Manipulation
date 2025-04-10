"""
카메라 모듈

다양한 카메라 장치에 대한 추상화 인터페이스와 기본 구현을 제공합니다.
카메라 프레임을 캡처하고 처리하기 위한 도구를 포함합니다.
"""

import abc
import time
import threading
import logging
import queue
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from modules.camera.frame_processor import FrameProcessor, FrameProcessorChain

# 로깅 설정
logger = logging.getLogger(__name__)


class Camera(abc.ABC):
    """카메라 추상 클래스
    
    모든 카메라 구현체가 따라야 하는 인터페이스를 정의합니다.
    """
    
    def __init__(self, camera_id: str, width: int = 640, height: int = 480, fps: int = 30):
        """카메라 초기화
        
        Args:
            camera_id: 카메라 식별자
            width: 프레임 너비 (기본값: 640)
            height: 프레임 높이 (기본값: 480)
            fps: 초당 프레임 수 (기본값: 30)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_open = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.frame_processors = FrameProcessorChain()
        self.frame_handlers: List[Callable[[np.ndarray, Dict[str, Any]], None]] = []
    
    @abc.abstractmethod
    def open(self) -> bool:
        """카메라 열기
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abc.abstractmethod
    def close(self) -> None:
        """카메라 닫기"""
        pass
    
    @abc.abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        """프레임 읽기
        
        Returns:
            Tuple[bool, np.ndarray]: 성공 여부와 프레임
        """
        pass
    
    def read_processed(self) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """처리된 프레임 읽기
        
        프레임을 읽고 모든 프레임 프로세서를 적용합니다.
        
        Returns:
            Tuple[bool, np.ndarray, Dict[str, Any]]: 성공 여부, 처리된 프레임, 메타데이터
        """
        success, frame = self.read()
        if not success:
            return False, np.array([]), {"error": "프레임 읽기 실패"}
        
        processed_frame, metadata = self.frame_processors.process(frame)
        
        # 프레임 핸들러 호출
        for handler in self.frame_handlers:
            try:
                handler(processed_frame, metadata)
            except Exception as e:
                logger.error(f"프레임 핸들러 오류: {e}")
        
        # 메타데이터 업데이트
        self.frame_count += 1
        current_time = time.time()
        metadata["camera_id"] = self.camera_id
        metadata["frame_count"] = self.frame_count
        metadata["timestamp"] = current_time
        
        if self.last_frame_time > 0:
            metadata["frame_interval"] = current_time - self.last_frame_time
            metadata["fps"] = 1.0 / (current_time - self.last_frame_time)
        else:
            metadata["frame_interval"] = 0
            metadata["fps"] = 0
        
        self.last_frame_time = current_time
        
        return True, processed_frame, metadata
    
    def add_processor(self, processor: FrameProcessor) -> bool:
        """프레임 프로세서 추가
        
        Args:
            processor: 추가할 프레임 프로세서
            
        Returns:
            bool: 성공 여부
        """
        return self.frame_processors.add_processor(processor)
    
    def remove_processor(self, processor_id: str) -> Optional[FrameProcessor]:
        """프레임 프로세서 제거
        
        Args:
            processor_id: 제거할 프로세서 ID
            
        Returns:
            Optional[FrameProcessor]: 제거된 프로세서 (없으면 None)
        """
        return self.frame_processors.remove_processor(processor_id)
    
    def get_processor(self, processor_id: str) -> Optional[FrameProcessor]:
        """프레임 프로세서 가져오기
        
        Args:
            processor_id: 가져올 프로세서 ID
            
        Returns:
            Optional[FrameProcessor]: 프로세서 (없으면 None)
        """
        return self.frame_processors.get_processor(processor_id)
    
    def get_processors(self) -> List[FrameProcessor]:
        """모든 프레임 프로세서 가져오기
        
        Returns:
            List[FrameProcessor]: 프로세서 목록
        """
        return self.frame_processors.get_processors()
    
    def clear_processors(self) -> None:
        """모든 프레임 프로세서 제거"""
        self.frame_processors.clear()
    
    def add_frame_handler(self, handler: Callable[[np.ndarray, Dict[str, Any]], None]) -> None:
        """프레임 핸들러 추가
        
        Args:
            handler: 프레임 핸들러 함수 (processed_frame, metadata) -> None
        """
        if handler not in self.frame_handlers:
            self.frame_handlers.append(handler)
    
    def remove_frame_handler(self, handler: Callable[[np.ndarray, Dict[str, Any]], None]) -> bool:
        """프레임 핸들러 제거
        
        Args:
            handler: 제거할 프레임 핸들러
            
        Returns:
            bool: 성공 여부
        """
        if handler in self.frame_handlers:
            self.frame_handlers.remove(handler)
            return True
        return False
    
    def clear_frame_handlers(self) -> None:
        """모든 프레임 핸들러 제거"""
        self.frame_handlers.clear()
    
    def get_metadata(self) -> Dict[str, Any]:
        """카메라 메타데이터 가져오기
        
        Returns:
            Dict[str, Any]: 카메라 메타데이터
        """
        return {
            "camera_id": self.camera_id,
            "type": self.__class__.__name__,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_open": self.is_open,
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time,
            "processors": self.frame_processors.get_metadata(),
            "frame_handler_count": len(self.frame_handlers)
        }


class OpenCVCamera(Camera):
    """OpenCV 카메라 클래스
    
    OpenCV를 사용하여 카메라 장치에 접근합니다.
    """
    
    def __init__(self, camera_id: Union[str, int] = 0, width: int = 640, height: int = 480, fps: int = 30):
        """OpenCV 카메라 초기화
        
        Args:
            camera_id: 카메라 ID 또는 경로 (기본값: 0, 첫 번째 카메라)
            width: 프레임 너비 (기본값: 640)
            height: 프레임 높이 (기본값: 480)
            fps: 초당 프레임 수 (기본값: 30)
        """
        # 정수로 변환 시도 (카메라 인덱스인 경우)
        if isinstance(camera_id, str) and camera_id.isdigit():
            camera_id = int(camera_id)
        
        self.cap = None
        self.device_id = camera_id
        super().__init__(str(camera_id), width, height, fps)
    
    def open(self) -> bool:
        """카메라 열기
        
        Returns:
            bool: 성공 여부
        """
        if self.is_open:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                logger.error(f"카메라를 열 수 없습니다: {self.device_id}")
                return False
            
            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 실제 설정된 값으로 업데이트
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.is_open = True
            logger.info(f"카메라 열기 성공: {self.device_id} ({self.width}x{self.height}@{self.fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"카메라 열기 실패: {self.device_id}, 오류: {e}")
            return False
    
    def close(self) -> None:
        """카메라 닫기"""
        if self.is_open and self.cap is not None:
            self.cap.release()
            self.is_open = False
            logger.info(f"카메라 닫기: {self.device_id}")
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """프레임 읽기
        
        Returns:
            Tuple[bool, np.ndarray]: 성공 여부와 프레임
        """
        if not self.is_open or self.cap is None:
            if not self.open():
                return False, np.array([])
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"프레임 읽기 실패: {self.device_id}")
                return False, np.array([])
            return True, frame
        except Exception as e:
            logger.error(f"프레임 읽기 오류: {self.device_id}, 오류: {e}")
            return False, np.array([])
    
    def __del__(self):
        """소멸자"""
        self.close()


class CameraStream:
    """카메라 스트림 클래스
    
    카메라에서 지속적으로 프레임을 읽고 처리하는 스트림을 관리합니다.
    """
    
    def __init__(self, camera: Camera, buffer_size: int = 1, auto_start: bool = False):
        """카메라 스트림 초기화
        
        Args:
            camera: 카메라 인스턴스
            buffer_size: 버퍼 크기 (기본값: 1)
            auto_start: 자동 시작 여부 (기본값: False)
        """
        self.camera = camera
        self.buffer_size = max(1, buffer_size)
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.is_running = False
        self.thread = None
        self.latest_frame = None
        self.latest_metadata = None
        
        if auto_start:
            self.start()
    
    def _stream_worker(self) -> None:
        """스트림 작업자 스레드"""
        while self.is_running:
            success, frame, metadata = self.camera.read_processed()
            if not success:
                time.sleep(0.1)  # 오류 시 잠시 대기
                continue
            
            # 버퍼가 가득 차면 가장 오래된 프레임 제거
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_buffer.put_nowait((frame, metadata))
                self.latest_frame = frame
                self.latest_metadata = metadata
            except queue.Full:
                pass
    
    def start(self) -> bool:
        """스트림 시작
        
        Returns:
            bool: 성공 여부
        """
        if self.is_running:
            return True
        
        if not self.camera.open():
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()
        logger.info(f"카메라 스트림 시작: {self.camera.camera_id}")
        return True
    
    def stop(self) -> None:
        """스트림 중지"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.camera.close()
        
        # 버퍼 비우기
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"카메라 스트림 중지: {self.camera.camera_id}")
    
    def read(self) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """프레임 읽기
        
        Returns:
            Tuple[bool, np.ndarray, Dict[str, Any]]: 성공 여부, 프레임, 메타데이터
        """
        if not self.is_running:
            return False, np.array([]), {"error": "스트림이 실행 중이 아님"}
        
        try:
            frame, metadata = self.frame_buffer.get(timeout=1.0)
            return True, frame, metadata
        except queue.Empty:
            if self.latest_frame is not None:
                return True, self.latest_frame, self.latest_metadata or {}
            return False, np.array([]), {"error": "사용 가능한 프레임 없음"}
    
    def get_latest(self) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """최신 프레임 가져오기
        
        Returns:
            Tuple[bool, np.ndarray, Dict[str, Any]]: 성공 여부, 프레임, 메타데이터
        """
        if not self.is_running or self.latest_frame is None:
            return False, np.array([]), {"error": "사용 가능한 프레임 없음"}
        
        return True, self.latest_frame, self.latest_metadata or {}
    
    def __del__(self):
        """소멸자"""
        self.stop() 