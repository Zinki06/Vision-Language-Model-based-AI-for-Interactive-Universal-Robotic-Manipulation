"""
스레드 기반 웹캠 모듈

WebcamCapture와 FrameProcessor를 통합하여 스레드 기반으로 웹캠 영상을 캡처하고
처리하는 기능을 제공합니다.
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from modules.webcam_capture import WebcamCapture
from modules.frame_processor import FrameProcessor

# 로거 설정
logger = logging.getLogger(__name__)

class ThreadedWebcam:
    """스레드 기반 웹캠 클래스
    
    WebcamCapture와 FrameProcessor를 통합하여 별도의 스레드에서
    웹캠 영상을 캡처하고 처리하는 기능을 제공합니다.
    
    Attributes:
        webcam (WebcamCapture): 웹캠 캡처 객체
        processor (FrameProcessor): 프레임 처리 객체
        is_running (bool): 실행 중 여부
    """
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, 
                 fps: int = 30, queue_size: int = 5, queue_policy: str = 'discard_oldest',
                 skip_frames: int = 1, enable_resize: bool = True, 
                 processing_width: int = 640, processing_height: int = 360):
        """ThreadedWebcam 클래스 초기화
        
        Args:
            camera_id (int): 웹캠 장치 ID (기본값: 0)
            width (int): 캡처 영상 너비 (기본값: 1280)
            height (int): 캡처 영상 높이 (기본값: 720)
            fps (int): 초당 프레임 수 (기본값: 30)
            queue_size (int): 프레임 큐 크기 (기본값: 5)
            queue_policy (str): 큐 정책 (기본값: 'discard_oldest')
            skip_frames (int): 처리할 때 건너뛸 프레임 수 (기본값: 1)
            enable_resize (bool): 프레임 리사이즈 활성화 여부 (기본값: True)
            processing_width (int): 처리용 프레임 너비 (기본값: 640)
            processing_height (int): 처리용 프레임 높이 (기본값: 360)
            
        Raises:
            RuntimeError: 초기화 실패 시 발생
        """
        # 웹캠 초기화
        try:
            self.webcam = WebcamCapture(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps
            )
        except Exception as e:
            raise RuntimeError(f"웹캠 초기화 실패: {e}")
        
        # 프레임 프로세서 초기화
        self.processor = FrameProcessor(
            frame_source=self.webcam.capture_frame,
            queue_size=queue_size,
            queue_policy=queue_policy,
            skip_frames=skip_frames
        )
        
        self.is_running = False
        
        # 콘슈머 스레드
        self._consumer_thread = None
        self._shutdown_event = threading.Event()
        
        # 작업 동기화 객체
        self._condition = threading.Condition()
        
        # 최신 프레임 캐시
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        
        # 프레임 처리 콜백 목록
        self._frame_callbacks = []
        
        # 성능 관련 설정 추가
        self.enable_resize = enable_resize
        self.processing_width = processing_width
        self.processing_height = processing_height
        
        logger.debug(f"ThreadedWebcam 초기화: camera_id={camera_id}, "
                   f"resolution={width}x{height}, fps={fps}, queue_size={queue_size}")
    
    def start(self) -> bool:
        """웹캠 캡처 및 프레임 처리 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        with self._condition:
            if self.is_running:
                logger.warning("이미 실행 중입니다.")
                return False
            
            # 프레임 프로세서 시작
            if not self.processor.start():
                logger.error("프레임 프로세서 시작 실패")
                return False
            
            self._shutdown_event.clear()
            self.is_running = True
            
            # 콘슈머 스레드 시작
            self._consumer_thread = threading.Thread(
                target=self._consumer_task,
                daemon=True,
                name="FrameConsumerThread"
            )
            self._consumer_thread.start()
            
            logger.info("스레드 기반 웹캠 처리 시작됨")
            return True
    
    def stop(self, timeout: float = 2.0) -> bool:
        """웹캠 캡처 및 프레임 처리 중지
        
        Args:
            timeout (float): 스레드 종료 대기 시간(초) (기본값: 2.0)
            
        Returns:
            bool: 중지 성공 여부
        """
        with self._condition:
            if not self.is_running:
                logger.warning("실행 중이 아닙니다.")
                return False
            
            self._shutdown_event.set()
            self.is_running = False
            
            # 컨디션 변수 통지
            self._condition.notify_all()
        
        # 프레임 프로세서 중지
        if not self.processor.stop(timeout=timeout):
            logger.warning("프레임 프로세서 중지 실패")
        
        # 콘슈머 스레드 종료 대기
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)
            if self._consumer_thread.is_alive():
                logger.warning(f"콘슈머 스레드가 {timeout}초 내에 종료되지 않았습니다.")
                return False
        
        logger.info("스레드 기반 웹캠 처리 중지됨")
        return True
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, float, Dict], None]) -> None:
        """프레임 처리 콜백 추가
        
        Args:
            callback (Callable): 프레임 처리 콜백 함수
                callback(frame, timestamp, metadata)
        """
        # 프로세서에 콜백 추가
        self.processor.add_frame_callback(callback)
        
        # 내부 콜백 목록에도 추가
        self._frame_callbacks.append(callback)
        
        logger.debug(f"프레임 처리 콜백 추가됨. 현재 콜백 수: {len(self._frame_callbacks)}")
    
    def remove_frame_callback(self, callback: Callable) -> bool:
        """프레임 처리 콜백 제거
        
        Args:
            callback (Callable): 제거할 콜백 함수
            
        Returns:
            bool: 제거 성공 여부
        """
        # 프로세서에서 콜백 제거
        result = self.processor.remove_frame_callback(callback)
        
        # 내부 콜백 목록에서도 제거
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
            logger.debug(f"프레임 처리 콜백 제거됨. 현재 콜백 수: {len(self._frame_callbacks)}")
        
        return result
    
    def _update_latest_frame(self) -> bool:
        """최신 프레임 업데이트
        
        프레임 프로세서에서 최신 프레임을 가져와 캐시에 저장합니다.
        
        Returns:
            bool: 업데이트 성공 여부
        """
        frame_data = self.processor.get_latest_frame(timeout=0.01)
        if frame_data is None:
            return False
        
        # 프레임 리사이징 추가
        if frame_data is not None and self.enable_resize:
            original_frame = frame_data['frame']
            # 원본 프레임은 보존하고 처리용 리사이즈 프레임 추가
            resized_frame = cv2.resize(original_frame, 
                                       (self.processing_width, self.processing_height),
                                       interpolation=cv2.INTER_AREA)
            frame_data['processing_frame'] = resized_frame
        
        with self._latest_frame_lock:
            self._latest_frame = frame_data
        
        return True
    
    def _consumer_task(self) -> None:
        """콘슈머 스레드 작업
        
        주기적으로 프레임을 처리하고 최신 프레임을 업데이트합니다.
        """
        logger.debug("콘슈머 스레드 시작됨")
        
        try:
            while not self._shutdown_event.is_set():
                # 최신 프레임 업데이트
                self._update_latest_frame()
                
                # 프레임 처리
                self.processor.process_frames(timeout=0.01, max_frames=1)
                
                # 짧은 대기 시간 (CPU 점유율 감소)
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"콘슈머 스레드에서 예기치 않은 오류 발생: {e}")
        
        finally:
            logger.debug("콘슈머 스레드 종료됨")
    
    def get_latest_frame(self) -> Optional[Dict]:
        """최신 프레임 가져오기
        
        캐시된 최신 프레임을 반환합니다.
        
        Returns:
            Optional[Dict]: 최신 프레임 데이터 또는 None
        """
        with self._latest_frame_lock:
            if self._latest_frame is None:
                return None
            return {
                'frame': self._latest_frame['frame'].copy(),
                'timestamp': self._latest_frame['timestamp'],
                'metadata': self._latest_frame['metadata'].copy()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보 딕셔너리
        """
        # 프로세서의 통계 정보 가져오기
        stats = self.processor.get_stats()
        
        # 웹캠 정보 추가
        if self.webcam and self.webcam.is_initialized:
            try:
                camera_props = self.webcam.get_camera_properties()
                stats['camera'] = {
                    'width': camera_props.get('width', 0),
                    'height': camera_props.get('height', 0),
                    'fps': camera_props.get('fps', 0)
                }
            except Exception as e:
                logger.error(f"카메라 속성 가져오기 실패: {e}")
                stats['camera'] = {'error': str(e)}
        else:
            stats['camera'] = {'initialized': False}
        
        return stats
    
    def capture_snapshot(self) -> Optional[Dict]:
        """현재 프레임의 스냅샷 캡처
        
        현재 캐시된 최신 프레임의 스냅샷을 반환합니다.
        
        Returns:
            Optional[Dict]: 스냅샷 데이터 또는 None
        """
        if not self.is_running:
            logger.warning("실행 중이 아닙니다.")
            return None
        
        # 최신 프레임 가져오기
        latest_frame = self.get_latest_frame()
        if latest_frame is None:
            logger.warning("캡처할 프레임이 없습니다.")
            return None
        
        # 타임스탬프 업데이트
        latest_frame['snapshot_time'] = time.time()
        
        return latest_frame
    
    def save_snapshot(self, filename: str = None) -> Optional[str]:
        """현재 프레임의 스냅샷을 파일로 저장
        
        Args:
            filename (str, optional): 저장할 파일명. 지정하지 않으면 자동 생성됨.
            
        Returns:
            Optional[str]: 저장된 파일 경로 또는 None
        """
        snapshot = self.capture_snapshot()
        if snapshot is None:
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, snapshot['frame'])
            logger.info(f"스냅샷 저장됨: {filename}")
            return filename
        except Exception as e:
            logger.error(f"스냅샷 저장 실패: {e}")
            return None
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """카메라 속성 정보 반환
        
        Returns:
            Dict[str, Any]: 카메라 속성 정보 딕셔너리
            
        Raises:
            RuntimeError: 카메라가 초기화되지 않은 경우 발생
        """
        return self.webcam.get_camera_properties()
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """카메라 속성 설정
        
        Args:
            property_id (int): 설정할 속성 ID (cv2.CAP_PROP_* 상수)
            value (float): 설정할 값
            
        Returns:
            bool: 설정 성공 여부
        """
        return self.webcam.set_camera_property(property_id, value)
    
    def release(self) -> None:
        """리소스 해제
        
        모든 리소스를 해제합니다.
        """
        logger.info("스레드 기반 웹캠 리소스 해제 중...")
        
        # 실행 중이라면 중지
        if self.is_running:
            self.stop()
        
        # 웹캠 리소스 해제
        self.webcam.release()
        
        logger.info("스레드 기반 웹캠 리소스 해제 완료")
    
    def __del__(self) -> None:
        """소멸자
        
        객체가 소멸될 때 리소스를 자동으로 해제합니다.
        """
        self.release()
