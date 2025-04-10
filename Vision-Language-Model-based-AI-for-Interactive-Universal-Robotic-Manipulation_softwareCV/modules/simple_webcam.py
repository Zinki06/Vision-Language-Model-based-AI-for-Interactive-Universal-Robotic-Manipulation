"""
단일 스레드 기반 웹캠 모듈

WebcamCapture를 확장하여 스레드 없이 간단한 프레임 처리 기능을 제공합니다.
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from modules.webcam_capture import WebcamCapture
from utils.profiler import timeit, FPSCounter

# 로거 설정
logger = logging.getLogger(__name__)

class SimpleWebcam(WebcamCapture):
    """단일 스레드 기반 웹캠 클래스
    
    WebcamCapture를 확장하여 스레드 없이 프레임 처리 기능을 추가합니다.
    
    Attributes:
        enable_resize (bool): 프레임 리사이징 활성화 여부
        processing_width (int): 처리용 프레임 너비
        processing_height (int): 처리용 프레임 높이
        frame_callbacks (list): 프레임 처리 콜백 함수 목록
        fps_counter (FPSCounter): FPS 측정용 카운터
        last_frame (dict): 마지막으로 처리된 프레임 정보
        frame_count (int): 처리된 총 프레임 수
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, 
                 fps: int = 15, auto_init: bool = True, 
                 enable_resize: bool = True, processing_width: int = 320, 
                 processing_height: int = 240, skip_frames: int = 1):
        """SimpleWebcam 클래스 초기화
        
        Args:
            camera_id (int): 웹캠 장치 ID (기본값: 0)
            width (int): 캡처 영상 너비 (기본값: 640)
            height (int): 캡처 영상 높이 (기본값: 480)
            fps (int): 초당 프레임 수 (기본값: 15)
            auto_init (bool): 자동 초기화 여부 (기본값: True)
            enable_resize (bool): 프레임 리사이징 활성화 여부 (기본값: True)
            processing_width (int): 처리용 프레임 너비 (기본값: 320)
            processing_height (int): 처리용 프레임 높이 (기본값: 240)
            skip_frames (int): 처리할 때 건너뛸 프레임 수 (기본값: 1)
        """
        # 부모 클래스 초기화
        super().__init__(camera_id, width, height, fps, auto_init)
        
        # 성능 관련 설정
        self.enable_resize = enable_resize
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.skip_frames = skip_frames
        
        # 콜백 및 상태 관리
        self.frame_callbacks = []
        self.fps_counter = FPSCounter(window_size=15)
        self.last_frame = None
        self.frame_count = 0
        self.skip_count = 0
        
        logger.info(f"SimpleWebcam 초기화: camera_id={camera_id}, "
                   f"resolution={width}x{height}, fps={fps}")
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, float, Dict], None]) -> None:
        """프레임 처리 콜백 추가
        
        Args:
            callback: 프레임 처리 콜백 함수
                callback(frame, timestamp, metadata)
        """
        self.frame_callbacks.append(callback)
        logger.debug(f"프레임 처리 콜백 추가됨. 현재 콜백 수: {len(self.frame_callbacks)}")
    
    def remove_frame_callback(self, callback: Callable) -> bool:
        """프레임 처리 콜백 제거
        
        Args:
            callback: 제거할 콜백 함수
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            self.frame_callbacks.remove(callback)
            logger.debug(f"프레임 처리 콜백 제거됨. 현재 콜백 수: {len(self.frame_callbacks)}")
            return True
        except ValueError:
            logger.warning("제거하려는 콜백이 목록에 없습니다.")
            return False
    
    @timeit
    def capture_and_process(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 캡처 및 처리
        
        프레임을 캡처하고 등록된 콜백으로 처리합니다.
        
        Returns:
            tuple: (성공 여부, 처리된 프레임 또는 원본 프레임)
            
        Raises:
            RuntimeError: 카메라가 초기화되지 않은 경우 발생
        """
        # 프레임 캡처
        ret, frame = super().capture_frame()
        
        if not ret or frame is None:
            return False, None
        
        # 프레임 건너뛰기 (skip_frames 적용)
        self.frame_count += 1
        self.skip_count = (self.skip_count + 1) % (self.skip_frames + 1)
        if self.skip_count != 0:
            return ret, frame
        
        # 타임스탬프 기록
        timestamp = time.time()
        
        # 처리용 프레임 준비
        processing_frame = frame
        if self.enable_resize:
            processing_frame = cv2.resize(
                frame, 
                (self.processing_width, self.processing_height),
                interpolation=cv2.INTER_AREA
            )
        
        # 메타데이터 생성
        metadata = {
            'frame_number': self.frame_count,
            'timestamp': timestamp,
            'original_shape': frame.shape,
            'processing_frame': processing_frame if self.enable_resize else None
        }
        
        # 콜백 함수 호출 (직접 처리)
        processed_frame = frame
        for callback in self.frame_callbacks:
            try:
                # 콜백이 프레임을 반환하면 사용, 아니면 원본 유지
                result = callback(processing_frame, timestamp, metadata)
                if result is not None and isinstance(result, np.ndarray):
                    processed_frame = result
            except Exception as e:
                logger.error(f"프레임 처리 콜백 실행 중 오류 발생: {e}")
        
        # 마지막 프레임 정보 저장
        self.last_frame = {
            'frame': frame.copy(),
            'processed_frame': processed_frame if processed_frame is not frame else frame.copy(),
            'timestamp': timestamp,
            'metadata': metadata
        }
        
        # FPS 업데이트
        fps = self.fps_counter.update()
        if fps > 0:
            metadata['fps'] = fps
        
        return ret, processed_frame
    
    def get_latest_frame(self) -> Optional[Dict]:
        """최신 프레임 정보 가져오기
        
        Returns:
            dict: 최신 프레임 정보 또는 None
        """
        return self.last_frame
    
    def save_snapshot(self, filename: str = None, use_processed: bool = True) -> Optional[str]:
        """현재 프레임의 스냅샷을 파일로 저장
        
        Args:
            filename: 저장할 파일명 (기본값: None, 자동 생성)
            use_processed: 처리된 프레임 사용 여부 (기본값: True)
            
        Returns:
            str: 저장된 파일 경로 또는 None
        """
        if self.last_frame is None:
            logger.warning("저장할 프레임이 없습니다.")
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            prefix = "processed" if use_processed else "original"
            filename = f"{prefix}_snapshot_{timestamp}.jpg"
        
        try:
            # 처리된 프레임 또는 원본 프레임 선택
            if use_processed and 'processed_frame' in self.last_frame:
                image = self.last_frame['processed_frame']
            else:
                image = self.last_frame['frame']
                
            cv2.imwrite(filename, image)
            logger.info(f"스냅샷 저장됨: {filename}")
            return filename
        except Exception as e:
            logger.error(f"스냅샷 저장 실패: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """상태 정보 반환
        
        Returns:
            dict: 상태 정보
        """
        stats = {
            'fps': self.fps_counter.get_fps(),
            'frame_count': self.frame_count,
            'callbacks': len(self.frame_callbacks)
        }
        
        # 카메라 속성 추가
        if self.is_initialized and self.cap is not None:
            try:
                props = self.get_camera_properties()
                stats['camera'] = props
            except Exception as e:
                logger.error(f"카메라 속성 가져오기 실패: {e}")
        
        return stats
