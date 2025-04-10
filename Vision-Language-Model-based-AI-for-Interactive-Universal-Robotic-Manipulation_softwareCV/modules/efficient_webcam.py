"""
효율적인 단일 스레드 기반 웹캠 모듈

타이밍 관리와 메모리 최적화를 통해 효율적인 웹캠 프레임 처리를 제공합니다.
"""

import logging
import time
import os
from typing import Callable, Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np

from modules.webcam_capture import WebcamCapture
from utils.frame_controller import FrameController
from utils.profiler import timeit, FPSCounter

# 로거 설정
logger = logging.getLogger(__name__)

class EfficientWebcam:
    """효율적인 단일 스레드 기반 웹캠 클래스
    
    타이밍 관리, 메모리 최적화, 적응형 프레임 스킵을 통해
    단일 스레드에서 효율적인 웹캠 프레임 처리를 제공합니다.
    
    Attributes:
        webcam (WebcamCapture): 웹캠 캡처 객체
        frame_controller (FrameController): 프레임 레이트 제어 객체
        enable_resize (bool): 프레임 리사이징 활성화 여부
        processing_width (int): 처리용 프레임 너비
        processing_height (int): 처리용 프레임 높이
        frame_callbacks (List): 프레임 처리 콜백 함수 목록
        last_frame (Dict): 마지막 처리된 프레임 정보
        frame_count (int): 프레임 카운터
        is_initialized (bool): 초기화 여부
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, 
                 fps: int = 15, enable_resize: bool = True, processing_width: int = 320, 
                 processing_height: int = 240, render_wait_time: int = 30,
                 adaptive_skip: bool = True):
        """EfficientWebcam 초기화
        
        Args:
            camera_id (int): 웹캠 장치 ID (기본값: 0)
            width (int): 캡처 영상 너비 (기본값: 640)
            height (int): 캡처 영상 높이 (기본값: 480)
            fps (int): 초당 프레임 수 (기본값: 15)
            enable_resize (bool): 프레임 리사이징 활성화 여부 (기본값: True)
            processing_width (int): 처리용 프레임 너비 (기본값: 320)
            processing_height (int): 처리용 프레임 높이 (기본값: 240)
            render_wait_time (int): 렌더링 대기 시간(ms) (기본값: 30)
            adaptive_skip (bool): 적응형 프레임 스킵 활성화 여부 (기본값: True)
        """
        # 웹캠 초기화
        try:
            self.webcam = WebcamCapture(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps
            )
            self.is_initialized = True
        except Exception as e:
            logger.error(f"웹캠 초기화 실패: {e}")
            self.is_initialized = False
            raise RuntimeError(f"웹캠 초기화 실패: {e}")
        
        # 프레임 레이트 제어 초기화
        self.frame_controller = FrameController(
            target_fps=fps,
            adaptive_skip=adaptive_skip,
            wait_key_ms=render_wait_time
        )
        
        # 프레임 처리 관련 설정
        self.enable_resize = enable_resize
        self.processing_width = processing_width
        self.processing_height = processing_height
        
        # 콜백 및 상태 관리
        self.frame_callbacks = []
        self.last_frame = None
        self.frame_count = 0
        
        # 렌더링 최적화를 위한 플래그
        self._windows_initialized = False
        
        logger.info(f"EfficientWebcam 초기화: camera_id={camera_id}, "
                   f"resolution={width}x{height}, fps={fps}")
    
    def _ensure_windows_initialized(self, window_name: str = 'EfficientWebcam') -> None:
        """디스플레이 창 초기화 보장
        
        Args:
            window_name (str): 창 이름 (기본값: 'EfficientWebcam')
        """
        if not self._windows_initialized:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self._windows_initialized = True
    
    def add_frame_callback(self, callback: Callable) -> None:
        """프레임 처리 콜백 추가
        
        Args:
            callback: 프레임 처리 콜백 함수
                callback(frame, timestamp, metadata) -> Optional[np.ndarray]
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
        
        효율적인 타이밍 관리와 함께 프레임을 캡처하고 처리합니다.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (성공 여부, 처리된 프레임)
        """
        if not self.is_initialized:
            logger.error("웹캠이 초기화되지 않았습니다.")
            return False, None
            
        # 1. 타이밍 관리: 일정한 FPS 유지
        self.frame_controller.wait_for_next_frame()
        
        # 2. 프레임 스킵 결정: 부하에 따라 동적으로 조정
        process_this_frame = self.frame_controller.should_process_frame()
        
        # 3. 프레임 캡처
        try:
            ret, frame = self.webcam.capture_frame()
            
            if not ret or frame is None:
                logger.warning("프레임 캡처 실패")
                return False, None
            
            # 프레임 카운트 증가
            self.frame_count += 1
            
            # 스킵 결정된 프레임은 최소 처리만 수행
            if not process_this_frame:
                return True, frame
            
            # 4. 프레임 처리 시작 시간 기록
            start_time = time.time()
            timestamp = start_time
            
            # 5. 처리용 프레임 준비 (리사이징으로 메모리 사용량 절감)
            if self.enable_resize:
                processing_frame = cv2.resize(
                    frame, 
                    (self.processing_width, self.processing_height),
                    interpolation=cv2.INTER_AREA  # 축소에 최적화된 보간법
                )
            else:
                processing_frame = frame  # 원본 사용 (복사 없음)
            
            # 6. 메타데이터 생성
            fps = self.frame_controller.update_fps()
            fc_stats = self.frame_controller.get_stats()
            
            metadata = {
                'timestamp': timestamp,
                'fps': fps,
                'frame_number': self.frame_count,
                'original_shape': frame.shape,
                'skip_frames': fc_stats['current_skip_frames'],
                'is_resized': self.enable_resize
            }
            
            # 7. 콜백 함수 처리 (복사 최소화)
            processed_frame = None
            
            for callback in self.frame_callbacks:
                try:
                    result = callback(processing_frame, timestamp, metadata)
                    if result is not None and isinstance(result, np.ndarray):
                        processed_frame = result
                except Exception as e:
                    logger.error(f"콜백 실행 중 오류 발생: {e}")
            
            # 8. 결과 프레임 결정 (처리된 프레임이 없으면 원본 사용)
            result_frame = processed_frame if processed_frame is not None else frame
            
            # 9. 처리 시간 계산 및 적응형 스킵 업데이트
            processing_time = time.time() - start_time
            self.frame_controller.update_skip_frames(processing_time)
            
            # 10. 마지막 프레임 정보 저장 (복사 최소화)
            self.last_frame = {
                'frame': frame,  # 원본 프레임 (복사 없음)
                'processed_frame': result_frame,  # 처리된 프레임
                'timestamp': timestamp,
                'metadata': metadata,
                'processing_time': processing_time
            }
            
            # 11. 창 초기화 확인
            self._ensure_windows_initialized()
            
            return True, result_frame
            
        except Exception as e:
            logger.error(f"프레임 처리 중 오류 발생: {e}")
            return False, None
    
    def display_frame(self, frame: np.ndarray, window_name: str = 'EfficientWebcam') -> int:
        """프레임 화면에 표시
        
        화면 깜빡임을 최소화하는 방식으로 프레임을 표시합니다.
        
        Args:
            frame: 표시할 프레임
            window_name: 창 이름 (기본값: 'EfficientWebcam')
            
        Returns:
            int: 키보드 입력 값
        """
        # 창 생성 보장
        self._ensure_windows_initialized(window_name)
        
        # 프레임 표시
        cv2.imshow(window_name, frame)
        
        # 깜빡임 방지를 위한 최적화된 waitKey 시간 사용
        wait_ms = self.frame_controller.get_wait_key_ms()
        return cv2.waitKey(wait_ms) & 0xFF
    
    def get_latest_frame(self) -> Optional[Dict]:
        """최신 프레임 정보 가져오기
        
        Returns:
            Optional[Dict]: 최신 프레임 정보 또는 None
        """
        return self.last_frame
    
    def save_snapshot(self, filename: str = None, use_processed: bool = True,
                     snapshots_dir: str = "snapshots") -> Optional[str]:
        """현재 프레임의 스냅샷을 파일로 저장
        
        Args:
            filename (str, optional): 저장할 파일명
            use_processed (bool): 처리된 프레임 사용 여부 (기본값: True)
            snapshots_dir (str): 스냅샷 저장 디렉토리 (기본값: "snapshots")
            
        Returns:
            Optional[str]: 저장된 파일 경로 또는 None
        """
        if self.last_frame is None:
            logger.warning("저장할 프레임이 없습니다.")
            return None
        
        # 스냅샷 디렉토리 생성
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # 파일명 자동 생성
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            prefix = "processed" if use_processed else "original"
            filename = f"{snapshots_dir}/{prefix}_snapshot_{timestamp}.jpg"
        elif not filename.startswith(f"{snapshots_dir}/"):
            filename = f"{snapshots_dir}/{filename}"
        
        try:
            # 저장할 프레임 선택
            if use_processed and 'processed_frame' in self.last_frame:
                image = self.last_frame['processed_frame']
            else:
                image = self.last_frame['frame']
                
            # 이미지 저장
            cv2.imwrite(filename, image)
            logger.info(f"스냅샷 저장됨: {filename}")
            
            # 메타데이터 저장 (선택 사항)
            metadata_file = f"{filename.rsplit('.', 1)[0]}.json"
            self._save_metadata(metadata_file, self.last_frame)
            
            return filename
            
        except Exception as e:
            logger.error(f"스냅샷 저장 실패: {e}")
            return None
    
    def _save_metadata(self, filename: str, frame_data: Dict) -> None:
        """프레임 메타데이터 저장
        
        Args:
            filename: 저장할 파일명
            frame_data: 프레임 데이터
        """
        try:
            import json
            
            # 메타데이터 추출 (numpy 배열 제외)
            metadata = {
                'timestamp': frame_data.get('timestamp', 0),
                'processing_time': frame_data.get('processing_time', 0),
                'frame_shape': frame_data['frame'].shape if 'frame' in frame_data else None,
                'metadata': frame_data.get('metadata', {})
            }
            
            # JSON으로 저장
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """상태 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        stats = {
            # FPS 및 프레임 정보
            'frame_count': self.frame_count,
            'callbacks': len(self.frame_callbacks)
        }
        
        # 프레임 컨트롤러 통계 추가
        stats.update(self.frame_controller.get_stats())
        
        # 마지막 프레임 처리 시간 추가
        if self.last_frame and 'processing_time' in self.last_frame:
            stats['last_processing_time_ms'] = self.last_frame['processing_time'] * 1000
        
        # 카메라 속성 추가
        if self.is_initialized:
            try:
                props = self.webcam.get_camera_properties()
                stats['camera'] = props
            except Exception as e:
                logger.error(f"카메라 속성 가져오기 실패: {e}")
        
        return stats
    
    def release(self) -> None:
        """리소스 해제"""
        logger.info("리소스 해제 중...")
        
        # 웹캠 리소스 해제
        if self.is_initialized:
            self.webcam.release()
            self.is_initialized = False
        
        # 창 닫기 (있는 경우)
        if self._windows_initialized:
            cv2.destroyAllWindows()
            self._windows_initialized = False
