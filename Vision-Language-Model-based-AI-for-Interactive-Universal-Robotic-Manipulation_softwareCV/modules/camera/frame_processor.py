"""
프레임 프로세서 모듈

카메라 프레임을 처리하기 위한 프로세서 인터페이스와 체인을 제공합니다.
다양한 프레임 프로세서를 연결하여 처리 파이프라인을 구성할 수 있습니다.
"""

import abc
import time
import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

# 로깅 설정
logger = logging.getLogger(__name__)


class FrameProcessor(abc.ABC):
    """프레임 프로세서 추상 클래스
    
    카메라 프레임을 처리하기 위한 인터페이스를 정의합니다.
    모든 프레임 프로세서는 이 클래스를 상속해야 합니다.
    """
    
    def __init__(self, processor_id: Optional[str] = None, enabled: bool = True):
        """프레임 프로세서 초기화
        
        Args:
            processor_id: 프로세서 ID (기본값: 무작위 UUID)
            enabled: 활성화 여부 (기본값: True)
        """
        self.processor_id = processor_id or str(uuid.uuid4())
        self.enabled = enabled
        self.processed_count = 0
        self.total_process_time = 0
        self.last_process_time = 0
        self.last_process_timestamp = 0
        self.avg_process_time = 0
    
    def enable(self) -> None:
        """프로세서 활성화"""
        self.enabled = True
    
    def disable(self) -> None:
        """프로세서 비활성화"""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """프로세서 활성화 상태 확인
        
        Returns:
            bool: 활성화 상태
        """
        return self.enabled
    
    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임 처리
        
        Args:
            frame: 처리할 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 처리된 프레임과 메타데이터
        """
        pass
    
    def safe_process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """안전하게 프레임 처리
        
        프레임 처리 중 오류가 발생해도 원본 프레임을 반환합니다.
        
        Args:
            frame: 처리할 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 처리된 프레임과 메타데이터
        """
        if not self.enabled or frame is None or frame.size == 0:
            return frame, {"error": "프레임 없음 또는 프로세서 비활성화"}
        
        # 처리 시간 측정
        start_time = time.time()
        
        try:
            # 프레임 처리
            processed_frame, metadata = self.process(frame)
            
            # 처리 시간 계산
            end_time = time.time()
            process_time = end_time - start_time
            
            # 성능 측정 업데이트
            self.processed_count += 1
            self.total_process_time += process_time
            self.last_process_time = process_time
            self.last_process_timestamp = end_time
            self.avg_process_time = self.total_process_time / self.processed_count
            
            # 메타데이터에 성능 정보 추가
            metadata.update({
                "processor_id": self.processor_id,
                "processor_type": self.__class__.__name__,
                "process_time": process_time,
                "process_timestamp": end_time,
            })
            
            return processed_frame, metadata
            
        except Exception as e:
            logger.error(f"프레임 처리 오류 ({self.__class__.__name__}): {e}")
            # 오류 발생 시 원본 프레임 반환
            return frame, {
                "processor_id": self.processor_id,
                "processor_type": self.__class__.__name__,
                "error": str(e),
                "process_timestamp": time.time(),
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """프로세서 메타데이터 가져오기
        
        Returns:
            Dict[str, Any]: 프로세서 메타데이터
        """
        return {
            "processor_id": self.processor_id,
            "type": self.__class__.__name__,
            "enabled": self.enabled,
            "processed_count": self.processed_count,
            "avg_process_time": self.avg_process_time,
            "last_process_time": self.last_process_time,
            "last_process_timestamp": self.last_process_timestamp,
        }


class FrameProcessorChain:
    """프레임 프로세서 체인
    
    여러 프레임 프로세서를 순차적으로 실행하는 체인을 관리합니다.
    """
    
    def __init__(self):
        """프레임 프로세서 체인 초기화"""
        self.processors: List[FrameProcessor] = []
        self.processor_map: Dict[str, FrameProcessor] = {}
        self.processed_count = 0
        self.total_process_time = 0
        self.last_process_time = 0
        self.last_process_timestamp = 0
    
    def add_processor(self, processor: FrameProcessor) -> bool:
        """프로세서 추가
        
        Args:
            processor: 추가할 프로세서
            
        Returns:
            bool: 성공 여부
        """
        if processor.processor_id in self.processor_map:
            logger.warning(f"이미 존재하는 프로세서 ID: {processor.processor_id}")
            return False
        
        self.processors.append(processor)
        self.processor_map[processor.processor_id] = processor
        return True
    
    def remove_processor(self, processor_id: str) -> Optional[FrameProcessor]:
        """프로세서 제거
        
        Args:
            processor_id: 제거할 프로세서 ID
            
        Returns:
            Optional[FrameProcessor]: 제거된 프로세서 (없으면 None)
        """
        if processor_id not in self.processor_map:
            return None
        
        processor = self.processor_map[processor_id]
        self.processors.remove(processor)
        del self.processor_map[processor_id]
        return processor
    
    def get_processor(self, processor_id: str) -> Optional[FrameProcessor]:
        """프로세서 가져오기
        
        Args:
            processor_id: 가져올 프로세서 ID
            
        Returns:
            Optional[FrameProcessor]: 프로세서 (없으면 None)
        """
        return self.processor_map.get(processor_id)
    
    def get_processors(self) -> List[FrameProcessor]:
        """모든 프로세서 가져오기
        
        Returns:
            List[FrameProcessor]: 프로세서 목록
        """
        return self.processors.copy()
    
    def clear(self) -> None:
        """모든 프로세서 제거"""
        self.processors.clear()
        self.processor_map.clear()
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임 처리
        
        모든 프로세서를 순차적으로
        
        Args:
            frame: 처리할 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 처리된 프레임과 메타데이터
        """
        if frame is None or frame.size == 0:
            return frame, {"error": "입력 프레임 없음"}
        
        # 처리 시간 측정
        start_time = time.time()
        current_frame = frame.copy()
        all_metadata: Dict[str, Any] = {}
        processor_metadata: List[Dict[str, Any]] = []
        
        # 모든 프로세서 순차 실행
        for processor in self.processors:
            processed_frame, metadata = processor.safe_process(current_frame)
            processor_metadata.append({
                "processor_id": processor.processor_id,
                "processor_type": processor.__class__.__name__,
                "metadata": metadata
            })
            
            # 처리된 프레임을 다음 프로세서의 입력으로 사용
            if processed_frame is not None and processed_frame.size > 0:
                current_frame = processed_frame
        
        # 처리 시간 계산
        end_time = time.time()
        process_time = end_time - start_time
        
        # 성능 측정 업데이트
        self.processed_count += 1
        self.total_process_time += process_time
        self.last_process_time = process_time
        self.last_process_timestamp = end_time
        
        # 메타데이터 준비
        all_metadata = {
            "chain_process_time": process_time,
            "chain_process_timestamp": end_time,
            "chain_processed_count": self.processed_count,
            "chain_processor_count": len(self.processors),
            "processors": processor_metadata
        }
        
        return current_frame, all_metadata
    
    def get_metadata(self) -> Dict[str, Any]:
        """체인 메타데이터 가져오기
        
        Returns:
            Dict[str, Any]: 체인 메타데이터
        """
        processor_metadata = []
        for processor in self.processors:
            processor_metadata.append(processor.get_metadata())
        
        return {
            "processor_count": len(self.processors),
            "processed_count": self.processed_count,
            "avg_process_time": self.total_process_time / self.processed_count if self.processed_count > 0 else 0,
            "last_process_time": self.last_process_time,
            "last_process_timestamp": self.last_process_timestamp,
            "processors": processor_metadata
        } 