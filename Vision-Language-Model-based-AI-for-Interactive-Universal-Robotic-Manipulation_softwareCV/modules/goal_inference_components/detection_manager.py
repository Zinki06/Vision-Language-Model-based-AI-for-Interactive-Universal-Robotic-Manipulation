"""
객체 감지 관리 모듈

이 모듈은 객체 감지 관련 로직을 담당하는 DetectionManager 클래스를 포함합니다.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

from ..yolo_detector import YOLODetector

class DetectionManager:
    """
    객체 감지 관리 클래스
    
    이 클래스는 다음과 같은 객체 감지 관련 기능을 담당합니다:
    - YOLO 객체 감지기 초기화 및 관리
    - 이미지에서 객체 감지 수행
    - 감지된 객체 결과 필터링 및 처리
    """
    
    def __init__(self, 
                yolo_model: str = "yolov8n.pt",
                conf_threshold: Optional[float] = None,
                iou_threshold: Optional[float] = None,
                hand_classes: List[str] = None,
                logger: Optional[logging.Logger] = None):
        """
        DetectionManager 초기화
        
        Args:
            yolo_model: 사용할 YOLOv8 모델 경로 또는 이름
            conf_threshold: 객체 감지 신뢰도 임계값
            iou_threshold: 객체 감지 IoU 임계값
            hand_classes: 손으로 인식할 클래스 목록
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 손 클래스 정의 - YOLO에서 손으로 인식할 클래스 목록
        self.hand_classes = hand_classes or ["hand", "person"]
        
        # YOLOv8 객체 감지기 초기화
        self.logger.info(f"YOLO: {yolo_model} 모델 로드 중...")
        self.yolo_detector = YOLODetector(
            model_name=yolo_model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        self.logger.info(f"YOLO: {yolo_model} 모델 로드 완료")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 객체 감지 수행
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            List[Dict[str, Any]]: 감지된 객체 목록
        """
        try:
            # YOLO 객체 감지 수행
            self.logger.debug("객체 감지 수행 중...")
            detections = self.yolo_detector.detect(image)
            self.logger.info(f"객체 감지 완료: {len(detections)}개 객체 감지됨")
            return detections
        except Exception as e:
            self.logger.error(f"객체 감지 중 오류 발생: {e}")
            return []
    
    def filter_non_hand_objects(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        감지된 객체 중 손이 아닌 객체만 필터링
        
        Args:
            detections: 감지된 객체 목록
            
        Returns:
            List[Dict[str, Any]]: 손이 아닌 객체 목록
        """
        if not detections:
            return []
            
        non_hand_objects = []
        for obj in detections:
            class_name = obj.get("class_name", "").lower()
            is_hand = any(hand_class.lower() in class_name for hand_class in self.hand_classes)
            
            if not is_hand:
                non_hand_objects.append(obj)
                
        self.logger.debug(f"손 필터링 결과: {len(non_hand_objects)}/{len(detections)} 객체 남음")
        return non_hand_objects
    
    def get_object_by_index(self, detections: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
        """
        인덱스를 기반으로 감지된 객체 가져오기
        
        Args:
            detections: 감지된 객체 목록
            index: 객체 인덱스
            
        Returns:
            Optional[Dict[str, Any]]: 해당 인덱스의 객체 또는 None
        """
        if not detections or index < 0 or index >= len(detections):
            return None
        return detections[index]
    
    def find_objects_by_class(self, detections: List[Dict[str, Any]], 
                             class_names: List[str]) -> List[Dict[str, Any]]:
        """
        클래스 이름을 기반으로 객체 필터링
        
        Args:
            detections: 감지된 객체 목록
            class_names: 찾을 클래스 이름 목록
            
        Returns:
            List[Dict[str, Any]]: 해당 클래스의 객체 목록
        """
        if not detections or not class_names:
            return []
            
        result = []
        for obj in detections:
            class_name = obj.get("class_name", "").lower()
            if any(target_class.lower() in class_name for target_class in class_names):
                result.append(obj)
                
        return result
    
    def get_object_bbox(self, obj: Dict[str, Any]) -> List[float]:
        """
        객체의 바운딩 박스 정보 가져오기
        
        Args:
            obj: 객체 정보
            
        Returns:
            List[float]: [x1, y1, x2, y2] 형태의 바운딩 박스
        """
        # 다양한 키 이름 지원 (bbox, box 등)
        for key in ["bbox", "box"]:
            if key in obj:
                return obj[key]
        
        # 기본값 사용
        return [0, 0, 0, 0] 