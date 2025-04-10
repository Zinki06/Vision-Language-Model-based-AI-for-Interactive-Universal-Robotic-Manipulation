"""
제스처 인식 및 해석 모듈

이 모듈은 제스처 인식 및 해석을 담당하는 GestureHandler 클래스를 포함합니다.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from ..gesture_recognizer import GestureRecognizer, GestureType
from ..utils.geometry import calculate_iou, calculate_box_from_points, calculate_angle_2d

class GestureHandler:
    """
    제스처 인식 및 해석 클래스
    
    이 클래스는 다음과 같은 제스처 관련 기능을 담당합니다:
    - 제스처 인식기 초기화 및 관리
    - 손 포인팅 인식 및 처리
    - 제스처와 객체 간 관계 분석
    """
    
    def __init__(self, 
                gesture_model_path: str = "models/hand_landmarker.task",
                logger: Optional[logging.Logger] = None):
        """
        GestureHandler 초기화
        
        Args:
            gesture_model_path: MediaPipe 제스처 인식 모델 경로
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 제스처 인식기 초기화
        self.logger.info(f"GestureRecognizer 초기화 중... ({gesture_model_path})")
        try:
            self.gesture_recognizer = GestureRecognizer(
                model_path=gesture_model_path
            )
            self.logger.info("GestureRecognizer 초기화 완료")
        except Exception as e:
            self.logger.error(f"GestureRecognizer 초기화 실패: {e}")
            self.gesture_recognizer = None
        
        # 제스처 모드 설정 - 초기값 False
        self.gesture_mode_active = False
    
    def recognize_gestures(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 제스처 인식 수행
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            List[Dict[str, Any]]: 인식된 제스처 정보 목록
        """
        if not self.gesture_recognizer:
            self.logger.warning("제스처 인식기가 초기화되지 않았습니다.")
            return []
            
        try:
            gesture_results = self.gesture_recognizer.process_frame(image)
            self.logger.debug(f"제스처 인식 완료: {len(gesture_results) if gesture_results else 0}개 감지됨")
            return gesture_results or []
        except Exception as e:
            self.logger.error(f"제스처 인식 중 오류 발생: {e}")
            return []
    
    def set_gesture_mode(self, active: bool) -> None:
        """
        제스처 모드 활성화/비활성화
        
        Args:
            active: 제스처 모드 활성화 여부
        """
        self.gesture_mode_active = active
        self.logger.info(f"제스처 모드: {'활성화' if active else '비활성화'}")
    
    def find_pointing_gestures(self, gesture_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        포인팅 제스처 필터링
        
        Args:
            gesture_results: 인식된 모든 제스처 결과 목록
            
        Returns:
            List[Dict[str, Any]]: 포인팅 제스처로 분류된 결과 목록
        """
        if not gesture_results:
            return []
            
        pointing_gestures = []
        
        for gesture in gesture_results:
            # 제스처 타입 체크
            gesture_type = gesture.get('gesture_type')
            
            # 포인팅 제스처만 필터링
            if gesture_type == GestureType.POINTING:
                pointing_gestures.append(gesture)
                
        self.logger.debug(f"포인팅 제스처 필터링: {len(pointing_gestures)}/{len(gesture_results)} 개 감지")
        return pointing_gestures
    
    def find_target_by_gesture(self, 
                              gesture_results: List[Dict[str, Any]], 
                              detections: List[Dict[str, Any]],
                              hand_classes: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        제스처 기반으로 타겟 객체 찾기
        
        Args:
            gesture_results: 인식된 제스처 정보 목록
            detections: 감지된 객체 목록
            hand_classes: 손으로 분류할 클래스 이름 목록 (기본값: ["hand", "person"])
            
        Returns:
            Optional[Dict[str, Any]]: 포인팅된 타겟 객체 또는 None
        """
        if not gesture_results or not detections:
            return None
            
        # 손 클래스 정의
        hand_classes = hand_classes or ["hand", "person"]
        
        # 포인팅 제스처 필터링
        pointing_gestures = self.find_pointing_gestures(gesture_results)
        if not pointing_gestures:
            self.logger.info("포인팅 제스처를 찾을 수 없습니다.")
            return None
            
        # 객체 중 손이 아닌 것만 필터링
        non_hand_objects = []
        for obj in detections:
            class_name = obj.get("class_name", "").lower()
            is_hand = any(hand_class.lower() in class_name for hand_class in hand_classes)
            
            if not is_hand:
                non_hand_objects.append(obj)
        
        if not non_hand_objects:
            self.logger.info("손이 아닌 객체를 찾을 수 없습니다.")
            return None
        
        # IoU 기반 객체 필터링
        iou_candidates = []
        for pointing_gesture in pointing_gestures:
            # 손 랜드마크에서 바운딩 박스 계산
            hand_box = calculate_box_from_points(pointing_gesture['all_points_2d'])
            
            for obj in non_hand_objects:
                # 객체 박스
                obj_box = obj.get('bbox') or obj.get('box', [0, 0, 0, 0])
                
                # IoU 계산
                iou = calculate_iou(obj_box, hand_box)
                self.logger.debug(f"손-객체 IoU: {iou:.4f}, 객체: {obj.get('class_name')}")
                
                # IoU가 임계값보다 큰 경우 (겹침이 있는 경우)
                if iou > 0.05:  # 5% 이상 겹침
                    iou_candidates.append({
                        'object': obj,
                        'gesture': pointing_gesture,
                        'iou': iou
                    })
        
        # IoU 기반 후보가 없는 경우 각도 기반 필터링
        if not iou_candidates:
            self.logger.info("IoU 기반 겹침이 없어 포인팅 방향 각도로 찾기")
            
            angle_candidates = []
            for pointing_gesture in pointing_gestures:
                # 포인팅 벡터
                if 'pointing_vector_2d' not in pointing_gesture:
                    continue
                    
                pointing_vector_2d = pointing_gesture['pointing_vector_2d']
                
                for obj in non_hand_objects:
                    # 객체 바운딩 박스 중심
                    obj_box = obj.get('bbox') or obj.get('box', [0, 0, 0, 0])
                    obj_center_x = (obj_box[0] + obj_box[2]) / 2
                    obj_center_y = (obj_box[1] + obj_box[3]) / 2
                    
                    # 손 랜드마크에서 객체 중심으로 향하는 벡터
                    index_finger_tip = pointing_gesture.get('index_finger_tip', [0, 0])
                    obj_vector = [obj_center_x - index_finger_tip[0], obj_center_y - index_finger_tip[1]]
                    
                    # 두 벡터 간 각도 계산
                    angle = calculate_angle_2d(pointing_vector_2d, obj_vector)
                    self.logger.debug(f"손-객체 각도: {angle:.2f}도, 객체: {obj.get('class_name')}")
                    
                    # 각도가 임계값보다 작은 경우 (포인팅 방향과 유사한 경우)
                    if angle < 45.0:  # 45도 이내
                        angle_candidates.append({
                            'object': obj,
                            'gesture': pointing_gesture,
                            'angle': angle
                        })
        
            # 각도 기반 타겟 선택 (가장 작은 각도)
            if angle_candidates:
                best_candidate = min(angle_candidates, key=lambda x: x['angle'])
                self.logger.info(f"각도 기반 타겟 선택: {best_candidate['object'].get('class_name')}, 각도: {best_candidate['angle']:.2f}도")
                return best_candidate['object']
            else:
                self.logger.info("각도 기반 타겟을 찾을 수 없습니다.")
                return None
        else:
            # IoU 기반 타겟 선택 (가장 큰 IoU)
            best_candidate = max(iou_candidates, key=lambda x: x['iou'])
            self.logger.info(f"IoU 기반 타겟 선택: {best_candidate['object'].get('class_name')}, IoU: {best_candidate['iou']:.4f}")
            return best_candidate['object'] 