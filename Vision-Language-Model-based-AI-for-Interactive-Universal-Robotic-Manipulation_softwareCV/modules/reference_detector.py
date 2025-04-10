import re
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import math
from datetime import datetime

class ReferenceDetector:
    """레퍼런스 객체 감지 및 분석 클래스"""
    
    def __init__(self, spatial_analyzer, logger=None):
        """
        ReferenceDetector 초기화
        
        Args:
            spatial_analyzer: 공간 관계 분석기 (SpatialAnalyzer 인스턴스)
            logger: 로깅을 위한 로거 객체. None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("ReferenceDetector")
        self.spatial_analyzer = spatial_analyzer
        
        # 신뢰도 임계값
        self.HIGH_CONFIDENCE = 0.8
        self.MID_CONFIDENCE = 0.5
        self.LOW_CONFIDENCE = 0.3
        
        # 손으로 간주될 클래스 이름 목록 (YOLO 모델에 따라 조정 필요)
        self.HAND_CLASSES = ["hand", "person"] # 예시, 실제 사용하는 YOLO 모델의 클래스 이름 확인 필요
        
        # 한국어 전치사구 패턴
        self.preposition_patterns = {
            # 위치 전치사
            '위에': {'direction': 'above', 'weight': 1.0},
            '아래': {'direction': 'below', 'weight': 1.0},
            '아래에': {'direction': 'below', 'weight': 1.0},
            '앞에': {'direction': 'front', 'weight': 1.0},
            '앞으로': {'direction': 'front', 'weight': 1.0},
            '뒤에': {'direction': 'back', 'weight': 1.0},
            '뒤로': {'direction': 'back', 'weight': 1.0},
            '왼쪽': {'direction': 'left', 'weight': 1.0},
            '왼쪽에': {'direction': 'left', 'weight': 1.0},
            '왼쪽으로': {'direction': 'left', 'weight': 1.0},
            '오른쪽': {'direction': 'right', 'weight': 1.0},
            '오른쪽에': {'direction': 'right', 'weight': 1.0},
            '오른쪽으로': {'direction': 'right', 'weight': 1.0},
            '옆에': {'direction': None, 'weight': 0.8},  # 명확한 방향 없음
            '옆으로': {'direction': None, 'weight': 0.8},
            '근처에': {'direction': None, 'weight': 0.7},
            '가까이': {'direction': None, 'weight': 0.7},
            '멀리': {'direction': None, 'weight': 0.6}
        }
        
        # 영어 전치사구 패턴
        self.en_preposition_patterns = {
            'in front of': {'direction': 'front', 'weight': 1.0},
            'behind': {'direction': 'back', 'weight': 1.0},
            'above': {'direction': 'above', 'weight': 1.0},
            'over': {'direction': 'above', 'weight': 0.9},
            'below': {'direction': 'below', 'weight': 1.0},
            'under': {'direction': 'below', 'weight': 0.9},
            'to the left of': {'direction': 'left', 'weight': 1.0},
            'to the right of': {'direction': 'right', 'weight': 1.0},
            'next to': {'direction': None, 'weight': 0.8},
            'near': {'direction': None, 'weight': 0.7},
            'beside': {'direction': None, 'weight': 0.8},
            'between': {'direction': None, 'weight': 0.7}
        }
        
        # 방향 표현 정규화
        self.direction_mapping = {
            'front': spatial_analyzer.DIRECTION_FRONT,
            'back': spatial_analyzer.DIRECTION_BACK,
            'left': spatial_analyzer.DIRECTION_LEFT,
            'right': spatial_analyzer.DIRECTION_RIGHT,
            'above': spatial_analyzer.DIRECTION_ABOVE,
            'below': spatial_analyzer.DIRECTION_BELOW
        }
        
        # 지시대명사 패턴
        self.deictic_patterns = {
            'ko': r'(이거|저거|그거|이것|저것|그것|여기|저기|거기|이|저|그)',
            'en': r'\b(this|that|these|those|here|there)\b'
        }
        
        # 방향 추출 정규식
        self.direction_pattern = re.compile(
            r'(앞|뒤|위|아래|왼쪽|오른쪽|옆)(?:쪽|으로|에)?'
        )
        
        self.logger.info("ReferenceDetector 초기화 완료")
    
    def detect_reference(self, prompt: str, 
                        detected_objects: List[Dict], 
                        gesture_info: Optional[Dict] = None) -> Dict:
        """
        프롬프트와 감지된 객체 정보를 기반으로 레퍼런스 객체 감지
        
        Args:
            prompt: 사용자 프롬프트
            detected_objects: 감지된 객체 목록
            gesture_info: 제스처 정보 (옵션)
            
        Returns:
            Dict: {
                "reference_object": 레퍼런스 객체 또는 None,
                "confidence": 신뢰도,
                "direction": 방향 정보
            }
        """
        # 감지된 객체가 없으면 빈 결과 반환
        if not detected_objects:
            return {
                "reference_object": None,
                "confidence": 0.0,
                "direction": None
            }
        
        # 1. 언어적 분석 수행
        linguistic_result = self._linguistic_analysis(prompt, detected_objects)
        
        # 2. 시각적 분석 수행 (제스처 정보 있는 경우)
        visual_result = self._visual_analysis(gesture_info, detected_objects) if gesture_info else {
            "reference_object": None,
            "confidence": 0.0,
            "direction": None
        }
        
        # 3. 통합 판단
        return self._integrated_decision(linguistic_result, visual_result)
    
    def _linguistic_analysis(self, prompt: str, 
                           detected_objects: List[Dict]) -> Dict:
        """
        프롬프트의 언어적 분석을 통한 레퍼런스 객체 식별
        
        Args:
            prompt: 사용자 프롬프트
            detected_objects: 감지된 객체 목록
            
        Returns:
            Dict: {
                "reference_object": 레퍼런스 객체 또는 None,
                "confidence": 신뢰도,
                "direction": 방향 정보
            }
        """
        if not prompt:
            return {
                "reference_object": None,
                "confidence": 0.0,
                "direction": None
            }
        
        prompt_lower = prompt.lower()
        result = {
            "reference_object": None,
            "confidence": 0.0,
            "direction": None
        }
        
        # 1. 전치사구 포함 여부 확인
        preposition_result = self._extract_preposition_relation(prompt_lower, detected_objects)
        if preposition_result["reference_object"]:
            return preposition_result
        
        # 2. 방향 표현과 명사 조합 확인
        direction_noun_result = self._extract_direction_noun_relation(prompt_lower, detected_objects)
        if direction_noun_result["reference_object"]:
            return direction_noun_result
        
        # 3. 지시대명사 + 방향어 확인
        deictic_result = self._extract_deictic_relation(prompt_lower)
        if deictic_result["direction"]:
            result["direction"] = deictic_result["direction"]
            result["confidence"] = deictic_result["confidence"]
        
        # 4. 단일 목적어 확인 (레퍼런스 없음)
        if self._is_single_object_command(prompt_lower):
            result["confidence"] = 0.9  # 높은 신뢰도로 레퍼런스 없음 확인
        
        return result
    
    def _visual_analysis(self, gesture_info: Dict, 
                        detected_objects: List[Dict]) -> Dict:
        """
        시각적 정보(제스처 등)를 기반으로 레퍼런스 객체 식별
        
        Args:
            gesture_info: 제스처 분석 정보
            detected_objects: 감지된 객체 목록
            
        Returns:
            Dict: {
                "reference_object": 레퍼런스 객체 또는 None,
                "confidence": 신뢰도,
                "direction": 방향 정보
            }
        """
        result = {
            "reference_object": None,
            "confidence": 0.0,
            "direction": None
        }
        
        if not gesture_info:
            return result
        
        # 1. 포인팅 제스처 분석
        if gesture_info.get("gesture_type") == "pointing" and gesture_info.get("pointing_vector"):
            # 포인팅 대상 객체가 직접 타겟이 아니라 레퍼런스일 수 있음
            # 대상 객체와 다른 객체들의 공간 관계 분석
            
            # 타겟 객체가 이미 지정되어 있는지 확인
            target_object = gesture_info.get("target_object")
            
            # 포인팅 대상 객체를 레퍼런스로 사용할 수 있는지 확인
            pointing_object = None
            pointing_confidence = 0.0
            
            # 포인팅 벡터
            origin, direction = gesture_info["pointing_vector"]
            
            # 포인팅 벡터와 객체들의 교차 확인
            for obj in detected_objects:
                if target_object and obj == target_object:
                    continue  # 타겟 객체는 건너뜀
                
                # 객체 중심점 계산
                bbox = obj.get("bbox", [0, 0, 0, 0])
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # 포인팅 벡터와 객체 중심 사이의 거리 계산
                dx = center_x - origin[0]
                dy = center_y - origin[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # 벡터의 각도 계산
                angle = math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, 
                            (dx*direction[0] + dy*direction[1]) / 
                            (distance * math.sqrt(direction[0]**2 + direction[1]**2))
                        ))
                    )
                )
                
                # 일정 각도 이내인 경우 후보로 지정
                if angle < 30:  # 30도 이내
                    confidence = 1.0 - (angle / 30)  # 각도가 작을수록 신뢰도 높음
                    
                    # --- 손 오인식 방지 로직 시작 ---
                    obj_class_name = obj.get("class_name", "").lower()
                    is_hand_related = obj_class_name in self.HAND_CLASSES or 'hand' in obj_class_name
                    
                    if is_hand_related:
                        self.logger.debug(f"시각 분석: 객체 '{obj.get('class_name')}'(ID:{obj.get('id')})는 손 관련 클래스이므로 레퍼런스 후보에서 제외합니다.")
                        continue # 손 관련 객체는 건너뜀
                    # --- 손 오인식 방지 로직 끝 ---
                    
                    if confidence > pointing_confidence:
                        pointing_object = obj
                        pointing_confidence = confidence
                        self.logger.debug(f"시각 분석: 포인팅 후보 업데이트 - 객체 '{pointing_object.get('class_name')}'(ID:{pointing_object.get('id')}), 신뢰도 {pointing_confidence:.2f}")
            
            if pointing_object:
                result["reference_object"] = pointing_object
                result["confidence"] = pointing_confidence
                self.logger.info(f"시각 분석 결과 (포인팅): 레퍼런스 후보='{pointing_object.get('class_name')}'(ID:{pointing_object.get('id')}), 신뢰도={pointing_confidence:.2f}")
        
        # 2. 객체 간 공간 관계 분석
        # 이미 타겟과 다른 객체가 지정된 경우, 이들 간의 상대적 위치 분석
        if gesture_info.get("target_object") and len(detected_objects) > 1:
            target_obj = gesture_info["target_object"]
            
            # 타겟 객체가 아닌 객체들 중 가장 가까운 객체 찾기
            closest_obj = None
            min_distance = float('inf')
            
            target_bbox = target_obj.get("bbox", [0, 0, 0, 0])
            target_center = [(target_bbox[0] + target_bbox[2]) / 2, 
                             (target_bbox[1] + target_bbox[3]) / 2]
            
            for obj in detected_objects:
                if obj == target_obj:
                    continue
                
                obj_bbox = obj.get("bbox", [0, 0, 0, 0])
                obj_center = [(obj_bbox[0] + obj_bbox[2]) / 2, 
                             (obj_bbox[1] + obj_bbox[3]) / 2]
                
                # 중심점 간 거리 계산
                distance = math.sqrt(
                    (target_center[0] - obj_center[0])**2 +
                    (target_center[1] - obj_center[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_obj = obj
            
            if closest_obj:
                # 공간 관계 분석
                spatial_relation = self.spatial_analyzer.analyze_spatial_relation(
                    target_obj, closest_obj
                )
                
                # 신뢰도 계산 (거리 기반)
                distance_confidence = 1.0 - min(1.0, min_distance / 500)  # 500px 기준
                
                # --- 손 오인식 방지 로직 (근접 객체) ---
                closest_obj_class = closest_obj.get("class_name", "").lower()
                is_closest_hand_related = closest_obj_class in self.HAND_CLASSES or 'hand' in closest_obj_class
                if is_closest_hand_related:
                    self.logger.debug(f"시각 분석: 가장 가까운 객체 '{closest_obj.get('class_name')}'(ID:{closest_obj.get('id')})는 손 관련 클래스이므로 레퍼런스로 고려하지 않습니다.")
                    distance_confidence = 0.0 # 신뢰도 0으로 설정
                # --- 손 오인식 방지 로직 끝 ---
                
                # 이미 더 높은 신뢰도의 레퍼런스가 있는지 확인
                if distance_confidence > result["confidence"]:
                    result["reference_object"] = closest_obj
                    result["confidence"] = distance_confidence
                    self.logger.info(f"시각 분석 결과 (근접 객체): 레퍼런스 후보='{closest_obj.get('class_name')}'(ID:{closest_obj.get('id')}), 신뢰도={distance_confidence:.2f}")
                    
                    # 공간 관계로부터 방향 추출
                    if "depth" in spatial_relation and spatial_relation["depth"]["relation"] != "same":
                        result["direction"] = spatial_relation["depth"]["relation"]
                    elif "horizontal" in spatial_relation:
                        result["direction"] = spatial_relation["horizontal"]["relation"]
                    elif "vertical" in spatial_relation:
                        result["direction"] = spatial_relation["vertical"]["relation"]
        
        # 3. 시선 추적 (향후 확장)
        # 현재는 구현하지 않음
        
        return result
    
    def _integrated_decision(self, linguistic_result: Dict, 
                           visual_result: Dict) -> Dict:
        """
        언어적 분석과 시각적 분석 결과를 통합하여 최종 결정
        
        Args:
            linguistic_result: 언어적 분석 결과
            visual_result: 시각적 분석 결과
            
        Returns:
            Dict: 통합된 결정 결과
        """
        self.logger.debug(f"통합 결정 시작: 언어 결과={linguistic_result}, 시각 결과={visual_result}")
        
        # 1. 언어적 분석 결과가 매우 확실한 경우 (HIGH_CONFIDENCE 이상)
        #   - 시각적 분석 결과가 손이거나 신뢰도가 매우 낮지 않은 이상, 언어적 결과를 우선 사용
        if linguistic_result["confidence"] >= self.HIGH_CONFIDENCE:
            # 시각적 결과가 손이거나 신뢰도가 매우 낮은지 확인
            visual_is_hand = False
            if visual_result["reference_object"]:
                visual_class = visual_result["reference_object"].get("class_name", "").lower()
                visual_is_hand = visual_class in self.HAND_CLASSES or 'hand' in visual_class
            
            if not (visual_is_hand and visual_result["confidence"] > self.LOW_CONFIDENCE):
                self.logger.info(f"통합 결정: 언어 분석 신뢰도 높음 ({linguistic_result['confidence']:.2f}). 언어 분석 결과 사용.")
                return linguistic_result
            else:
                self.logger.info(f"통합 결정: 언어 분석 신뢰도 높지만, 시각 분석 결과가 신뢰도 높은 손({visual_result['confidence']:.2f}). 추가 검토 필요.")
                # 이 경우는 드물지만, 일단 언어 결과를 반환 (추후 정교화 가능)
                return linguistic_result

        # 2. 시각적 분석 결과가 매우 확실하고 손이 아닌 경우
        visual_is_hand_high_conf = False
        if visual_result["reference_object"]:
            visual_class_high = visual_result["reference_object"].get("class_name", "").lower()
            visual_is_hand_high_conf = visual_class_high in self.HAND_CLASSES or 'hand' in visual_class_high
            
        if visual_result["confidence"] >= self.HIGH_CONFIDENCE and not visual_is_hand_high_conf:
            # 언어적 결과가 없거나 신뢰도가 매우 낮으면 시각적 결과 사용
            if linguistic_result["confidence"] < self.LOW_CONFIDENCE:
                self.logger.info(f"통합 결정: 시각 분석 신뢰도 높고 손 아님 ({visual_result['confidence']:.2f}), 언어 분석 신뢰도 낮음. 시각 분석 결과 사용.")
                return visual_result
            else:
                # 언어 결과도 어느정도 신뢰도 있으면, 아래 로직에서 처리 (여기서는 시각 우선하지 않음)
                self.logger.debug("통합 결정: 시각 분석 신뢰도 높지만, 언어 분석도 유효하여 아래에서 처리.")

        # 3. 두 분석 결과가 일치하고 손이 아닌 경우
        if (linguistic_result["reference_object"] and 
            linguistic_result["reference_object"] == visual_result["reference_object"] and
            not visual_is_hand_high_conf): # 시각 결과가 손이 아닌 경우만 해당
            # 신뢰도는 두 분석의 평균
            combined_confidence = (linguistic_result["confidence"] + 
                                 visual_result["confidence"]) / 2
            # 방향은 언어적 분석 우선
            direction = linguistic_result["direction"] or visual_result["direction"]
            
            self.logger.info(f"통합 결정: 언어/시각 분석 결과 일치 (객체 ID: {linguistic_result['reference_object'].get('id')}). 신뢰도={combined_confidence:.2f}")
            return {
                "reference_object": linguistic_result["reference_object"],
                "confidence": combined_confidence,
                "direction": direction
            }
        
        # 4. 언어적 분석 결과 신뢰도가 중간 이상인 경우 우선 고려
        if linguistic_result["confidence"] >= self.MID_CONFIDENCE:
            self.logger.info(f"통합 결정: 언어 분석 신뢰도 중간 이상 ({linguistic_result['confidence']:.2f}). 언어 분석 결과 우선 고려.")
            # 시각적 결과가 손이 아니고 더 확실하지 않으면 언어적 결과 사용
            if not (visual_result["confidence"] > linguistic_result["confidence"] and not visual_is_hand_high_conf):
                return linguistic_result
            else: # 시각적 결과가 더 확실하고 손이 아니면 시각적 결과 사용 (아래 로직에서 처리됨)
                self.logger.debug("통합 결정: 언어 신뢰도 중간 이상이나, 시각 결과가 더 확실하여 아래에서 처리.")

        # 5. 시각적 분석 결과 신뢰도가 중간 이상이고 손이 아닌 경우
        if visual_result["confidence"] >= self.MID_CONFIDENCE and not visual_is_hand_high_conf:
            self.logger.info(f"통합 결정: 시각 분석 신뢰도 중간 이상 ({visual_result['confidence']:.2f})이고 손 아님. 시각 분석 결과 사용.")
            return visual_result
            
        # 6. 둘 다 신뢰도가 낮거나 애매한 경우: 언어적 분석 결과를 따르되 신뢰도는 낮춤
        self.logger.info("통합 결정: 언어/시각 분석 모두 신뢰도 낮음. 언어 분석 결과 사용 (신뢰도 조정).")
        # 언어적 결과 반환하되, 신뢰도 조정 (예: 원래 신뢰도 * 0.5)
        linguistic_result["confidence"] *= 0.5 
        return linguistic_result
    
    def _extract_preposition_relation(self, prompt: str, 
                                     detected_objects: List[Dict]) -> Dict:
        """
        프롬프트에서 전치사구를 분석하여 레퍼런스 객체와 방향 식별
        예: "컵을 책상 위에 놓아줘" -> 책상(레퍼런스), 위(방향)
        
        Args:
            prompt: 사용자 프롬프트
            detected_objects: 감지된 객체 목록
            
        Returns:
            Dict: 추출된 레퍼런스 객체와 방향
        """
        result = {
            "reference_object": None,
            "confidence": 0.0,
            "direction": None
        }
        self.logger.debug(f"언어 분석 (전치사구): 프롬프트='{prompt}'")
        
        # 객체 클래스 이름 목록 추출
        object_names = [obj["class_name"].lower() for obj in detected_objects if "class_name" in obj]
        if not object_names:
            return result
        
        # 1. 한국어 전치사 패턴 매칭
        for prep, info in self.preposition_patterns.items():
            # 문자열 연결 방식으로 패턴 생성
            pattern_str = r'(' + '|'.join(re.escape(name) for name in object_names) + r')\s+(' + re.escape(prep) + r')(\b|$)'
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                match = pattern.search(prompt)
            except re.error as e:
                self.logger.error(f"한국어 전치사 패턴 컴파일 오류: {e} (패턴: {pattern_str})")
                continue # 오류 발생 시 다음 전치사로 이동
        
        if match:
            ref_name = match.group(1).strip()
            matched_prep = match.group(2).strip()
            self.logger.debug(f"언어 분석 (전치사구): 패턴 '{matched_prep}' 매칭됨, 레퍼런스 후보 '{ref_name}'")
            
            # 객체 이름으로 실제 객체 찾기
            reference_obj = next((obj for obj in detected_objects if obj.get("class_name", "").lower() == ref_name), None)
            
            if reference_obj:
                self.logger.info(f"언어 분석 (전치사구): 레퍼런스 객체 '{ref_name}'(ID:{reference_obj.get('id')}) 식별됨, 방향 '{info['direction']}'")
                return {
                    "reference_object": reference_obj,
                    "confidence": self.HIGH_CONFIDENCE * info['weight'], # 패턴 가중치 적용
                    "direction": info['direction']
                }
        
        # 2. 영어 전치사 패턴 매칭 (동일 로직 적용)
        for prep, info in self.en_preposition_patterns.items():
            # 문자열 연결 방식으로 패턴 생성
            pattern_str_en = r'(' + '|'.join(re.escape(name) for name in object_names) + r')\s+(' + re.escape(prep) + r')(\b|$)'
            try:
                pattern_en = re.compile(pattern_str_en, re.IGNORECASE)
                match_en = pattern_en.search(prompt)
            except re.error as e:
                self.logger.error(f"영어 전치사 패턴 컴파일 오류: {e} (패턴: {pattern_str_en})")
                continue # 오류 발생 시 다음 전치사로 이동
                
            if match_en:
                ref_name_en = match_en.group(1).strip()
                matched_prep_en = match_en.group(2).strip()
                self.logger.debug(f"언어 분석 (전치사구): 영어 패턴 '{matched_prep_en}' 매칭됨, 레퍼런스 후보 '{ref_name_en}'")
                reference_obj_en = next((obj for obj in detected_objects if obj.get("class_name", "").lower() == ref_name_en), None)
                if reference_obj_en:
                    self.logger.info(f"언어 분석 (전치사구): 영어 레퍼런스 객체 '{ref_name_en}'(ID:{reference_obj_en.get('id')}) 식별됨, 방향 '{info['direction']}'")
                    return {
                        "reference_object": reference_obj_en,
                        "confidence": self.HIGH_CONFIDENCE * info['weight'],
                        "direction": info['direction']
                    }
        
        self.logger.debug("언어 분석 (전치사구): 매칭되는 패턴 없음")
        return result
    
    def _extract_direction_noun_relation(self, prompt: str, 
                                       detected_objects: List[Dict]) -> Dict:
        """
        프롬프트에서 방향어와 명사 조합을 분석하여 레퍼런스 객체 식별
        예: "컵 왼쪽으로"
        
        Args:
            prompt: 사용자 프롬프트
            detected_objects: 감지된 객체 목록
            
        Returns:
            Dict: 추출된 레퍼런스 객체와 방향
        """
        result = {
            "reference_object": None,
            "confidence": 0.0,
            "direction": None
        }
        self.logger.debug(f"언어 분석 (방향어+명사): 프롬프트='{prompt}'")
        
        # 객체 이름 목록
        object_names = [obj["class_name"].lower() for obj in detected_objects if "class_name" in obj]
        if not object_names:
            return result
        
        # 패턴: 명사 + (공백) + 방향어(쪽/으로/에 등 포함)
        # 예: bottle (왼쪽|왼쪽으로|왼쪽에)
        direction_keywords = ["앞", "뒤", "위", "아래", "왼쪽", "오른쪽", "옆"]
        pattern_str = r'(' + '|'.join(re.escape(name) for name in object_names) + r')\s*(' + '|'.join(direction_keywords) + r')(?:쪽|으로|에)?(\b|$)'
        pattern = re.compile(pattern_str, re.IGNORECASE)
        match = pattern.search(prompt)
        
        if match:
            ref_name = match.group(1).strip()
            direction_keyword = match.group(2).strip()
            self.logger.debug(f"언어 분석 (방향어+명사): 패턴 매칭됨, 레퍼런스 후보 '{ref_name}', 방향 키워드 '{direction_keyword}'")
            
            reference_obj = next((obj for obj in detected_objects if obj.get("class_name", "").lower() == ref_name), None)
            
            if reference_obj:
                # 방향 키워드를 표준 방향으로 변환
                direction = None
                if direction_keyword == "앞": direction = "front"
                elif direction_keyword == "뒤": direction = "back"
                elif direction_keyword == "위": direction = "above"
                elif direction_keyword == "아래": direction = "below"
                elif direction_keyword == "왼쪽": direction = "left"
                elif direction_keyword == "오른쪽": direction = "right"
                # "옆"은 명확한 방향 없음
                
                self.logger.info(f"언어 분석 (방향어+명사): 레퍼런스 객체 '{ref_name}'(ID:{reference_obj.get('id')}) 식별됨, 방향 '{direction}'")
                return {
                    "reference_object": reference_obj,
                    "confidence": self.HIGH_CONFIDENCE * 0.9, # 전치사구보다 약간 낮은 신뢰도
                    "direction": direction
                }
        
        self.logger.debug("언어 분석 (방향어+명사): 매칭되는 패턴 없음")
        return result
    
    def _extract_deictic_relation(self, prompt: str) -> Dict:
        """
        지시대명사와 방향어 조합 분석 (예: "이거 오른쪽")
        
        Args:
            prompt: 사용자 프롬프트
            
        Returns:
            Dict: 추출된 방향 정보
        """
        result = {"direction": None, "confidence": 0.0}
        self.logger.debug(f"언어 분석 (지시대명사+방향어): 프롬프트='{prompt}'")

        # 지시대명사 패턴 (한국어, 영어)
        deictic_ko = r"(이거|저거|그거|이것|저것|그것|여기|저기|거기|이|저|그)"
        deictic_en = r"\b(this|that|these|those|here|there)\b"
        
        # 방향어 패턴
        direction_keywords = ["앞", "뒤", "위", "아래", "왼쪽", "오른쪽", "옆"]
        direction_pattern_ko = r'(' + '|'.join(direction_keywords) + r')(?:쪽|으로|에)?'
        direction_pattern_en = r"\b(front|back|behind|above|over|below|under|left|right|next to|near|beside)\b"

        # 패턴: 지시대명사 + (공백) + 방향어
        pattern_ko = re.compile(deictic_ko + r'\s*' + direction_pattern_ko + r'(\b|$)', re.IGNORECASE)
        pattern_en = re.compile(deictic_en + r'\s*' + direction_pattern_en + r'(\b|$)', re.IGNORECASE)

        match_ko = pattern_ko.search(prompt)
        match_en = pattern_en.search(prompt)

        direction_keyword = None
        if match_ko:
            direction_keyword = match_ko.group(2).strip().replace("쪽", "").replace("으로", "").replace("에", "")
            self.logger.debug(f"언어 분석 (지시대명사+방향어): 한국어 패턴 매칭됨, 방향 키워드 '{direction_keyword}'")
        elif match_en:
            direction_keyword = match_en.group(2).strip()
            self.logger.debug(f"언어 분석 (지시대명사+방향어): 영어 패턴 매칭됨, 방향 키워드 '{direction_keyword}'")

        if direction_keyword:
            # 방향 키워드를 표준 방향으로 변환
            direction = None
            if direction_keyword in ["앞", "front"]: direction = "front"
            elif direction_keyword in ["뒤", "back", "behind"]: direction = "back"
            elif direction_keyword in ["위", "above", "over"]: direction = "above"
            elif direction_keyword in ["아래", "below", "under"]: direction = "below"
            elif direction_keyword == "왼쪽" or "left" in direction_keyword: direction = "left"
            elif direction_keyword == "오른쪽" or "right" in direction_keyword: direction = "right"
            
            if direction:
                self.logger.info(f"언어 분석 (지시대명사+방향어): 방향 '{direction}' 식별됨")
                return {"direction": direction, "confidence": self.MID_CONFIDENCE} # 중간 신뢰도

        self.logger.debug("언어 분석 (지시대명사+방향어): 매칭되는 패턴 없음")
        return result
    
    def _is_single_object_command(self, prompt: str) -> bool:
        """
        프롬프트가 단일 객체에 대한 명령인지 (레퍼런스 없이) 확인
        예: "컵 들어줘"
        
        Args:
            prompt: 사용자 프롬프트
            
        Returns:
            bool: 단일 목적어 명령 여부
        """
        # 전치사나 방향어가 없으면 단일 객체 명령으로 간주 (간단한 휴리스틱)
        has_preposition = any(prep in prompt for prep in list(self.preposition_patterns.keys()) + list(self.en_preposition_patterns.keys()))
        has_direction = self.direction_pattern.search(prompt) is not None
        
        is_single = not has_preposition and not has_direction
        self.logger.debug(f"언어 분석 (단일 객체 명령 확인): 결과={is_single}")
        return is_single
    
    def _find_object_by_name(self, name: str, object_classes: Dict[str, int]) -> Optional[int]:
        """
        객체 이름으로 클래스 ID 찾기 (대소문자 무시)
        
        Args:
            name: 객체 이름
            object_classes: 객체 클래스 매핑
            
        Returns:
            int|None: 찾은 객체 인덱스 또는 None
        """
        name_lower = name.lower()
        for class_name, class_id in object_classes.items():
            if class_name.lower() == name_lower:
                return class_id
        return None
    
    def _infer_direction_from_objects(self, target_obj: Dict, 
                                    reference_obj: Dict) -> Optional[str]:
        """
        두 객체의 상대적 위치로부터 방향 추론 (2D bbox 기준)
        
        Args:
            target_obj: 타겟 객체 정보
            reference_obj: 레퍼런스 객체 정보
            
        Returns:
            str: 방향 상수 또는 None
        """
        if not target_obj or not reference_obj:
            return None
        
        # 중심점 계산
        t_bbox = target_obj.get("bbox", [0,0,0,0])
        r_bbox = reference_obj.get("bbox", [0,0,0,0])
        t_center = np.array([(t_bbox[0] + t_bbox[2]) / 2, (t_bbox[1] + t_bbox[3]) / 2])
        r_center = np.array([(r_bbox[0] + r_bbox[2]) / 2, (r_bbox[1] + r_bbox[3]) / 2])
        
        # 벡터 계산 (레퍼런스 -> 타겟)
        vector = t_center - r_center
        
        if np.linalg.norm(vector) < 1e-6: # 두 객체가 거의 같은 위치
            return None
            
        angle = math.degrees(math.atan2(vector[1], vector[0])) # Y축 반전 고려 안 함 (이미지 좌표계)

        # 각도에 따른 방향 결정 (간단화된 버전)
        if -45 <= angle < 45: return "right"
        elif 45 <= angle < 135: return "below" # 이미지 Y축 아래쪽
        elif abs(angle) >= 135: return "left"
        elif -135 <= angle < -45: return "above" # 이미지 Y축 위쪽
            
        return None # Should not happen 