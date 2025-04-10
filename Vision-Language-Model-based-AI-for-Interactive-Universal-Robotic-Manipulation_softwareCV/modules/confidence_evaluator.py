import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math

class ConfidenceEvaluator:
    """신뢰도 평가 및 다중 후보 관리 클래스"""
    
    def __init__(self, logger=None):
        """
        ConfidenceEvaluator 초기화
        
        Args:
            logger: 로깅을 위한 로거 객체. None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("ConfidenceEvaluator")
        
        # 신뢰도 수준 정의
        self.CONFIDENCE_HIGH = 0.8
        self.CONFIDENCE_MID = 0.5
        self.CONFIDENCE_LOW = 0.3
        self.CONFIDENCE_VERY_LOW = 0.0
        
        # 후보 점수 계산을 위한 가중치
        self.WEIGHTS = {
            "linguistic_match": 0.3,  # 언어적 매칭 점수
            "spatial_proximity": 0.25,  # 공간적 근접성
            "gesture_alignment": 0.25,  # 제스처 일치도
            "historical_consistency": 0.2  # 과거 패턴 일관성
        }
        
        # 대안 간 최소 점수 차이 임계값
        self.ALTERNATIVES_THRESHOLD = 0.15
        
        self.logger.info("ConfidenceEvaluator 초기화 완료")
    
    def evaluate_confidence(self, target_obj: Optional[Dict], 
                           reference_obj: Optional[Dict], 
                           direction: Optional[str], 
                           prompt: Optional[str]) -> float:
        """
        추론 결과의 신뢰도 평가
        
        Args:
            target_obj: 타겟 객체 정보
            reference_obj: 레퍼런스 객체 정보
            direction: 방향 정보
            prompt: 사용자 프롬프트
            
        Returns:
            float: 0-1 사이의 신뢰도 점수
        """
        # 초기 신뢰도
        confidence = 0.5
        
        # 필수 객체 검증
        if not target_obj:
            self.logger.warning("타겟 객체가 없습니다.")
            return max(0.1, confidence - 0.4)
        
        # 객체 인식 신뢰도 반영
        target_conf = target_obj.get("confidence", 0.5)
        confidence *= (0.5 + 0.5 * target_conf)  # 객체 신뢰도 반영 (50% 이상 보장)
        
        # 레퍼런스 객체가 있는 경우
        if reference_obj:
            # 레퍼런스 객체 신뢰도 반영
            ref_conf = reference_obj.get("confidence", 0.5)
            confidence *= (0.7 + 0.3 * ref_conf)
            
            # 방향 정보 검증
            if not direction:
                self.logger.warning("레퍼런스 객체는 있으나 방향 정보가 없습니다.")
                confidence *= 0.8  # 방향 정보 없을 경우 약간 감소
        
        # 레퍼런스 객체가 없는 경우
        else:
            # 사용자 프롬프트에 지시대명사 확인
            if prompt and any(word in prompt.lower() for word in ["이거", "저거", "그거", "여기", "저기", "거기", "this", "that", "here", "there"]):
                # 지시대명사가 있으나 레퍼런스가 없는 경우, 제스처 필요
                if not any(word in prompt.lower() for word in ["앞", "뒤", "위", "아래", "왼쪽", "오른쪽"]):
                    self.logger.warning("지시대명사는 있으나 레퍼런스 객체와 방향 정보가 없습니다.")
                    confidence *= 0.7  # 신뢰도 감소
        
        # 최종 신뢰도 조정
        confidence = min(1.0, max(0.0, confidence))
        
        self.logger.info(f"최종 신뢰도 평가: {confidence:.2f}")
        return confidence
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        신뢰도 수준 분류
        
        Args:
            confidence: 신뢰도 점수 (0-1)
            
        Returns:
            str: 신뢰도 수준 ("high", "mid", "low", "very_low")
        """
        if confidence >= self.CONFIDENCE_HIGH:
            return "high"
        elif confidence >= self.CONFIDENCE_MID:
            return "mid"
        elif confidence >= self.CONFIDENCE_LOW:
            return "low"
        else:
            return "very_low"
    
    def get_action_for_confidence(self, confidence: float) -> Dict[str, Any]:
        """
        신뢰도 수준에 따른 권장 조치
        
        Args:
            confidence: 신뢰도 점수 (0-1)
            
        Returns:
            Dict: 조치 정보 {
                "action_type": 조치 유형,
                "description": 설명,
                "requires_feedback": 사용자 확인 필요 여부
            }
        """
        confidence_level = self.get_confidence_level(confidence)
        
        if confidence_level == "high":
            return {
                "action_type": "auto_execute",
                "description": "자동 수행",
                "requires_feedback": False
            }
        elif confidence_level == "mid":
            return {
                "action_type": "visual_confirm",
                "description": "시각적 피드백 제공 후 수행",
                "requires_feedback": False
            }
        elif confidence_level == "low":
            return {
                "action_type": "user_confirm",
                "description": "사용자에게 확인 요청",
                "requires_feedback": True
            }
        else:  # very_low
            return {
                "action_type": "suggest_alternatives",
                "description": "대안 제시 또는 재요청",
                "requires_feedback": True
            }
    
    def generate_alternatives(self, primary_result: Dict[str, Any], 
                            detected_objects: List[Dict]) -> List[Dict[str, Any]]:
        """
        대안 후보 생성
        
        Args:
            primary_result: 주요 추론 결과
            detected_objects: 감지된 객체 목록
            
        Returns:
            List[Dict]: 대안 결과 목록 (신뢰도 포함)
        """
        alternatives = []
        
        # 주요 결과가 없거나 충분한 객체가 없는 경우
        if not primary_result or len(detected_objects) < 2:
            return alternatives
        
        # 주요 결과 복사
        primary_target = primary_result.get("target_object")
        primary_reference = primary_result.get("reference_object")
        primary_direction = primary_result.get("direction")
        
        # 타겟 객체가 없는 경우, 대안 생성 불가
        if not primary_target:
            return alternatives
        
        # 1. 다른 레퍼런스 객체를 사용한 대안
        if primary_reference:
            for obj in detected_objects:
                # 자기 자신이나 주 레퍼런스는 제외
                if obj == primary_target or obj == primary_reference:
                    continue
                
                # 새로운 레퍼런스로 대안 생성
                alt_result = self._create_alternative_with_reference(
                    primary_target, obj, primary_direction
                )
                alternatives.append(alt_result)
        
        # 2. 방향 변경 대안
        if primary_reference and primary_direction:
            alt_directions = self._get_alternative_directions(primary_direction)
            
            for direction in alt_directions:
                alt_result = self._create_alternative_with_direction(
                    primary_target, primary_reference, direction
                )
                alternatives.append(alt_result)
        
        # 3. 레퍼런스 없는 대안
        if primary_reference:
            alt_result = self._create_alternative_without_reference(
                primary_target
            )
            alternatives.append(alt_result)
        
        # 4. 점수 기반 필터링 및 정렬
        scored_alternatives = []
        primary_score = self._calculate_alternative_score(primary_result)
        
        for alt in alternatives:
            alt_score = self._calculate_alternative_score(alt)
            alt["score"] = alt_score
            
            # 점수 차이가 임계값보다 작을 경우만 포함
            if primary_score - alt_score < self.ALTERNATIVES_THRESHOLD:
                scored_alternatives.append(alt)
        
        # 5. 점수 기준 정렬
        sorted_alternatives = sorted(
            scored_alternatives, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        # 6. 최대 3개까지 반환
        return sorted_alternatives[:3]
    
    def _create_alternative_with_reference(self, target_obj: Dict, 
                                         reference_obj: Dict, 
                                         direction: Optional[str]) -> Dict[str, Any]:
        """
        다른 레퍼런스 객체를 사용한 대안 생성
        
        Args:
            target_obj: 타겟 객체
            reference_obj: 새 레퍼런스 객체
            direction: 방향
            
        Returns:
            Dict: 대안 결과
        """
        alternative = {
            "target_object": target_obj,
            "reference_object": reference_obj,
            "direction": direction,
            "confidence": 0.0,  # 초기값, 후에 계산
            "alternative_type": "different_reference"
        }
        
        # 새 레퍼런스에 적합한 방향이 없으면 유추
        if not direction:
            # 간단한 기본 방향 (실제로는 더 정교한 방향 유추 필요)
            alternative["direction"] = "front"  # 임시 방향
        
        return alternative
    
    def _create_alternative_with_direction(self, target_obj: Dict, 
                                         reference_obj: Dict, 
                                         direction: str) -> Dict[str, Any]:
        """
        다른 방향을 사용한 대안 생성
        
        Args:
            target_obj: 타겟 객체
            reference_obj: 레퍼런스 객체
            direction: 새 방향
            
        Returns:
            Dict: 대안 결과
        """
        return {
            "target_object": target_obj,
            "reference_object": reference_obj,
            "direction": direction,
            "confidence": 0.0,  # 초기값, 후에 계산
            "alternative_type": "different_direction"
        }
    
    def _create_alternative_without_reference(self, target_obj: Dict) -> Dict[str, Any]:
        """
        레퍼런스 없는 대안 생성
        
        Args:
            target_obj: 타겟 객체
            
        Returns:
            Dict: 대안 결과
        """
        return {
            "target_object": target_obj,
            "reference_object": None,
            "direction": None,
            "confidence": 0.0,  # 초기값, 후에 계산
            "alternative_type": "no_reference"
        }
    
    def _get_alternative_directions(self, current_direction: str) -> List[str]:
        """
        대체 방향 목록 반환
        
        Args:
            current_direction: 현재 방향
            
        Returns:
            List[str]: 대안 방향 목록
        """
        all_directions = ["front", "back", "left", "right", "above", "below"]
        
        # 현재 방향의 반대 방향 찾기
        opposite_direction = None
        if current_direction == "front":
            opposite_direction = "back"
        elif current_direction == "back":
            opposite_direction = "front"
        elif current_direction == "left":
            opposite_direction = "right"
        elif current_direction == "right":
            opposite_direction = "left"
        elif current_direction == "above":
            opposite_direction = "below"
        elif current_direction == "below":
            opposite_direction = "above"
        
        if opposite_direction:
            alternatives = [opposite_direction]
            for direction in all_directions:
                if direction != current_direction and direction != opposite_direction:
                    alternatives.append(direction)
            return alternatives
        else:
            return [d for d in all_directions if d != current_direction]
    
    def _calculate_alternative_score(self, alternative: Dict[str, Any]) -> float:
        """
        대안의 점수 계산
        
        Args:
            alternative: 대안 결과
            
        Returns:
            float: 점수 (0-1)
        """
        # 기본 점수
        score = 0.5
        
        # 객체 인식 신뢰도 반영
        target_obj = alternative.get("target_object")
        if target_obj:
            target_conf = target_obj.get("confidence", 0.5)
            score *= (0.5 + 0.5 * target_conf)
        else:
            return 0.0  # 타겟 없으면 0점
        
        # 레퍼런스 존재 여부 및 신뢰도 반영
        ref_obj = alternative.get("reference_object")
        direction = alternative.get("direction")
        
        if ref_obj:
            # 레퍼런스 신뢰도
            ref_conf = ref_obj.get("confidence", 0.5)
            score *= (0.7 + 0.3 * ref_conf)
            
            # 방향 정보 유무
            if not direction:
                score *= 0.8
        else:
            # 레퍼런스 없는 경우, 낮은 점수
            score *= 0.7
        
        # 대안 유형별 추가 조정
        alt_type = alternative.get("alternative_type")
        if alt_type == "different_reference":
            # 다른 레퍼런스 사용 대안
            score *= 0.9
        elif alt_type == "different_direction":
            # 다른 방향 사용 대안
            score *= 0.85
        elif alt_type == "no_reference":
            # 레퍼런스 없는 대안
            score *= 0.7
        
        return min(1.0, max(0.0, score))
    
    def provide_feedback(self, result: Dict[str, Any], 
                        visualizer=None) -> Dict[str, Any]:
        """
        결과에 대한 피드백 처리
        
        Args:
            result: 추론 결과
            visualizer: 시각화 엔진 (옵션)
            
        Returns:
            Dict: 피드백 처리 결과
        """
        confidence = result.get("confidence", 0.0)
        confidence_action = self.get_action_for_confidence(confidence)
        
        feedback_result = {
            "original_result": result,
            "action": confidence_action,
            "requires_user_confirmation": confidence_action["requires_feedback"],
            "visualization": None
        }
        
        # 시각화 필요 시 생성
        if visualizer and (confidence_action["action_type"] == "visual_confirm" or 
                         confidence_action["action_type"] == "user_confirm"):
            # 시각화 생성 (실제 구현에 맞게 조정 필요)
            visualization = None
            try:
                # 예시: 시각화 함수 호출
                visualization = visualizer.visualize_inference_result(result)
            except Exception as e:
                self.logger.error(f"시각화 생성 중 오류 발생: {e}")
            
            feedback_result["visualization"] = visualization
        
        # 대안 필요 시 추가
        if confidence_action["action_type"] == "suggest_alternatives":
            alternatives = self.generate_alternatives(
                result, result.get("detected_objects", [])
            )
            feedback_result["alternatives"] = alternatives
        
        return feedback_result 