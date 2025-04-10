"""
결과 데이터 통합 및 생성 모듈

이 모듈은 결과 데이터 통합 및 생성을 담당하는 ResultAggregator 클래스를 포함합니다.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from ..result_storage import ResultStorage
from ..utils.geometry import calculate_iou

class ResultAggregator:
    """
    결과 데이터 통합 및 생성 클래스
    
    이 클래스는 다음과 같은 결과 관련 기능을 담당합니다:
    - 다양한 소스에서 얻은 결과 데이터 통합
    - 최종 결과 생성 및 저장
    - 결과 데이터 형식 변환 및 검증
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        ResultAggregator 초기화
        
        Args:
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 결과 저장소 초기화
        self.logger.info("ResultStorage 초기화 중...")
        try:
            self.result_storage = ResultStorage(logger=self.logger)
            self.logger.info("ResultStorage 초기화 완료")
        except Exception as e:
            self.logger.error(f"ResultStorage 초기화 실패: {e}")
            self.result_storage = None
    
    def aggregate_results(self,
                         target_object: Optional[Dict[str, Any]] = None,
                         reference_object: Optional[Dict[str, Any]] = None,
                         goal_point_result: Optional[Dict[str, Any]] = None,
                         target_inference_result: Optional[Dict[str, Any]] = None,
                         gesture_mode: bool = False,
                         user_prompt: Optional[str] = None,
                         processing_times: Optional[Dict[str, float]] = None,
                         detections: Optional[List[Dict[str, Any]]] = None,
                         img_width: Optional[int] = None,
                         img_height: Optional[int] = None) -> Dict[str, Any]:
        """
        결과 데이터 통합
        
        Args:
            target_object: 타겟 객체 정보
            reference_object: 레퍼런스 객체 정보
            goal_point_result: 목표 위치 결과
            target_inference_result: 타겟 추론 결과
            gesture_mode: 제스처 모드 여부
            user_prompt: 사용자 프롬프트 (텍스트)
            processing_times: 단계별 처리 시간
            detections: 감지된 객체 목록
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            Dict[str, Any]: 통합된 결과 데이터
        """
        # 기본 결과 구조 생성
        result = {
            "timestamp": time.time(),
            "success": False,
            "gesture_mode": gesture_mode,
            "user_prompt": user_prompt,
            "method": "unknown",
            "processing_times": processing_times or {},
            "image_info": {
                "width": img_width,
                "height": img_height
            }
        }
        
        # 타겟/레퍼런스 객체 정보 추가
        if target_object:
            result["target_object"] = self._extract_object_info(target_object)
            result["success"] = True
        
        if reference_object:
            result["reference_object"] = self._extract_object_info(reference_object)
        
        # 타겟 추론 결과 추가
        if target_inference_result:
            # 기존 데이터 보존을 위해 단순 업데이트
            for key, value in target_inference_result.items():
                if key not in result:
                    result[key] = value
            
            if "method" in target_inference_result:
                result["method"] = target_inference_result["method"]
        
        # 목표 위치 결과 추가
        if goal_point_result:
            if "goal_point" in goal_point_result:
                result["goal_point"] = goal_point_result["goal_point"]
            
            if "direction" in goal_point_result:
                result["direction"] = goal_point_result["direction"]
            
            if "method" in goal_point_result:
                # 기존 방법이 unknown인 경우에만 업데이트
                if result["method"] == "unknown":
                    result["method"] = goal_point_result["method"]
            
            # 겹침 정보 복사
            for key in ["overlap_iou", "has_overlap"]:
                if key in goal_point_result:
                    result[key] = goal_point_result[key]
        
        # 감지된 객체 목록 (옵션)
        if detections:
            result["detections_count"] = len(detections)
        
        # 결과 저장
        self._save_result(result)
        
        return result
    
    def _extract_object_info(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        객체 정보에서 필요한 부분만 추출
        
        Args:
            obj: 객체 전체 정보
            
        Returns:
            Dict[str, Any]: 핵심 객체 정보
        """
        # 기본 필드
        result = {
            "id": obj.get("id", 0),
            "class_name": obj.get("class_name", "unknown"),
            "confidence": obj.get("confidence", 0.0)
        }
        
        # 바운딩 박스
        for key in ["bbox", "box"]:
            if key in obj:
                result["bbox"] = obj[key]
                break
        
        # 3D 좌표
        if "3d_coords" in obj:
            result["3d_coords"] = obj["3d_coords"]
        
        return result
    
    def _save_result(self, result: Dict[str, Any]) -> bool:
        """
        결과 저장
        
        Args:
            result: 저장할 결과 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.result_storage:
            self.logger.warning("ResultStorage가 초기화되지 않아 결과를 저장할 수 없습니다.")
            return False
        
        try:
            self.result_storage.store_result(result)
            return True
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {e}")
            return False
    
    def evaluate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 평가
        
        Args:
            result: 평가할 결과 데이터
            
        Returns:
            Dict[str, Any]: 평가 결과가 추가된 데이터
        """
        evaluated_result = result.copy()
        
        # 성공 여부 확인
        success = result.get("success", False)
        
        # 타겟/레퍼런스 객체 존재 여부
        has_target = "target_object" in result
        has_reference = "reference_object" in result
        
        # 목표 위치 존재 여부
        has_goal_point = "goal_point" in result
        
        # 방향 정보 존재 여부
        has_direction = "direction" in result
        
        # 겹침 여부
        has_overlap = result.get("has_overlap", False)
        
        # 평가 결과 계산
        if not success:
            confidence = 0.0
            status = "failed"
        elif not has_target or not has_reference:
            confidence = 0.2
            status = "incomplete"
        elif not has_goal_point:
            confidence = 0.3
            status = "incomplete"
        elif has_overlap:
            confidence = 0.7
            status = "partial" 
        else:
            confidence = 0.9
            status = "success"
        
        # 평가 결과 추가
        evaluated_result["evaluation"] = {
            "status": status,
            "confidence": confidence,
            "has_target": has_target,
            "has_reference": has_reference,
            "has_goal_point": has_goal_point,
            "has_direction": has_direction,
            "has_overlap": has_overlap
        }
        
        return evaluated_result
    
    def clear_storage(self) -> bool:
        """
        결과 저장소 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        if not self.result_storage:
            self.logger.warning("ResultStorage가 초기화되지 않아 초기화할 수 없습니다.")
            return False
        
        try:
            self.result_storage.clear()
            return True
        except Exception as e:
            self.logger.error(f"저장소 초기화 중 오류: {e}")
            return False 