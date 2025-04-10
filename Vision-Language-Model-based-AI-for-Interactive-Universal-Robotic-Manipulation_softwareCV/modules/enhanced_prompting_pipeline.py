#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EnhancedPromptingPipeline 모듈

3단계 LLM 호출을 통해 타겟/레퍼런스/목표 지점을 추론하는 향상된 프롬프팅 파이프라인
1단계: 지시대명사 해석 (이것, 저것 등을 구체적인 객체로 변환)
2단계: 사용자 명령 구체화 (지시대명사가 해석된 후 명확한 명령으로 변환)
3단계: 타겟/레퍼런스/목표 지점 추론 (구체화된 명령을 기반으로 최종 판단)
"""

import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import traceback

class EnhancedPromptingPipeline:
    """
    3단계 LLM 호출을 통해 타겟/레퍼런스/목표 지점을 추론하는 파이프라인
    """
    
    def __init__(self, llm_interface, logger=None):
        """
        EnhancedPromptingPipeline 초기화
        
        Args:
            llm_interface: LLM 인터페이스 객체 (generate_text 메서드 필요)
            logger: 로깅을 위한 로거 객체 (선택 사항)
        """
        self.llm = llm_interface
        self.logger = logger or logging.getLogger(__name__)
        self.default_img_resolution = (1920, 1080)  # 기본 이미지 해상도
        
        # 프롬프트 강화 패턴
        self.demonstrative_pattern = re.compile(r'(이거|저거|그거|이것|저것|그것)\\s*\\(([^)]+)\\)') # 예: 이거(에어팟)
        self.spatial_pattern = re.compile(r'(\S+)(?:을|를|\s+)(\S+)\s+(앞으로|뒤로|위로|아래로|왼쪽으로|오른쪽으로|앞에|뒤에|위에|아래에|왼쪽에|오른쪽에)') # 예: 병 앞으로
        self.direction_map = {
            "앞으로": "front", "앞에": "front",
            "뒤로": "back", "뒤에": "back",
            "위로": "above", "위에": "above",
            "아래로": "below", "아래에": "below",
            "왼쪽으로": "left", "왼쪽에": "left",
            "오른쪽으로": "right", "오른쪽에": "right",
        }
        
    def process(self, user_command: str, detected_objects: List[Dict], 
              target_object_id: Optional[int] = None,
              target_object_name: Optional[str] = None, 
              img_resolution: Optional[Tuple[int, int]] = None, 
              advanced_nlp: bool = False) -> Dict[str, Any]:
        """
        사용자 명령과 감지된 객체를 처리하여 결과를 반환합니다.
        
        Args:
            user_command: 사용자 명령어
            detected_objects: 감지된 객체 목록
            target_object_id: 대상 객체 ID (선택 사항)
            target_object_name: 대상 객체 이름 (선택 사항)
            img_resolution: 이미지 해상도 (width, height) (선택 사항)
            advanced_nlp: 고급 자연어 처리 사용 여부
            
        Returns:
            Dict: 처리 결과
        """
        try:
            self.logger.info(f"프롬프트 처리 시작: '{user_command}'")
            start_time = time.time()
            
            if not img_resolution:
                img_resolution = self.default_img_resolution
                
            if not detected_objects:
                self.logger.warning("감지된 객체가 없습니다.")
                return {
                    "success": False, 
                    "error": "감지된 객체가 없습니다."
                }
                
            # 고급 자연어 처리 사용 시 3단계 파이프라인 실행
            if advanced_nlp:
                self.logger.info("고급 자연어 처리 경로 사용")
                return self.process_natural_language_command(
                    user_command=user_command,
                    detected_objects=detected_objects,
                    target_object_id=target_object_id,
                    target_object_name=target_object_name,
                    img_resolution=img_resolution
                )
            
            # 기존 처리 로직 실행
            self.logger.info("기존 처리 경로 사용")
            
            # 참조 객체 탐지
            reference_detector_result = self.reference_detector.detect(
                command=user_command,
                detected_objects=detected_objects
            )
            
            reference_object = None
            reference_object_id = -1
            
            if reference_detector_result["has_reference"]:
                reference_object = reference_detector_result["reference_object"]
                reference_object_id = reference_object["id"]
                self.logger.info(f"참조 객체 탐지됨: {reference_object['name']} (ID: {reference_object_id})")
            
            # 방향 분석
            direction_info = self.direction_analyzer.analyze(
                command=user_command,
                target_object_id=target_object_id,
                target_object_name=target_object_name,
                reference_object=reference_object
            )
            
            self.logger.info(f"방향 분석 결과: {direction_info}")
            
            # 프롬프트 결과 구성
            direction_type = direction_info["type"]
            direction_value = direction_info["value"]
            
            # 목표 지점 추론
            goal_point_result = {}
            
            # 참조 객체가 있는 경우
            if reference_object:
                self.logger.info(f"참조 객체 기반 목표 지점 추론 시작")
                goal_point_result = self.goal_inference.infer_with_reference(
                    user_command=user_command,
                    reference_object=reference_object,
                    target_object_id=target_object_id,
                    target_object_name=target_object_name,
                    direction_type=direction_type,
                    direction_value=direction_value,
                    img_resolution=img_resolution
                )
            # 참조 객체가 없는 경우
            else:
                self.logger.info(f"참조 객체 없는 목표 지점 추론 시작")
                goal_point_result = self.goal_inference.infer_without_reference(
                    user_command=user_command,
                    target_object_id=target_object_id,
                    target_object_name=target_object_name,
                    direction_type=direction_type,
                    direction_value=direction_value,
                    img_resolution=img_resolution
                )
            
            process_time = time.time() - start_time
            self.logger.info(f"프롬프트 처리 완료 (소요 시간: {process_time:.2f}초)")
            
            # 최종 결과 구성
            result = {
                "success": True,
                "target_object_id": target_object_id if target_object_id is not None else -1,
                "reference_object_id": reference_object_id,
                "direction": {
                    "type": direction_type,
                    "value": direction_value,
                    "confidence": direction_info.get("confidence", 0.5)
                },
                **goal_point_result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"프롬프트 처리 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"success": False, "error": f"프롬프트 처리 오류: {str(e)}"}

    def _contains_demonstrative_pronoun(self, text: str) -> bool:
        """
        텍스트에 지시대명사 포함 여부 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            bool: 지시대명사 포함 여부
        """
        # 한국어 지시대명사 패턴
        patterns = [
            r'\b이것\b', r'\b저것\b', r'\b그것\b',
            r'\b이거\b', r'\b저거\b', r'\b그거\b',
            r'\b여기\b', r'\b저기\b', r'\b거기\b',
            r'\b이쪽\b', r'\b저쪽\b', r'\b그쪽\b',
            # 영어 지시대명사 패턴
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bhere\b', r'\bthere\b', r'\bit\b'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _find_pointed_object(self, gesture_results: List[Dict], detections: List[Dict]) -> Optional[int]:
        """
        제스처 결과를 기반으로 가리키는 객체 식별
        
        Args:
            gesture_results: 제스처 인식 결과
            detections: 감지된 객체 목록
            
        Returns:
            Optional[int]: 가리키는 객체의 ID (없으면 None)
        """
        if not gesture_results or not detections:
            return None
            
        # 제스처 결과에서 POINTING 제스처 찾기
        pointing_gestures = [g for g in gesture_results if g.get("gesture_type") == "POINTING"]
        if not pointing_gestures:
            return None
            
        # 가장 확실한 포인팅 제스처 선택
        pointing_gesture = max(pointing_gestures, key=lambda g: g.get("confidence", 0))
        
        # 제스처 방향 벡터와 객체 사이의 각도 계산
        pointing_vector = pointing_gesture.get("direction_vector")
        pointing_origin = pointing_gesture.get("origin")
        
        if not pointing_vector or not pointing_origin:
            return None
            
        # 가장 작은 각도를 가진 객체 찾기
        min_angle = float('inf')
        best_match_idx = None
        
        for idx, obj in enumerate(detections):
            # 객체 중심점 계산
            bbox = obj.get("bbox", [0, 0, 0, 0])
            obj_center_x = (bbox[0] + bbox[2]) / 2
            obj_center_y = (bbox[1] + bbox[3]) / 2
            
            # 제스처 원점에서 객체 중심까지의 벡터
            object_vector = [obj_center_x - pointing_origin[0], obj_center_y - pointing_origin[1]]
            
            # 벡터 크기 계산
            object_vector_magnitude = np.sqrt(object_vector[0]**2 + object_vector[1]**2)
            pointing_vector_magnitude = np.sqrt(pointing_vector[0]**2 + pointing_vector[1]**2)
            
            # 벡터가 유효한지 확인
            if object_vector_magnitude > 0 and pointing_vector_magnitude > 0:
                # 내적을 사용하여 각도 계산
                dot_product = (object_vector[0] * pointing_vector[0] + 
                              object_vector[1] * pointing_vector[1])
                cos_angle = dot_product / (object_vector_magnitude * pointing_vector_magnitude)
                cos_angle = max(-1, min(cos_angle, 1))  # 범위 제한 (-1 ~ 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                if angle < min_angle and angle < 45:  # 45도 내에 있는 객체만 고려
                    min_angle = angle
                    best_match_idx = idx
        
        return best_match_idx
        
    def _get_object_position_description(self, obj: Dict, detections: List[Dict]) -> str:
        """
        객체의 상대적 위치 설명 생성
        
        Args:
            obj: 위치를 설명할 객체
            detections: 감지된 모든 객체 목록
            
        Returns:
            str: 객체 위치 설명
        """
        if not obj or not detections:
            return "Unknown location"
            
        # 이미지에서의 상대적 위치 설명 (왼쪽/오른쪽/위/아래)
        bbox = obj.get("bbox", [0, 0, 0, 0])
        obj_center_x = (bbox[0] + bbox[2]) / 2
        obj_center_y = (bbox[1] + bbox[3]) / 2
        
        # 다른 객체들과의 관계 파악
        other_objs = [d for d in detections if d != obj]
        
        # 기본 위치 설명
        position_parts = []
        
        # 좌우 위치
        if obj_center_x < 1/3 * 640:  # 이미지 너비 가정
            position_parts.append("왼쪽")
        elif obj_center_x > 2/3 * 640:
            position_parts.append("오른쪽")
        else:
            position_parts.append("중앙")
            
        # 상하 위치
        if obj_center_y < 1/3 * 480:  # 이미지 높이 가정
            position_parts.append("상단")
        elif obj_center_y > 2/3 * 480:
            position_parts.append("하단")
        else:
            position_parts.append("중앙")
            
        position = " ".join(position_parts)
        
        # 크기 정보 (다른 객체와 비교)
        obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        other_areas = [((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) for d in other_objs if "bbox" in d]
        
        if other_areas:
            avg_area = sum(other_areas) / len(other_areas)
            if obj_area > 1.5 * avg_area:
                size = "큰"
            elif obj_area < 0.67 * avg_area:
                size = "작은"
            else:
                size = "중간 크기의"
        else:
            size = "중간 크기의"
            
        # 깊이 정보 (가까움/멀음)
        depth_info = ""
        if "depth" in obj and "avg_depth" in obj["depth"]:
            avg_depth = obj["depth"]["avg_depth"]
            other_depths = [d["depth"]["avg_depth"] for d in other_objs if "depth" in d and "avg_depth" in d["depth"]]
            
            if other_depths:
                avg_other_depth = sum(other_depths) / len(other_depths)
                if avg_depth < 0.8 * avg_other_depth:
                    depth_info = "가까운"
                elif avg_depth > 1.2 * avg_other_depth:
                    depth_info = "먼"
        
        # 최종 설명 조합
        description = f"{position}에 있는 {depth_info} {size} {obj['class_name']}"
        return description.strip()
    
    def _resolve_demonstrative_pronouns(self, user_prompt: str, 
                                       detections: List[Dict], 
                                       gesture_results: Optional[List[Dict]] = None,
                                       context_str: str = "") -> Dict:
        """
        지시대명사(이것, 저것 등)를 해석하여 구체적인 객체로 변환
        
        Args:
            user_prompt: 사용자 입력 명령
            detections: 감지된 객체 목록
            gesture_results: 제스처 인식 결과 (옵션)
            context_str: 추가 컨텍스트 정보
            
        Returns:
            Dict: 처리 결과
                - resolved_command: 해석된 명령
                - confidence: 해석 신뢰도
                - reasoning: 추론 근거
        """
        self.logger.info(f"1단계: 지시대명사 해석 시작 - '{user_prompt}'")
        
        # 지시대명사 패턴 확인
        has_demonstrative = self._contains_demonstrative_pronoun(user_prompt)
        if not has_demonstrative:
            self.logger.info("지시대명사가 없습니다. 1단계 처리 건너뜀")
            return {
                "resolved_command": user_prompt,
                "confidence": 1.0,
                "reasoning": "지시대명사가 없습니다."
            }
        
        # 객체 정보 구성
        objects_info = "\n".join([
            f"- 객체 {i+1}: {obj.get('class_name', '알 수 없는 객체')}"
            for i, obj in enumerate(detections)
        ])
        
        # 제스처 정보 추가
        gesture_info = ""
        if gesture_results and len(gesture_results) > 0:
            pointing_info = []
            for g in gesture_results:
                if g.get("gesture_type") == "pointing":
                    obj_id = g.get("pointed_object_id")
                    obj_class = next((d.get("class_name", "알 수 없는 객체") for d in detections 
                                     if d.get("id") == obj_id), "없음")
                    pointing_info.append(f"손가락으로 가리킨 객체: {obj_class} (ID: {obj_id})")
            
            if pointing_info:
                gesture_info = "제스처 정보:\n" + "\n".join(pointing_info)
        
        # 시스템 프롬프트 구성
        system_prompt = f"""당신은 자연어 처리 전문가입니다. 사용자의 명령에서 지시대명사(이것, 저것, 그것 등)를 정확한 객체로 변환해주세요.

현재 장면에는 다음 객체들이 있습니다:
{objects_info}

{gesture_info}

{context_str}

사용자의 명령에서 지시대명사를 찾아 실제 객체로 치환한 명령을 생성해주세요. 

응답은 JSON 형식으로 다음과 같은 구조로 제공해주세요:
```json
{{
  "resolved_command": "변환된 명령",
  "confidence": 0.9,  // 확신도 (0.0~1.0)
  "reasoning": "변환 이유와 근거"
}}
```

JSON 형식 외에 다른 텍스트는 제외하고 응답하세요.
"""
        
        try:
            # LLM 호출
            response = self.llm.generate_text(system_prompt, user_prompt, temperature=0.1)
            
            try:
                # JSON 파싱 및 추출
                result = self._extract_json_from_response(response)
                
                self.logger.info(f"지시대명사 해석 완료: '{result.get('resolved_command', user_prompt)}'")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM 응답을 JSON으로 파싱할 수 없음: {e}")
                return {
                    "resolved_command": user_prompt,  # 원래 명령 유지
                    "confidence": 0.0,
                    "reasoning": f"JSON 파싱 실패: {str(e)}"
                }
        except Exception as e:
            self.logger.error(f"지시대명사 해석 중 오류: {e}")
            return {
                "resolved_command": user_prompt,  # 원래 명령 유지
                "confidence": 0.0,
                "reasoning": f"오류 발생: {str(e)}"
            }
            
    def _enhance_prompt(self, user_prompt: str, detections: List[Dict], context_str: str = "") -> Dict:
        """
        사용자 명령을 구체화하여 타겟/레퍼런스 객체 관계를 명확하게 표현
        
        Args:
            user_prompt: 사용자 입력 명령
            detections: 감지된 객체 목록
            context_str: 추가 컨텍스트 정보
            
        Returns:
            Dict: 처리 결과
                - enhanced_command: 구체화된 명령
                - confidence: 구체화 신뢰도
                - reasoning: 추론 근거
        """
        self.logger.info(f"2단계: 프롬프트 구체화 시작 - '{user_prompt}'")
        
        # 객체 정보 구성
        objects_info = "\n".join([
            f"- 객체 {i+1}: {obj.get('class_name', '알 수 없는 객체')}"
            for i, obj in enumerate(detections)
        ])
        
        # 시스템 프롬프트 구성
        system_prompt = f"""당신은 자연어 처리 전문가입니다. 사용자의 모호한 명령을 구체적인 명령으로 변환해주세요.

현재 장면에는 다음 객체들이 있습니다:
{objects_info}

{context_str}

사용자의 명령을 분석하여, 무엇을 어디에 배치하고 싶은지 명확하게 표현한 구체적인 명령으로 변환해주세요.
예를 들어:
- "컵을 테이블 위에 놓아줘" -> "컵을 테이블 위에 놓아줘"
- "옮겨줘" -> "컵을 테이블 위에 옮겨줘"

응답은 JSON 형식으로 다음과 같은 구조로 제공해주세요:
```json
{{
  "enhanced_command": "구체화된 명령",
  "confidence": 0.9,  // 확신도 (0.0~1.0)
  "reasoning": "구체화 이유와 근거"
}}
```

JSON 형식 외에 다른 텍스트는 제외하고 응답하세요.
"""
        
        try:
            # LLM 호출
            response = self.llm.generate_text(system_prompt, user_prompt, temperature=0.1)
            
            try:
                # JSON 파싱 및 추출
                result = self._extract_json_from_response(response)
                
                self.logger.info(f"프롬프트 구체화 완료: '{result.get('enhanced_command', user_prompt)}'")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM 응답을 JSON으로 파싱할 수 없음: {e}")
                return {
                    "enhanced_command": user_prompt,  # 원래 명령 유지
                    "confidence": 0.0,
                    "reasoning": f"JSON 파싱 실패: {str(e)}"
                }
        except Exception as e:
            self.logger.error(f"프롬프트 구체화 중 오류: {e}")
            return {
                "enhanced_command": user_prompt,  # 원래 명령 유지
                "confidence": 0.0,
                "reasoning": f"오류 발생: {str(e)}"
            }
            
    def _infer_target_reference_goal(self, user_prompt: str, detections: List[Dict], 
                                    depth_data: Optional[np.ndarray] = None,
                                    context_str: str = "",
                                    explicit_direction: Optional[str] = None,
                                    explicit_reference_name: Optional[str] = None) -> Dict:
        """
        타겟 객체, 레퍼런스 객체, 목표 위치 추론
        
        Args:
            user_prompt: 사용자 입력 명령
            detections: 감지된 객체 목록
            depth_data: 깊이 맵 데이터 (옵션)
            context_str: 추가 컨텍스트 정보
            explicit_direction: 명시적 방향 (옵션)
            explicit_reference_name: 명시적 레퍼런스 객체 이름 (옵션)
            
        Returns:
            Dict: 처리 결과
                - target_object_id: 타겟 객체 ID
                - reference_object_id: 레퍼런스 객체 ID
                - goal_position: 목표 위치 [x, y, z]
                - confidence: 추론 신뢰도
                - reasoning: 추론 근거
                - method: 추론 방법
                - direction: 방향 정보
        """
        self.logger.info(f"3단계: 타겟/레퍼런스/목표 추론 시작 - '{user_prompt}'")
        
        # 객체 정보 구성
        objects_info = []
        for i, obj in enumerate(detections):
            obj_desc = f"- 객체 {i+1} (ID: {obj.get('id', i)}):\n"
            obj_desc += f"  - 클래스: {obj.get('class_name', '알 수 없는 객체')}\n"
            obj_desc += f"  - 바운딩 박스: {obj.get('bbox', [0, 0, 0, 0])}\n"
            
            # 깊이 정보 추가 (있는 경우)
            if 'depth_stats' in obj:
                depth_stats = obj['depth_stats']
                obj_desc += f"  - 깊이 정보: 중앙값={depth_stats.get('median_depth', 0):.4f}, "
                obj_desc += f"평균={depth_stats.get('mean_depth', 0):.4f}\n"
            
            objects_info.append(obj_desc)
        
        objects_info_str = "\n".join(objects_info)
        
        # 명시적 방향 정보 추가
        direction_info = ""
        if explicit_direction and explicit_reference_name:
            direction_info = f"\n사용자가 '{explicit_reference_name}' 객체의 '{explicit_direction}' 방향을 명시했습니다."
        
        # 시스템 프롬프트 구성
        system_prompt = f"""당신은 로봇 조작 계획을 담당하는 전문가입니다. 사용자의 명령을 분석하여 어떤 객체(타겟)를 어떤 객체(레퍼런스) 기준으로 어디에 배치할지 결정해주세요.

현재 장면에는 다음 객체들이 있습니다:
{objects_info_str}

{context_str}
{direction_info}

사용자의 명령을 분석하여, 다음을 결정해주세요:
1. 타겟 객체 (이동해야 할 객체)
2. 레퍼런스 객체 (기준이 되는 객체)
3. 레퍼런스 객체를 기준으로 한 목표 위치 (상대적 좌표)

응답은 JSON 형식으로 다음과 같은 구조로 제공해주세요:
```json
{{
  "target_object_id": 1,  // 타겟 객체 ID
  "reference_object_id": 2,  // 레퍼런스 객체 ID
  "goal_position": [0.5, 0.5, 0.5],  // 목표 위치 [x, y, z] (0~1 사이 정규화된 값)
  "direction": "front",  // 방향 정보 (front, back, left, right, above, below, side 중 하나)
  "confidence": 0.9,  // 확신도 (0.0~1.0)
  "reasoning": "추론 근거와 이유",
  "method": "semantic"  // 추론 방법 (semantic, relative, absolute 등)
}}
```

JSON 형식 외에 다른 텍스트는 제외하고 응답하세요.
"""
        
        try:
            # LLM 호출
            response = self.llm.generate_text(system_prompt, user_prompt, temperature=0.1)
            
            try:
                # JSON 파싱 및 추출
                response_json = self._extract_json_from_response(response)
                
                # 방향 정보가 명시적으로 제공된 경우 오버라이드
                if explicit_direction:
                    response_json['direction'] = explicit_direction
                    response_json['confidence'] = 0.95  # 명시적 정보가 있으므로 높은 신뢰도
                
                # 결과 로깅
                target_id = response_json.get('target_object_id')
                ref_id = response_json.get('reference_object_id')
                target_name = next((d.get('class_name', '?') for d in detections if d.get('id') == target_id), '?')
                ref_name = next((d.get('class_name', '?') for d in detections if d.get('id') == ref_id), '?')
                
                self.logger.info(f"타겟/레퍼런스/목표 추론 완료: 타겟={target_name}(ID:{target_id}), 레퍼런스={ref_name}(ID:{ref_id}), 방향={response_json.get('direction', 'unknown')}")
                
                return response_json
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM 응답을 JSON으로 파싱할 수 없음: {e}")
                # 감지된 객체가 1개 또는 2개일 경우 기본 값 결정
                if len(detections) == 1:
                    # 객체가 하나뿐인 경우, 그 객체가 타겟이자 레퍼런스
                    obj_id = detections[0].get('id', 0)
                    return {
                        "target_object_id": obj_id,
                        "reference_object_id": obj_id,
                        "goal_position": [0.5, 0.5, 0.5],  # 중앙 위치
                        "direction": "front",  # 기본 방향
                        "confidence": 0.5,
                        "reasoning": "객체가 1개뿐이라 해당 객체를 기준으로 판단했습니다.",
                        "method": "default" 
                    }
                elif len(detections) == 2:
                    # 객체가 2개인 경우, 첫 번째가 타겟, 두 번째가 레퍼런스로 가정
                    target_id = detections[0].get('id', 0)
                    ref_id = detections[1].get('id', 1)
                    return {
                        "target_object_id": target_id,
                        "reference_object_id": ref_id,
                        "goal_position": [0.5, 0.5, 0.5],  # 중앙 위치
                        "direction": explicit_direction or "right",  # 명시적 방향 또는 기본 오른쪽
                        "confidence": 0.3,
                        "reasoning": "JSON 파싱에 실패했으나, 객체가 2개이므로 첫 번째 객체를 타겟, 두 번째를 레퍼런스로 가정했습니다.",
                        "method": "default"
                    }
                else:
                    # 3개 이상인 경우 첫 두 개 객체를 선택
                    target_id = detections[0].get('id', 0)
                    ref_id = detections[1].get('id', 1)
                    return {
                        "target_object_id": target_id,
                        "reference_object_id": ref_id,
                        "goal_position": [0.5, 0.5, 0.5],  # 중앙 위치
                        "direction": explicit_direction or "right",  # 명시적 방향 또는 기본 오른쪽
                        "confidence": 0.1,
                        "reasoning": "JSON 파싱에 실패했으며, 임의로 처음 두 객체를 선택했습니다.",
                        "method": "default"
                    }
        except Exception as e:
            self.logger.error(f"타겟/레퍼런스/목표 추론 중 오류: {e}")
            return {
                "target_object_id": 0,
                "reference_object_id": 0,
                "goal_position": [0.5, 0.5, 0.5],  # 중앙 위치
                "direction": "front",  # 기본 방향
                "confidence": 0.0,
                "reasoning": f"오류 발생: {str(e)}",
                "method": "error"
            }
            
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 부분을 추출하고 파싱
        
        Args:
            response_text: LLM의 텍스트 응답
            
        Returns:
            파싱된 JSON 데이터 (딕셔너리)
            
        Raises:
            json.JSONDecodeError: JSON 파싱 실패 시 발생
        """
        if not response_text or not isinstance(response_text, str):
            raise ValueError("응답 텍스트가 비어있거나, 문자열이 아닙니다.")
            
        # 1. markdown 코드 블록 제거
        clean_text = response_text.strip()
        if "```json" in clean_text:
            # JSON 코드 블록 추출
            pattern = r"```json\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, clean_text)
            if matches:
                clean_text = matches[0].strip()
        elif "```" in clean_text:
            # 일반 코드 블록 추출
            pattern = r"```\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, clean_text)
            if matches:
                clean_text = matches[0].strip()
                
        # 2. 첫 번째 { 와 마지막 } 사이의 내용 추출 (일부 응답에서 JSON 앞뒤에 텍스트가 있는 경우)
        if '{' in clean_text and '}' in clean_text:
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1
            clean_text = clean_text[start_idx:end_idx]
                
        # 3. JSON 파싱
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {e}, 원본 텍스트: {clean_text[:100]}...")
            raise

    def confirm_reference(self, 
                      user_command: str,
                      target_object: Dict[str, Any],
                      reference_object: Dict[str, Any]) -> Dict[str, Any]:
        """
        타겟 객체와 자동으로 결정된 레퍼런스 객체의 유효성을 LLM에게 확인받는 함수
        
        Args:
            user_command: 사용자 명령
            target_object: 타겟 객체 정보
            reference_object: 레퍼런스 객체 정보
        
        Returns:
            Dict[str, Any]: 확인 결과
                - 'is_valid': 레퍼런스 객체 유효성 (True/False)
                - 'reference_id': 확인된 레퍼런스 객체 ID
                - 'confidence': 확인 신뢰도
                - 'explanation': 확인 설명
                - 'original_response': LLM의 원본 응답
        """
        self.logger.info(f"레퍼런스 객체 확인 시작: 타겟={target_object.get('class_name')}, 레퍼런스={reference_object.get('class_name')}")
        
        # 결과 초기화
        result = {
            'is_valid': False,
            'reference_id': None, 
            'confidence': 0.0,
            'explanation': "",
            'original_response': ""
        }
        
        # LLM에 전달할 프롬프트 구성
        prompt = f"""
사용자 명령은 '{user_command}'입니다.
타겟 객체는 '{target_object.get('class_name', '알 수 없는 객체')}'(ID: {target_object.get('id', '?')})입니다.
장면에는 '{reference_object.get('class_name', '알 수 없는 객체')}'(ID: {reference_object.get('id', '?')}) 객체가 남아있습니다.

위 정보를 고려할 때, 이 '{reference_object.get('class_name', '알 수 없는 객체')}'가 레퍼런스 객체가 맞습니까?
사용자 명령과 상황을 고려하여, 이 객체가 레퍼런스 객체로 적합한지 판단해 주세요.

응답은 반드시 다음 JSON 형식으로 제공해 주세요:
{{
  "is_valid": true or false,
  "reference_id": {reference_object.get('id', 0)},
  "confidence": 0.0부터 1.0 사이의 값,
  "explanation": "선택 이유에 대한 간단한 설명"
}}

다른 설명 없이 위 JSON 형식으로만 응답해 주세요.
"""
        
        try:
            # LLM 호출
            response = self.llm.generate_text(prompt, user_prompt="레퍼런스 객체 확인", temperature=0.1)
            result['original_response'] = response
            
            try:
                # JSON 파싱 및 추출
                response_json = self._extract_json_from_response(response)
                
                # 필요한 필드 추출
                result['is_valid'] = response_json.get('is_valid', False)
                result['reference_id'] = response_json.get('reference_id', reference_object.get('id'))
                result['confidence'] = float(response_json.get('confidence', 0.5))  # 기본값 0.5
                result['explanation'] = response_json.get('explanation', "")
                
                self.logger.info(f"레퍼런스 객체 확인 결과: 유효={result['is_valid']}, ID={result['reference_id']}, 신뢰도={result['confidence']}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM 응답을 JSON으로 파싱할 수 없음: {e}")
                # 파싱 실패 시, 기본적으로 레퍼런스 객체를 유효하다고 가정
                result['is_valid'] = True
                result['reference_id'] = reference_object.get('id')
                result['confidence'] = 0.5  # 중간 신뢰도
                result['explanation'] = "JSON 파싱 실패로 인해 기본값으로 설정됨"
        
        except Exception as e:
            self.logger.error(f"레퍼런스 객체 확인 중 오류: {e}", exc_info=True)
            # 오류 발생 시, 기본적으로 레퍼런스 객체를 유효하다고 가정
            result['is_valid'] = True
            result['reference_id'] = reference_object.get('id') 
            result['confidence'] = 0.5  # 중간 신뢰도
            result['explanation'] = f"오류 발생: {str(e)}"
        
        return result

    def _create_direction_inference_prompt(self, user_command: str, target_obj: Dict[str, Any], 
                                          reference_obj: Dict[str, Any], img_dims: Tuple[int, int]) -> str:
        """
        방향 추론을 위한 프롬프트 생성
        
        Args:
            user_command: 사용자 명령
            target_obj: 타겟 객체 정보
            reference_obj: 레퍼런스 객체 정보
            img_dims: 이미지 크기 (width, height)
            
        Returns:
            방향 추론 프롬프트 문자열
        """
        ref_bbox = reference_obj.get('bbox', [0, 0, 0, 0])
        target_class = target_obj.get('class_name', '알 수 없는 객체')
        reference_class = reference_obj.get('class_name', '알 수 없는 객체')
        target_id = target_obj.get('id', 0)
        reference_id = reference_obj.get('id', 1)
        
        return f"""
사용자의 명령은 {user_command}입니다.
타겟 객체는 {target_class}(ID: {target_id})이고, 레퍼런스 객체는 {reference_class}(ID: {reference_id})입니다.
이미지 해상도는 {img_dims[0]}x{img_dims[1]}입니다.
레퍼런스 객체의 바운딩 박스는 {ref_bbox}입니다.

이미지는 2D이지만, 실제 공간을 반영하고 있습니다.
**이미지에서 y값이 낮을수록 카메라와 더 가까운 위치**를 의미합니다.
즉, 사용자의 명령에서 말하는 "앞쪽"은 **이미지 기준으로 더 위쪽(y값이 더 작음)** 방향입니다.

방향 유형은 다음과 같이 분류됩니다:
1. 단일 방향: 'front'(앞), 'back'(뒤), 'left'(왼쪽), 'right'(오른쪽), 'above'(위), 'below'(아래), 'side'(옆)
2. 랜덤 방향: 'random_lr'(좌우 랜덤), 'random_fb'(앞뒤 랜덤), 'random'(완전 랜덤)

사용자 명령을 분석하여 방향 유형과 구체적인 방향을 결정해주세요.
명령에 "랜덤" 또는 "무작위"라는 표현이 있고 특정 축(좌우/앞뒤)이 언급되면 그에 맞는 랜덤 방향을 선택하세요.

결과는 다음 형식의 JSON으로 정확히 반환해주세요:
{{
  "target_object_id": {target_id},       // 타겟 객체 ID (그대로 유지)
  "reference_object_id": {reference_id}, // 레퍼런스 객체 ID (그대로 유지)
  "direction_type": "simple",            // 'simple', 'random', 'composite' 중 하나
  "direction": "front",                  // 단일 방향인 경우 방향값
  "options": ["left", "right"],          // 랜덤 방향인 경우 선택지 목록
  "weights": [0.5, 0.5],                 // 각 선택지의 확률 가중치 (합이 1이 되도록)
  "confidence": 0.9                      // 결정에 대한 신뢰도 (0-1)
}}

**다른 설명 없이 반드시 위 형식만 반환**해 주세요.
"""
    
    def _process_direction_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 응답에서 방향 정보 추출 및 처리
        
        Args:
            response_json: LLM 응답에서 파싱된 JSON 데이터
        
        Returns:
            Dict: 처리된 방향 정보
        """
        try:
            # 기본 검증
            if "direction_type" not in response_json:
                response_json["direction_type"] = "simple"
                
            if response_json["direction_type"] == "simple":
                # 단일 방향 처리
                direction = response_json.get("direction", "front")
                return {
                    "type": "simple",
                    "value": direction,
                    "confidence": response_json.get("confidence", 0.8)
                }
            elif response_json["direction_type"] == "random":
                # 랜덤 방향 처리
                options = response_json.get("options", ["left", "right"])
                weights = response_json.get("weights", [1.0/len(options)] * len(options))
                return {
                    "type": "random",
                    "options": options,
                    "weights": weights,
                    "confidence": response_json.get("confidence", 0.8)
                }
            else:
                # 알 수 없는 유형 처리
                return {
                    "type": "simple",
                    "value": "front",
                    "confidence": 0.5
                }
        except Exception as e:
            self.logger.error(f"방향 응답 처리 중 오류: {e}")
            return {
                "type": "simple",
                "value": "front",
                "confidence": 0.3
            }
    
    def _analyze_random_direction_intent(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        사용자 프롬프트에서 랜덤 방향 의도 분석
        
        Args:
            user_prompt: 사용자 입력 프롬프트
            
        Returns:
            Optional[Dict[str, Any]]: 랜덤 방향 정보 또는 None
        """
        if not user_prompt:
            return None
            
        user_prompt = user_prompt.lower()
        
        # 랜덤/무작위 키워드 확인
        random_keywords = ["랜덤", "무작위", "random", "randomly", "either", "any", "아무", "어느", "어디든"]
        has_random = any(keyword in user_prompt for keyword in random_keywords)
        
        if not has_random:
            return None
        
        # 축 방향 확인
        lr_keywords = ["좌우", "좌 우", "left right", "왼쪽 오른쪽", "왼쪽이나 오른쪽"]
        fb_keywords = ["앞뒤", "앞 뒤", "front back", "앞쪽 뒤쪽", "앞쪽이나 뒤쪽"]
        
        if any(keyword in user_prompt for keyword in lr_keywords):
            return {
                "type": "random",
                "options": ["left", "right"],
                "weights": [0.5, 0.5],
                "confidence": 0.9
            }
        elif any(keyword in user_prompt for keyword in fb_keywords):
            return {
                "type": "random", 
                "options": ["front", "back"],
                "weights": [0.5, 0.5],
                "confidence": 0.9
            }
        
        # 일반 랜덤 (방향 미지정)
        return {
            "type": "random",
            "options": ["front", "back", "left", "right"],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "confidence": 0.8
        }
    
    def infer_goal_bounding_box(self,
                            user_command: str,
                            target_object: Dict[str, Any],
                            reference_object: Dict[str, Any],
                            img_width: int,
                            img_height: int,
                            direction: str = "front") -> Dict[str, Any]:
        """
        타겟 객체와 레퍼런스 객체 간의 의미론적 관계를 추론하는 함수
        
        Args:
            user_command: 사용자 명령
            target_object: 타겟 객체 정보 (클래스명, 바운딩 박스 등)
            reference_object: 레퍼런스 객체 정보 (클래스명, 바운딩 박스 등)
            img_width: 이미지 너비
            img_height: 이미지 높이
            direction: 방향 (기본값: "front" = 앞쪽)
        
        Returns:
            Dict[str, Any]: 추론 결과
                - 'target_object_id': 타겟 객체 ID
                - 'reference_object_id': 레퍼런스 객체 ID
                - 'direction': 위치 방향 정보 (단순 문자열 또는 복합 객체)
                - 'confidence': 추론 신뢰도
                - 'original_response': LLM의 원본 응답
                - 'prompt': LLM에 전달된 프롬프트
                - 'parse_success': JSON 파싱 성공 여부
                - 'target_box': (이전 버전과의 호환성) 추정되는 목표 위치 바운딩 박스 [x1, y1, x2, y2]
        """
        self.logger.info(f"의미론적 관계 추론 시작: 타겟={target_object.get('class_name')}, 레퍼런스={reference_object.get('class_name')}")
        
        # 결과 초기화
        result = {
            'target_object_id': None,
            'reference_object_id': None,
            'direction': direction,  # 기본 방향 유지
            'confidence': 0.0,
            'original_response': "",
            'prompt': "",
            'parse_success': False,
            'target_box': None  # 이전 버전과의 호환성을 위해 유지
        }
        
        # 사용자 프롬프트에서 랜덤 방향 의도 직접 분석
        random_direction = self._analyze_random_direction_intent(user_command)
        if random_direction:
            self.logger.info(f"사용자 프롬프트에서 랜덤 방향 의도 감지: {random_direction}")
            result['direction'] = random_direction
            result['confidence'] = random_direction.get('confidence', 0.8)
            result['parse_success'] = True
            return result
        
        # 확장된 방향 추론 프롬프트 생성
        prompt = self._create_direction_inference_prompt(
            user_command, 
            target_object, 
            reference_object, 
            (img_width, img_height)
        )
        
        # 프롬프트 저장
        result['prompt'] = prompt
        self.logger.debug(f"LLM 프롬프트: {prompt}")
        
        try:
            # LLM 호출 (온도 0.1로 결정성 높임) - user_prompt 전달 추가
            response = self.llm.generate_text(prompt, user_prompt=user_command, temperature=0.1)
            result['original_response'] = response
            
            try:
                # JSON 파싱 및 추출
                response_json = self._extract_json_from_response(response)
                result['parse_success'] = True
                
                # 필드 추출
                result['target_object_id'] = response_json.get('target_object_id')
                result['reference_object_id'] = response_json.get('reference_object_id')
                result['goal_position'] = response_json.get('goal_position')
                
                # 확장된 방향 정보 처리
                if 'direction_type' in response_json:
                    direction_data = self._process_direction_response(response_json)
                    result['direction'] = direction_data
                else:
                    # 기존 단순 방향 호환성 유지
                    result['direction'] = response_json.get('direction', 'right')
                
                result['confidence'] = response_json.get('confidence', 0.8)  # 기본 신뢰도 0.8
                
                self.logger.info(f"미래 행동 예측 결과: 타겟={result['target_object_id']}, 레퍼런스={result['reference_object_id']}, 방향={result['direction']}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM 응답을 JSON으로 파싱할 수 없음: {e}")
                # 파싱 실패 시 기본값 설정
                result['parse_success'] = False
                result['confidence'] = 0.2  # 낮은 신뢰도
                # 기본 방향 유지, 이미 direction 기본값으로 설정됨
        
        except Exception as e:
            self.logger.error(f"의미론적 관계 추론 중 오류: {e}", exc_info=True)
            
        # 이전 버전과의 호환성을 위해 target_box가 None인 경우 기본값 설정
        if result['parse_success'] and result['target_box'] is None:
            # 레퍼런스 객체의 바운딩 박스 추출
            ref_bbox = reference_object.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2] 형식
            x1, y1, x2, y2 = ref_bbox
            width = x2 - x1
            height = y2 - y1
            
            # direction이 딕셔너리인 경우 (확장된 방향 형식)
            direction_value = result['direction']
            if isinstance(direction_value, dict):
                if direction_value.get('type') == 'simple':
                    # 단일 방향인 경우
                    direction_str = direction_value.get('value', 'right')
                elif direction_value.get('type') == 'random':
                    # 랜덤 방향인 경우 첫 번째 옵션 사용 (단순 호환성 목적)
                    options = direction_value.get('options', ['right'])
                    direction_str = options[0] if options else 'right'
                else:
                    direction_str = 'right'  # 기본값
            else:
                # 기존 문자열 방향
                direction_str = direction_value
            
            # 방향에 따른 target_box 생성
            if direction_str == 'front':
                # 앞쪽 (y 감소)
                result['target_box'] = [x1, max(0, y1 - height), x2, max(height, y2 - height)]
            elif direction_str == 'back':
                # 뒤쪽 (y 증가)
                result['target_box'] = [x1, y1 + height, x2, min(img_height, y2 + height)]
            elif direction_str == 'left':
                # 왼쪽 (x 감소)
                result['target_box'] = [max(0, x1 - width), y1, max(width, x2 - width), y2]
            elif direction_str == 'right' or direction_str == 'side':
                # 오른쪽 또는 옆 (x 증가)
                result['target_box'] = [x1 + width, y1, min(img_width, x2 + width), y2]
            elif direction_str == 'above':
                # 위쪽 (이미지 상에서는 y 감소, 실제로는 z 증가)
                result['target_box'] = [x1, max(0, y1 - height), x2, max(height, y2 - height)]
            elif direction_str == 'below':
                # 아래쪽 (이미지 상에서는 y 증가, 실제로는 z 감소)
                result['target_box'] = [x1, y1 + height, x2, min(img_height, y2 + height)]
            else:
                # 알 수 없는 방향 - 오른쪽 기본값 사용
                result['target_box'] = [x1 + width, y1, min(img_width, x2 + width), y2]
                
            self.logger.info(f"방향에 따른 가상 바운딩 박스 생성: {result['target_box']}")
        
        return result 

    def create_intent_extraction_prompt(self, user_command: str) -> str:
        """
        사용자 명령에서 방향 및 배치 의도를 추출하기 위한 프롬프트를 생성합니다.
        
        Args:
            user_command: 사용자 명령어
            
        Returns:
            str: 의도 추출을 위한 프롬프트
        """
        directions_kr = [
            "앞", "앞으로", "앞에", "전방", "전방으로", "앞쪽", "앞쪽으로", "앞쪽에",
            "뒤", "뒤로", "뒤에", "후방", "후방으로", "뒤쪽", "뒤쪽으로", "뒤쪽에",
            "위", "위로", "위에", "상단", "상단으로", "상단에", "위쪽", "위쪽으로", "위쪽에",
            "아래", "아래로", "아래에", "하단", "하단으로", "하단에", "아래쪽", "아래쪽으로", "아래쪽에",
            "왼쪽", "왼쪽으로", "왼쪽에", "좌측", "좌측으로", "좌측에", "왼편", "왼편으로", "왼편에",
            "오른쪽", "오른쪽으로", "오른쪽에", "우측", "우측으로", "우측에", "오른편", "오른편으로", "오른편에",
            "옆", "옆으로", "옆에", "측면", "측면으로", "측면에", "사이", "사이에", "중앙", "중앙에", "가운데", "가운데에"
        ]
        
        directions_en = [
            "front", "in front", "in front of", "forward", "ahead", 
            "back", "behind", "backward", "in the back", "in the back of",
            "top", "above", "over", "up", "on top", "on top of", "upward",
            "bottom", "below", "under", "underneath", "down", "downward",
            "left", "to the left", "on the left", "leftward",
            "right", "to the right", "on the right", "rightward",
            "side", "beside", "next to", "adjacent", "between", "middle", "center", "in the middle", "in the center"
        ]
        
        # 모든 방향 키워드를 하나의 문자열로 결합
        all_directions = ", ".join([f'"{d}"' for d in directions_kr + directions_en])
        
        return f"""당신은 자연어 명령에서 방향과 배치 의도를 정확하게 분석하는 전문가입니다.

# 입력 명령
"{user_command}"

# 방향 키워드
다음은 방향을 나타내는 키워드들입니다:
{all_directions}

# 분석 지침
1. 입력 명령에서 방향 정보(앞, 뒤, 위, 아래, 왼쪽, 오른쪽 등)를 추출하세요.
2. 명령에 들어있는 방향 표현이 영어나 한국어 어떤 언어로도 있을 수 있습니다.
3. 방향 표현이 명확하지 않은 경우 문맥에서 방향을 유추하세요.
4. 방향 유형이 '절대적'인지 '상대적'인지 결정하세요:
   - '절대적': 특정 참조 객체 없이 공간상의 일반적인 방향 (예: "위로 옮겨줘")
   - '상대적': 특정 참조 객체를 기준으로 하는 방향 (예: "컵 옆에 놓아줘")
5. '무작위'나 '아무데나'와 같은 표현이 있다면 'random' 방향으로 분류하세요.

# 출력 형식
다음 형식의 JSON으로만 응답하세요:
```json
{{
  "direction_type": "상대적 또는 절대적 또는 random",
  "direction": "front, back, above, below, left, right, random 중 하나",
  "confidence": 신뢰도(0.0~1.0),
  "keywords": ["검출된 모든 방향 키워드"],
  "reasoning": "방향을 결정한 이유에 대한 간략한 설명"
}}
```

위 출력 형식 외의 다른 텍스트는 포함하지 마세요.
"""

    def create_relationship_extraction_prompt(self, user_command: str, detected_objects: List[Dict], 
                                           direction_info: Dict) -> str:
        """
        사용자 명령과 감지된 객체에서 객체 간 관계를 추출하기 위한 프롬프트를 생성합니다.
        
        Args:
            user_command: 사용자 명령어
            detected_objects: 감지된 객체 목록
            direction_info: 의도 분석에서 추출된 방향 정보
            
        Returns:
            str: 관계 추출을 위한 프롬프트
        """
        # 감지된 객체 목록 문자열 생성
        objects_list = []
        for obj in detected_objects:
            obj_id = obj.get('id', -1)
            obj_name = obj.get('class_name', obj.get('name', 'unknown'))
            position = obj.get('position', obj.get('xyz', [0, 0, 0]))
            position_str = f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]" if position else "[0, 0, 0]"
            
            objects_list.append(f"ID: {obj_id}, 이름: {obj_name}, 위치: {position_str}")
        
        objects_text = "\n".join(objects_list)
        
        # 방향 정보에서 추출된 키워드
        direction_keywords = direction_info.get('keywords', [])
        keywords_text = ", ".join([f'"{k}"' for k in direction_keywords]) if direction_keywords else "없음"
        
        # 방향 정보
        direction = direction_info.get('direction', 'front')
        direction_type = direction_info.get('direction_type', 'absolute')
        
        return f"""당신은 사용자 명령과 감지된 객체 목록에서 객체 간 관계를 추출하는 전문가입니다.

# 입력 명령
"{user_command}"

# 감지된 객체 목록
{objects_text}

# 방향 정보
방향: {direction}
방향 유형: {direction_type}
검출된 키워드: {keywords_text}

# 분석 지침
1. 입력 명령에서 대상 객체(옮기거나 놓을 물체)와 참조 객체(기준점)를 식별하세요.
2. "이것", "저것", "그것" 등의 지시대명사는 문맥과 객체 목록을 고려하여 가장 적합한 객체로 해석하세요.
3. 객체를 식별할 때는 객체 이름과 사용자 명령의 유사성을 고려하세요.
4. 명령에서 언급되지 않은 객체는 포함하지 마세요.
5. 참조 객체가 명시적으로 언급되지 않았지만 문맥에서 추론 가능하다면 식별하세요.

# 출력 형식
다음 형식의 JSON으로만 응답하세요:
```json
{{
  "target_object": {{
    "name": "대상 객체 이름",
    "id": 대상 객체 ID,
    "confidence": 신뢰도(0.0~1.0)
  }},
  "reference_object": {{
    "name": "참조 객체 이름",
    "id": 참조 객체 ID,
    "confidence": 신뢰도(0.0~1.0)
  }},
  "confidence": 전체 추출의 신뢰도(0.0~1.0),
  "reasoning": "관계 추출에 대한 간략한 설명"
}}
```

참조 객체가 명확하지 않거나 없는 경우에는 다음과 같이 응답하세요:
```json
{{
  "target_object": {{
    "name": "대상 객체 이름",
    "id": 대상 객체 ID,
    "confidence": 신뢰도(0.0~1.0)
  }},
  "reference_object": null,
  "confidence": 전체 추출의 신뢰도(0.0~1.0),
  "reasoning": "참조 객체가 없는 이유에 대한 설명"
}}
```

위 형식 외의 다른 텍스트는 포함하지 마세요.
"""

    def create_final_integration_prompt(self, user_command: str, direction_info: Dict, 
                                    relationship_info: Dict, detected_objects: List[Dict]) -> str:
        """
        의도 분석과 관계 추출 결과를 통합하여 최종 결과를 생성하기 위한 프롬프트를 생성합니다.
        
        Args:
            user_command: 사용자 명령어
            direction_info: 의도 분석에서 추출된 방향 정보
            relationship_info: 관계 추출에서 추출된 객체 관계 정보
            detected_objects: 감지된 객체 목록
            
        Returns:
            str: 최종 통합을 위한 프롬프트
        """
        # 대상 객체와 참조 객체 정보 추출
        target_obj = None
        reference_obj = None
        
        # 관계 정보에서 대상 객체와 참조 객체 ID 추출
        target_obj_info = relationship_info.get('target_object', {})
        target_id = target_obj_info.get('id', -1) if target_obj_info else -1
        
        reference_obj_info = relationship_info.get('reference_object', {})
        reference_id = reference_obj_info.get('id', -1) if reference_obj_info else -1
        
        # 객체 ID로 대상 객체와 참조 객체 찾기
        for obj in detected_objects:
            obj_id = obj.get('id', -1)
            if obj_id == target_id:
                target_obj = obj
            if obj_id == reference_id:
                reference_obj = obj
        
        # 객체 정보 문자열 생성
        target_info = "없음"
        if target_obj:
            target_name = target_obj.get('class_name', target_obj.get('name', 'unknown'))
            target_position = target_obj.get('position', target_obj.get('xyz', [0, 0, 0]))
            target_position_str = f"[{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]" if target_position else "[0, 0, 0]"
            target_info = f"ID: {target_id}, 이름: {target_name}, 위치: {target_position_str}"
        
        reference_info = "없음"
        if reference_obj:
            reference_name = reference_obj.get('class_name', reference_obj.get('name', 'unknown'))
            reference_position = reference_obj.get('position', reference_obj.get('xyz', [0, 0, 0]))
            reference_position_str = f"[{reference_position[0]:.2f}, {reference_position[1]:.2f}, {reference_position[2]:.2f}]" if reference_position else "[0, 0, 0]"
            reference_info = f"ID: {reference_id}, 이름: {reference_name}, 위치: {reference_position_str}"
        
        # 방향 정보
        direction = direction_info.get('direction', 'front')
        direction_type = direction_info.get('direction_type', 'absolute')
        direction_confidence = direction_info.get('confidence', 0.5)
        
        # 방향 키워드
        direction_keywords = direction_info.get('keywords', [])
        keywords_text = ", ".join([f'"{k}"' for k in direction_keywords]) if direction_keywords else "없음"
        
        # 신뢰도 계산
        target_confidence = target_obj_info.get('confidence', 0.5) if target_obj_info else 0.5
        reference_confidence = reference_obj_info.get('confidence', 0.5) if reference_obj_info else 0.5
        relationship_confidence = relationship_info.get('confidence', 0.5)
        
        # 평균 신뢰도 계산
        avg_confidence = (target_confidence + reference_confidence + direction_confidence + relationship_confidence) / 4 if reference_obj_info else (target_confidence + direction_confidence) / 2
        
        # 목표 위치 추론을 위한 프롬프트 생성
        return f"""당신은 방향 정보와 객체 관계를 기반으로 최종 목표 위치를 결정하는 전문가입니다.

# 입력 명령
"{user_command}"

# 방향 정보
방향: {direction}
방향 유형: {direction_type}
신뢰도: {direction_confidence}
키워드: {keywords_text}

# 객체 관계
대상 객체: {target_info}
참조 객체: {reference_info}
관계 추론: {relationship_info.get('reasoning', '추론 정보 없음')}

# 분석 지침
1. 사용자 명령, 방향 정보, 객체 관계를 종합적으로 분석하세요.
2. 대상 객체를 위한 최적의 목표 위치를 3D 좌표 [x, y, z]로 결정하세요.
3. 참조 객체가 있다면 해당 객체를 기준으로 방향에 맞는 목표 위치를 계산하세요.
4. 참조 객체가 없다면 현재 대상 객체의 위치를 기준으로 방향에 맞는 목표 위치를 계산하세요.
5. 방향이 'random'이면 무작위 위치를 제안하세요.
6. 신뢰도가 낮으면 보수적인 위치를 선택하세요.

# 출력 형식
다음 형식의 JSON으로만 응답하세요:
```json
{{
  "goal_point": [x, y, z],
  "confidence": 신뢰도(0.0~1.0),
  "reasoning": "목표 위치 결정에 대한 간략한 설명",
  "direction_type": "{direction_type}",
  "direction": "{direction}"
}}
```

위 형식 외의 다른 텍스트는 포함하지 마세요.
"""

    def process_natural_language_command(self, user_command: str, detected_objects: List[Dict],
                               target_object_id: Optional[int] = None,
                               target_object_name: Optional[str] = None,
                               img_resolution: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        고급 자연어 처리를 통해 사용자 명령을 분석하고 목표 지점을 추론합니다.
        3단계 프롬프트 파이프라인을 통해 진행됩니다.
        
        Args:
            user_command: 사용자 명령어
            detected_objects: 감지된 객체 목록
            target_object_id: 대상 객체 ID (선택 사항)
            target_object_name: 대상 객체 이름 (선택 사항)
            img_resolution: 이미지 해상도 (width, height)
            
        Returns:
            Dict: 처리 결과
        """
        self.logger.info("자연어 처리 파이프라인 시작")
        
        if not img_resolution:
            img_resolution = self.default_img_resolution
        
        try:
            # 단계 1: 의도 분석 - 방향 및 배치 유형 추출
            self.logger.info("단계 1: 의도 분석 시작")
            intent_prompt = self.create_intent_extraction_prompt(user_command)
            
            intent_response = self.llm.generate_text(intent_prompt, user_prompt=user_command, temperature=0.1)
            
            intent_data = self._extract_json_from_response(intent_response)
            if not intent_data:
                self.logger.error("의도 분석 결과를 추출할 수 없습니다.")
                return {"success": False, "error": "의도 분석 결과를 추출할 수 없습니다."}
                
            self.logger.info(f"의도 분석 결과: {intent_data}")
            
            # 단계 2: 관계 추출 - 대상 객체와 참조 객체 간 관계 파악
            self.logger.info("단계 2: 관계 추출 시작")
            relationship_prompt = self.create_relationship_extraction_prompt(
                user_command=user_command,
                detected_objects=detected_objects,
                direction_info=intent_data
            )
            
            relationship_response = self.llm.generate_text(
                relationship_prompt,
                user_prompt=user_command,
                temperature=0.1
            )
            
            relationship_data = self._extract_json_from_response(relationship_response)
            if not relationship_data:
                self.logger.error("관계 추출 결과를 추출할 수 없습니다.")
                # 의도 데이터만으로 부분적인 결과 반환 시도
                return self._convert_to_compatibility_format(intent_data, {}, detected_objects)
                
            self.logger.info(f"관계 추출 결과: {relationship_data}")
            
            # 단계 3: 최종 통합
            self.logger.info("단계 3: 최종 통합 시작")
            integration_prompt = self.create_final_integration_prompt(
                user_command=user_command,
                direction_info=intent_data,
                relationship_info=relationship_data,
                detected_objects=detected_objects
            )
            
            integration_response = self.llm.generate_text(
                integration_prompt,
                user_prompt=user_command,
                temperature=0.1
            )
            
            final_data = self._extract_json_from_response(integration_response)
            if not final_data:
                self.logger.error("최종 통합 결과를 추출할 수 없습니다.")
                # 의도와 관계 데이터로 부분적인 결과 반환
                return self._convert_to_compatibility_format(intent_data, relationship_data, detected_objects)
            
            self.logger.info(f"최종 통합 결과: {final_data}")
            
            # 결과를 기존 코드와 호환되는 형식으로 변환
            return self._convert_to_compatibility_format(intent_data, relationship_data, detected_objects, final_data)
            
        except Exception as e:
            self.logger.error(f"자연어 처리 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"success": False, "error": f"자연어 처리 오류: {str(e)}"}
            
    def _convert_to_compatibility_format(self, intent_data: Dict, relationship_data: Dict,
                                   detected_objects: List[Dict], final_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        3단계 파이프라인의 결과를 기존 코드베이스와 호환되는 형식으로 변환합니다.
        
        Args:
            intent_data: 의도 분석 결과
            relationship_data: 관계 추출 결과
            detected_objects: 감지된 객체 목록
            final_data: 최종 통합 결과 (선택 사항)
            
        Returns:
            Dict: 호환 가능한 형식의 결과
        """
        # 기본 결과 템플릿
        result = {
            "success": True,
            "goal_point": None,
            "confidence": 0.0,
            "reasoning": "",
            "direction_type": "absolute",
            "direction": "front"
        }
        
        # 방향 정보 설정
        if intent_data:
            result["direction_type"] = intent_data.get("direction_type", "absolute")
            result["direction"] = intent_data.get("direction", "front")
            
            # 신뢰도와 이유 설정
            if "confidence" in intent_data:
                result["confidence"] = intent_data.get("confidence", 0.0)
            if "reasoning" in intent_data:
                result["reasoning"] = intent_data.get("reasoning", "")
        
        # 관계 정보 설정
        if relationship_data:
            # 참조 객체 ID 설정
            reference_object = relationship_data.get("reference_object", {})
            if reference_object and "id" in reference_object:
                result["reference_object_id"] = reference_object.get("id")
                
            # 신뢰도와 이유 업데이트 (있는 경우)
            if "confidence" in relationship_data:
                result["confidence"] = max(result["confidence"], relationship_data.get("confidence", 0.0))
            if "reasoning" in relationship_data and relationship_data.get("reasoning"):
                result["reasoning"] += " " + relationship_data.get("reasoning", "")
        
        # 최종 통합 결과가 제공된 경우
        if final_data:
            # 목표 지점 설정
            if "goal_point" in final_data and final_data["goal_point"]:
                result["goal_point"] = final_data["goal_point"]
                
            # 신뢰도와 이유 업데이트
            if "confidence" in final_data:
                result["confidence"] = final_data.get("confidence", result["confidence"])
            if "reasoning" in final_data and final_data.get("reasoning"):
                result["reasoning"] = final_data.get("reasoning", result["reasoning"])
                
            # 방향 타입과 방향 업데이트
            if "direction_type" in final_data:
                result["direction_type"] = final_data.get("direction_type", result["direction_type"])
            if "direction" in final_data:
                result["direction"] = final_data.get("direction", result["direction"])
        
        return result
        