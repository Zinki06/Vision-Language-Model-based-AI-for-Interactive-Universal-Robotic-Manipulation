#!/usr/bin/env python
"""
타겟 물체 추론 정확도 향상 모듈

이 모듈은 mediapipe, yolo, midas의 결과를 통합하여
타겟 물체 추론 정확도를 향상시키는 기능을 제공합니다.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import re

class TargetInferenceEnhancer:
    """
    타겟 물체 추론 정확도 향상 클래스
    
    mediapipe, yolo, midas의 결과를 연계하여 사용자 의도에 맞는
    타겟 물체를 더 정확히 추론하는 기능을 제공합니다.
    """
    
    def __init__(self, llm=None, logger: Optional[logging.Logger] = None):
        """
        TargetInferenceEnhancer 초기화
        
        Args:
            llm: LLM 인터페이스 객체
            logger: 로거 객체 (기본값: None, 내부 생성)
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("TargetInferenceEnhancer")
        
        # LLM 설정
        self.llm = llm
        
        # 디버깅 로그를 저장할 내부 리스트
        self.debug_logs = []
        
        self.logger.info("TargetInferenceEnhancer 초기화 완료")
    
    def enhance_target_inference(self, 
                               detections: List[Dict[str, Any]], 
                               gesture_results: List[Dict[str, Any]], 
                               depth_map: np.ndarray,
                               user_prompt: Optional[str] = None,
                               img_width: int = 1280,
                               img_height: int = 720) -> Dict[str, Any]:
        """
        3개 센서(mediapipe, yolo, midas) 결과의 연계성을 강화하여 타겟 물체 추론
        
        Args:
            detections: YOLO로 감지된 객체 목록
            gesture_results: MediaPipe로 감지된 제스처 결과
            depth_map: MiDaS로 생성된 깊이 맵
            user_prompt: 사용자 프롬프트 (기본값: None)
            img_width: 이미지 너비 (기본값: 1280)
            img_height: 이미지 높이 (기본값: 720)
            
        Returns:
            추론 결과 {"target_idx": idx, "reference_idx": idx, "confidence": float, "method": str, "logs": List[str]}
        """
        # 초기화
        result = {
            "target_idx": 0,  # 기본값: 첫 번째 객체
            "reference_idx": 1 if len(detections) > 1 else 0,  # 기본값: 두 번째 객체 (없으면 첫 번째)
            "confidence": 0.0,  # 신뢰도
            "method": "default",  # 추론 방법
        }
        target_idx, reference_idx = result["target_idx"], result["reference_idx"]
        confidence = 0.0
        method = "default"
        
        self.debug_logs = []  # 로그 초기화
        self._log(f"타겟 추론 강화 시작: 감지된 객체 {len(detections)}개, 제스처 {len(gesture_results)}개")
        
        # 감지된 객체가 없으면 빈 결과 반환
        if not detections:
            self._log("감지된 객체가 없습니다.")
            result["logs"] = self.debug_logs
            return result

        # 객체가 하나만 있으면 자기 자신을 타겟/레퍼런스로 설정하고 종료
        if len(detections) == 1:
            self._log("객체가 하나만 감지됨. 타겟/레퍼런스로 설정.")
            result["target_idx"] = 0
            result["reference_idx"] = 0
            result["method"] = "single_object"
            result["confidence"] = 0.5 # 단일 객체는 신뢰도 중간
            result["logs"] = self.debug_logs
            return result
            
        # 각 추론 방법별 결과 저장
        inference_candidates = []

        # 1. 제스처 기반 추론
        gesture_based_target = self._infer_target_from_gesture(
            detections, gesture_results, depth_map, img_width, img_height
        )
        if gesture_based_target["confidence"] > 0:
            inference_candidates.append(gesture_based_target)
            self._log(f"제스처 기반 후보: T={gesture_based_target['target_idx']}, R={gesture_based_target['reference_idx']}, C={gesture_based_target['confidence']:.2f}")
            
        # 2. 프롬프트 기반 추론
        if user_prompt:
            prompt_based_target = self._infer_target_from_prompt(
                detections, user_prompt, gesture_based_target # 제스처 결과를 힌트로 사용 가능
            )
            if prompt_based_target["confidence"] > 0:
                inference_candidates.append(prompt_based_target)
                self._log(f"프롬프트 기반 후보: T={prompt_based_target['target_idx']}, R={prompt_based_target['reference_idx']}, C={prompt_based_target['confidence']:.2f}")
        
        # 3. 깊이 기반 추론 (가장 가까운 객체)
        depth_based_target = self._infer_target_from_depth(detections)
        if depth_based_target["confidence"] > 0:
            inference_candidates.append(depth_based_target)
            self._log(f"깊이 기반 후보: T={depth_based_target['target_idx']}, R={depth_based_target['reference_idx']}, C={depth_based_target['confidence']:.2f}")
        
        # 4. 기본 선택 후보 추가 (낮은 신뢰도)
        default_target = {
            "target_idx": 0,
            "reference_idx": 1,
            "confidence": 0.1, # 기본 선택은 매우 낮은 신뢰도
            "method": "default"
        }
        inference_candidates.append(default_target)
        self._log(f"기본 후보: T=0, R=1, C=0.1")

        # --- 최종 결정: 신뢰도가 가장 높은 후보 선택 --- 
        if inference_candidates:
            # 신뢰도 기준으로 정렬 (내림차순)
            inference_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            best_candidate = inference_candidates[0]
            
            target_idx = best_candidate["target_idx"]
            reference_idx = best_candidate["reference_idx"]
            confidence = best_candidate["confidence"]
            method = best_candidate["method"]
            self._log(f"최종 후보 선택됨 (신뢰도 기반): 방법='{method}', T={target_idx}, R={reference_idx}, C={confidence:.2f}")
        else:
             # 후보가 없는 비정상적인 경우, 기본값 사용
             self._log("경고: 추론 후보가 없음. 기본값 사용.")
             target_idx, reference_idx = 0, 1
             confidence = 0.0
             method = "fallback_default"

        # --- 일관성 검증 및 최종 교정 --- 
        verified_target_idx, verified_reference_idx = self._verify_target_reference_consistency(
            target_idx, reference_idx, detections, user_prompt
        )
        
        if target_idx != verified_target_idx or reference_idx != verified_reference_idx:
             self._log(f"일관성 검증 후 교정됨: T={target_idx}->{verified_target_idx}, R={reference_idx}->{verified_reference_idx}")
             method += "_verified" # 교정되었음을 명시
        
        # 최종 결과 업데이트
        result["target_idx"] = verified_target_idx
        result["reference_idx"] = verified_reference_idx
        result["confidence"] = confidence # 신뢰도는 교정 전 최고값 유지
        result["method"] = method
        result["logs"] = self.debug_logs
        
        self._log(f"타겟 추론 강화 완료: 최종 T={result['target_idx']}, 최종 R={result['reference_idx']}, 방법='{result['method']}', 신뢰도={result['confidence']:.2f}")
        return result
    
    def _infer_target_from_gesture(self, 
                                 detections: List[Dict[str, Any]], 
                                 gesture_results: List[Dict[str, Any]], 
                                 depth_map: np.ndarray,
                                 img_width: int,
                                 img_height: int) -> Dict[str, Any]:
        """
        제스처(POINTING, GRABBING)와 3D 정보를 이용하여 타겟 추론
        
        Args:
            detections: YOLO로 감지된 객체 목록
            gesture_results: MediaPipe로 감지된 제스처 결과
            depth_map: MiDaS로 생성된 깊이 맵
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            추론 결과 {"target_idx": idx, "reference_idx": idx, "confidence": float, "method": str}
        """
        result = {
            "target_idx": 0,
            "reference_idx": 1 if len(detections) > 1 else 0,
            "confidence": 0.0,
            "method": "gesture"
        }
        
        # 제스처 또는 감지된 객체가 없으면 낮은 신뢰도 반환
        if not gesture_results or not detections:
            self._log("제스처 기반 추론: 제스처 또는 객체 없음")
            return result
        
        # 우선 POINTING 제스처 확인
        pointing_gesture = next((g for g in gesture_results if g.get('gesture') == 'POINTING' and g.get('pointing_vector') is not None), None)
        
        if pointing_gesture and 'pointing_vector' in pointing_gesture and 'landmarks_3d' in pointing_gesture:
            self._log("POINTING 제스처 감지됨")
            
            # 가리키는 방향 벡터와 손목 위치 가져오기
            landmarks_3d = pointing_gesture['landmarks_3d']
            mp_wrist = np.array([landmarks_3d[0]['x'], landmarks_3d[0]['y'], landmarks_3d[0]['z']])
            mp_pointing_vector = pointing_gesture['pointing_vector']
            
            # 개선 1: MediaPipe 좌표계 변환 명확화 (Y, Z 부호 반전)
            ray_origin = np.array([
                mp_wrist[0],          # x는 그대로
                -mp_wrist[1],         # y축 반전
                -mp_wrist[2]          # z축 반전
            ])
            
            ray_direction = np.array([
                mp_pointing_vector[0],  # x는 그대로
                -mp_pointing_vector[1], # y축 반전
                -mp_pointing_vector[2]  # z축 반전
            ])
            
            # 개선 2: 방향 벡터 정규화 검증
            norm_dir = np.linalg.norm(ray_direction)
            if norm_dir < 1e-6:
                self._log("경고: 방향 벡터가 너무 작습니다. 기본 방향 벡터(카메라 방향)로 대체합니다.")
                # 기본 방향 벡터 (카메라 방향으로 전방 지향)
                ray_direction = np.array([0, 0, -1])
                norm_dir = 1.0
            else:
                ray_direction /= norm_dir
            
            # 각 객체와의 각도 및 거리 계산
            best_match_idx = -1
            min_score = float('inf')
            angle_threshold_deg = 30.0  # 최대 각도 편차
            distance_threshold_m = 2.5  # 최대 거리
            
            # 개선 3: 각도 및 거리 가중치 재조정
            angle_weight = 2.0        # 각도에 더 높은 가중치 부여
            distance_weight = 1.0     # 거리 가중치는 낮게 유지
            
            # 화면 중앙 좌표 (이미지 좌표계)
            screen_center = np.array([img_width/2, img_height/2])
            
            for idx, obj in enumerate(detections):
                if '3d_coords' not in obj or not isinstance(obj['3d_coords'], dict):
                    continue
                    
                coords = obj['3d_coords']
                if not all(k in coords for k in ['x_cm', 'y_cm', 'z_cm']):
                    continue
                
                # 객체 중심 좌표 (미터 단위)
                obj_center_m = np.array([
                    coords.get('x_cm', 0) / 100.0,
                    coords.get('y_cm', 0) / 100.0,
                    coords.get('z_cm', 0) / 100.0
                ])
                
                # 손에서 객체까지 벡터
                vector_to_obj = obj_center_m - ray_origin
                distance_to_obj = np.linalg.norm(vector_to_obj)
                
                # 거리가 0이거나 임계값을 초과하면 건너뛰기
                if distance_to_obj < 1e-6 or distance_to_obj > distance_threshold_m:
                    continue
                
                # 방향 벡터와 객체 벡터 사이 각도 계산
                dot_product = np.dot(ray_direction, vector_to_obj / distance_to_obj)
                angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                
                # 개선 4: 시선-손 일치성 검증 (객체가 손과 시선 사이에 있는지)
                in_sight_line = True
                if 'bbox' in obj:
                    bbox = obj['bbox']
                    obj_center_2d = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    
                    # 중심점에서 객체 중심까지의 벡터와 중심점에서 손까지의 벡터 각도 계산
                    # 이 각도가 작을수록 객체가 손과 시선 사이에 있을 가능성 높음
                    obj_to_center_dist = np.linalg.norm(np.array(obj_center_2d) - screen_center)
                    
                    # 시선 일치도를 가중치에 반영 (화면 중앙에 가까운 객체 선호)
                    sight_alignment = obj_to_center_dist / (np.sqrt(img_width**2 + img_height**2) / 2)
                    # 0~1 범위로 정규화 (0: 완벽한 일치, 1: 최대 불일치)
                    sight_alignment = min(1.0, sight_alignment)
                else:
                    sight_alignment = 0.5  # 기본값
                
                self._log(f"객체 {idx} ({obj.get('class_name', 'unknown')}): 거리={distance_to_obj:.2f}m, 각도={angle:.1f}도, 시선일치도={sight_alignment:.2f}")
                
                # 각도가 임계값 이내인 경우 점수 계산 (가중치 적용)
                if angle <= angle_threshold_deg:
                    # 개선된 점수 계산: 각도, 거리, 시선 일치도를 고려
                    score = (angle * angle_weight) + (distance_to_obj * distance_weight) + (sight_alignment * 3.0)
                    
                    if score < min_score:
                        min_score = score
                        best_match_idx = idx
                        self._log(f" -> 새로운 최적 매치: 점수={score:.2f} (각도={angle:.1f}×{angle_weight}, 거리={distance_to_obj:.2f}×{distance_weight}, 시선={sight_alignment:.2f}×3.0)")
            
            # 최적 매치 발견된 경우
            if best_match_idx >= 0:
                self._log(f"POINTING 제스처 최적 타겟: 인덱스={best_match_idx}, 객체={detections[best_match_idx].get('class_name', 'unknown')}")
                result["target_idx"] = best_match_idx
                
                # 레퍼런스 객체 선택 (가장 가까운 다른 객체)
                if len(detections) > 1:
                    reference_candidates = [(i, self._calculate_3d_distance(detections[best_match_idx], obj)) 
                                          for i, obj in enumerate(detections) if i != best_match_idx]
                    if reference_candidates:
                        reference_candidates.sort(key=lambda x: x[1])  # 거리순 정렬
                        result["reference_idx"] = reference_candidates[0][0]
                
                # 신뢰도 계산: 최대 점수 대비 실제 점수의 역수 (낮은 점수가 높은 신뢰도)
                max_possible_score = angle_threshold_deg * angle_weight + distance_threshold_m * distance_weight + 1.0 * 3.0
                confidence = 1.0 - (min_score / max_possible_score)
                result["confidence"] = min(0.9, max(0.7, confidence))  # 0.7~0.9 범위로 제한
                return result
        
        # GRABBING 제스처 확인
        grabbing_gesture = next((g for g in gesture_results if g.get('gesture') == 'GRABBING' and g.get('hand_center_3d') is not None), None)
        
        if grabbing_gesture and 'hand_center_3d' in grabbing_gesture:
            self._log("GRABBING 제스처 감지됨")
            
            # 손 중심 위치 가져오기
            mp_hand_center = grabbing_gesture['hand_center_3d']
            
            # 개선 1: MediaPipe 좌표계 변환 명확화 (Y, Z 부호 반전)
            hand_center = np.array([
                mp_hand_center[0],     # x는 그대로
                -mp_hand_center[1],    # y축 반전
                -mp_hand_center[2]     # z축 반전
            ])
            
            # 각 객체와의 거리 계산
            best_match_idx = -1
            min_dist = float('inf')
            distance_threshold_m = 0.25  # 최대 거리 (미터)
            
            # 화면 중앙 좌표 (이미지 좌표계) - 시선 검증 사용
            screen_center = np.array([img_width/2, img_height/2])
            
            for idx, obj in enumerate(detections):
                if '3d_coords' not in obj or not isinstance(obj['3d_coords'], dict):
                    continue
                    
                coords = obj['3d_coords']
                if not all(k in coords for k in ['x_cm', 'y_cm', 'z_cm']):
                    continue
                
                # 객체 중심 좌표 (미터 단위)
                obj_center_m = np.array([
                    coords.get('x_cm', 0) / 100.0,
                    coords.get('y_cm', 0) / 100.0,
                    coords.get('z_cm', 0) / 100.0
                ])
                
                # 손과 객체 사이 거리
                distance = np.linalg.norm(obj_center_m - hand_center)
                
                # 개선: 시선 방향과의 일치도 확인
                sight_factor = 1.0
                if 'bbox' in obj:
                    bbox = obj['bbox']
                    obj_center_2d = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    obj_to_center_dist = np.linalg.norm(np.array(obj_center_2d) - screen_center)
                    
                    # 화면 중앙에 가까운 객체일수록 가중치 감소 (거리 점수 향상)
                    sight_factor = 0.8 + (0.4 * (obj_to_center_dist / (np.sqrt(img_width**2 + img_height**2) / 2)))
                    sight_factor = min(1.2, max(0.8, sight_factor))  # 0.8~1.2 범위로 제한
                
                # 시선 가중치를 적용한 거리
                adjusted_distance = distance * sight_factor
                
                self._log(f"객체 {idx} ({obj.get('class_name', 'unknown')}): 손과의 거리={distance:.2f}m, 시선 가중치={sight_factor:.2f}, 보정 거리={adjusted_distance:.2f}m")
                
                if adjusted_distance < min_dist and distance <= distance_threshold_m:
                    min_dist = adjusted_distance
                    best_match_idx = idx
                    self._log(f" -> 새로운 최적 매치: 보정 거리={adjusted_distance:.2f}m")
            
            # 최적 매치 발견된 경우
            if best_match_idx >= 0:
                self._log(f"GRABBING 제스처 최적 타겟: 인덱스={best_match_idx}, 객체={detections[best_match_idx].get('class_name', 'unknown')}")
                result["target_idx"] = best_match_idx
                
                # 레퍼런스 객체 선택 (가장 가까운 다른 객체)
                if len(detections) > 1:
                    reference_candidates = [(i, self._calculate_3d_distance(detections[best_match_idx], obj)) 
                                          for i, obj in enumerate(detections) if i != best_match_idx]
                    if reference_candidates:
                        reference_candidates.sort(key=lambda x: x[1])  # 거리순 정렬
                        result["reference_idx"] = reference_candidates[0][0]
                
                # 신뢰도 계산: 임계 거리 대비 실제 거리 (원래 거리 사용)
                dist_confidence = 1.0 - (min_dist / (distance_threshold_m * 1.2))  # 최대 보정 거리 고려
                result["confidence"] = min(0.9, max(0.7, dist_confidence))  # 0.7~0.9 범위로 제한
                return result
        
        self._log("적합한 제스처 없음 또는 매칭되는 객체 없음")
        return result
    
    def _infer_target_from_prompt(self, 
                                detections: List[Dict[str, Any]], 
                                user_prompt: str,
                                gesture_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 프롬프트와 YOLO 라벨 매칭을 통한 타겟 추론
        
        Args:
            detections: YOLO로 감지된 객체 목록
            user_prompt: 사용자 프롬프트
            gesture_result: 제스처 기반 추론 결과 (보조 자료로 활용)
            
        Returns:
            추론 결과 {"target_idx": idx, "reference_idx": idx, "confidence": float, "method": str}
        """
        result = {
            "target_idx": 0,
            "reference_idx": 1 if len(detections) > 1 else 0,
            "confidence": 0.0,
            "method": "prompt"
        }
        
        # 프롬프트나 감지된 객체가 없으면 낮은 신뢰도 반환
        if not user_prompt or not detections:
            self._log("프롬프트 기반 추론: 프롬프트 또는 객체 없음")
            return result
        
        # 프롬프트 소문자 변환 및 정규화
        prompt_lower = user_prompt.lower()
        
        # 객체 클래스 이름 추출 및 소문자 변환
        class_names = [obj.get("class_name", "").lower() for obj in detections]
        
        # 개선 1: 한국어 문법 패턴 기반 타겟/레퍼런스 추출
        # 한국어 문장 패턴: "A를 B 위에/옆에/앞에/뒤에 놓아줘" 형태
        # A: 타겟 객체, B: 레퍼런스 객체
        target_ref_patterns = [
            # "A를 B 위치에" 패턴
            r'(.+?)[을를](.*?)([옆위아래앞뒤쪽안]에|[옆위아래앞뒤]쪽에|주변에|근처에|가까이에|멀리에)',
            # "A를 B에 놓아/두어/올려" 패턴
            r'(.+?)[을를](.*?)에\s*(?:놓|두|올려|넣|이동|배치|위치)'
        ]
        
        # 이동 동작 관련 단어
        moving_actions = ["놓", "두", "올려", "이동", "옮겨", "배치", "가져다", "넣"]
        contains_moving_action = any(action in prompt_lower for action in moving_actions)
        
        # 우선 패턴 매칭 시도
        pattern_matched = False
        target_name = ""
        reference_name = ""
        
        for pattern in target_ref_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                target_name = match.group(1).strip()
                reference_group = match.group(2).strip()
                
                # 레퍼런스 객체 추출 (중간 그룹에서 전치사 제거)
                for prep in ["을", "를", "에게", "에서", "에"]:
                    if prep in reference_group:
                        reference_name = reference_group.split(prep)[0].strip()
                        break
                else:
                    reference_name = reference_group.strip()
                
                self._log(f"문법 패턴 매칭 성공: 타겟='{target_name}', 레퍼런스='{reference_name}'")
                pattern_matched = True
                break
        
        # 패턴 매칭 성공 시 객체와 매칭
        if pattern_matched and (target_name or reference_name):
            target_candidates = []
            reference_candidates = []
            
            # 객체명 매칭 (부분 일치 허용)
            for idx, name in enumerate(class_names):
                if name and target_name and name in target_name or target_name in name:
                    similarity = len(set(name) & set(target_name)) / max(len(name), len(target_name))
                    target_candidates.append((idx, similarity))
                    self._log(f"타겟 후보: '{name}' (인덱스 {idx}, 유사도 {similarity:.2f})")
                    
                if name and reference_name and name in reference_name or reference_name in name:
                    similarity = len(set(name) & set(reference_name)) / max(len(name), len(reference_name))
                    reference_candidates.append((idx, similarity))
                    self._log(f"레퍼런스 후보: '{name}' (인덱스 {idx}, 유사도 {similarity:.2f})")
            
            # 후보가 있으면 가장 유사도가 높은 것을 선택
            if target_candidates:
                target_candidates.sort(key=lambda x: x[1], reverse=True)
                target_idx = target_candidates[0][0]
                
                if reference_candidates:
                    reference_candidates.sort(key=lambda x: x[1], reverse=True)
                    # 타겟과 같은 객체가 레퍼런스로 선택되지 않도록
                    for ref_idx, _ in reference_candidates:
                        if ref_idx != target_idx:
                            reference_idx = ref_idx
                            break
                else:
                    # 레퍼런스 후보가 없으면 타겟이 아닌 가장 가까운 객체
                    reference_idx = next((i for i, _ in enumerate(detections) if i != target_idx), target_idx)
                
                self._log(f"패턴 기반 타겟: 인덱스={target_idx}, 객체={detections[target_idx].get('class_name', 'unknown')}")
                self._log(f"패턴 기반 레퍼런스: 인덱스={reference_idx}, 객체={detections[reference_idx].get('class_name', 'unknown')}")
                
                result["target_idx"] = target_idx
                result["reference_idx"] = reference_idx
                result["confidence"] = 0.85  # 문법 패턴 매칭으로 높은 신뢰도
                result["method"] = "prompt_pattern"
                return result
        
        # 패턴 매칭 실패 시 기존 방식 시도 (단순 언급 찾기)
        self._log("문법 패턴 매칭 실패, 단순 언급 기반 추론 시도")
        
        # 프롬프트에서 언급된 객체 탐색
        mentioned_objects = []
        for idx, name in enumerate(class_names):
            if name and name in prompt_lower:
                mentioned_objects.append((idx, prompt_lower.index(name)))  # (인덱스, 언급 위치)
                self._log(f"프롬프트에서 객체 발견: '{name}' (인덱스 {idx})")
        
        # 언급된 객체가 있는 경우
        if mentioned_objects:
            # 개선 2: 이동 동작 단어와 함께 판단
            if contains_moving_action and len(mentioned_objects) > 1:
                # 이동 동작이 있으면 첫 번째 언급 객체가 타겟일 가능성이 높음
                mentioned_objects.sort(key=lambda x: x[1])
                
                # 첫 번째 언급 객체를 타겟으로, 두 번째 언급 객체를 레퍼런스로 설정
                target_idx = mentioned_objects[0][0]
                reference_idx = mentioned_objects[1][0]
                self._log(f"이동 동작 있음, 첫 번째 언급 객체를 타겟으로 선택")
            else:
                # 이동 동작이 없으면 크기나 위치 기반으로 판단
                
                # 객체 크기 비교 (일반적으로 더 작은 물체가 타겟)
                if len(mentioned_objects) > 1:
                    # 언급된 객체 인덱스 목록
                    mentioned_indices = [idx for idx, _ in mentioned_objects]
                    
                    # 면적 계산 (바운딩 박스 기준)
                    areas = []
                    for idx in mentioned_indices:
                        bbox = detections[idx].get("bbox", [0, 0, 1, 1])
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        areas.append((idx, area))
                    
                    # 면적 순으로 정렬 (작은 것부터)
                    areas.sort(key=lambda x: x[1])
                    
                    # 작은 객체를 타겟으로, 큰 객체를 레퍼런스로 설정 (일반적 상황)
                    target_idx = areas[0][0]
                    reference_idx = areas[1][0]
                    
                    # 크기 차이가 뚜렷한 경우만 적용 (2배 이상 차이)
                    if areas[1][1] > areas[0][1] * 2:
                        self._log(f"면적 비교: 작은 객체를 타겟으로 선택 (타겟={areas[0][1]}, 레퍼런스={areas[1][1]})")
                    else:
                        # 크기 차이가 크지 않으면 언급 순서대로
                        mentioned_objects.sort(key=lambda x: x[1])
                        target_idx = mentioned_objects[0][0]
                        reference_idx = mentioned_objects[1][0]
                        self._log(f"면적 차이가 작음, 언급 순서대로 선택")
                else:
                    # 언급된 객체가 하나뿐이면 그것을 타겟으로
                    target_idx = mentioned_objects[0][0]
                    reference_idx = next((i for i, _ in enumerate(detections) if i != target_idx), target_idx)
            
            self._log(f"프롬프트 기반 타겟: 인덱스={target_idx}, 객체={detections[target_idx].get('class_name', 'unknown')}")
            self._log(f"프롬프트 기반 레퍼런스: 인덱스={reference_idx}, 객체={detections[reference_idx].get('class_name', 'unknown')}")
            
            result["target_idx"] = target_idx
            result["reference_idx"] = reference_idx
            result["confidence"] = 0.8  # 명시적 언급이므로 높은 신뢰도
            return result
        
        # 개선 3: 객체 언급 없이 지시대명사만 있는 경우 (이거, 저거 등)
        demonstratives = ["이거", "저거", "그거", "이것", "저것", "그것", "얘", "쟤", "걔"]
        has_demonstrative = any(d in prompt_lower for d in demonstratives)
        
        if has_demonstrative and contains_moving_action:
            self._log(f"지시대명사와 이동 동작 감지됨")
            
            # 제스처 결과가 있으면 그것을 활용
            if gesture_result["confidence"] > 0.0:
                self._log(f"지시대명사 + 이동 동작 + 제스처 결과 활용 (신뢰도: {gesture_result['confidence']:.2f})")
                adjusted_confidence = min(0.8, gesture_result["confidence"] + 0.15)  # 향상된 신뢰도
                result["target_idx"] = gesture_result["target_idx"]
                result["reference_idx"] = gesture_result["reference_idx"]
                result["confidence"] = adjusted_confidence
                result["method"] = "demonstrative+gesture"
                return result
            
            # 제스처 결과가 없으면 화면에 더 가까운 물체를 타겟으로
            if len(detections) > 1:
                # 깊이 기준 정렬
                depth_sorted = []
                for idx, obj in enumerate(detections):
                    if "depth" in obj and "center_depth" in obj["depth"]:
                        depth_sorted.append((idx, obj["depth"]["center_depth"]))
                
                if depth_sorted:
                    depth_sorted.sort(key=lambda x: x[1])  # 가까운 순서대로
                    target_idx = depth_sorted[0][0]
                    reference_idx = depth_sorted[1][0]
                    
                    self._log(f"지시대명사 + 이동 동작 + 깊이 정보: 가까운 객체를 타겟으로 선택")
                    result["target_idx"] = target_idx
                    result["reference_idx"] = reference_idx
                    result["confidence"] = 0.7
                    result["method"] = "demonstrative+depth"
                    return result
        
        # 지시대명사만 있고 이동 동작이 없으면 기존 처리
        elif has_demonstrative and gesture_result["confidence"] > 0.0:
            # 제스처 결과를 보완 자료로 활용
            self._log(f"지시대명사 감지됨, 제스처 결과 활용 (신뢰도: {gesture_result['confidence']:.2f})")
            adjusted_confidence = min(0.7, gesture_result["confidence"] + 0.1)  # 약간 향상된 신뢰도
            result["target_idx"] = gesture_result["target_idx"]
            result["reference_idx"] = gesture_result["reference_idx"]
            result["confidence"] = adjusted_confidence
            result["method"] = "prompt+gesture"
            return result
        
        # 매칭되는 객체가 없으면 낮은 신뢰도 반환
        self._log("프롬프트에서 객체 언급 또는 지시대명사 발견 실패")
        return result
    
    def _infer_target_from_depth(self, 
                               detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        깊이 정보를 이용한 타겟 추론 (가장 가까운 객체를 타겟으로 선택)
        
        Args:
            detections: YOLO로 감지된 객체 목록
            
        Returns:
            추론 결과 {"target_idx": idx, "reference_idx": idx, "confidence": float, "method": str}
        """
        result = {
            "target_idx": 0,
            "reference_idx": 1 if len(detections) > 1 else 0,
            "confidence": 0.0,
            "method": "depth"
        }
        
        # 감지된 객체가 없으면 낮은 신뢰도 반환
        if not detections:
            self._log("깊이 기반 추론: 객체 없음")
            return result
        
        # 깊이 정보를 이용하여 객체 정렬 (가장 가까운 객체부터)
        sorted_indices = []
        for idx, obj in enumerate(detections):
            if "depth" in obj and "center_depth" in obj["depth"]:
                depth = obj["depth"]["center_depth"]
                sorted_indices.append((idx, depth))
                self._log(f"객체 {idx} ({obj.get('class_name', 'unknown')}): 깊이={depth:.2f}")
        
        # 깊이 정보가 있는 객체가 없으면 기본값 반환
        if not sorted_indices:
            self._log("깊이 정보가 있는 객체가 없습니다.")
            return result
        
        # 깊이 기준 정렬
        sorted_indices.sort(key=lambda x: x[1])
        
        # 가장 가까운 객체를 타겟으로, 두 번째로 가까운 객체를 레퍼런스로 설정
        target_idx = sorted_indices[0][0]
        reference_idx = sorted_indices[1][0] if len(sorted_indices) > 1 else target_idx
        
        self._log(f"깊이 기반 타겟: 인덱스={target_idx}, 객체={detections[target_idx].get('class_name', 'unknown')}")
        self._log(f"깊이 기반 레퍼런스: 인덱스={reference_idx}, 객체={detections[reference_idx].get('class_name', 'unknown')}")
        
        result["target_idx"] = target_idx
        result["reference_idx"] = reference_idx
        result["confidence"] = 0.6  # 중간 정도 신뢰도
        return result
    
    def _calculate_3d_distance(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> float:
        """
        두 객체 간의 3D 거리 계산
        
        Args:
            obj1: 첫 번째 객체
            obj2: 두 번째 객체
            
        Returns:
            두 객체 간의 3D 거리 (미터)
        """
        # 3D 좌표 정보가 없으면 큰 값(무한대) 반환
        if '3d_coords' not in obj1 or '3d_coords' not in obj2:
            return float('inf')
            
        coords1 = obj1['3d_coords']
        coords2 = obj2['3d_coords']
        
        if not all(k in coords1 for k in ['x_cm', 'y_cm', 'z_cm']) or not all(k in coords2 for k in ['x_cm', 'y_cm', 'z_cm']):
            return float('inf')
        
        # 3D 좌표 추출 (미터 단위)
        p1 = np.array([coords1['x_cm'] / 100.0, coords1['y_cm'] / 100.0, coords1['z_cm'] / 100.0])
        p2 = np.array([coords2['x_cm'] / 100.0, coords2['y_cm'] / 100.0, coords2['z_cm'] / 100.0])
        
        # 유클리드 거리 계산
        return np.linalg.norm(p1 - p2)
    
    def _log(self, message: str) -> None:
        """
        디버깅 로그 추가
        
        Args:
            message: 로그 메시지
        """
        self.logger.debug(message)
        self.debug_logs.append(message)
    
    def get_debug_logs(self) -> List[str]:
        """
        디버깅 로그 반환
        
        Returns:
            디버깅 로그 리스트
        """
        return self.debug_logs.copy()
    
    def save_debug_logs(self, filepath: str) -> bool:
        """
        디버깅 로그를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.debug_logs))
            self.logger.info(f"디버깅 로그 저장 완료: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"디버깅 로그 저장 실패: {e}")
            return False

    def _verify_target_reference_consistency(self, 
                                        target_idx: int, 
                                        reference_idx: int, 
                                        detections: List[Dict[str, Any]], 
                                        user_prompt: Optional[str] = None) -> Tuple[int, int]:
        """
        추론된 타겟/레퍼런스 인덱스의 일관성을 검증하고 필요한 경우 교체합니다.
        크기, 중앙성, 깊이, 고정성, 프롬프트 키워드 등을 기반으로 휴리스틱 검증 수행.

        Args:
            target_idx: 초기 추론된 타겟 인덱스
            reference_idx: 초기 추론된 레퍼런스 인덱스
            detections: 전체 객체 감지 목록
            user_prompt: 사용자 프롬프트

        Returns:
            (검증/교체된 타겟 인덱스, 검증/교체된 레퍼런스 인덱스)
        """
        self._log("타겟-레퍼런스 일관성 검증 시작...")
        if target_idx == reference_idx or len(detections) < 2:
            self._log("검증 불필요 (동일 인덱스 또는 객체 부족)")
            return target_idx, reference_idx
        
        target_obj = detections[target_idx]
        ref_obj = detections[reference_idx]
        
        swap_score = 0 # 스왑을 지지하는 점수
        log_reasons = [] # 스왑 결정 이유 로깅
        
        # --- 이동 동사 감지 --- 
        move_verbs = ["놓아", "옮겨", "가져와", "넣어", "올려", "내려", "두어", "move", "put", "place", "bring", "take"]
        is_move_action = False
        if user_prompt:
            for verb in move_verbs:
                if verb in user_prompt.lower():
                    is_move_action = True
                    log_reasons.append(f"이동 동사 감지('{verb}')")
                    self._log(f"이동 동사 감지: {verb}")
                    break
        # ---------------------

        # 1. 크기 기반 검증
        target_size = (target_obj['bbox'][2] - target_obj['bbox'][0]) * (target_obj['bbox'][3] - target_obj['bbox'][1])
        ref_size = (ref_obj['bbox'][2] - ref_obj['bbox'][0]) * (ref_obj['bbox'][3] - ref_obj['bbox'][1])
        # 이동 동작이 있을 때, 타겟이 레퍼런스보다 크면 스왑 점수 증가 (작은 것을 옮기는 경향)
        if is_move_action and target_size > ref_size * 1.5: 
            swap_score += 1
            log_reasons.append("크기: 이동 시 타겟이 레퍼런스보다 큼")
            self._log("크기 검증: 이동 동사 있고 타겟이 레퍼런스보다 큼 -> 스왑 점수 +1")
        elif not is_move_action and ref_size > target_size * 1.5:
            # 이동 동작 없을 때, 레퍼런스가 타겟보다 크면 스왑 점수 증가 (일반적) 
            # swap_score += 0.5 # 약한 증거
            # self._log("크기 검증: 이동 동사 없고 레퍼런스가 타겟보다 큼 -> 스왑 점수 +0.5")
            pass # 크기만으로는 애매할 수 있어 점수 부여 보류

        # 2. 중앙성 검증
        img_center_x = 640 # 임시값, 실제 이미지 너비 필요
        img_center_y = 360 # 임시값, 실제 이미지 높이 필요
        # TODO: 이미지 크기를 함수 인자로 받도록 수정 필요
        
        def calculate_center_distance(obj):
            center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
            center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
            return np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)

        target_center_dist = calculate_center_distance(target_obj)
        ref_center_dist = calculate_center_distance(ref_obj)
        # 이동 동작이 있을 때, 타겟이 레퍼런스보다 중앙에 가까우면 스왑 점수 증가 (보통 주변부를 옮김)
        if is_move_action and target_center_dist < ref_center_dist:
            swap_score += 0.5
            log_reasons.append("중앙성: 이동 시 타겟이 레퍼런스보다 중앙에 가까움")
            self._log("중앙성 검증: 이동 동사 있고 타겟이 중앙에 더 가까움 -> 스왑 점수 +0.5")
        elif not is_move_action and ref_center_dist < target_center_dist:
            # 이동 동작 없을 때, 레퍼런스가 중앙에 더 가까우면 스왑 점수 증가 (자연스러움)
            swap_score += 0.5
            log_reasons.append("중앙성: 이동 없을 시 레퍼런스가 타겟보다 중앙에 가까움")
            self._log("중앙성 검증: 이동 동사 없고 레퍼런스가 중앙에 더 가까움 -> 스왑 점수 +0.5")

        # 3. 깊이 검증
        target_depth = target_obj.get('depth', {}).get('avg_depth', 0.5)
        ref_depth = ref_obj.get('depth', {}).get('avg_depth', 0.5)
        # 이동 동작이 있을 때, 타겟이 레퍼런스보다 뒤에 있으면 스왑 점수 증가 (보통 앞의 것을 뒤로 옮김)
        if is_move_action and target_depth > ref_depth:
            swap_score += 1
            log_reasons.append("깊이: 이동 시 타겟이 레퍼런스보다 뒤에 있음")
            self._log("깊이 검증: 이동 동사 있고 타겟이 레퍼런스보다 뒤 -> 스왑 점수 +1")
        elif not is_move_action and target_depth > ref_depth:
             # 이동 없을 때, 타겟이 뒤에 있으면 자연스러움 (점수 없음)
             pass

        # 4. 고정 객체 검증 (예: 테이블, 책상, 모니터 등)
        fixed_objects = ["table", "desk", "monitor", "tv", "chair", "sofa", "bed", "bench", "shelf", "counter", "sink", "refrigerator"]
        target_is_fixed = any(fixed in target_obj.get("class_name", "") for fixed in fixed_objects)
        ref_is_fixed = any(fixed in ref_obj.get("class_name", "") for fixed in fixed_objects)
        # 이동 동작이 있을 때, 타겟이 고정 객체이고 레퍼런스는 아니면 스왑 점수 크게 증가
        if is_move_action and target_is_fixed and not ref_is_fixed:
            swap_score += 2
            log_reasons.append("고정성: 이동 시 타겟이 고정 객체임")
            self._log("고정성 검증: 이동 동사 있고 타겟이 고정 객체 -> 스왑 점수 +2")
        # 이동 동작 없을 때, 레퍼런스가 고정 객체이고 타겟은 아니면 자연스러움 (점수 없음)

        # 5. 언급 순서 검증 (LLM 결과가 있을 경우 활용)
        # TODO: LLM 분석 결과에서 target_name, reference_name을 받아와 비교하는 로직 추가
        # if llm_result and user_prompt:
        #    target_mention_index = user_prompt.lower().find(llm_result.get('target_name', ''))
        #    ref_mention_index = user_prompt.lower().find(llm_result.get('reference_name', ''))
        #    if target_mention_index != -1 and ref_mention_index != -1 and ref_mention_index < target_mention_index:
        #        # 레퍼런스가 타겟보다 먼저 언급되면 스왑 점수 약간 증가 (일반적이지 않은 경우)
        #        swap_score += 0.3
        #        log_reasons.append("언급순서: 레퍼런스가 타겟보다 먼저 언급됨")
        #        self._log("언급순서 검증: 레퍼런스가 먼저 언급됨 -> 스왑 점수 +0.3")
        
        self._log(f"최종 스왑 점수: {swap_score:.1f}")
        
        # 최종 결정: 스왑 점수가 임계값(예: 1.5) 이상이면 스왑
        if swap_score >= 1.5:
            self._log(f"일관성 검증 결과: 스왑 결정 (점수: {swap_score:.1f}, 이유: {', '.join(log_reasons)})")
            return reference_idx, target_idx
        else:
            self._log("일관성 검증 결과: 스왑하지 않음")
            return target_idx, reference_idx 