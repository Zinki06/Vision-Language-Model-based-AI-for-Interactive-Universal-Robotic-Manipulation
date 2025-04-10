#!/usr/bin/env python
"""
목표지점 추론 엔진 모듈

이 모듈은 객체 감지, 깊이 추정, LLM을 통합하여 목표지점을 추론하는 기능을 제공합니다.
데모 버전에서 모듈화된 구조로 재설계되었습니다.
"""

import os
import time
import json
import logging
import re  # For pronoun detection
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union
import math

from .yolo_detector import YOLODetector
from .depth_3d_mapper import Depth3DMapper
from .gpt4o_builder import GPT4oBuilder
from .spatial_analyzer import SpatialAnalyzer
from .visualization import VisualizationEngine
from .gesture_recognizer import GestureRecognizer, GestureType
from .result_storage import ResultStorage
from .target_inference_enhancer import TargetInferenceEnhancer
from .enhanced_prompting_pipeline import EnhancedPromptingPipeline
from .utils.geometry import calculate_iou, calculate_box_from_points, calculate_angle_2d
from .goal_inference_components.direction_factory import adapt_direction

logger = logging.getLogger('goal_inference')

class GoalPointInferenceEngine:
    """
    목표 위치 추론 엔진 클래스
    참조 객체와 방향에 기반하여 목표 위치를 3D 공간에서 계산하고
    이를 2D 이미지 상의 바운딩 박스로 변환하는 기능을 제공합니다.
    """
    
    def __init__(self, 
                 yolo_model="yolov8n.pt", 
                 llm_type="gpt4o",
                 conf_threshold=None, 
                 iou_threshold=None,
                 gesture_model_path="models/hand_landmarker.task",
                 logger=None):
        """
        GoalPointInferenceEngine 초기화
        
        Args:
            yolo_model: 사용할 YOLOv8 모델
            llm_type: 사용할 LLM 유형 ('gpt4o' 또는 'gemini')
            conf_threshold: 객체 감지 신뢰도 임계값
            iou_threshold: 객체 감지 IoU 임계값
            gesture_model_path: MediaPipe 제스처 인식 모델 경로
            logger: 로깅을 위한 로거 객체. None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("GoalInference")
        
        # 손 클래스 정의 - YOLO에서 손으로 인식할 클래스 목록
        self.HAND_CLASSES = ["hand", "person"]
        
        # 제스처 모드 설정 - 초기값 False
        self.gesture_mode_active = False
        
        # 3D 설정
        self.min_depth_cm = 30.0
        self.max_depth_cm = 300.0
        
        # 오프셋 상수 추가
        self.OFFSET_X_CM = 15.0  # 좌/우 오프셋 (cm)
        self.OFFSET_Z_CM = 10.0  # 앞쪽 오프셋 (카메라 기준 Z축) (cm)
        self.PLACEMENT_MARGIN_CM = 2.0  # 겹침 조정 시 사용 (cm)
        
        # 2D 방향 오프셋 정의 (화면 좌표계 기준)
        offset_x = 50  # 예시: 좌우 오프셋 픽셀
        offset_y = 70  # 예시: 상하 오프셋 픽셀
        self.direction_2d_offsets = {
            'front': (0, +offset_y),  # 앞으로 (화면 아래쪽) -> y 증가
            'back':  (0, -offset_y),  # 뒤로 (화면 위쪽) -> y 감소
            'left':  (-offset_x, 0),  # 왼쪽 -> x 감소
            'right': (+offset_x, 0),  # 오른쪽 -> x 증가
            'above': (0, -offset_y),  # 위로 (화면 위쪽) -> y 감소
            'below': (0, +offset_y),  # 아래로 (화면 아래쪽) -> y 증가
            'side':  (+offset_x, 0)   # 옆은 오른쪽과 동일하게 처리 (기존 관례 따름)
        }
        
        # YOLOv8 객체 감지기 초기화
        self.logger.info(f"YOLO: {yolo_model} 모델 로드 중...")
        self.yolo_detector = YOLODetector(
            model_name=yolo_model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        self.logger.info(f"YOLO: {yolo_model} 모델 로드 완료")
        
        # 깊이 추정기 초기화
        self.logger.info("Depth3DMapper 초기화 중...")
        self.depth_mapper = Depth3DMapper()
        self.logger.info("Depth3DMapper 초기화 완료")
        
        # 방향 벡터 정의 (X, Y, Z)
        # X: 좌우 방향 (왼쪽 -, 오른쪽 +)
        # Y: 상하 방향 (아래 -, 위 +)
        # Z: 전후 방향 (뒤 -, 앞 +)
        self.direction_vectors = {
            'front': [0, 0, 1],    # 앞
            'back': [0, 0, -1],    # 뒤
            'left': [-1, 0, 0],    # 왼쪽
            'right': [1, 0, 0],    # 오른쪽
            'above': [0, 1, 0],    # 위
            'below': [0, -1, 0],   # 아래
            'front_left': [-0.7, 0, 0.7],    # 앞 왼쪽
            'front_right': [0.7, 0, 0.7],    # 앞 오른쪽
            'back_left': [-0.7, 0, -0.7],    # 뒤 왼쪽
            'back_right': [0.7, 0, -0.7],    # 뒤 오른쪽
        }
        
        # 방향별 기본 거리 (미터 단위)
        self.default_distances = {
            'front': 0.5,      # 앞 0.5m
            'back': 0.5,       # 뒤 0.5m
            'left': 0.5,       # 왼쪽 0.5m
            'right': 0.5,      # 오른쪽 0.5m
            'above': 0.3,      # 위 0.3m
            'below': 0.3,      # 아래 0.3m
            'front_left': 0.5, # 앞 왼쪽 0.5m
            'front_right': 0.5, # 앞 오른쪽 0.5m
            'back_left': 0.5,  # 뒤 왼쪽 0.5m
            'back_right': 0.5, # 뒤 오른쪽 0.5m
        }
        
        # LLM 인터페이스 초기화
        self.logger.info(f"LLM: {llm_type} 인터페이스 초기화 중...")
        self.llm = GPT4oBuilder.create_gpt4o(
            temperature=0.1,
            logger=self.logger
        )
        self.logger.info(f"LLM: {llm_type} 인터페이스 초기화 완료")
        
        # 제스처 인식기 초기화
        self.logger.info(f"GestureRecognizer 초기화 중... ({gesture_model_path})")
        self.gesture_recognizer = GestureRecognizer(
            model_path=gesture_model_path
        )
        self.logger.info("GestureRecognizer 초기화 완료")
        
        # 시각화 엔진 초기화
        self.visualization = VisualizationEngine(logger=self.logger)
        
        # 공간 분석기 초기화
        self.spatial_analyzer = SpatialAnalyzer(logger=self.logger)
        
        # 결과 저장 엔진 초기화
        self.result_storage = ResultStorage(logger=self.logger)
        
        # 타깃 추론 강화 엔진 초기화
        self.target_enhancer = TargetInferenceEnhancer(
            llm=self.llm,
            logger=self.logger
        )
        
        # 향상된 프롬프팅 파이프라인 초기화
        self.prompting_pipeline = EnhancedPromptingPipeline(
            llm_interface=self.llm,
            logger=self.logger
        )
    
    def calculate_goal_3d_position(self, reference_object, direction, detections=None):
        """
        참조 객체와 방향을 기반으로 3D 공간에서 목표 위치를 계산합니다.
        
        Args:
            reference_object: 참조 객체 정보가 담긴 딕셔너리
            direction: 참조 객체로부터의 방향 문자열 또는 방향 객체
            detections: 감지된 객체 목록 (선택적)
            
        Returns:
            3D 목표 위치 정보가 담긴 딕셔너리
        """
        # 방향 객체 형식 변환
        direction_obj = adapt_direction(direction)
        
        # 새로운 계산 메서드 사용
        if direction_obj.get("type") != "simple":
            return self.calculate_goal_3d_position_v2(reference_object, direction_obj, detections)
            
        try:
            # 참조 객체의 3D 중심 좌표
            if '3d_coords' not in reference_object or 'center' not in reference_object['3d_coords']:
                self.logger.error(f"참조 객체에 3D 좌표 정보가 없습니다: {reference_object}")
                return None
            
            reference_center = reference_object['3d_coords']['center']
            
            # 참조 객체의 크기 정보
            if 'dimensions' in reference_object['3d_coords']:
                reference_dimensions = reference_object['3d_coords']['dimensions']
            else:
                reference_dimensions = [0.3, 0.3, 0.3]  # 기본 크기 가정
                self.logger.warning(f"참조 객체 크기 정보가 없어 기본값을 사용합니다: {reference_dimensions}")
            
            # 방향 벡터 가져오기
            if direction not in self.direction_vectors:
                self.logger.warning(f"알 수 없는 방향입니다: {direction}. 'front'를 사용합니다.")
                direction = 'front'
                
            direction_vector = self.direction_vectors[direction]
            
            # 방향별 오프셋 계산
            offset = self._calculate_placement_offset(direction)
            
            # 참조 객체의 크기 계산
            ref_width = reference_dimensions[0]
            ref_height = reference_dimensions[1]
            ref_depth = reference_dimensions[2]
            
            # 참조 객체의 중심 좌표
            ref_x = reference_center[0]
            ref_y = reference_center[1]
            ref_z = reference_center[2]
            
            # 바닥면 Y 좌표 추정 (객체 바닥면에 배치하기 위함)
            base_y = self.estimate_base_y(reference_object)
            
            # 목표 위치 계산 - 방향별 오프셋 적용
            goal_x = ref_x + offset[0]
            goal_y = base_y  # 바닥면 Y좌표 사용 (방향이 'above'나 'below'면 다르게 처리 필요)
            goal_z = ref_z + offset[2]
            
            # 위/아래 방향 특수 처리
            if direction == 'above':
                goal_y = ref_y + ref_height/2 + self.PLACEMENT_MARGIN_CM  # 객체 위에 배치
            elif direction == 'below':
                goal_y = base_y - self.PLACEMENT_MARGIN_CM  # 객체 아래에 배치 (바닥보다 약간 아래)
            
            # 결과 정보 생성
            result = {
                "success": True,
                "position": [goal_x, goal_y, goal_z],
                "reference_center": reference_center,
                "direction": direction,
                "direction_vector": direction_vector,
                "offset": offset,
                "timestamp": time.time()
            }
            
            self.logger.info(f"목표 3D 위치 계산 완료: {[goal_x, goal_y, goal_z]}, 방향: {direction}")
            return result
        except Exception as e:
            self.logger.error(f"목표 3D 위치 계산 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "position": [0, 0, 100],  # 기본 위치
                "direction": direction
            }
    
    def calculate_goal_3d_position_v2(self, reference_object, direction_obj, detections=None):
        """
        확장된 방향 객체를 사용하여 3D 공간에서 목표 위치를 계산합니다.
        
        Args:
            reference_object: 참조 객체 정보가 담긴 딕셔너리
            direction_obj: 참조 객체로부터의 방향 객체
            detections: 감지된 객체 목록 (선택적)
            
        Returns:
            3D 목표 위치 정보가 담긴 딕셔너리
        """
        try:
            # 참조 객체의 3D 중심 좌표
            if '3d_coords' not in reference_object or 'center' not in reference_object['3d_coords']:
                self.logger.error(f"참조 객체에 3D 좌표 정보가 없습니다: {reference_object}")
                return {
                    "success": False,
                    "error": "참조 객체에 3D 좌표 정보가 없습니다",
                    "position": [0, 0, 100]  # 기본 위치
                }
            
            reference_center = reference_object['3d_coords']['center']
            
            # 참조 객체의 크기 정보
            if 'dimensions' in reference_object['3d_coords']:
                reference_dimensions = reference_object['3d_coords']['dimensions']
            else:
                reference_dimensions = [0.3, 0.3, 0.3]  # 기본 크기 가정
                self.logger.warning(f"참조 객체 크기 정보가 없어 기본값을 사용합니다: {reference_dimensions}")
            
            # 방향 객체에서 정보 추출
            dir_type = direction_obj.get("type", "simple")
            direction_vector = None
            direction_name = None
            
            # 방향 타입에 따른 처리
            if dir_type == "simple":
                direction_value = direction_obj.get("value", "front")
                if direction_value not in self.direction_vectors:
                    self.logger.warning(f"알 수 없는 방향입니다: {direction_value}. 'front'를 사용합니다.")
                    direction_value = 'front'
                
                direction_vector = self.direction_vectors[direction_value]
                direction_name = direction_value
            elif dir_type == "random":
                # 랜덤 방향 처리
                options = direction_obj.get("options", [])
                weights = direction_obj.get("weights", [1.0/len(options)] * len(options))
                
                if not options:
                    self.logger.warning("랜덤 방향 옵션이 비어있습니다. 'front'를 사용합니다.")
                    direction_vector = self.direction_vectors["front"]
                    direction_name = "front"
                else:
                    # 옵션 중 유효한 방향만 필터링
                    valid_options = [opt for opt in options if opt in self.direction_vectors]
                    valid_weights = []
                    
                    # 유효한 옵션에 대한 가중치 조정
                    if valid_options:
                        valid_indices = [options.index(opt) for opt in valid_options]
                        valid_weights = [weights[idx] if idx < len(weights) else 1.0 for idx in valid_indices]
                        # 가중치 합이 1이 되도록 정규화
                        weight_sum = sum(valid_weights)
                        if weight_sum > 0:
                            valid_weights = [w / weight_sum for w in valid_weights]
                    
                    if not valid_options:
                        # 유효한 옵션이 없으면 기본 방향 사용
                        self.logger.warning("유효한 랜덤 방향 옵션이 없습니다. 'front'를 사용합니다.")
                        direction_vector = self.direction_vectors["front"]
                        direction_name = "front"
                    else:
                        # 가중치에 따른 랜덤 선택
                        import random
                        selected_direction = random.choices(valid_options, weights=valid_weights, k=1)[0]
                        self.logger.info(f"랜덤 방향이 선택되었습니다: {selected_direction}")
                        
                        direction_vector = self.direction_vectors[selected_direction]
                        direction_name = selected_direction
            else:
                # 알 수 없는 타입은 기본 방향 사용
                self.logger.warning(f"알 수 없는 방향 타입입니다: {dir_type}. 'simple'을 사용합니다.")
                direction_vector = self.direction_vectors["front"]
                direction_name = "front"
            
            # 방향별 오프셋 계산
            offset = self._calculate_placement_offset(direction_name)
            
            # 참조 객체의 중심 좌표
            ref_x = reference_center[0]
            ref_y = reference_center[1]
            ref_z = reference_center[2]
            
            # 참조 객체의 크기 계산
            ref_width = reference_dimensions[0]
            ref_height = reference_dimensions[1]
            ref_depth = reference_dimensions[2]
            
            # 바닥면 Y 좌표 추정 (객체 바닥면에 배치하기 위함)
            base_y = self.estimate_base_y(reference_object)
            
            # 목표 위치 계산 - 방향별 오프셋 적용
            goal_x = ref_x + offset[0]
            goal_y = base_y  # 바닥면 Y좌표 사용 (방향이 'above'나 'below'면 다르게 처리 필요)
            goal_z = ref_z + offset[2]
            
            # 위/아래 방향 특수 처리
            if direction_name == 'above':
                goal_y = ref_y + ref_height/2 + self.PLACEMENT_MARGIN_CM  # 객체 위에 배치
            elif direction_name == 'below':
                goal_y = base_y - self.PLACEMENT_MARGIN_CM  # 객체 아래에 배치 (바닥보다 약간 아래)
            
            # 결과 정보 생성
            result = {
                "success": True,
                "position": [goal_x, goal_y, goal_z],
                "reference_center": reference_center,
                "direction": direction_obj,
                "direction_vector": direction_vector,
                "direction_name": direction_name,
                "offset": offset,
                "timestamp": time.time()
            }
            
            self.logger.info(f"목표 3D 위치 계산 완료: {[goal_x, goal_y, goal_z]}, 방향: {direction_name}")
            return result
        except Exception as e:
            self.logger.error(f"목표 3D 위치 계산 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "position": [0, 0, 100],  # 기본 위치
                "direction": direction_obj.get("value", "front") if dir_type == "simple" else "front"
            }
    
    def project_3d_to_2d(self, position_3d, image_width, image_height):
        """
        3D 좌표를 2D 이미지 평면에 투영합니다.
        
        Args:
            position_3d: 3D 좌표 [x, y, z]
            image_width: 이미지 가로 크기
            image_height: 이미지 세로 크기
            
        Returns:
            2D 좌표와 유효성이 담긴 딕셔너리
        """
        try:
            # 카메라 파라미터 가져오기
            camera_params = self.depth_mapper.camera_params
            fx = camera_params.get('fx', 500.0)  # 기본값 500
            fy = camera_params.get('fy', 500.0)  # 기본값 500
            cx = camera_params.get('cx', image_width / 2)
            cy = camera_params.get('cy', image_height / 2)
            
            # 3D 좌표 추출
            x, y, z = position_3d
            
            # z가 0이거나 음수인 경우 (카메라 뒤에 있는 경우) 투영 불가능
            if z <= 0:
                self.logger.warning(f"3D 좌표 {position_3d}는 카메라 뒤에 있어 투영할 수 없습니다.")
                return {
                    'coordinates': [cx, cy],  # 기본적으로 이미지 중앙 반환
                    'is_valid': False,
                    'error': 'behind_camera'
                }
            
            # 3D 좌표를 2D 좌표로 투영
            x_2d = (x * fx / z) + cx
            y_2d = (y * fy / z) + cy
            
            # 투영된 좌표가 이미지 안에 있는지 확인
            is_valid = (0 <= x_2d < image_width) and (0 <= y_2d < image_height)
            
            if not is_valid:
                self.logger.warning(f"투영된 2D 좌표 ({x_2d}, {y_2d})가 이미지 ({image_width}x{image_height}) 범위를 벗어납니다.")
                
                # 이미지 경계 내로 좌표 조정
                x_2d = max(0, min(x_2d, image_width - 1))
                y_2d = max(0, min(y_2d, image_height - 1))
            
            return {
                'coordinates': [x_2d, y_2d],
                'is_valid': is_valid,
                'original_3d': position_3d
            }
            
        except Exception as e:
            self.logger.error(f"3D 좌표 투영 중 오류 발생: {str(e)}")
            return {
                'coordinates': [image_width / 2, image_height / 2],  # 기본적으로 이미지 중앙 반환
                'is_valid': False,
                'error': str(e)
            }
    
    def generate_goal_bounding_box(self, goal_3d_position, reference_object, image_width, image_height):
        """
        3D 목표 위치를 기반으로 2D 바운딩 박스를 생성합니다.
        
        Args:
            goal_3d_position: 3D 목표 위치 정보가 담긴 딕셔너리
            reference_object: 참조 객체 정보가 담긴 딕셔너리
            image_width: 이미지 가로 크기
            image_height: 이미지 세로 크기
            
        Returns:
            2D 바운딩 박스 정보가 담긴 딕셔너리
        """
        try:
            # 3D 좌표 추출
            position_3d = goal_3d_position['position']
            
            # 3D 좌표를 2D로 투영
            projection_result = self.project_3d_to_2d(position_3d, image_width, image_height)
            
            if not projection_result['is_valid']:
                self.logger.warning(f"목표 위치가 이미지 밖에 있습니다. 조정된 좌표 사용: {projection_result['coordinates']}")
            
            # 2D 좌표 추출
            x_2d, y_2d = projection_result['coordinates']
            
            # 참조 객체의 크기와 깊이 정보를 기반으로 바운딩 박스 크기 계산
            if 'bbox' in reference_object:
                ref_bbox = reference_object['bbox']
                ref_width = ref_bbox[2] - ref_bbox[0]
                ref_height = ref_bbox[3] - ref_bbox[1]
            else:
                # 기본 크기 사용
                ref_width = 100
                ref_height = 100
                self.logger.warning(f"참조 객체 바운딩 박스 정보가 없어 기본값을 사용합니다: {ref_width}x{ref_height}")
            
            # 참조 객체의 깊이와 목표의 깊이 비율에 따라 크기 조정
            ref_depth = reference_object['3d_coords']['center'][2]
            goal_depth = position_3d[2]
            
            depth_ratio = ref_depth / goal_depth if goal_depth > 0 else 1.0
            
            # 목표 바운딩 박스 크기 계산 (깊이에 비례)
            goal_width = ref_width * depth_ratio * 0.5  # 참조 객체의 절반 크기로 설정
            goal_height = ref_height * depth_ratio * 0.5
            
            # 방향에 따라 크기 조정 (특정 방향은 더 작게 표시)
            direction = goal_3d_position['direction']
            if direction in ['front', 'back']:
                goal_width *= 0.8
                goal_height *= 0.8
            
            # 바운딩 박스 좌표 계산
            x1 = x_2d - goal_width / 2
            y1 = y_2d - goal_height / 2
            x2 = x_2d + goal_width / 2
            y2 = y_2d + goal_height / 2
            
            # 바운딩 박스가 이미지 경계 내에 있도록 조정
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))
            
            # 최종 바운딩 박스 생성
            bbox = [x1, y1, x2, y2]
            
            # 결과 생성
            result = {
                'bbox': bbox,
                'center': [x_2d, y_2d],
                'width': goal_width,
                'height': goal_height,
                'confidence': 0.85,  # 기본 신뢰도
                'depth': goal_depth
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"바운딩 박스 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 바운딩 박스 반환
            return {
                'bbox': [image_width/2 - 50, image_height/2 - 50, image_width/2 + 50, image_height/2 + 50],
                'center': [image_width/2, image_height/2],
                'width': 100,
                'height': 100,
                'confidence': 0.5,
                'error': str(e)
            }
    
    def infer_goal_placement(self, reference_object, direction, image_width, image_height, detections=None):
        """
        참조 객체와 방향을 기반으로 목표 위치를 추론합니다.
        
        Args:
            reference_object: 참조 객체 정보가 담긴 딕셔너리
                {
                    'id': '객체_ID',
                    'label': '객체_레이블',
                    'bbox': [x1, y1, x2, y2],
                    'center': [x, y],
                    '3d_coords': {
                        'center': [x, y, z],
                        'dimensions': [width, height, depth]
                    }
                }
            direction: 참조 객체로부터의 방향 문자열
                ('front', 'back', 'left', 'right', 'above', 'below' 등)
            image_width: 이미지 가로 크기
            image_height: 이미지 세로 크기
            detections: 현재 장면에서 감지된 다른 객체들의 목록 (선택적)
            
        Returns:
            목표 위치 정보가 담긴 딕셔너리
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': [x, y],
                    'dimensions': [width, height],
                    'confidence': 0.9,
                    '3d_position': [x, y, z],
                    'reference_object': {...},
                    'direction': 'front',
                    'method': 'reference_offset',
                    'timestamp': 1234567890
                }
        """
        self.logger.info(f"목표 위치 추론 시작: 참조={reference_object.get('label', 'unknown')}, 방향={direction}")
        
        try:
            # Phase 1: 참조 객체 및 방향 파싱
            if not reference_object:
                self.logger.error("참조 객체가 제공되지 않았습니다.")
                return None
            
            if 'bbox' not in reference_object:
                self.logger.error(f"참조 객체에 바운딩 박스 정보가 없습니다: {reference_object}")
                return None
            
            # Phase 2: 3D 목표 위치 계산
            goal_3d_position = self.calculate_goal_3d_position(reference_object, direction)
            
            if not goal_3d_position:
                self.logger.error("3D 목표 위치를 계산할 수 없습니다.")
                return None
            
            # Phase 3: 3D 위치를 2D 바운딩 박스로 변환
            bbox_result = self.generate_goal_bounding_box(
                goal_3d_position,
                reference_object,
                image_width,
                image_height
            )
            
            if not bbox_result:
                self.logger.error("2D 바운딩 박스를 생성할 수 없습니다.")
                return None
            
            # 최종 결과 생성
            result = {
                'bbox': bbox_result['bbox'],
                'center': bbox_result['center'],
                'dimensions': [bbox_result['width'], bbox_result['height']],
                'confidence': bbox_result['confidence'],
                '3d_position': goal_3d_position['position'],
                'reference_object': {
                    'id': reference_object.get('id', ''),
                    'label': reference_object.get('label', 'unknown'),
                    'bbox': reference_object['bbox'],
                    'center': reference_object.get('center', [0, 0]),
                    '3d_center': reference_object.get('3d_coords', {}).get('center', [0, 0, 0])
                },
                'direction': direction,
                'method': goal_3d_position['method'],
                'timestamp': int(time.time())
            }
            
            self.logger.info(f"목표 위치 추론 완료: 방향={direction}, 위치={result['bbox']}")
            return result
            
        except Exception as e:
            self.logger.error(f"목표 위치 추론 중 오류 발생: {str(e)}")
            return None
    
    def _contains_demonstrative(self, text: Optional[str]) -> bool:
        """주어진 텍스트에 지시대명사(이거, 저거 등)가 포함되어 있는지 확인합니다."""
        if not text:
            return False
        # 간단한 한국어 지시대명사 목록 (확장 가능)
        pronouns = ["이거", "저거", "그거", "얘", "쟤", "걔", "이것", "저것", "그것"]
        # 정규 표현식 사용 가능 (예: r"\b(이|그|저)[것거]" 등)
        for pronoun in pronouns:
            if pronoun in text:
                return True
        return False

    # Helper function for angle calculation
    def _angle_between_vectors(self, v1, v2):
        """Calculate the angle in degrees between two 3D vectors."""
        # Handle potential zero vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 180.0 # Return max angle if one vector is zero
        
        v1_u = v1 / norm_v1
        v2_u = v2 / norm_v2
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def _find_target_by_gesture(self,
                                gesture_results: List[Dict[str, Any]],
                                detections: List[Dict[str, Any]],
                                depth_map: np.ndarray, # Not directly used here, but could be for visibility checks
                                img_width: int,
                                img_height: int
                               ) -> Optional[Dict[str, Any]]:
        """
        제스처 정보와 객체 감지 결과를 사용하여 타겟 객체를 식별합니다.
        특히 포인팅 제스처를 사용하여 객체를 가리키는 경우를 처리합니다.

        Args:
            gesture_results: GestureRecognizer의 결과 리스트.
            detections: 감지된 객체 리스트 (3D 좌표 포함).
            depth_map: 깊이 맵.
            img_width: 이미지 너비.
            img_height: 이미지 높이.

        Returns:
            Dict[str, Any]: 식별된 타겟 객체 딕셔너리 또는 None (찾지 못한 경우).
        """
        # 제스처 또는 감지 결과가 없는 경우 None 반환
        if not gesture_results or not detections:
            self.logger.debug("_find_target_by_gesture: 제스처 또는 감지 결과 없음.")
            return None
            
        # 포인팅 제스처 찾기
        pointing_gesture = None
        for gesture in gesture_results:
            if gesture.get('gesture') and gesture['gesture'] == GestureType.POINTING:
                pointing_gesture = gesture
                break
                
        if not pointing_gesture:
            self.logger.debug("_find_target_by_gesture: 포인팅 제스처를 찾을 수 없음.")
            return None
            
        # 손 객체 필터링
        valid_detections = []
        hand_objects = []
        
        for obj in detections:
            # 클래스 이름이 손 목록에 있는지 확인
            if obj['class_name'].lower() in [cls.lower() for cls in self.HAND_CLASSES]:
                # 손 랜드마크와 YOLO 객체 간의 IoU 계산
                if pointing_gesture.get('all_points_2d'):
                    obj_box = obj['bbox']  # [x1, y1, x2, y2]
                    
                    # 손 랜드마크에서 바운딩 박스 계산
                    hand_box = calculate_box_from_points(pointing_gesture['all_points_2d'])
                    
                    # IoU 계산
                    iou = calculate_iou(obj_box, hand_box)
                    self.logger.debug(f"손 IoU: {iou:.4f}, 객체: {obj['class_name']}")
                    
                    # IoU가 임계값보다 크면 이 객체는 '실제 손'
                    if iou > 0.3:  # IoU 임계값 (조정 가능)
                        self.logger.debug(f"'실제 손' 감지: {obj['class_name']}, IoU={iou:.4f}")
                        hand_objects.append(obj)
                        continue
            
            # 손이 아닌 경우 유효한 감지 객체로 추가
            valid_detections.append(obj)
            
        # 유효한 객체가 없는 경우
        if not valid_detections:
            self.logger.warning("_find_target_by_gesture: 손을 제외한 유효한 객체가 없음.")
            return None
            
        # 정확히 두 개의 객체가 감지되었는지 확인 (프로토타입 요구사항)
        if len(valid_detections) != 2:
            self.logger.warning(f"_find_target_by_gesture: 정확히 2개 객체가 필요하나, {len(valid_detections)}개 감지됨.")
            # 두 개 이상인 경우 각도 기반으로 계속 진행
        
        # 손목에서 검지 끝으로 향하는 2D 포인팅 벡터 계산
        if not pointing_gesture.get('pointing_points') or len(pointing_gesture['pointing_points']) != 2:
            self.logger.warning("_find_target_by_gesture: 유효한 포인팅 좌표 없음.")
            return None
            
        wrist_2d = pointing_gesture['pointing_points'][0]
        index_tip_2d = pointing_gesture['pointing_points'][1]
        
        # 2D 포인팅 벡터 계산 (검지 끝 - 손목)
        pointing_vector_2d = [index_tip_2d[0] - wrist_2d[0], index_tip_2d[1] - wrist_2d[1]]
        
        # 각 객체에 대해 각도 계산
        target_candidates = []
        
        for obj in valid_detections:
            # 객체 중심점 계산
            obj_center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
            obj_center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
            
            # 손목에서 객체 중심으로 향하는 벡터
            obj_vector = [obj_center_x - wrist_2d[0], obj_center_y - wrist_2d[1]]
            
            # 두 벡터 간 각도 계산
            angle = calculate_angle_2d(pointing_vector_2d, obj_vector)
            
            # 후보 추가
            target_candidates.append({
                'object': obj,
                'angle': angle,
                'distance': np.sqrt((obj_center_x - wrist_2d[0])**2 + (obj_center_y - wrist_2d[1])**2)
            })
            
            self.logger.debug(f"객체 '{obj['class_name']}': 각도={angle:.2f}°, 좌표=({obj_center_x:.1f}, {obj_center_y:.1f})")
        
        # 각도가 가장 작은 객체 선택
        if target_candidates:
            # 각도로 정렬
            target_candidates.sort(key=lambda x: x['angle'])
            best_candidate = target_candidates[0]
            
            self.logger.info(f"제스처 기반 타겟 객체 선택: {best_candidate['object']['class_name']}, 각도={best_candidate['angle']:.2f}°")
            
            # 로깅: 포인팅 벡터, 객체 방향 벡터, 각도 차이, 최종 선택
            self.logger.debug(f"포인팅 벡터: {pointing_vector_2d}")
            obj = best_candidate['object']
            obj_center = [(obj['bbox'][0] + obj['bbox'][2]) / 2, (obj['bbox'][1] + obj['bbox'][3]) / 2]
            obj_vector = [obj_center[0] - wrist_2d[0], obj_center[1] - wrist_2d[1]]
            self.logger.debug(f"객체 벡터: {obj_vector}")
            self.logger.debug(f"각도 차이: {best_candidate['angle']:.2f}°")
            self.logger.debug(f"선택된 타겟: {obj['class_name']}, ID={obj.get('id', '?')}")
            
            return best_candidate['object']
        
        return None

    def process_image(self, image, user_prompt=None):
        """
        이미지를 처리하여 객체 감지, 목표 지점 추론, 결과 시각화
        
        Args:
            image: 입력 이미지
            user_prompt: 사용자 명령 (선택적)
            
        Returns:
            Dict: 처리 결과 (타겟 객체, 레퍼런스 객체, 목표 바운딩 박스 등 포함)
        """
        try:
            # 시작 시간 기록
            start_time = time.time()
        
            # --- 1. 객체 감지 ---
            self.logger.info("객체 감지 시작...")
            detection_start = time.time()
            detections = self.yolo_detector.detect(image)
            detection_time = time.time() - detection_start
            self.logger.info(f"객체 감지 완료: {len(detections)}개 감지됨")
            
            # --- 2. 깊이 추정 ---
            self.logger.info("깊이 맵 추정 시작...")
            depth_start = time.time()
            depth_map, colored_depth = self.depth_mapper.estimate_depth(image)
            
            # 이미지 크기 가져오기
            img_h, img_w = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
            # 기본 카메라 매트릭스를 이미지 크기에 맞게 업데이트
            self.depth_mapper._prepare_default_camera_matrix(img_w, img_h)
            
            # 각 객체에 깊이 정보 추가
            for obj in detections:
                obj.update(self.depth_mapper.get_object_depth(obj["bbox"], depth_map))
                obj.update({"3d_coords": self.depth_mapper.get_object_3d_position(obj, depth_map, img_w, img_h)})
            depth_time = time.time() - depth_start
            self.logger.info("객체 깊이 추정 완료")
            
            # --- 3. 제스처 인식 ---
            self.logger.info("제스처 인식 시작...")
            gesture_start = time.time()
            gesture_results = None
            if self.gesture_recognizer:
                try:
                    gesture_results = self.gesture_recognizer.process_frame(image)
                    self.logger.info(f"제스처 인식 완료: {len(gesture_results) if gesture_results else 0}개 감지됨")
                except Exception as e:
                    self.logger.error(f"제스처 인식 중 오류: {e}")
            gesture_time = time.time() - gesture_start
            
            # --- 4. 타겟/레퍼런스 객체 추론 ---
            self.logger.info("타겟/레퍼런스 객체 추론 시작...")
            inference_start = time.time()
            
            # 초기화: 결과 변수들
            target_object = None
            reference_object = None
            target_inference_result = None
            goal_point_result = None
            
            # 제스처 모드가 활성화되어 있는지 확인
            if self.gesture_mode_active:
                self.logger.info("*** 제스처 모드 활성화됨 - 4단계 통합 프로세스 실행 ***")
                
                # ========== 1단계: 타겟 객체 식별 로직 ==========
                self.logger.info("1단계: 제스처 기반 타겟 객체 식별")
                target_object = self._find_target_by_gesture(
                    gesture_results=gesture_results,
                    detections=detections,
                    depth_map=depth_map,
                    img_width=img_w,
                    img_height=img_h
                )
                
                # 타겟 객체를 찾은 경우에만 계속 진행
                if target_object:
                    # 타겟 인덱스 찾기
                    target_idx = next((i for i, d in enumerate(detections) if d == target_object), None)
                    self.logger.info(f"타겟 객체 식별 성공: {target_object['class_name']} (인덱스: {target_idx})")
                    
                    # ========== 2단계: 레퍼런스 객체 식별 및 LLM 확인 ==========
                    self.logger.info("2단계: 레퍼런스 객체 자동 결정 및 LLM 확인")
                    
                    # 타겟 외 객체 필터링 (손 제외)
                    non_target_objects = [
                        (i, d) for i, d in enumerate(detections) 
                        if d != target_object and 
                        d['class_name'].lower() not in [cls.lower() for cls in self.HAND_CLASSES]
                    ]
                    
                    reference_idx = None
                    reference_validation_result = None
                    
                    # 레퍼런스 객체 자동 선택 및 LLM 확인
                    if len(non_target_objects) == 1:
                        reference_idx, reference_object = non_target_objects[0]
                        self.logger.info(f"단일 레퍼런스 후보 자동 선택: {reference_object['class_name']} (인덱스: {reference_idx})")
                        
                        # LLM으로 레퍼런스 객체 확인
                        if hasattr(self, 'prompting_pipeline'):
                            self.logger.info("LLM을 사용하여 자동 선택된 레퍼런스 객체 확인")
                            reference_validation_result = self.prompting_pipeline.confirm_reference(
                                user_command="", # 제스처 모드에서는 사용자 명령 없음
                                target_object=target_object,
                                reference_object=reference_object
                            )
                            
                            if reference_validation_result['is_valid']:
                                self.logger.info(f"LLM 확인 완료: 레퍼런스 객체 '{reference_object['class_name']}' 유효함 (신뢰도: {reference_validation_result['confidence']:.2f})")
                                self.logger.debug(f"LLM 설명: {reference_validation_result['explanation']}")
                            else:
                                self.logger.warning(f"LLM 확인 실패: 레퍼런스 객체 '{reference_object['class_name']}' 부적합 (신뢰도: {reference_validation_result['confidence']:.2f})")
                                self.logger.debug(f"LLM 설명: {reference_validation_result['explanation']}")
                                # 실패 시에도 유일한 객체이므로 그대로 사용
                        else:
                            self.logger.warning("prompting_pipeline이 없어 LLM 확인 생략")
                            
                    elif len(non_target_objects) > 1:
                        # 여러 후보 중에서 선택
                        self.logger.info(f"여러 레퍼런스 후보 발견: {len(non_target_objects)}개")
                        
                        # 초기 후보를 첫 번째 객체로 설정
                        reference_idx, reference_object = non_target_objects[0]
                        best_confidence = 0.0
                        
                        # LLM으로 모든 후보 객체 확인
                        if hasattr(self, 'prompting_pipeline'):
                            self.logger.info("LLM을 사용하여 최적의 레퍼런스 객체 선택")
                            
                            candidates_results = []
                            for idx, (obj_idx, obj) in enumerate(non_target_objects):
                                self.logger.info(f"후보 {idx+1}/{len(non_target_objects)} 확인 중: {obj['class_name']}")
                                
                                confirmation_result = self.prompting_pipeline.confirm_reference(
                                    user_command="", # 제스처 모드에서는 사용자 명령 없음
                                    target_object=target_object,
                                    reference_object=obj
                                )
                                
                                candidates_results.append({
                                    'obj_idx': obj_idx,
                                    'object': obj,
                                    'result': confirmation_result
                                })
                                
                                # 더 나은 후보 발견 시 업데이트
                                if confirmation_result['is_valid'] and confirmation_result['confidence'] > best_confidence:
                                    best_confidence = confirmation_result['confidence']
                                    reference_idx = obj_idx
                                    reference_object = obj
                                    reference_validation_result = confirmation_result
                                    self.logger.info(f"더 나은 레퍼런스 후보 발견: {obj['class_name']} (신뢰도: {best_confidence:.2f})")
                            
                            # 최종 선택 로깅
                            self.logger.info(f"최종 레퍼런스 객체 선택: {reference_object['class_name']} (인덱스: {reference_idx}, 신뢰도: {best_confidence:.2f})")
                            
                            # 후보 비교 상세 로깅
                            for candidate in candidates_results:
                                self.logger.debug(f"후보 {candidate['object']['class_name']}: 유효성={candidate['result']['is_valid']}, 신뢰도={candidate['result']['confidence']:.2f}")
                        else:
                            # LLM 확인 불가능한 경우 첫 번째 객체 사용
                            self.logger.warning(f"prompting_pipeline이 없어 첫 번째 객체를 레퍼런스로 선택: {reference_object['class_name']}")
                    else:
                        self.logger.warning("레퍼런스 객체 후보가 없음")
                        
                    # 레퍼런스 객체가 있을 경우 계속 진행
                    if reference_object:
                        # 제스처 기반 타겟 추론 결과 생성
                        target_inference_result = {
                            "target_idx": target_idx,
                            "reference_idx": reference_idx,
                            "method": "gesture_based",
                            "confidence": 0.9,  # 제스처 기반이므로 높은 신뢰도 부여
                            "direction": "front",  # 기본 방향 (이후 단계에서 수정 가능)
                            "logs": [
                                f"제스처 모드로 타겟 선택: {target_object['class_name']}",
                                f"레퍼런스 객체: {reference_object['class_name'] if reference_object else 'None'}",
                                f"레퍼런스 객체 자동 선택 및 LLM 확인 완료" if reference_validation_result else "레퍼런스 객체 자동 선택 (LLM 확인 없음)"
                            ],
                            "reference_validation": reference_validation_result is not None,
                            "reference_validation_confidence": reference_validation_result['confidence'] if reference_validation_result else 0.0
                        }
                        
                        # ========== 3단계: 목표 지점 바운딩 박스 추론 ==========
                        self.logger.info("3단계: LLM 기반 목표 위치 추론")
                        
                        if hasattr(self, 'prompting_pipeline'):
                            # LLM에서 타겟-레퍼런스 객체 간의 방향 관계 추론
                            bbox_result = self.prompting_pipeline.infer_goal_bounding_box(
                                user_command=user_prompt,  # 제스처 모드에서도 사용자 프롬프트 전달
                                target_object=target_object,
                                reference_object=reference_object,
                                img_width=img_w,
                                img_height=img_h,
                                direction="front"  # 기본 방향 (앞쪽)
                            )
                            
                            if bbox_result['parse_success']:
                                # LLM이 추론한 방향 정보 추출
                                inferred_direction = bbox_result.get('direction', 'front')
                                self.logger.info(f"LLM이 추론한 방향: {inferred_direction}")
                                
                                # ===== 새로운 2D 기반 목표 바운딩 박스 계산 =====
                                self.logger.info(f"2D 기반 목표 바운딩 박스 계산 시작 ({inferred_direction} 방향)")
                                
                                # 2D 직접 오프셋 방식으로 목표 바운딩 박스 계산
                                goal_bbox_2d_result = self.generate_goal_bounding_box_2d(
                                    reference_object=reference_object,
                                    direction=inferred_direction,
                                    image_width=img_w,
                                    image_height=img_h,
                                    detections=detections
                                )
                                
                                # 2D 방식 성공 여부 확인
                                if goal_bbox_2d_result and goal_bbox_2d_result['bbox']:
                                    # 2D 방식 성공 시 결과 사용
                                    self.logger.info("2D 기반 목표 바운딩 박스 계산 성공")
                                    
                                    # 2D 바운딩 박스 추출
                                    goal_bbox = goal_bbox_2d_result['bbox']
                                    
                                    # 겹침 확인 정보
                                    iou_value = goal_bbox_2d_result['overlap_iou']
                                    has_overlap = goal_bbox_2d_result['has_overlap']
                                    
                                    if has_overlap:
                                        self.logger.warning(f"목표 bbox와 레퍼런스 bbox 겹침 감지 (IoU: {iou_value:.2f})")
                                    
                                    # 3D 좌표 형식 변환 (2D 바운딩 박스 기반)
                                    # 바운딩 박스 중심점
                                    bbox_center_x = (goal_bbox[0] + goal_bbox[2]) / 2
                                    bbox_center_y = (goal_bbox[1] + goal_bbox[3]) / 2
                                    
                                    # 예상 깊이는 레퍼런스 객체와 동일하게 설정
                                    ref_depth = reference_object['3d_coords'].get('z_cm', 100.0)
                                    
                                    goal_3d_coords = {
                                        "x_cm": float(bbox_center_x),  # 2D x 좌표 (간단히 변환)
                                        "y_cm": float(bbox_center_y),  # 2D y 좌표 (간단히 변환)
                                        "z_cm": float(ref_depth),      # 레퍼런스 객체와 동일한 깊이
                                        "direction": inferred_direction
                                    }
                                    
                                    # goal_point_result 생성 - 2D 기반 계산 결과 사용
                                    goal_point_result = {
                                        "goal_point": {
                                            "3d_coords": goal_3d_coords,
                                            "2d_bbox": goal_bbox,
                                            "method": "direct_2d_offset"
                                        },
                                        "direction": inferred_direction,
                                        "confidence": bbox_result['confidence'],
                                        "method": "direct_2d_offset",
                                        "llm_response": bbox_result['original_response'],
                                        "calculation_details": goal_bbox_2d_result,
                                        "overlap_iou": iou_value,
                                        "has_overlap": has_overlap
                                    }
                                    
                                    self.logger.info(f"2D 기반 목표 바운딩 박스: {goal_bbox}")
                                elif bbox_result.get('target_box'):
                                    # LLM이 생성한 바운딩 박스 사용
                                    target_box = bbox_result['target_box']
                                    self.logger.info(f"LLM이 생성한 목표 바운딩 박스: {target_box}")
                                    
                                    # 바운딩 박스 기반 goal_point_result 생성
                                    goal_point_result = {
                                        "goal_point": {
                                            "2d_bbox": target_box,
                                            "method": "llm_direct_bbox"
                                        },
                                        "direction": inferred_direction,
                                        "confidence": bbox_result['confidence'],
                                        "method": "llm_direct_bbox",
                                        "llm_response": bbox_result['original_response']
                                    }
                                    
                                    # 깊이 정보가 있는 경우 3D 좌표 계산 시도
                                    if depth_map is not None:
                                        try:
                                            # 바운딩 박스 중심점 계산
                                            center_x = (target_box[0] + target_box[2]) / 2
                                            center_y = (target_box[1] + target_box[3]) / 2
                                            
                                            # 깊이 정보에서 z값 추출
                                            if 0 <= int(center_y) < depth_map.shape[0] and 0 <= int(center_x) < depth_map.shape[1]:
                                                depth_value = depth_map[int(center_y), int(center_x)]
                                                
                                                # 3D 좌표 계산
                                                goal_3d_coords = {
                                                    "x_cm": center_x,
                                                    "y_cm": center_y,
                                                    "z_cm": float(depth_value) * (self.max_depth_cm - self.min_depth_cm) + self.min_depth_cm,
                                                    "direction": inferred_direction
                                                }
                                                
                                                goal_point_result["goal_point"]["3d_coords"] = goal_3d_coords
                                                self.logger.info(f"목표 위치 3D 좌표 계산: {goal_3d_coords}")
                                        except Exception as e:
                                            self.logger.error(f"3D 좌표 계산 중 오류: {e}")
                                else:
                                    # LLM 추론 결과가 없는 경우 기존 방식으로 폴백
                                    self.logger.warning("2D 방식과 LLM 바운딩 박스 모두 실패. 기존 3D 방식으로 폴백")
                                    goal_point_result = self.spatial_analyzer.calculate_goal_point_3d(
                                        target_object=target_object,
                                        reference_object=reference_object,
                                        direction=inferred_direction
                                    )
                        else:
                            # prompting_pipeline이 없는 경우
                            self.logger.warning("prompting_pipeline이 없어 기존 방식으로 목표 위치 계산")
                            goal_point_result = self.spatial_analyzer.calculate_goal_point_3d(
                                target_object=target_object,
                                reference_object=reference_object,
                                direction="front"  # 기본 방향 (앞쪽)
                            )
                    else:
                        self.logger.warning("레퍼런스 객체를 찾을 수 없어 목표 지점 추론 생략")
                else:
                    # 제스처 기반 타겟 찾기 실패 시 기존 방식 사용
                    self.logger.warning("제스처 기반 타겟 찾기 실패 - 기존 방식으로 대체")
                    
                    # 사용자 프롬프트가 있으면 향상된 프롬프팅 파이프라인 사용
                    if user_prompt:
                        # --- 기존 프롬프트 기반 처리로 폴백 ---
                        self.logger.info(f"사용자 프롬프트로 대체: '{user_prompt}'")
                        
                        # --- 향상된 프롬프팅 파이프라인 사용 ---
                        self.logger.info("향상된 프롬프팅 파이프라인 사용")
                        pipeline_result = self.prompting_pipeline.process(
                            user_prompt=user_prompt,
                            detections=detections,
                            gesture_results=gesture_results,
                            depth_data=depth_map
                        )
                        
                        # 결과 검증
                        self.logger.info(f"파이프라인 처리 결과: 타겟={pipeline_result['final_target_id']}, 레퍼런스={pipeline_result['final_reference_id']}")
                        
                        # 유효한 타겟/레퍼런스 인덱스 확인
                        target_idx = pipeline_result.get("final_target_id")
                        reference_idx = pipeline_result.get("final_reference_id")
                        
                        # 인덱스가 유효한지 확인
                        if target_idx is not None and target_idx >= 0 and target_idx < len(detections):
                            target_object = detections[target_idx]
                            self.logger.info(f"타겟 객체: {target_object['class_name']}")
                        else:
                            target_object = None
                            self.logger.warning(f"유효하지 않은 타겟 인덱스: {target_idx}")
                            
                        if reference_idx is not None and reference_idx >= 0 and reference_idx < len(detections):
                            reference_object = detections[reference_idx]
                            self.logger.info(f"레퍼런스 객체: {reference_object['class_name']}")
                        else:
                            reference_object = None
                            self.logger.warning(f"유효하지 않은 레퍼런스 인덱스: {reference_idx}")
                            
                        # 유효한 목표 위치 확인
                        goal_position = pipeline_result.get("final_goal_position")
                        
                        # 향상된 추론 결과 생성
                        target_inference_result = {
                            "target_idx": target_idx,
                            "reference_idx": reference_idx,
                            "method": "enhanced_prompting_pipeline",
                            "confidence": pipeline_result.get("confidence", 0.0),
                            "direction": pipeline_result.get("direction", "right"),
                            "logs": [
                                f"원본 프롬프트: {user_prompt}",
                                f"1단계 결과: {pipeline_result.get('stage1_result')}",
                                f"2단계 결과: {pipeline_result.get('stage2_result')}",
                                f"3단계 결과: {pipeline_result.get('stage3_result')}",
                            ]
                        }
                        
                        # 목표 지점 계산
                        if goal_position is not None:
                            # 3D 좌표가 리스트 형태인지 확인
                            if isinstance(goal_position, list) and len(goal_position) >= 3:
                                # 파이프라인에서 계산된 3D 좌표 사용
                                goal_3d = {
                                    "x_cm": float(goal_position[0]),
                                    "y_cm": float(goal_position[1]),
                                    "z_cm": float(goal_position[2]),
                                    "direction": pipeline_result.get("direction", "right")
                                }
                                
                                # 초기 goal_point_result 생성
                                goal_point_result = {
                                    "goal_point": {
                                        "3d_coords": goal_3d
                                    },
                                    "direction": pipeline_result.get("direction", "right"),
                                    "confidence": pipeline_result.get("confidence", 0.0),
                                    "method": "enhanced_prompting_pipeline"
                                }
                                self.logger.info(f"파이프라인에서 계산된 목표 위치 사용: {goal_3d}")
                                
                                # 3D 좌표를 바탕으로 2D 바운딩 박스 생성
                                if reference_object and 'bbox' in reference_object:
                                    try:
                                        goal_3d_position = {
                                            'position': goal_position,
                                            'direction': pipeline_result.get("direction", "right")
                                        }
                                        
                                        # 2D 바운딩 박스 생성
                                        goal_bbox_result = self.generate_goal_bounding_box(
                                            goal_3d_position=goal_3d_position,
                                        reference_object=reference_object,
                                            image_width=img_w,
                                            image_height=img_h
                                        )
                                        
                                        # 생성된 2D 바운딩 박스 추가
                                        if goal_bbox_result and 'bbox' in goal_bbox_result:
                                            goal_point_result["goal_point"]["2d_bbox"] = goal_bbox_result['bbox']
                                            self.logger.info(f"3D 좌표에서 2D 바운딩 박스 생성 완료: {goal_bbox_result['bbox']}")
                                    except Exception as e:
                                        self.logger.error(f"3D 좌표에서 2D 바운딩 박스 생성 실패: {e}")
                        else:
                            self.logger.warning(f"유효하지 않은 목표 위치 형식: {goal_position}. 기존 방식으로 계산.")
                            goal_point_result = self.spatial_analyzer.calculate_goal_point_3d(
                                target_object=target_object,
                                reference_object=reference_object,
                                direction=pipeline_result.get("direction", "right")
                            )
                    else:
                        # 파이프라인에서 목표 위치를 제공하지 않은 경우 새로운 LLM 직접 바운딩 박스 추론
                        self.logger.warning("파이프라인에서 목표 위치를 제공하지 않음. LLM 직접 바운딩 박스 추론 시도")
                        
                        # 새로운 LLM 직접 바운딩 박스 추론 적용
                        if target_object and reference_object and hasattr(self, 'prompting_pipeline'):
                            # LLM을 사용한 목표 바운딩 박스 직접 추론
                            self.logger.info("LLM을 사용하여 목표 바운딩 박스 직접 추론 시작")
                            bbox_result = self.prompting_pipeline.infer_goal_bounding_box(
                                user_command=user_prompt,
                                target_object=target_object, 
                                reference_object=reference_object,
                                img_width=img_w,
                                img_height=img_h,
                                direction=pipeline_result.get("direction", "front")
                            )
                            
                            if bbox_result['parse_success'] and bbox_result['target_box']:
                                # LLM이 생성한 바운딩 박스 사용
                                target_box = bbox_result['target_box']
                                self.logger.info(f"LLM이 생성한 목표 바운딩 박스: {target_box}")
                                
                                # 바운딩 박스 기반 goal_point_result 생성
                                goal_point_result = {
                                    "goal_point": {
                                        "2d_bbox": target_box,
                                        "method": "llm_direct_bbox"
                                    },
                                    "direction": pipeline_result.get("direction", "front"),
                                    "confidence": bbox_result['confidence'],
                                    "method": "llm_direct_bbox",
                                    "llm_response": bbox_result['original_response']
                                }
                                
                                # 깊이 정보가 있는 경우 3D 좌표 계산 시도
                                if depth_map is not None:
                                    try:
                                        # 바운딩 박스 중심점 계산
                                        center_x = (target_box[0] + target_box[2]) / 2
                                        center_y = (target_box[1] + target_box[3]) / 2
                                        
                                        # 깊이 정보에서 z값 추출
                                        if 0 <= int(center_y) < depth_map.shape[0] and 0 <= int(center_x) < depth_map.shape[1]:
                                            depth_value = depth_map[int(center_y), int(center_x)]
                                            
                                            # 3D 좌표 계산
                                            goal_3d_coords = {
                                                "x_cm": center_x,
                                                "y_cm": center_y,
                                                "z_cm": float(depth_value) * (self.max_depth_cm - self.min_depth_cm) + self.min_depth_cm,
                                                "direction": pipeline_result.get("direction", "front")
                                            }
                                            
                                            goal_point_result["goal_point"]["3d_coords"] = goal_3d_coords
                                            self.logger.info(f"목표 위치 3D 좌표 계산: {goal_3d_coords}")
                                    except Exception as e:
                                        self.logger.error(f"3D 좌표 계산 중 오류: {e}")
                                
                                # 로깅: LLM 프롬프트와 응답 기록
                                self.logger.debug(f"LLM 프롬프트: {bbox_result['prompt']}")
                                self.logger.debug(f"LLM 응답: {bbox_result['original_response']}")
                            else:
                                # LLM 추론 실패 시 기존 방식으로 폴백
                                self.logger.warning("LLM 목표 바운딩 박스 추론 실패, 기존 방식으로 폴백")
                                goal_point_result = self.spatial_analyzer.calculate_goal_point_3d(
                                    target_object=target_object,
                                    reference_object=reference_object,
                                    direction=pipeline_result.get("direction", "right")
                                )
                        else:
                            # LLM 추론을 위한 조건이 충족되지 않은 경우 기존 방식 사용
                            self.logger.warning("LLM 추론 조건 미충족 (타겟/레퍼런스 객체 없음 또는 prompting_pipeline 없음)")
                            if target_object and reference_object:
                                goal_point_result = self.spatial_analyzer.calculate_goal_point_3d(
                                    target_object=target_object,
                                    reference_object=reference_object,
                                    direction="right"  # 기본 방향
                                )
            
            # 타겟 추론 시간 계산
            inference_time = time.time() - inference_start
            self.logger.info("타겟/레퍼런스 객체 추론 완료")
            
            # --- 5. 결과 시각화 ---
            self.logger.info("결과 시각화 시작...")
            visualization_start = time.time()
            
            # goal_point_result에서 좌표 정보 추출
            goal_point_coords = None
            goal_bounding_box = None
            
            if goal_point_result and "goal_point" in goal_point_result:
                # 3D 좌표 정보 추출
                if "3d_coords" in goal_point_result["goal_point"]:
                    coords_3d = goal_point_result["goal_point"]["3d_coords"]
                    # 3D 좌표를 numpy 배열로 변환 
                    if isinstance(coords_3d, dict) and "x_cm" in coords_3d and "y_cm" in coords_3d and "z_cm" in coords_3d:
                        goal_point_coords = np.array([coords_3d["x_cm"], coords_3d["y_cm"], coords_3d["z_cm"]])
                    elif isinstance(coords_3d, list) and len(coords_3d) >= 3:
                        goal_point_coords = np.array(coords_3d[:3])
                
                # 2D 바운딩 박스 정보 추출 (LLM 직접 추론)
                if "2d_bbox" in goal_point_result["goal_point"]:
                    goal_bounding_box = goal_point_result["goal_point"]["2d_bbox"]
                    self.logger.info(f"목표 바운딩 박스 시각화: {goal_bounding_box}")
            
            visualization = self.visualization.draw_results(
                image,
                detections,
                target_object=target_object,
                reference_object=reference_object,
                goal_point_coords=goal_point_coords,
                goal_bounding_box=goal_bounding_box,
                gesture_results=gesture_results
            )
            visualization_time = time.time() - visualization_start
            self.logger.info("결과 시각화 완료")
            
            # --- 6. 결과 저장 ---
            self.logger.info("결과 저장 시작...")
            
            # 처리 시간 결과
            timings = {
                "detection": detection_time,
                "depth_estimation": depth_time,
                "gesture_recognition": gesture_time,
                "target_inference": inference_time,
                "visualization": visualization_time,
                "total": time.time() - start_time
            }
            
            # 결과 저장용 데이터 구성
            analysis_data_to_save = {
                "detections": detections,
                "target_inference": target_inference_result,
                "goal_point": goal_point_result,
                "user_prompt": user_prompt,
                "gesture_results": gesture_results,
                "timestamp": time.time(),
                "timing": timings
            }
            
            # 향상된 프롬프팅 결과 저장 (존재하는 경우)
            if user_prompt and 'pipeline_result' in locals():
                analysis_data_to_save["enhanced_prompting"] = {
                    "original_prompt": pipeline_result.get("original_prompt"),
                    "stage1_result": pipeline_result.get("stage1_result"),
                    "stage2_result": pipeline_result.get("stage2_result"),
                    "stage3_result": pipeline_result.get("stage3_result"),
                    "processing_time": pipeline_result.get("processing_time")
                }
            
            # 결과 저장
            saved_paths = None
            try:
                saved_paths = self.result_storage.save_analysis_results(
                    results_data=analysis_data_to_save,
                    snapshot_image=image, # 원본 이미지
                    result_image=visualization, # 시각화된 이미지
                    depth_map=colored_depth # 색상 깊이 맵 저장
                )
                self.logger.info(f"결과 저장 완료: {saved_paths.get('json', '경로 정보 없음')}")
            except Exception as e:
                self.logger.error(f"결과 저장 중 오류: {e}", exc_info=True)
            
            # --- 7. 최종 결과 반환 ---
            return {
                "detections": detections,
                "target_object": target_object,
                "reference_object": reference_object,
                "target_inference": target_inference_result, # 상세 추론 결과 포함
                "goal_point": goal_point_result,
                "goal_bounding_box": goal_bounding_box, # 명시적으로 goal_bounding_box 추가
                "visualization": visualization,
                "saved_paths": saved_paths,
                "timing": timings
            }
        
        except Exception as e:
            self.logger.error(f"이미지 처리 중 오류: {e}", exc_info=True)
            # 처리 중 오류 발생 시 가능한 정보 반환
            return {
                "error": str(e),
                "detections": [] if 'detections' not in locals() else detections,
                "visualization": image if 'visualization' not in locals() else visualization
            }
    
    def export_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        결과를 JSON 형식으로 내보내기
        
        Args:
            results: 처리 결과
            output_path: 출력 파일 경로
        """
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [],
            "target_object": None,
            "reference_object": None,
            "goal_point": results["goal_point"] if results["goal_point"] else None,
            "timing": results["timing"]
        }
        
        # 객체 정보
        for i, obj in enumerate(results["detections"]):
            object_data = {
                "id": i,
                "class_name": obj["class_name"],
                "confidence": float(obj["confidence"]),
                "bbox": [float(x) for x in obj["bbox"]],
                "depth_info": {
                    "center_depth": float(obj["depth"]["center_depth"]),
                    "avg_depth": float(obj["depth"]["avg_depth"]),
                    "estimated_distance_cm": int(self.min_depth_cm + 
                                               obj["depth"]["avg_depth"] * (self.max_depth_cm - self.min_depth_cm))
                }
            }
            
            # 3D 좌표 정보 추가
            if "3d_coords" in obj:
                object_data["3d_coords"] = {
                    "center": [
                        float(obj["3d_coords"]["x_cm"]),
                        float(obj["3d_coords"]["y_cm"]),
                        float(obj["3d_coords"]["z_cm"])
                    ],
                    "dimensions": {
                        "width": float(obj["3d_coords"].get("width_cm", 0)),
                        "height": float(obj["3d_coords"].get("height_cm", 0)),
                        "depth": float(obj["3d_coords"].get("depth_cm", 0))
                    }
                }
            
            export_data["objects"].append(object_data)
            
            # 타겟/레퍼런스 객체 표시
            if obj == results["target_object"]:
                export_data["target_object"] = object_data
            elif obj == results["reference_object"]:
                export_data["reference_object"] = object_data
        
        # 목표 지점 3D 좌표 추가
        if results["goal_point"] and "goal_point" in results["goal_point"] and "3d_coords" in results["goal_point"]["goal_point"]:
            goal_3d = results["goal_point"]["goal_point"]["3d_coords"]
            export_data["goal_point"]["3d_coords"] = {
                "center": [
                    float(goal_3d["x_cm"]),
                    float(goal_3d["y_cm"]),
                    float(goal_3d["z_cm"])
                ]
            }
            
        # 2D 바운딩 박스 정보 추가
        if "goal_bounding_box" in results and results["goal_bounding_box"]:
            export_data["goal_point"]["2d_bbox"] = [float(x) for x in results["goal_bounding_box"]]
        elif results["goal_point"] and "goal_point" in results["goal_point"] and "2d_bbox" in results["goal_point"]["goal_point"]:
            export_data["goal_point"]["2d_bbox"] = [float(x) for x in results["goal_point"]["goal_point"]["2d_bbox"]]
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"결과 내보내기 완료: {output_path}")
    
    # 속성 설정 메서드
    def set_gesture_mode(self, active: bool):
        """
        제스처 모드 활성화/비활성화 설정
        
        Args:
            active: 제스처 모드 활성화 여부 (True/False)
        """
        self.gesture_mode_active = active
        self.logger.info(f"제스처 모드 {'활성화' if active else '비활성화'}")
    
    def is_gesture_mode_active(self) -> bool:
        """
        현재 제스처 모드 상태를 반환합니다.
        
        Returns:
            제스처 모드 활성화 여부
        """
        return self.gesture_mode_active 

    def _calculate_placement_offset(self, direction: str) -> list:
        """
        방향에 따른 오프셋 벡터 계산
        
        Args:
            direction: 방향 문자열
            
        Returns:
            list: [x_offset, y_offset, z_offset] 형태의 오프셋 벡터 (cm)
        """
        # 기본 오프셋
        x_offset, y_offset, z_offset = 0.0, 0.0, 0.0
        
        # 방향에 따른 오프셋 계산
        if direction == 'front':
            z_offset = self.OFFSET_Z_CM
        elif direction == 'back':
            z_offset = -self.OFFSET_Z_CM
        elif direction == 'left':
            x_offset = -self.OFFSET_X_CM
        elif direction == 'right':
            x_offset = self.OFFSET_X_CM
        elif direction == 'side':  # 'side'는 'right'와 동일하게 처리
            x_offset = self.OFFSET_X_CM
        elif direction == 'front_left':
            x_offset = -self.OFFSET_X_CM * 0.7
            z_offset = self.OFFSET_Z_CM * 0.7
        elif direction == 'front_right':
            x_offset = self.OFFSET_X_CM * 0.7
            z_offset = self.OFFSET_Z_CM * 0.7
        elif direction == 'back_left':
            x_offset = -self.OFFSET_X_CM * 0.7
            z_offset = -self.OFFSET_Z_CM * 0.7
        elif direction == 'back_right':
            x_offset = self.OFFSET_X_CM * 0.7
            z_offset = -self.OFFSET_Z_CM * 0.7
        
        # 상하 방향은 Y축 오프셋 사용
        # (실제 계산은 calculate_goal_3d_position에서 처리)
        
        return [x_offset, y_offset, z_offset]
    
    def estimate_base_y(self, obj: Dict[str, Any]) -> float:
        """
        객체의 바닥면 Y좌표 추정 (객체가 바닥에 닿는 위치)
        
        Args:
            obj: 객체 정보 딕셔너리 ('3d_coords', 'bbox' 포함)
            
        Returns:
            float: 추정된 바닥면 Y좌표 (cm 단위)
        """
        # 3D 좌표가 없는 경우 기본값 반환
        if not obj or '3d_coords' not in obj:
            self.logger.warning("바닥면 Y좌표 추정 실패: 객체에 3D 좌표 정보가 없습니다. 기본값 0.0 반환.")
            return 0.0
        
        try:
            # 객체의 3D 중심 좌표
            center_3d = obj['3d_coords']
            center_y = float(center_3d.get('y_cm', 0.0))
            depth_z = float(center_3d.get('z_cm', 100.0))  # 깊이 (기본값 100cm)
            
            # 객체의 2D 바운딩 박스 정보
            bbox = obj.get('bbox', [0, 0, 0, 0])
            if len(bbox) != 4:
                self.logger.warning("바운딩 박스 형식이 올바르지 않습니다. 기본 바닥면 Y좌표 반환.")
                return center_y
            
            # 바운딩 박스 높이 계산
            bbox_height_px = bbox[3] - bbox[1]
            
            # 추정을 위한 가정: 객체의 실제 높이는 바운딩 박스 높이와 비례
            # 깊이에 따른 투영 비율 계산 (간단한 근사치)
            # 가정: 깊이 100cm에서 1픽셀 ≈ 0.2cm (이 값은 카메라 특성에 따라 조정 필요)
            px_to_cm_ratio = 0.2 * (depth_z / 100.0)
            
            # 바운딩 박스 높이를 cm로 변환
            bbox_height_cm = bbox_height_px * px_to_cm_ratio
            
            # 바닥면 Y좌표 = 중심 Y좌표 + 바운딩 박스 높이의 절반
            # (바운딩 박스 하단이 객체의 바닥이라고 가정)
            base_y = center_y + (bbox_height_cm / 2.0)
            
            self.logger.info(f"객체 '{obj.get('class_name', 'unknown')}'의 바닥면 Y좌표 추정: {base_y:.2f}cm (중심 Y: {center_y:.2f}cm, 높이: {bbox_height_cm:.2f}cm)")
            return base_y
            
        except Exception as e:
            self.logger.error(f"바닥면 Y좌표 추정 중 오류: {e}", exc_info=True)
            return float(obj['3d_coords'].get('y_cm', 0.0))  # 오류 시 중심 Y 좌표 반환
    
    def calculate_goal_3d_position(self, reference_object: Dict[str, Any], direction: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        레퍼런스 객체와 방향을 기반으로 목표 3D 위치 계산
        
        Args:
            reference_object: 레퍼런스 객체 정보
            direction: 방향 ('front', 'back', 'left', 'right', 'above', 'below', 'side' 등)
            detections: 감지된 객체 목록 (IoU 계산 등에 사용)
            
        Returns:
            Dict[str, Any]: 목표 3D 위치 정보
                - 'position': [x, y, z] 형태의 3D 좌표 (cm 단위)
                - 'direction': 계산에 사용된 방향
                - 'reference_center': 레퍼런스 객체의 중심 3D 좌표
                - 'offset_vector': 적용된 오프셋 벡터
                - 'base_y': 적용된 바닥면 Y좌표
                - 'method': 계산 방법 설명
        """
        # 결과 초기화
        result = {
            'position': None,        # 최종 3D 좌표
            'direction': direction,  # 사용된 방향
            'reference_center': None,# 레퍼런스 중심 좌표
            'offset_vector': None,   # 적용된 오프셋 벡터
            'base_y': None,          # 바닥면 Y좌표
            'method': 'goal_3d_calculation_v2'  # 계산 방법
        }
        
        try:
            # 레퍼런스 객체 확인
            if not reference_object or '3d_coords' not in reference_object:
                self.logger.error("목표 3D 위치 계산 실패: 레퍼런스 객체가 없거나 3D 좌표 정보가 없습니다.")
                return result
            
            # 레퍼런스 객체의 3D 중심 좌표
            ref_center_3d = reference_object['3d_coords']
            ref_center = np.array([
                float(ref_center_3d.get('x_cm', 0.0)),
                float(ref_center_3d.get('y_cm', 0.0)),
                float(ref_center_3d.get('z_cm', 100.0))
            ])
            result['reference_center'] = ref_center.tolist()
            
            # 바닥면 Y좌표 추정
            ref_base_y = self.estimate_base_y(reference_object)
            result['base_y'] = ref_base_y
            
            # 방향에 따른 오프셋 벡터 계산
            offset_vector = self._calculate_placement_offset(direction)
            result['offset_vector'] = offset_vector.tolist()
            
            # 원시 목표 위치 계산 (방향에 따른 오프셋 적용)
            goal_3d_raw = ref_center + offset_vector
            
            # 최종 목표 위치 (바닥면 Y좌표 적용)
            goal_3d = np.array([
                goal_3d_raw[0],
                ref_base_y,  # 레퍼런스의 바닥면 Y좌표 사용
                goal_3d_raw[2]
            ])
            
            result['position'] = goal_3d.tolist()
            self.logger.info(f"목표 3D 위치 계산 완료: {goal_3d.tolist()} (방향: {direction}, 바닥면 Y: {ref_base_y:.2f}cm)")
            
            # TODO: IoU 계산 및 겹침 확인 (Phase 4에서 구현 예정)
            
            return result
            
        except Exception as e:
            self.logger.error(f"목표 3D 위치 계산 중 오류: {e}", exc_info=True)
            return result
    
    def project_3d_to_2d(self, position_3d: List[float], image_width: int, image_height: int) -> Dict[str, float]:
        """
        3D 좌표를 2D 이미지 평면에 투영
        
        Args:
            position_3d: [x, y, z] 형태의 3D 좌표 (cm 단위)
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            Dict[str, float]: 투영된 2D 좌표 (픽셀 단위)
                - 'x': x 좌표 (픽셀)
                - 'y': y 좌표 (픽셀)
                - 'depth': 깊이 정보 (cm)
                - 'is_valid': 투영 성공 여부
        """
        result = {
            'x': 0.0,
            'y': 0.0,
            'depth': 0.0,
            'is_valid': False
        }
        
        try:
            # 입력 확인
            if not position_3d or len(position_3d) != 3:
                self.logger.error(f"3D 투영 실패: 유효하지 않은 3D 좌표 형식: {position_3d}")
                return result
            
            # 3D 좌표 분리
            x_cm, y_cm, z_cm = position_3d
            
            # 깊이 정보 저장
            result['depth'] = z_cm
            
            # 카메라 내부 파라미터 설정 (depth_3d_mapper의 값과 동일하게 유지)
            # 실제 애플리케이션에서는 카메라 캘리브레이션 값을 사용해야 함
            fx = 1000.0  # 카메라 초점 거리 X
            fy = 1000.0  # 카메라 초점 거리 Y
            cx = image_width / 2.0  # 주점 X (이미지 중심)
            cy = image_height / 2.0  # 주점 Y (이미지 중심)
            
            # 깊이가 0이면 유효하지 않음
            if z_cm <= 0:
                self.logger.warning(f"3D 투영 실패: 깊이 값이 0 이하입니다: {z_cm}cm")
                return result
            
            # 카메라 좌표계를 이미지 좌표계로 변환
            # X' = (X * fx / Z) + cx
            # Y' = (Y * fy / Z) + cy
            # Y 좌표 반전 (3D 좌표계와 이미지 좌표계의 Y축 방향이 반대)
            x_px = (x_cm * fx / z_cm) + cx
            y_px = (-y_cm * fy / z_cm) + cy  # Y축 방향 반전
            
            # 결과 저장
            result['x'] = x_px
            result['y'] = y_px
            result['is_valid'] = True
            
            self.logger.info(f"3D 좌표 ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f})cm -> 2D 좌표 ({x_px:.2f}, {y_px:.2f})px 투영 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"3D->2D 투영 중 오류: {e}", exc_info=True)
            return result
    
    def generate_goal_bounding_box(self, goal_3d_position: Dict[str, Any], reference_object: Dict[str, Any], 
                                  image_width: int, image_height: int) -> Dict[str, Any]:
        """
        목표 3D 위치를 바탕으로 2D 바운딩 박스 생성
        
        Args:
            goal_3d_position: 목표 3D 위치 정보 딕셔너리
            reference_object: 레퍼런스 객체 정보
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            Dict[str, Any]: 목표 바운딩 박스 정보
                - 'bbox': [x1, y1, x2, y2] 형태의 바운딩 박스 (픽셀 단위)
                - 'center': [x, y] 형태의 중심 좌표 (픽셀 단위)
                - 'width': 바운딩 박스 너비 (픽셀)
                - 'height': 바운딩 박스 높이 (픽셀)
                - 'confidence': 신뢰도 점수 (0-1)
                - 'method': 생성 방법 설명
                - '3d_position': 원본 3D 좌표
        """
        # 결과 초기화
        result = {
            'bbox': [0, 0, 0, 0],
            'center': [0, 0],
            'width': 0,
            'height': 0,
            'confidence': 0.0,
            'method': 'projection_from_3d',
            '3d_position': None
        }
        
        try:
            # 3D 위치 확인
            if not goal_3d_position or not goal_3d_position.get('position'):
                self.logger.error("바운딩 박스 생성 실패: 유효한 3D 위치 정보가 없습니다.")
                return result
            
            # 3D 좌표 가져오기
            position_3d = goal_3d_position['position']
            result['3d_position'] = position_3d
            
            # 3D->2D 투영
            projected_2d = self.project_3d_to_2d(position_3d, image_width, image_height)
            
            if not projected_2d['is_valid']:
                self.logger.error("바운딩 박스 생성 실패: 3D->2D 투영에 실패했습니다.")
                return result
            
            # 중심 좌표
            center_x, center_y = projected_2d['x'], projected_2d['y']
            
            # 깊이 정보 (cm) - 가까울수록 큰 바운딩 박스
            depth_cm = projected_2d['depth']
            
            # 레퍼런스 객체의 크기 정보 가져오기
            ref_bbox = reference_object.get('bbox', [0, 0, 1, 1])
            ref_width = ref_bbox[2] - ref_bbox[0]
            ref_height = ref_bbox[3] - ref_bbox[1]
            
            # 레퍼런스 객체의 깊이 정보
            ref_depth = reference_object.get('3d_coords', {}).get('z_cm', 100.0)
            
            # 깊이에 따른 크기 조정 (가까울수록 크게, 멀수록 작게)
            # 동일한 물리적 크기의 물체는 거리가 2배 멀어지면 화면상 크기가 1/2로 감소
            if ref_depth > 0 and depth_cm > 0:
                depth_ratio = ref_depth / depth_cm
            else:
                depth_ratio = 1.0
                
            # 목표 바운딩 박스 크기 계산
            # 방향에 따라 크기를 다르게 조정할 수 있음
            direction = goal_3d_position.get('direction', 'front')
            
            # 기본 크기: 레퍼런스 객체의 80%
            base_width_ratio = 0.8
            base_height_ratio = 0.8
            
            # 방향에 따른 크기 조정 (선택적)
            if direction in ['front', 'back']:
                # 앞/뒤에 놓는 경우 비슷한 크기
                width_ratio = base_width_ratio
                height_ratio = base_height_ratio
            elif direction in ['left', 'right', 'side']:
                # 옆에 놓는 경우 약간 작게
                width_ratio = base_width_ratio * 0.9
                height_ratio = base_height_ratio
            else:
                # 기본값
                width_ratio = base_width_ratio
                height_ratio = base_height_ratio
            
            # 최종 바운딩 박스 크기 (깊이 비율 적용)
            bbox_width = ref_width * width_ratio * depth_ratio
            bbox_height = ref_height * height_ratio * depth_ratio
            
            # 중심점에서 바운딩 박스 좌표 계산
            x1 = center_x - (bbox_width / 2)
            y1 = center_y - (bbox_height / 2)
            x2 = center_x + (bbox_width / 2)
            y2 = center_y + (bbox_height / 2)
            
            # 이미지 경계 내로 제한
            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))
            
            # 결과 저장
            result['bbox'] = [x1, y1, x2, y2]
            result['center'] = [center_x, center_y]
            result['width'] = bbox_width
            result['height'] = bbox_height
            
            # 신뢰도 점수 계산 (거리에 반비례)
            # 가까운 물체일수록 높은 신뢰도
            confidence = min(1.0, max(0.1, 1.0 - (depth_cm / 300.0)))
            result['confidence'] = confidence
            
            self.logger.info(f"목표 바운딩 박스 생성 완료: {result['bbox']}, 중심: {result['center']}, 크기: {result['width']}x{result['height']}, 신뢰도: {confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"목표 바운딩 박스 생성 중 오류: {e}", exc_info=True)
            return result
    
    def infer_goal_placement(self, reference_object: Dict[str, Any], direction: str,
                            image_width: int, image_height: int,
                            detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        레퍼런스 객체를 기준으로 지정된 방향에 배치될 목표 위치를 추론합니다.
        
        Args:
            reference_object: 레퍼런스 객체 정보 (바운딩 박스, 클래스, 3D 좌표 등)
            direction: 방향 ('front', 'back', 'left', 'right', 'above', 'below', 'side' 등)
            image_width: 이미지 너비
            image_height: 이미지 높이
            detections: 현재 감지된 객체 목록 (IoU 계산 등에 사용)
            
        Returns:
            Dict[str, Any]: 목표 위치 정보
                - 'bbox': [x1, y1, x2, y2] 형태의 추론된 바운딩 박스
                - 'center': [x, y] 형태의 중심 좌표
                - '3d_position': 3D 위치 정보
                - 추가 메타데이터...
        """
        if not reference_object:
            self.logger.error("목표 위치 추론 실패: 레퍼런스 객체가 없습니다.")
            return None
            
        try:
            # Phase 1: 레퍼런스 객체와 방향 파싱
            if 'bbox' not in reference_object:
                self.logger.error("목표 위치 추론 실패: 레퍼런스 객체에 바운딩 박스가 없습니다.")
                return None
                
            # Phase 2: 3D 목표 위치 계산
            goal_3d_position = self.calculate_goal_3d_position(reference_object, direction, detections or [])
            
            # Phase 3: 3D 위치를 2D 바운딩 박스로 변환
            goal_bbox = self.generate_goal_bounding_box(
                goal_3d_position,
                reference_object,
                image_width,
                image_height
            )
            
            # 최종 결과 조합
            result = {
                'bbox': goal_bbox['bbox'],
                'center': goal_bbox['center'],
                'width': goal_bbox['width'],
                'height': goal_bbox['height'],
                'confidence': goal_bbox['confidence'],
                '3d_position': goal_3d_position.get('position'),
                'reference_object': {
                    'id': reference_object.get('id'),
                    'label': reference_object.get('label'),
                    'bbox': reference_object.get('bbox'),
                    'center': reference_object.get('center'),
                    '3d_coords': reference_object.get('3d_coords')
                },
                'direction': direction,
                'method': f"3d_projection_v1.{goal_bbox['method']}",
                'timestamp': time.time()
            }
            
            # Phase 4: 겹침 확인 및 로깅
            from modules.utils import calculate_iou
            
            # 레퍼런스 바운딩 박스와 목표 바운딩 박스 간의 IoU 계산
            ref_bbox = reference_object.get('bbox', [0, 0, 1, 1])
            goal_bbox_coords = result['bbox']
            
            iou_value = calculate_iou(ref_bbox, goal_bbox_coords)
            result['overlap_iou'] = iou_value  # 결과에 IoU 값 추가
            
            # IoU 임계값 설정 (0.1 = 10% 겹침)
            IOI_THRESHOLD = 0.1
            
            if iou_value > IOI_THRESHOLD:
                self.logger.warning(f"목표 bbox와 레퍼런스 bbox 겹침 감지 (IoU: {iou_value:.2f}). 조정 로직 비활성화됨.")
                result['has_overlap'] = True
                
                # TODO: 향후 겹침 조정 로직 추가 위치
                # 예: adjusted_bbox = self.adjust_bbox_for_overlap(goal_bbox_coords, ref_bbox, direction)
                # 1) 방향을 고려하여 겹치는 반대 방향으로 바운딩 박스 이동
                # 2) 이동 시 이미지 경계 확인
                # 3) 더 정교한 3D 공간 기반 조정 가능성 고려
            else:
                result['has_overlap'] = False
            
            self.logger.info(f"목표 위치 추론 완료: {result['bbox']}, 방향: {direction}, 레퍼런스: {reference_object.get('label')}")
            return result
            
        except Exception as e:
            self.logger.error(f"목표 위치 추론 중 오류 발생: {str(e)}", exc_info=True)
            return None 
    
    def generate_goal_bounding_box_2d(self, reference_object: Dict[str, Any], direction: Union[str, Dict[str, Any]], 
                                     image_width: int, image_height: int, 
                                     detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        2D 좌표계에서 직접 목표 바운딩 박스를 생성합니다.
        
        Args:
            reference_object: 레퍼런스 객체 정보
            direction: 방향 ('front', 'back', 'left', 'right', 'above', 'below', 'side' 등) 또는
                    방향 객체 {'type': 'simple', 'value': 방향값, ...}
            image_width: 이미지 가로 크기
            image_height: 이미지 세로 크기
            detections: 감지된 객체 목록 (겹침 확인에 사용)
            
        Returns:
            Dict[str, Any]: 목표 바운딩 박스 정보
        """
        # direction 매개변수 처리
        direction_value = direction
        if isinstance(direction, dict):
            # 방향 객체에서 direction 값 추출
            if direction.get('type') == 'simple':
                direction_value = direction.get('value', 'front')
            else:
                # 복잡한 타입은 기본값 사용
                self.logger.warning(f"지원되지 않는 방향 타입: {direction.get('type')}. 기본값 'front'로 대체합니다.")
                direction_value = 'front'
        
        # 결과 초기화
        result = {
            'bbox': None,            # 최종 2D 바운딩 박스
            'center': None,          # 중심 좌표
            'dimensions': None,      # [width, height]
            'direction': direction,  # 원본 방향 (딕셔너리 또는 문자열)
            'direction_value': direction_value,  # 추출된 방향 값 (문자열)
            'reference_bbox': None,  # 레퍼런스 바운딩 박스
            'method': 'direct_2d_offset',  # 계산 방법
            'overlap_iou': 0.0,      # 겹침 IoU 값
            'has_overlap': False     # 겹침 여부
        }
        
        try:
            # 레퍼런스 객체 확인
            if not reference_object or 'bbox' not in reference_object:
                self.logger.error("목표 바운딩 박스 계산 실패: 레퍼런스 객체가 없거나 바운딩 박스 정보가 없습니다.")
                return result
            
            # 레퍼런스 객체의 2D 바운딩 박스 정보
            ref_bbox = reference_object['bbox']
            if len(ref_bbox) != 4:
                self.logger.error(f"레퍼런스 바운딩 박스 형식이 올바르지 않습니다: {ref_bbox}")
                return result
            
            result['reference_bbox'] = ref_bbox
            
            # 레퍼런스 객체의 중심점과 크기 계산
            ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
            ref_width = ref_x2 - ref_x1
            ref_height = ref_y2 - ref_y1
            ref_center_x = (ref_x1 + ref_x2) / 2
            ref_center_y = (ref_y1 + ref_y2) / 2
            
            # 방향에 따른 오프셋 가져오기
            if direction_value not in self.direction_2d_offsets:
                self.logger.warning(f"알 수 없는 방향: '{direction_value}'. 기본값 'front'로 대체합니다.")
                direction_value = 'front'
            
            offset_x, offset_y = self.direction_2d_offsets[direction_value]
            
            # 객체 크기에 비례한 오프셋 스케일링
            size_scale = min(max(ref_width, ref_height) / 300.0, 2.0)  # 최대 2배까지만 스케일링
            scaled_offset_x = offset_x * size_scale
            scaled_offset_y = offset_y * size_scale
            
            # 목표 바운딩 박스 중심점 계산
            goal_center_x = ref_center_x + scaled_offset_x
            goal_center_y = ref_center_y + scaled_offset_y
            
            # 목표 바운딩 박스 크기 (레퍼런스 객체의 2/3 크기)
            goal_width = ref_width * 0.67
            goal_height = ref_height * 0.67
            
            # 목표 바운딩 박스 좌표 계산
            goal_x1 = goal_center_x - goal_width / 2
            goal_y1 = goal_center_y - goal_height / 2
            goal_x2 = goal_center_x + goal_width / 2
            goal_y2 = goal_center_y + goal_height / 2
            
            # 이미지 경계 내로 좌표 클리핑
            goal_x1 = max(0, min(goal_x1, image_width - 1))
            goal_y1 = max(0, min(goal_y1, image_height - 1))
            goal_x2 = max(goal_x1 + 10, min(goal_x2, image_width))  # 최소 10픽셀 너비 보장
            goal_y2 = max(goal_y1 + 10, min(goal_y2, image_height))  # 최소 10픽셀 높이 보장
            
            # 최종 바운딩 박스
            goal_bbox = [goal_x1, goal_y1, goal_x2, goal_y2]
            
            # 겹침 확인
            from modules.utils import calculate_iou
            iou_value = calculate_iou(ref_bbox, goal_bbox)
            has_overlap = iou_value > 0.1  # 10% 이상 겹치면 겹침으로 간주
            
            # 겹침 발생 시 오프셋 증가 재계산 (최대 3회)
            attempt = 0
            while has_overlap and attempt < 3:
                attempt += 1
                # 방향에 따라 오프셋 증가
                additional_offset = 30 * attempt  # 추가 오프셋 (30픽셀씩 증가)
                if direction_value == 'front' or direction_value == 'below': # 앞 또는 아래 방향
                    goal_y1 += additional_offset
                    goal_y2 += additional_offset
                elif direction_value == 'back' or direction_value == 'above': # 뒤 또는 위 방향
                    goal_y1 -= additional_offset
                    goal_y2 -= additional_offset
                elif direction_value == 'left':
                    goal_x1 -= additional_offset
                    goal_x2 -= additional_offset
                elif direction_value == 'right' or direction_value == 'side':
                    goal_x1 += additional_offset
                    goal_x2 += additional_offset
                # 'above' 와 'below' 는 위에서 이미 처리됨
                
                # 이미지 경계 내로 좌표 클리핑
                goal_x1 = max(0, min(goal_x1, image_width - 1))
                goal_y1 = max(0, min(goal_y1, image_height - 1))
                goal_x2 = max(goal_x1 + 10, min(goal_x2, image_width))
                goal_y2 = max(goal_y1 + 10, min(goal_y2, image_height))
                
                # 새 바운딩 박스로 겹침 다시 확인
                goal_bbox = [goal_x1, goal_y1, goal_x2, goal_y2]
                iou_value = calculate_iou(ref_bbox, goal_bbox)
                has_overlap = iou_value > 0.1
                
                if not has_overlap:
                    self.logger.info(f"겹침 해결됨 (시도 {attempt}): IoU {iou_value:.3f}")
                    break
            
            # 최종 결과 설정
            result['bbox'] = goal_bbox
            result['center'] = [goal_center_x, goal_center_y]
            result['dimensions'] = [goal_width, goal_height]
            result['overlap_iou'] = iou_value
            result['has_overlap'] = has_overlap
            
            if has_overlap:
                self.logger.warning(f"목표 바운딩 박스와 레퍼런스 바운딩 박스 겹침 감지 (IoU: {iou_value:.3f}). 재시도 {attempt}회 후에도 해결되지 않음.")
            else:
                self.logger.info(f"목표 바운딩 박스 생성 완료: {goal_bbox} (방향: {direction_value})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"목표 바운딩 박스 계산 중 오류: {e}", exc_info=True)
            return result