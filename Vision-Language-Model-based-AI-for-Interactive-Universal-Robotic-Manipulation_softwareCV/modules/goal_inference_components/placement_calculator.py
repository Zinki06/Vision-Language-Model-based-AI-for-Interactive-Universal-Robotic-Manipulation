"""
목표 위치 계산 모듈

이 모듈은 목표 위치 계산을 담당하는 PlacementCalculator 클래스를 포함합니다.
"""

import logging
import time
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from ..utils.geometry import calculate_iou

class PlacementCalculator:
    """
    목표 위치 계산 클래스
    
    이 클래스는 다음과 같은 목표 위치 계산 관련 기능을 담당합니다:
    - 3D 위치 계산
    - 목표 바운딩 박스 생성
    - 방향 벡터 관리
    - 목표 위치와 레퍼런스 객체 간 겹침 확인
    """
    
    def __init__(self,
                min_depth_cm: float = 30.0,
                max_depth_cm: float = 300.0,
                offset_x_cm: float = 15.0,
                offset_z_cm: float = 10.0,
                placement_margin_cm: float = 2.0,
                logger: Optional[logging.Logger] = None):
        """
        PlacementCalculator 초기화
        
        Args:
            min_depth_cm: 최소 깊이 값 (cm)
            max_depth_cm: 최대 깊이 값 (cm)
            offset_x_cm: 좌/우 오프셋 (cm)
            offset_z_cm: 앞쪽 오프셋 (카메라 기준 Z축) (cm)
            placement_margin_cm: 겹침 조정 시 사용 (cm)
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 깊이 범위 설정
        self.min_depth_cm = min_depth_cm
        self.max_depth_cm = max_depth_cm
        
        # 오프셋 상수
        self.OFFSET_X_CM = offset_x_cm
        self.OFFSET_Z_CM = offset_z_cm
        self.PLACEMENT_MARGIN_CM = placement_margin_cm
        
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
            'side':  (+offset_x, 0)   # 옆은 오른쪽과 동일하게 처리
        }
        
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
    
    def calculate_goal_3d_position(self, 
                                 reference_object: Dict[str, Any], 
                                 direction: str,
                                 detections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        참조 객체와 방향을 기반으로 3D 공간에서 목표 위치를 계산합니다.
        
        Args:
            reference_object: 참조 객체 정보가 담긴 딕셔너리
            direction: 참조 객체로부터의 방향 문자열 또는 방향 객체
            detections: 감지된 객체 목록 (선택적)
            
        Returns:
            Dict[str, Any]: 3D 목표 위치 정보가 담긴 딕셔너리
        """
        try:
            # 참조 객체의 3D 중심 좌표
            if '3d_coords' not in reference_object or 'center' not in reference_object['3d_coords']:
                self.logger.error(f"참조 객체에 3D 좌표 정보가 없습니다: {reference_object}")
                return {"success": False, "error": "참조 객체에 3D 좌표 정보가 없습니다"}
            
            reference_center = reference_object['3d_coords']['center']
            
            # 참조 객체의 크기 정보
            if 'dimensions' in reference_object['3d_coords']:
                reference_dimensions = reference_object['3d_coords']['dimensions']
            else:
                reference_dimensions = [0.3, 0.3, 0.3]  # 기본 크기 가정
                self.logger.warning(f"참조 객체 크기 정보가 없어 기본값을 사용합니다: {reference_dimensions}")
            
            # 방향 벡터 및 오프셋 계산
            direction_vector = None
            direction_name = None
            
            # 방향 객체 형식 처리
            if isinstance(direction, dict):
                offset = self._calculate_placement_offset_from_direction_object(direction)
                
                dir_type = direction.get("type", "simple")
                if dir_type == "simple":
                    direction_name = direction.get("value", "front")
                    if direction_name in self.direction_vectors:
                        direction_vector = self.direction_vectors[direction_name]
                    else:
                        direction_vector = self.direction_vectors["front"]
                        direction_name = "front"
                elif dir_type == "random":
                    # 선택된 방향에 대한 정보를 랜덤 벡터 함수 내에서 얻어옴
                    options = direction.get("options", [])
                    if options and all(opt in self.direction_vectors for opt in options):
                        selected_option = options[0]  # 기본값 설정
                        weights = direction.get("weights", [1.0/len(options)] * len(options))
                        
                        # 랜덤 선택 (offset 계산 시와 동일한 방식으로)
                        import random
                        selected_option = random.choices(options, weights=weights, k=1)[0]
                        
                        direction_vector = self.direction_vectors[selected_option]
                        direction_name = selected_option
                    else:
                        direction_vector = self.direction_vectors["front"]
                        direction_name = "front"
            else:
                # 문자열 방향 처리 (기존 로직)
                if direction not in self.direction_vectors:
                    self.logger.warning(f"알 수 없는 방향입니다: {direction}. 'front'를 사용합니다.")
                    direction = 'front'
                    
                direction_vector = self.direction_vectors[direction]
                direction_name = direction
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
            base_y = self._estimate_base_y(reference_object)
            
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
                "direction": direction,
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
                "direction": direction if isinstance(direction, str) else "front"
            }
    
    def _calculate_placement_offset(self, direction: str) -> List[float]:
        """
        방향에 따른 오프셋 벡터 계산
        
        Args:
            direction: 방향 문자열
            
        Returns:
            List[float]: [x_offset, y_offset, z_offset] 형태의 오프셋 벡터 (cm)
        """
        # 방향 객체 형식인 경우 처리
        if isinstance(direction, dict):
            return self._calculate_placement_offset_from_direction_object(direction)
            
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
    
    def _calculate_placement_offset_from_direction_object(self, direction_obj: Dict[str, Any]) -> List[float]:
        """
        방향 객체 형식에 따른 오프셋 벡터 계산
        
        Args:
            direction_obj: 방향 객체 딕셔너리
            
        Returns:
            List[float]: [x_offset, y_offset, z_offset] 형태의 오프셋 벡터 (cm)
        """
        dir_type = direction_obj.get("type", "simple")
        
        if dir_type == "simple":
            # 단일 방향 처리
            direction_value = direction_obj.get("value", "front")
            return self._calculate_placement_offset(direction_value)
            
        elif dir_type == "random":
            # 랜덤 방향 처리
            return self._calculate_random_direction_vector(direction_obj)
            
        # 기본 방향 (앞)
        return [0.0, 0.0, self.OFFSET_Z_CM]
    
    def _calculate_random_direction_vector(self, direction_data: Dict[str, Any]) -> List[float]:
        """
        랜덤 방향에 대한 벡터 계산
        
        Args:
            direction_data: 랜덤 방향 객체 정보
            
        Returns:
            List[float]: [x_offset, y_offset, z_offset] 형태의 오프셋 벡터 (cm)
        """
        import random
        
        options = direction_data.get("options", [])
        weights = direction_data.get("weights", [1.0/len(options)] * len(options))
        
        # 옵션이 없는 경우 기본 방향(앞) 사용
        if not options:
            self.logger.warning("랜덤 방향 옵션이 비어있습니다. 기본 방향(앞)을 사용합니다.")
            return [0.0, 0.0, self.OFFSET_Z_CM]
        
        # 가중치에 따른 랜덤 선택
        selected_direction = random.choices(options, weights=weights, k=1)[0]
        self.logger.info(f"랜덤 방향이 선택되었습니다: {selected_direction}")
        
        # 선택된 방향에 대한 오프셋 계산
        return self._calculate_placement_offset(selected_direction)
    
    def _estimate_base_y(self, obj: Dict[str, Any]) -> float:
        """
        객체의 바닥면 Y좌표 추정
        
        Args:
            obj: 객체 정보 (3D 좌표 포함)
            
        Returns:
            float: 바닥면 Y좌표 (cm)
        """
        try:
            # 3D 중심 좌표 가져오기
            center_3d = obj.get('3d_coords', {}).get('center', [0, 0, 100])
            center_y = center_3d[1] if len(center_3d) > 1 else 0
            
            # 3D 높이 가져오기
            dimensions = obj.get('3d_coords', {}).get('dimensions', [30, 30, 30])
            height_3d = dimensions[1] if len(dimensions) > 1 else 30
            
            # 바닥면 Y좌표 계산 (중심에서 높이의 절반만큼 아래)
            base_y = center_y - (height_3d / 2)
            
            return base_y
        except Exception as e:
            self.logger.error(f"바닥면 Y좌표 추정 중 오류: {e}")
            return 0.0
    
    def generate_goal_bounding_box_2d(self,
                                    reference_object: Dict[str, Any],
                                    direction: str,
                                    image_width: int,
                                    image_height: int,
                                    detections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        2D 직접 오프셋 방식으로 목표 바운딩 박스 생성
        
        Args:
            reference_object: 레퍼런스 객체 정보
            direction: 방향 문자열
            image_width: 이미지 너비
            image_height: 이미지 높이
            detections: 다른 감지된 객체 목록 (선택적)
            
        Returns:
            Dict[str, Any]: 생성된 바운딩 박스 및 관련 정보
        """
        try:
            # 레퍼런스 객체 바운딩 박스 가져오기
            ref_bbox = reference_object.get('bbox') or reference_object.get('box')
            if not ref_bbox or len(ref_bbox) != 4:
                self.logger.error("레퍼런스 객체의 바운딩 박스 정보가 없거나 잘못되었습니다.")
                return None
                
            # 방향 오프셋 가져오기
            if direction not in self.direction_2d_offsets:
                self.logger.warning(f"알 수 없는 방향: {direction}, 'front' 사용")
                direction = 'front'
                
            offset_x, offset_y = self.direction_2d_offsets[direction]
            
            # 레퍼런스 박스 정보
            ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
            ref_width = ref_x2 - ref_x1
            ref_height = ref_y2 - ref_y1
            ref_center_x = (ref_x1 + ref_x2) / 2
            ref_center_y = (ref_y1 + ref_y2) / 2
            
            # 목표 박스 중심점 계산 (오프셋 적용)
            goal_center_x = ref_center_x + offset_x
            goal_center_y = ref_center_y + offset_y
            
            # 목표 박스 크기 (레퍼런스와 동일하게 가정)
            goal_width = ref_width
            goal_height = ref_height
            
            # 목표 바운딩 박스 계산
            goal_x1 = goal_center_x - (goal_width / 2)
            goal_y1 = goal_center_y - (goal_height / 2)
            goal_x2 = goal_center_x + (goal_width / 2)
            goal_y2 = goal_center_y + (goal_height / 2)
            
            # 이미지 경계 내로 제한
            goal_x1 = max(0, min(image_width - 1, goal_x1))
            goal_y1 = max(0, min(image_height - 1, goal_y1))
            goal_x2 = max(0, min(image_width - 1, goal_x2))
            goal_y2 = max(0, min(image_height - 1, goal_y2))
            
            # 최종 바운딩 박스
            goal_bbox = [goal_x1, goal_y1, goal_x2, goal_y2]
            
            # 겹침 확인
            iou_value = calculate_iou(ref_bbox, goal_bbox)
            has_overlap = iou_value > 0.1  # 10% 이상 겹침
            
            if has_overlap:
                self.logger.warning(f"목표 bbox와 레퍼런스 bbox 겹침 감지 (IoU: {iou_value:.2f}). 조정 로직 비활성화됨.")
            
            # 결과 생성
            result = {
                "bbox": goal_bbox,
                "center": [goal_center_x, goal_center_y],
                "width": goal_width,
                "height": goal_height,
                "direction": direction,
                "reference_bbox": ref_bbox,
                "method": "direct_2d_offset",
                "overlap_iou": iou_value,
                "has_overlap": has_overlap
            }
            
            self.logger.debug(f"2D 직접 오프셋 방식의 목표 바운딩 박스: {goal_bbox}")
            return result
        except Exception as e:
            self.logger.error(f"2D 바운딩 박스 생성 중 오류: {e}")
            return None 