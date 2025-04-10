"""
공간 계산 관련 유틸리티 모듈

이 모듈은 바운딩 박스, 각도, 점 집합 등 공간적 계산과 관련된 함수들을 제공합니다.
"""

from typing import List
import numpy as np

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산
    
    Args:
        box1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
        box2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
        
    Returns:
        두 박스 간의 IoU (0~1 사이 값)
    """
    # 각 박스의 좌표
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역 좌표 계산
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 교집합 영역 넓이
    if x2_i < x1_i or y2_i < y1_i:
        # 겹치는 영역 없음
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 각 박스 영역 넓이
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 영역 넓이 = 두 영역의 합 - 교집합
    union_area = box1_area + box2_area - intersection_area
    
    # IoU 계산
    if union_area < 1e-6:  # 0으로 나누기 방지
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def calculate_box_from_points(points: List[List[float]]) -> List[float]:
    """점 집합에서 바운딩 박스 계산
    
    Args:
        points: 점 좌표 리스트 [[x1, y1], [x2, y2], ...]
        
    Returns:
        바운딩 박스 [xmin, ymin, xmax, ymax]
    """
    if not points:
        return [0, 0, 0, 0]
    
    # 각 축별 최소/최대 좌표
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    
    return [xmin, ymin, xmax, ymax]

def calculate_angle_2d(vector1: List[float], vector2: List[float]) -> float:
    """2D 벡터 간의 각도를 계산
    
    Args:
        vector1: 첫 번째 2D 벡터 [x, y]
        vector2: 두 번째 2D 벡터 [x, y]
        
    Returns:
        두 벡터 간의 각도 (도 단위, 0-180)
    """
    # 벡터 정규화
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # 0벡터 처리
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 180.0  # 한 벡터가 0이면 최대 각도 반환
    
    v1_norm = v1 / norm_v1
    v2_norm = v2 / norm_v2
    
    # 내적으로 각도 계산
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg 