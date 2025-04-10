#!/usr/bin/env python
"""
시각화 엔진 모듈

이 모듈은 목표지점 추론 결과를 시각화하는 기능을 제공합니다.
객체 감지, 깊이 맵, 3D 좌표, 목표지점 등을 통합적으로 시각화합니다.
"""

import logging
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

# MediaPipe 그리기 유틸리티 임포트
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

# GestureRecognizer에서 사용하는 Enum 임포트
from .gesture_recognizer import GestureType

# 전역 변수로 그리기 스타일 설정 (선택 사항)
_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5

# MediaPipe DrawingSpec 정의 (색상, 두께 등)
_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(121, 121, 121), thickness=1)
_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2) # 빨간색 랜드마크
_pointing_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2) # 초록색 가리키기
_grabbing_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2) # 파란색 쥐기

# 색상 상수 정의 (BGR 형식)
TARGET_COLOR = (255, 0, 0)        # 파란색
REFERENCE_COLOR = (255, 0, 255)   # 분홍색
GOAL_COLOR = (0, 255, 0)          # 초록색
GOAL_TEXT_COLOR = (255, 255, 255) # 흰색
DEFAULT_COLOR = (0, 140, 255)     # 주황색

class VisualizationEngine:
    """
    시각화 엔진 클래스
    
    목표지점 추론 결과를 통합적으로 시각화하는 기능을 제공합니다.
    """
    
    def __init__(self, 
                 min_depth_cm: float = 30.0, 
                 max_depth_cm: float = 300.0,
                 logger: Optional[logging.Logger] = None):
        """
        시각화 엔진 초기화
        
        Args:
            min_depth_cm: 최소 깊이값 (cm)
            max_depth_cm: 최대 깊이값 (cm)
            logger: 로거 객체 (None인 경우 기본 로거 생성)
        """
        self.min_depth_cm = min_depth_cm
        self.max_depth_cm = max_depth_cm
        
        # 색상 및 폰트 설정
        self.text_y_step = 25
        self.confidence_threshold = 0.4
        
        # 기본 색상 설정
        self.default_color = DEFAULT_COLOR
        self.high_conf_color = (0, 255, 0)   # 초록색
        self.low_conf_color = (0, 0, 255)    # 빨간색
        self.target_color = TARGET_COLOR
        self.reference_color = REFERENCE_COLOR
        self.goal_color = GOAL_COLOR
        self.goal_text_color = GOAL_TEXT_COLOR
        
        # 로거 설정
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("VisualizationEngine 초기화 완료")
        
        # 시각화 설정
        self.depth_colormap = cv2.COLORMAP_INFERNO
        # 제스처 시각화 색상 추가
        self.gesture_colors = {
            GestureType.POINTING: (0, 255, 0),  # 초록색
            GestureType.GRABBING: (255, 0, 0),  # 파란색
            GestureType.UNKNOWN: (0, 255, 255) # 노란색
        }
    
    def _get_object_center(self, obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        객체의 2D 중심 좌표 계산
        
        Args:
            obj: 객체 정보 딕셔너리 (bbox 키 포함)
            
        Returns:
            (center_x, center_y) 튜플 또는 None (좌표 계산 실패 시)
        """
        if obj is None or 'bbox' not in obj:
            return None
            
        bbox = obj['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return (center_x, center_y)
        except Exception as e:
            self.logger.error(f"객체 중심 좌표 계산 중 오류: {e}")
            return None
            
    def _project_3d_to_2d(self, coords_3d: Union[np.ndarray, List, Dict]) -> Optional[Tuple[float, float]]:
        """
        3D 좌표를 2D 이미지 좌표로 투영 (단순 투영)
        
        Args:
            coords_3d: 3D 좌표 (numpy 배열, 리스트, 또는 딕셔너리)
            
        Returns:
            (x, y) 튜플 또는 None (투영 실패 시)
        """
        if coords_3d is None:
            return None
            
        try:
            # Dict 타입 처리 (x_cm, y_cm 키 사용)
            if isinstance(coords_3d, dict):
                if 'x_cm' in coords_3d and 'y_cm' in coords_3d:
                    return (coords_3d['x_cm'], coords_3d['y_cm'])
                return None
                
            # 리스트/배열 타입 처리 (첫 두 요소 사용)
            if isinstance(coords_3d, (list, tuple, np.ndarray)):
                if len(coords_3d) >= 2:
                    return (coords_3d[0], coords_3d[1])
                    
            return None
        except Exception as e:
            self.logger.error(f"3D->2D 투영 중 오류: {e}")
            return None
            
    def _draw_hand_gestures(self, 
                            image: np.ndarray, 
                            gesture_results: List[Dict[str, Any]]):
        """손 랜드마크와 제스처 정보를 이미지에 그립니다."""
        if not gesture_results:
            return

        height, width = image.shape[:2]

        for hand_info in gesture_results:
            landmarks_2d_norm = hand_info.get('landmarks_2d') # 0-1 정규화된 좌표
            gesture_type = hand_info.get('gesture')
            handedness = hand_info.get('handedness', '')

            if not landmarks_2d_norm:
                continue

            # NormalizedLandmark 객체 리스트로 변환 (mp.solutions.drawing_utils 호환용)
            landmark_list_proto = landmark_pb2.NormalizedLandmarkList()
            for lm_norm in landmarks_2d_norm:
                 landmark = landmark_list_proto.landmark.add()
                 landmark.x = lm_norm['x']
                 landmark.y = lm_norm['y']
                 # visibility, presence는 없으므로 기본값 사용
                 landmark.visibility = 1.0
                 landmark.presence = 1.0

            # 랜드마크 및 연결선 그리기
            # 특정 제스처에 따라 랜드마크 색상 변경
            current_landmark_spec = _landmark_drawing_spec
            if gesture_type == GestureType.POINTING:
                current_landmark_spec = _pointing_drawing_spec
            elif gesture_type == GestureType.GRABBING:
                current_landmark_spec = _grabbing_drawing_spec
                
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                landmark_list_proto,
                _HAND_CONNECTIONS,
                current_landmark_spec, # 랜드마크 스타일
                _drawing_spec # 연결선 스타일
            )
            
            # 제스처 유형 텍스트 표시 (손목 근처)
            wrist_landmark = landmarks_2d_norm[0]
            text_x = int(wrist_landmark['x'] * width)
            text_y = int(wrist_landmark['y'] * height) + 20
            color = self.gesture_colors.get(gesture_type, (255, 255, 255))
            gesture_text = f"{handedness} Hand: {gesture_type.name if gesture_type else 'N/A'}"
            cv2.putText(image, gesture_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_goal_bounding_box(self, 
                              image: np.ndarray, 
                              goal_bounding_box: List[int]):
        """
        목표 지점 바운딩 박스를 그립니다.
        
        Args:
            image: 시각화할 이미지
            goal_bounding_box: 목표 지점 바운딩 박스 좌표 [x1, y1, x2, y2]
            
        Returns:
            바운딩 박스가 그려진 이미지
        """
        if goal_bounding_box is None or len(goal_bounding_box) != 4:
            return image
            
        vis_img = image.copy()
        x1, y1, x2, y2 = map(int, goal_bounding_box)
        
        # 가로선 점선 스타일 (10픽셀마다 5픽셀 선)
        for i in range(x1, x2, 10):
            cv2.line(vis_img, (i, y1), (i + 5, y1), self.goal_color, 2)
            cv2.line(vis_img, (i, y2), (i + 5, y2), self.goal_color, 2)
        
        # 세로선 점선 스타일 (10픽셀마다 5픽셀 선)
        for i in range(y1, y2, 10):
            cv2.line(vis_img, (x1, i), (x1, i + 5), self.goal_color, 2)
            cv2.line(vis_img, (x2, i), (x2, i + 5), self.goal_color, 2)
        
        # 텍스트 배경 추가 (더 나은 가독성을 위해)
        text = "목표 위치"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), self.goal_color, -1)
        
        # "목표 위치" 텍스트 표시 (흰색 텍스트로 변경)
        cv2.putText(
            vis_img, 
            text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            self.goal_text_color, 
            2
        )
        
        return vis_img

    def _draw_object(self, 
                    image: np.ndarray, 
                    detection: Dict[str, Any],
                    is_target: bool = False,
                    is_reference: bool = False):
        """
        감지된 객체를 이미지에 그립니다.
        
        Args:
            image: 시각화할 이미지
            detection: 감지된 객체 정보
            is_target: 타겟 객체 여부
            is_reference: 레퍼런스 객체 여부
            
        Returns:
            객체가 그려진 이미지
        """
        vis_img = image.copy()
        
        x1, y1, x2, y2 = [int(coord) for coord in detection.get("bbox", [0,0,0,0])]
        depth_value = detection.get("depth", {}).get("center_depth", 0.5)
        color_r = int(255 * (1 - depth_value))
        color_b = int(255 * depth_value)

        if is_target:
            box_color = self.target_color
            label = "Target"
        elif is_reference:
            box_color = self.reference_color
            label = "Reference"
        else:
            box_color = (color_b, 0, color_r)
            label = "Object"

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
        
        class_name = detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.0)
        depth_avg = detection.get("depth", {}).get("avg_depth", 0.5)
        approx_dist_cm = int(self.min_depth_cm + depth_avg * (self.max_depth_cm - self.min_depth_cm))
        
        text = f"{label}: {class_name} ({confidence:.2f}) ~{approx_dist_cm}cm"
        cv2.putText(vis_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_img, (center_x, center_y), 4, box_color, -1)

        # 3D 좌표 정보 표시 (간략화)
        coords_3d = detection.get('3d_coords')
        if coords_3d and isinstance(coords_3d, dict):
            z_cm = coords_3d.get('z_cm')
            if z_cm is not None:
                text_3d_z = f"Z={z_cm:.1f}cm"
                cv2.putText(vis_img, text_3d_z, (x1, y2 + self.text_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                
        return vis_img

    def _draw_dashed_arrow(self, 
                         image: np.ndarray, 
                         start_point: Tuple[int, int], 
                         end_point: Tuple[int, int], 
                         color: Tuple[int, int, int], 
                         thickness: int = 2,
                         segment_length: int = 5, 
                         gap_length: int = 3,
                         tip_length: float = 0.3):
        """
        점선 화살표를 그립니다.
        
        Args:
            image: 그림을 그릴 이미지
            start_point: 시작점 좌표 (x, y)
            end_point: 끝점 좌표 (x, y)
            color: 화살표 색상 (B, G, R)
            thickness: 선 두께
            segment_length: 실선 세그먼트 길이
            gap_length: 세그먼트 간격 길이
            tip_length: 화살표 팁 길이 비율
            
        Returns:
            화살표가 그려진 이미지
        """
        # 시작점과 끝점을 정수로 변환
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))
        
        # 벡터 계산
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        # 벡터 길이 계산
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude < 1e-6:
            return image  # 길이가 0에 가까우면 그리지 않음
        
        # 단위 벡터
        dx, dy = dx/magnitude, dy/magnitude
        
        # 화살표 몸통을 위한 끝 지점 (화살표 팁 앞)
        arrow_end_x = start_point[0] + dx * magnitude * (1 - tip_length)
        arrow_end_y = start_point[1] + dy * magnitude * (1 - tip_length)
        arrow_end = (int(arrow_end_x), int(arrow_end_y))
        
        # 현재 그리는 점
        current_point = np.array(start_point, dtype=float)
        
        # 화살표 몸통의 80%까지 점선으로 그리기
        total_length = 0.0
        max_length = magnitude * (1 - tip_length)
        
        while total_length < max_length:
            # 다음 세그먼트 끝점
            next_length = min(total_length + segment_length, max_length)
            segment_end = (
                int(start_point[0] + dx * next_length),
                int(start_point[1] + dy * next_length)
            )
            
            # 선 세그먼트 그리기
            cv2.line(image, tuple(current_point.astype(int)), segment_end, color, thickness)
            
            # 다음 세그먼트 시작점으로 이동 (간격 건너뛰기)
            total_length = next_length + gap_length
            current_point = np.array([
                start_point[0] + dx * total_length,
                start_point[1] + dy * total_length
            ])
        
        # 화살표 팁 그리기
        cv2.arrowedLine(image, arrow_end, end_point, color, thickness, tipLength=tip_length*3)
        
        return image
    
    def _get_direction_vector(self, direction: str) -> np.ndarray:
        """
        방향 이름에 해당하는 벡터를 반환합니다.
        
        Args:
            direction: 방향 문자열 ('front', 'back', 'left', 'right', 'above', 'below' 등)
            
        Returns:
            np.ndarray: 2D 방향 벡터 [dx, dy]
        """
        # 2D 화면 기준 방향 벡터 (x, y)
        direction_vectors = {
            'front': np.array([0, 1]),      # 아래쪽
            'back': np.array([0, -1]),      # 위쪽
            'left': np.array([-1, 0]),      # 왼쪽
            'right': np.array([1, 0]),      # 오른쪽
            'above': np.array([0, -1]),     # 위쪽
            'below': np.array([0, 1]),      # 아래쪽
            'front_left': np.array([-0.7, 0.7]),
            'front_right': np.array([0.7, 0.7]),
            'back_left': np.array([-0.7, -0.7]),
            'back_right': np.array([0.7, -0.7])
        }
        
        # 기본 방향 반환
        return direction_vectors.get(direction, np.array([0, 1]))
    
    def _calculate_end_point(self, 
                           start_point: Tuple[int, int], 
                           direction_vector: np.ndarray, 
                           distance: int) -> Tuple[int, int]:
        """
        시작점에서 방향 벡터와 거리에 따른 끝점을 계산합니다.
        
        Args:
            start_point: 시작점 좌표 (x, y)
            direction_vector: 방향 벡터 [dx, dy]
            distance: 화살표 길이 (픽셀)
            
        Returns:
            Tuple[int, int]: 끝점 좌표 (x, y)
        """
        # 방향 벡터 정규화
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            unit_vector = direction_vector / norm
        else:
            unit_vector = np.array([0, 1])  # 기본 방향 (아래쪽)
            
        # 끝점 계산
        end_x = int(start_point[0] + unit_vector[0] * distance)
        end_y = int(start_point[1] + unit_vector[1] * distance)
        
        return (end_x, end_y)
    
    def _draw_direction(self, 
                      image: np.ndarray, 
                      direction_data: Union[str, Dict[str, Any]], 
                      start_point: Tuple[int, int], 
                      distance: int = 100):
        """
        방향을 시각화합니다.
        
        Args:
            image: 그림을 그릴 이미지
            direction_data: 방향 정보 (문자열 또는 방향 객체)
            start_point: 화살표 시작점 좌표 (x, y)
            distance: 화살표 길이 (픽셀)
            
        Returns:
            방향이 그려진 이미지
        """
        # 문자열 방향인 경우 처리
        if isinstance(direction_data, str):
            vector = self._get_direction_vector(direction_data)
            end_point = self._calculate_end_point(start_point, vector, distance)
            return cv2.arrowedLine(image, start_point, end_point, self.goal_color, 2, tipLength=0.3)
        
        # 방향 객체인 경우 처리
        if isinstance(direction_data, dict):
            dir_type = direction_data.get("type", "simple")
            
            if dir_type == "simple":
                # 단일 방향 화살표 그리기
                direction_value = direction_data.get("value", "front")
                vector = self._get_direction_vector(direction_value)
                end_point = self._calculate_end_point(start_point, vector, distance)
                return cv2.arrowedLine(image, start_point, end_point, self.goal_color, 2, tipLength=0.3)
                
            elif dir_type == "random":
                # 랜덤 방향 다중 화살표 그리기 (점선으로)
                options = direction_data.get("options", [])
                if not options:
                    # 옵션이 없으면 기본 방향
                    vector = self._get_direction_vector("front")
                    end_point = self._calculate_end_point(start_point, vector, distance)
                    return cv2.arrowedLine(image, start_point, end_point, self.goal_color, 2, tipLength=0.3)
                
                # 모든 옵션에 대해 점선 화살표 그리기
                for option in options:
                    vector = self._get_direction_vector(option)
                    end_point = self._calculate_end_point(start_point, vector, distance)
                    image = self._draw_dashed_arrow(image, start_point, end_point, self.goal_color, 2)
                
                return image
        
        # 기본 방향 (앞쪽)
        vector = self._get_direction_vector("front")
        end_point = self._calculate_end_point(start_point, vector, distance)
        return cv2.arrowedLine(image, start_point, end_point, self.goal_color, 2, tipLength=0.3)

    def _draw_goal_marker(self, 
                         image: np.ndarray, 
                         goal_point_coords, 
                         target_object=None):
        """
        목표 지점 마커를 그립니다.
        
        Args:
            image: 시각화할 이미지
            goal_point_coords: 목표 지점 3D 좌표 (numpy array)
            target_object: 타겟 객체 정보 (선택 사항)
            
        Returns:
            마커가 그려진 이미지
        """
        if goal_point_coords is None:
            return image
            
        vis_img = image.copy()
        
        # 3D 좌표를 2D 이미지 좌표로 변환해야 함 (Depth3DMapper에 유사 기능 필요)
        # 임시: 간단하게 타겟 객체 근처에 표시 (정확한 투영 필요)
        goal_x_px, goal_y_px = -1, -1
        if target_object and 'bbox' in target_object:
             tx1, ty1, tx2, ty2 = [int(c) for c in target_object['bbox']]
             # 방향에 따라 위치 조정 (예시)
             # TODO: 정확한 3D -> 2D 투영 필요
             goal_x_px = (tx1 + tx2) // 2 + 50 # 임시 오프셋
             goal_y_px = (ty1 + ty2) // 2
             
        if goal_x_px > 0 and goal_y_px > 0:
            cv2.drawMarker(vis_img, (goal_x_px, goal_y_px), self.goal_color, 
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
            cv2.putText(vis_img, "Goal", (goal_x_px + 15, goal_y_px), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.goal_color, 2)
            
            # 방향 정보 확인
            direction = None
            if isinstance(goal_point_coords, dict):
                direction = goal_point_coords.get("direction")
            
            # 타겟 -> 목표 지점 화살표 (방향 객체 지원)
            if target_object and 'bbox' in target_object:
                 target_center_x = (tx1 + tx2) // 2
                 target_center_y = (ty1 + ty2) // 2
                 start_point = (target_center_x, target_center_y)
                 
                 # 방향을 사용하여 화살표 그리기
                 if direction is not None:
                     vis_img = self._draw_direction(vis_img, direction, start_point, 100)
                 else:
                     # 일반 화살표 그리기 (기존 방식)
                     cv2.arrowedLine(vis_img, start_point, (goal_x_px, goal_y_px),
                                    self.goal_color, 2, tipLength=0.1)
        
        # 목표 방향 표시 (3D 좌표가 있는 경우)
        if target_object and goal_point_coords is not None:
            # 타겟 중심에서 목표 지점까지 화살표 그리기
            target_center = self._get_object_center(target_object)
            if target_center:
                if isinstance(goal_point_coords, dict) and "position" in goal_point_coords:
                    # 새로운, 확장된 goal_point_coords 처리
                    goal_position = goal_point_coords.get("position")
                    goal_2d = self._project_3d_to_2d(goal_position)
                    direction = goal_point_coords.get("direction")
                    
                    if goal_2d is not None:
                        start_point = (int(target_center[0]), int(target_center[1]))
                        # 3D 좌표가 올바른 경우, 실제 위치에 화살표 그리기
                        if isinstance(direction, (str, dict)):
                            # 방향 시각화 사용
                            vis_img = self._draw_direction(vis_img, direction, start_point, 100)
                        else:
                            # 기존 방식
                            cv2.arrowedLine(
                                vis_img, 
                                start_point,
                                (int(goal_2d[0]), int(goal_2d[1])), 
                                self.goal_color, 
                                2, 
                                tipLength=0.3
                            )
                else:
                    # 기존 형식 처리
                    goal_2d = self._project_3d_to_2d(goal_point_coords)
                    if goal_2d is not None:
                        cv2.arrowedLine(
                            vis_img, 
                            (int(target_center[0]), int(target_center[1])), 
                            (int(goal_2d[0]), int(goal_2d[1])), 
                            self.goal_color, 
                            2, 
                            tipLength=0.3
                        )
        
        return vis_img

    def draw_results(self, 
                    image, 
                    detections, 
                    target_object=None, 
                    reference_object=None, 
                    goal_point_coords=None,
                    goal_bounding_box=None, 
                    gesture_results=None):
        """
        결과를 시각화합니다.
        
        Args:
            image: 원본 이미지
            detections: 감지된 객체 목록
            target_object: 타겟 객체
            reference_object: 레퍼런스 객체
            goal_point_coords: 목표 지점 3D 좌표
            goal_bounding_box: 목표 위치 바운딩 박스 [x1, y1, x2, y2]
            gesture_results: 제스처 인식 결과 (선택 사항)
            
        Returns:
            np.ndarray: 시각화된 이미지
        """
        self.logger.debug("결과 시각화 시작...")
        self.logger.debug(f"  - 입력 이미지 타입: {type(image)}")
        self.logger.debug(f"  - 감지된 객체 수: {len(detections)}")
        self.logger.debug(f"  - 타겟 객체: {target_object.get('class_name', 'None') if target_object else 'None'}")
        self.logger.debug(f"  - 레퍼런스 객체: {reference_object.get('class_name', 'None') if reference_object else 'None'}")
        self.logger.debug(f"  - 목표 지점 좌표: {goal_point_coords}")
        self.logger.debug(f"  - 제스처 결과 수: {len(gesture_results) if gesture_results else 0}")

        # --- 이미지 준비 --- 
        if isinstance(image, Image.Image):
            vis_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            vis_img = image.copy()
        else:
            self.logger.error(f"지원되지 않는 이미지 타입: {type(image)}")
            return np.zeros((480, 640, 3), dtype=np.uint8) # 기본 검은 화면 반환
        
        h, w = vis_img.shape[:2]

        # --- 객체 및 목표 지점 그리기 --- 
        for i, detection in enumerate(detections):
            if 'virtual' in detection.get('class_name', ''): continue # 가상 레퍼런스는 그리지 않음
            
            is_target = target_object and detection == target_object
            is_reference = reference_object and detection == reference_object
            
            vis_img = self._draw_object(vis_img, detection, is_target, is_reference)

        # 목표 지점 그리기 (점이 아니라 마커로)
        if goal_point_coords is not None:
            vis_img = self._draw_goal_marker(vis_img, goal_point_coords, target_object)

        # 목표 바운딩 박스 표시 (LLM 직접 생성)
        if goal_bounding_box is not None and len(goal_bounding_box) == 4:
            vis_img = self._draw_goal_bounding_box(vis_img, goal_bounding_box)

        # --- 손 제스처 그리기 --- 
        if gesture_results:
            try:
                self._draw_hand_gestures(vis_img, gesture_results)
                self.logger.debug("손 제스처 시각화 완료.")
            except Exception as e:
                self.logger.exception(f"손 제스처 시각화 중 오류 발생: {e}")

        # --- 깊이 맵 결합 (선택 사항, 필요 시 주석 해제) ---
        # if colored_depth is not None:
        #     colored_depth_resized = cv2.resize(colored_depth, (w, h))
        #     vis_img = np.hstack((vis_img, colored_depth_resized))

        return vis_img 