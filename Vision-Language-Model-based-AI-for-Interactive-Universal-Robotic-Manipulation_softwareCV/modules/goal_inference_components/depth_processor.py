"""
깊이 처리 및 3D 좌표 계산 모듈

이 모듈은 깊이 맵 처리 및 3D 좌표 변환을 담당하는 DepthProcessor 클래스를 포함합니다.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from ..depth_3d_mapper import Depth3DMapper

class DepthProcessor:
    """
    깊이 처리 및 3D 좌표 계산 클래스
    
    이 클래스는 다음과 같은 깊이 처리 관련 기능을 담당합니다:
    - 깊이 맵 추정
    - 3D 좌표 계산
    - 객체의 깊이 정보 추출
    - 3D <-> 2D 좌표 변환
    """
    
    def __init__(self, 
                min_depth_cm: float = 30.0,
                max_depth_cm: float = 300.0,
                logger: Optional[logging.Logger] = None):
        """
        DepthProcessor 초기화
        
        Args:
            min_depth_cm: 최소 깊이 값 (cm)
            max_depth_cm: 최대 깊이 값 (cm)
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 깊이 범위 설정
        self.min_depth_cm = min_depth_cm
        self.max_depth_cm = max_depth_cm
        
        # 깊이 추정기 초기화
        self.logger.info("Depth3DMapper 초기화 중...")
        self.depth_mapper = Depth3DMapper()
        self.logger.info("Depth3DMapper 초기화 완료")
    
    def estimate_depth(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지의 깊이 맵 추정
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (깊이 맵, 컬러 깊이 맵)
        """
        try:
            self.logger.debug("깊이 맵 추정 중...")
            depth_map, colored_depth = self.depth_mapper.estimate_depth(image)
            self.logger.debug("깊이 맵 추정 완료")
            return depth_map, colored_depth
        except Exception as e:
            self.logger.error(f"깊이 맵 추정 중 오류 발생: {e}")
            # 실패 시 빈 깊이 맵 반환
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.uint8)
            else:
                return np.zeros((100, 100), dtype=np.float32), np.zeros((100, 100, 3), dtype=np.uint8)
    
    def prepare_camera_matrix(self, image_width: int, image_height: int) -> None:
        """
        이미지 크기에 맞게 카메라 매트릭스 업데이트
        
        Args:
            image_width: 이미지 너비
            image_height: 이미지 높이
        """
        self.depth_mapper._prepare_default_camera_matrix(image_width, image_height)
    
    def get_object_depth(self, bbox: List[float], depth_map: np.ndarray) -> Dict[str, float]:
        """
        객체 바운딩 박스 영역의 깊이 정보 계산
        
        Args:
            bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
            depth_map: 깊이 맵
            
        Returns:
            Dict[str, float]: 깊이 정보 (평균, 최소, 최대 깊이값)
        """
        try:
            return self.depth_mapper.get_object_depth({"bbox": bbox}, depth_map)
        except Exception as e:
            self.logger.error(f"객체 깊이 계산 중 오류: {e}")
            return {"depth_avg": 0.5, "depth_min": 0.3, "depth_max": 0.7}
    
    def get_object_3d_position(self, obj: Dict[str, Any], depth_map: np.ndarray,
                              img_width: int, img_height: int) -> Dict[str, Any]:
        """
        객체의 3D 위치 계산
        
        Args:
            obj: 객체 정보
            depth_map: 깊이 맵
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            Dict[str, Any]: 3D 위치 정보
        """
        try:
            return self.depth_mapper.get_object_3d_position(obj, depth_map, img_width, img_height)
        except Exception as e:
            self.logger.error(f"객체 3D 위치 계산 중 오류: {e}")
            return {
                "center": [0, 0, 100],  # 기본 중심 좌표 (cm)
                "dimensions": [30, 30, 30],  # 기본 크기 (cm)
                "x_cm": 0,
                "y_cm": 0,
                "z_cm": 100
            }
    
    def project_3d_to_2d(self, position_3d: List[float], 
                         img_width: int, img_height: int) -> Dict[str, Any]:
        """
        3D 좌표를 2D 화면 좌표로 투영
        
        Args:
            position_3d: [x, y, z] 형태의 3D 좌표 (cm)
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            Dict[str, Any]: 2D 좌표 정보 (x, y, depth, is_valid)
        """
        try:
            # 기본 카메라 매트릭스 업데이트
            self.prepare_camera_matrix(img_width, img_height)
            
            # 3D -> 2D 투영 (해당 메서드가 있다고 가정)
            if hasattr(self.depth_mapper, 'project_3d_to_2d'):
                return self.depth_mapper.project_3d_to_2d(position_3d, img_width, img_height)
            
            # 기본 간단한 투영 구현 (실제로는 더 복잡한 카메라 모델 사용)
            # 이 부분은 Depth3DMapper 클래스 구현에 따라 수정 필요
            x_3d, y_3d, z_3d = position_3d
            
            # 깊이가 0이면 투영 불가
            if z_3d < 1.0:
                return {
                    "x": img_width / 2,
                    "y": img_height / 2,
                    "depth": 100.0,
                    "is_valid": False
                }
            
            # 간단한 원근 투영 (예시)
            focal_length = img_width  # 간단히 이미지 너비를 초점 거리로 가정
            x_2d = (x_3d * focal_length / z_3d) + (img_width / 2)
            y_2d = (y_3d * focal_length / z_3d) + (img_height / 2)
            
            return {
                "x": float(x_2d),
                "y": float(y_2d),
                "depth": float(z_3d),
                "is_valid": True
            }
        except Exception as e:
            self.logger.error(f"3D->2D 투영 중 오류: {e}")
            return {
                "x": img_width / 2,
                "y": img_height / 2,
                "depth": 100.0,
                "is_valid": False
            }
    
    def estimate_base_y(self, obj: Dict[str, Any]) -> float:
        """
        객체의 바닥면 Y좌표 추정
        
        Args:
            obj: 객체 정보 (바운딩 박스와 3D 좌표 포함)
            
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