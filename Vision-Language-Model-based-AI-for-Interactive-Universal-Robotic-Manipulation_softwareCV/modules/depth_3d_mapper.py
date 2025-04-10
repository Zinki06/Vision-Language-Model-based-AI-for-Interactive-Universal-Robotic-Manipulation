#!/usr/bin/env python
"""
단안 카메라 기반 3D 좌표 추정 모듈

이 모듈은 MiDaS 모델을 사용한 깊이 추정과 카메라 파라미터를 이용하여
단일 RGB 이미지로부터 3D 좌표를 추정하는 기능을 제공합니다.
"""

import os
import json
import time
import logging
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

class Depth3DMapper:
    """
    단안 카메라 기반 3D 좌표 추정 클래스
    
    MiDaS 깊이 추정 + 카메라 캘리브레이션 + 객체 크기 기반 스케일링을 통합하여
    단일 이미지에서 3D 좌표를 추정합니다.
    """
    
    def __init__(self, 
                 midas_model_type: str = "DPT_Large",
                 reference_object_size: Optional[float] = None,
                 min_depth_cm: float = 30.0,
                 max_depth_cm: float = 300.0,
                 logger: Optional[logging.Logger] = None):
        """
        초기화
        
        Args:
            midas_model_type: 사용할 MiDaS 모델 유형 ('DPT_Large', 'DPT_Hybrid', 'MiDaS_small')
            reference_object_size: 참조 객체 실제 크기 (cm) (없으면 상대적 깊이만 사용)
            min_depth_cm: 최소 깊이 범위 (cm)
            max_depth_cm: 최대 깊이 범위 (cm)
            logger: 로깅을 위한 로거 객체 (없으면 새로 생성)
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("Depth3DMapper")
        
        # 깊이 범위 설정
        self.min_depth_cm = min_depth_cm
        self.max_depth_cm = max_depth_cm
        
        # 참조 객체 크기 설정
        self.reference_object_size = reference_object_size
        
        # 깊이 스케일 인자 (초기값은 None, 참조 객체를 통해 설정됨)
        self.depth_scale_factor = None
        
        # 카메라 내부 파라미터 (기본값: 초점 거리 fx=fy=1000, 주점 cx=cy=이미지 중앙)
        # 실제 값은 나중에 설정하거나 기본값으로 대체
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 이전 좌표 히스토리 (평활화용)
        self.coordinate_history = {}
        
        # 3D 좌표 평활화 가중치 (0-1, 0: 히스토리 완전 의존, 1: 현재 값만 사용)
        self.smoothing_alpha = 0.7
        
        # 기본 카메라 매트릭스 준비 (나중에 로드됨)
        self._prepare_default_camera_matrix()
        
        # MiDaS 모델 설정은 필요할 때 초기화 (메모리 효율성)
        self.midas_model_type = midas_model_type
        self.midas_model = None
        self.midas_transform = None
        self.device = self._get_device()
        
        self.logger.info(f"Depth3DMapper 초기화 완료. 사용 장치: {self.device}, 모델: {midas_model_type}")
    
    def _get_device(self) -> torch.device:
        """적절한 계산 장치 결정"""
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _prepare_default_camera_matrix(self, image_width: int = 1280, image_height: int = 720) -> None:
        """
        기본 카메라 매트릭스 준비
        
        Args:
            image_width: 예상 이미지 넓이 (기본값: 1280)
            image_height: 예상 이미지 높이 (기본값: 720)
        """
        # 기본 내부 파라미터
        fx = fy = 1000.0  # 가상의 초점 거리
        cx = image_width / 2
        cy = image_height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 왜곡 계수 (왜곡 없음 가정)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        self.logger.info(f"기본 카메라 매트릭스 생성: 크기 {image_width}x{image_height}, 초점 거리 {fx}")
    
    def load_camera_params(self, param_file: str) -> bool:
        """
        카메라 파라미터 파일 로드
        
        Args:
            param_file: 카메라 파라미터가 저장된 JSON 또는 YAML 파일 경로
        
        Returns:
            성공 여부
        """
        if not os.path.exists(param_file):
            self.logger.warning(f"카메라 파라미터 파일을 찾을 수 없음: {param_file}")
            return False
        
        try:
            # 파일 확장자 확인
            _, ext = os.path.splitext(param_file)
            
            if ext.lower() == '.json':
                # JSON 파일 로드
                with open(param_file, 'r') as f:
                    params = json.load(f)
                
                camera_matrix = np.array(params.get('camera_matrix'), dtype=np.float32)
                dist_coeffs = np.array(params.get('dist_coeffs', [0, 0, 0, 0, 0]), dtype=np.float32)
                
            elif ext.lower() in ['.yml', '.yaml']:
                # YAML 파일 로드 (OpenCV FileStorage 사용)
                fs = cv2.FileStorage(param_file, cv2.FILE_STORAGE_READ)
                camera_matrix = fs.getNode('camera_matrix').mat()
                dist_coeffs = fs.getNode('dist_coeffs').mat()
                fs.release()
                
            else:
                self.logger.error(f"지원되지 않는 파일 형식: {ext}")
                return False
            
            # 매트릭스 설정
            if camera_matrix is not None and camera_matrix.shape == (3, 3):
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                self.logger.info(f"카메라 파라미터 로드 완료: {param_file}")
                return True
            else:
                self.logger.error("유효하지 않은 카메라 매트릭스 형식")
                return False
                
        except Exception as e:
            self.logger.error(f"카메라 파라미터 로드 중 오류: {str(e)}")
            return False
    
    def calibrate_camera(self, calibration_images: List[str], chessboard_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        체스보드 이미지를 사용한 카메라 캘리브레이션
        
        Args:
            calibration_images: 체스보드 패턴이 포함된 이미지 파일 경로 목록
            chessboard_size: 체스보드 내부 코너 개수 (가로, 세로)
        
        Returns:
            캘리브레이션 성공 여부
        """
        # 캘리브레이션을 위한 코너 객체 포인트 
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # 객체 포인트와 이미지 포인트 저장 배열
        objpoints = []  # 3D 공간의 점
        imgpoints = []  # 2D 이미지의 점
        
        for img_path in calibration_images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"이미지를 로드할 수 없음: {img_path}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 체스보드 코너 탐지
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                if ret:
                    objpoints.append(objp)
                    
                    # 코너 위치 세분화
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    
                    self.logger.info(f"체스보드 탐지 성공: {img_path}")
                else:
                    self.logger.warning(f"체스보드 코너를 탐지할 수 없음: {img_path}")
            except Exception as e:
                self.logger.error(f"이미지 처리 중 오류 ({img_path}): {str(e)}")
        
        # 충분한 이미지를 확보했는지 확인
        if len(objpoints) < 3:
            self.logger.error(f"캘리브레이션에 필요한 충분한 이미지가 없음. 확보한 이미지: {len(objpoints)}, 필요 개수: ≥3")
            return False
        
        # 카메라 캘리브레이션 수행
        img_shape = gray.shape[::-1]  # (width, height)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.logger.info(f"카메라 캘리브레이션 성공. 재투영 오차: {ret}")
            return True
        else:
            self.logger.error("카메라 캘리브레이션 실패")
            return False
    
    def set_scale_factor(self, reference_object: Dict[str, Any], known_size_cm: float) -> None:
        """
        참조 객체를 이용한 깊이 스케일 인자 설정
        
        Args:
            reference_object: 참조 객체 정보 (바운딩 박스, 깊이 등)
            known_size_cm: 알려진 객체 크기(cm)
        """
        try:
            # 깊이 정보 추출
            if "depth" not in reference_object or not isinstance(reference_object["depth"], dict):
                self.logger.error("참조 객체에 유효한 깊이 정보가 없음")
                return
            
            # 객체 크기 픽셀 단위로 계산
            bbox = reference_object["bbox"]
            object_width_px = bbox[2] - bbox[0]
            object_height_px = bbox[3] - bbox[1]
            
            # 더 작은 값을 기준으로 사용 (보통 객체의 특성 크기)
            object_size_px = min(object_width_px, object_height_px)
            
            # 깊이 값 (0-1 사이 값)
            depth_value = reference_object["depth"].get("center_depth", 
                          reference_object["depth"].get("avg_depth", 0.5))
            
            # 스케일 인자 계산:
            # 실제 거리 = depth_value * scale_factor
            # 이때 scale_factor는 객체 크기와 깊이에 기반하여 결정
            # 가정: 깊이가 작을수록(가까울수록) 객체는 크게 보임
            #       F = (px * Z) / real_size  (핀홀 카메라 모델 기준)
            #       여기서 F는 초점 거리, px는 픽셀 크기, Z는 거리, real_size는 실제 크기
            
            # 카메라 초점 거리 추출
            if self.camera_matrix is not None:
                focal_length = self.camera_matrix[0, 0]  # fx 사용
            else:
                focal_length = 1000.0  # 기본값
            
            # 스케일 인자 계산
            # 실제 거리 = 초점거리 * 실제크기 / 픽셀크기
            estimated_distance = (focal_length * known_size_cm) / object_size_px
            
            # 0-1 깊이 값을 실제 거리로 변환하는 스케일 인자
            self.depth_scale_factor = estimated_distance / depth_value
            
            # 정상적인 값인지 확인
            if self.depth_scale_factor <= 0 or self.depth_scale_factor > 1000:
                self.logger.warning(f"비정상적인 스케일 인자 ({self.depth_scale_factor}). 기본값으로 재설정")
                # 기본 스케일 인자 설정 (실험적으로 조정)
                self.depth_scale_factor = self.max_depth_cm - self.min_depth_cm
            
            self.logger.info(f"깊이 스케일 인자 설정됨: {self.depth_scale_factor:.2f}, "
                           f"참조 객체: {reference_object.get('class_name', 'unknown')}, "
                           f"크기: {known_size_cm}cm, 깊이: {depth_value:.3f}")
            
        except Exception as e:
            self.logger.error(f"스케일 인자 설정 중 오류: {str(e)}")
            # 안전한 기본값 설정
            self.depth_scale_factor = self.max_depth_cm - self.min_depth_cm
    
    def _initialize_midas(self) -> None:
        """MiDaS 모델 초기화"""
        if self.midas_model is not None:
            return  # 이미 초기화됨
        
        try:
            self.logger.info(f"MiDaS 모델 로드 중: {self.midas_model_type}")
            
            # MiDaS 모델 로드
            self.midas_model = torch.hub.load("intel-isl/MiDaS", self.midas_model_type)
            self.midas_model.to(self.device)
            self.midas_model.eval()
            
            # 전처리 파이프라인 설정
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            
            if self.midas_model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.midas_transform = Compose([
                    Resize((384, 384), antialias=True),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:  # MiDaS_small
                self.midas_transform = Compose([
                    Resize((256, 256), antialias=True),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            
            self.logger.info(f"MiDaS 모델 로드 완료: {self.midas_model_type}")
            
        except Exception as e:
            self.logger.error(f"MiDaS 모델 로드 실패: {str(e)}")
            raise
    
    def estimate_depth(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 깊이 맵 추정 및 시각화

        Args:
            image: 입력 이미지 (numpy 배열 또는 PIL 이미지)
            
        Returns:
            깊이 맵 및 시각화된 깊이 맵의 튜플
        """
        # 깊이 맵 추정
        depth_map = self.estimate_depth_map(image)
        
        # 시각화를 위해 컬러맵 적용
        depth_map_vis = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_INFERNO
        )
        
        return depth_map, depth_map_vis
    
    def estimate_depth_map(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        이미지에서 깊이 맵 추정
        
        Args:
            image: 입력 이미지 (numpy 배열 또는 PIL 이미지)
            
        Returns:
            깊이 맵 (0-1 사이 값의 2D numpy 배열)
        """
        # 필요시 MiDaS 모델 초기화
        if self.midas_model is None:
            self._initialize_midas()
        
        # 이미지 전처리
        if isinstance(image, np.ndarray):
            # OpenCV 이미지 (BGR)를 RGB로 변환
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_rgb)
            else:
                img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        original_size = img.size  # (width, height)
        
        # 이미지 변환 및 추론
        input_tensor = self.midas_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas_model(input_tensor)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            
            depth_map = prediction.squeeze().cpu().numpy()
        
        # 후처리: 최소-최대 정규화
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # 원본 이미지 크기로 리사이징
        depth_map = cv2.resize(depth_map, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
        
        return depth_map
    
    def get_object_depth(self, bbox: List[float], depth_map: np.ndarray) -> Dict[str, float]:
        """
        바운딩 박스 영역의 깊이 통계 계산
        
        Args:
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            depth_map: 깊이 맵
            
        Returns:
            깊이 통계 정보를 포함한 딕셔너리
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 바운딩 박스가 이미지 범위를 벗어나는지 확인
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            self.logger.warning(f"유효하지 않은 바운딩 박스: [{x1}, {y1}, {x2}, {y2}]")
            return {
                "center_depth": 0.5,
                "avg_depth": 0.5,
                "valid_ratio": 0.0
            }

        # bbox 영역 추출 및 유효한 깊이 값 필터링
        depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
        valid_depths = depth_roi[depth_roi > 0] # 0 이하 값 제외

        if valid_depths.size == 0:
            self.logger.warning(f"bbox 내 유효한 깊이 값 없음. bbox: {bbox}")
            return {
                "center_depth": 0.5,
                "avg_depth": 0.5,
                "valid_ratio": 0.0
            }

        # --- 개선: 이상치 제거 후 평균 계산 ---
        # 백분위수 기반 이상치 제거 (예: 10% ~ 90% 범위 사용)
        q10 = np.percentile(valid_depths, 10)
        q90 = np.percentile(valid_depths, 90)
        filtered_depths = valid_depths[(valid_depths >= q10) & (valid_depths <= q90)]

        if filtered_depths.size == 0:
            # 필터링 후 값이 없으면 전체 평균 사용 (fallback)
            avg_depth = np.mean(valid_depths)
            self.logger.warning(f"깊이 이상치 제거 후 유효 값 없음. 전체 평균 사용. bbox: {bbox}")
        else:
            avg_depth = np.mean(filtered_depths)
            
        # --- 중장기 개선 방안 (주석) ---
        # Kalman Filter 기반 depth smoothing:
        #   여러 프레임에 걸쳐 깊이 값을 추적하고 Kalman Filter를 적용하여 노이즈를 줄이고 안정화.
        #   객체 ID별로 필터 상태 유지 필요.
        # Multi-frame tracking:
        #   Optical flow나 객체 추적 알고리즘(예: SORT, DeepSORT)과 결합하여
        #   여러 프레임에서 동일 객체의 깊이 정보를 통합하고 평균화하여 정확도 향상.
        # Object mask 기반 평균 추출:
        #   객체 분할(segmentation) 모델을 사용하여 객체의 정확한 마스크를 얻고,
        #   마스크 영역 내의 픽셀들만 사용하여 평균 깊이 계산. YOLO만으로는 한계.
        # --------------------------------

        # 중심 깊이 (이전 방식 유지 또는 개선된 평균값 사용)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        center_depth = depth_map[int(center_y), int(center_x)]
        
        # 유효 픽셀 비율
        valid_ratio = valid_depths.size / depth_roi.size

        return {
            "center_depth": float(center_depth),
            "avg_depth": float(avg_depth), # 개선된 평균 깊이
            "valid_ratio": float(valid_ratio)
        }
    
    def pixel_to_3d(self, pixel_x: float, pixel_y: float, depth_value: float, 
                   image_width: int, image_height: int) -> Dict[str, float]:
        """
        픽셀 좌표 및 깊이 값을 3D 좌표로 변환
        
        Args:
            pixel_x, pixel_y: 이미지 상의 픽셀 좌표
            depth_value: 해당 픽셀의 깊이 값 (0-1 사이 값)
            image_width, image_height: 이미지 크기
            
        Returns:
            x, y, z 3D 좌표를 포함한 딕셔너리 (cm 단위)
        """
        # 카메라 매트릭스가 없는 경우 기본값으로 설정
        if self.camera_matrix is None:
            self._prepare_default_camera_matrix(image_width, image_height)
        
        # 카메라 내부 파라미터 추출
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 깊이 값을 실제 거리(cm)로 변환
        z_cm = self._depth_to_distance(depth_value)
        
        # 핀홀 카메라 모델에 따른 3D 좌표 계산
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x_cm = (pixel_x - cx) * z_cm / fx
        y_cm = (pixel_y - cy) * z_cm / fy
        
        # 주의: y 좌표 반전 (이미지는 위에서 아래로, 3D 공간은 아래에서 위로)
        y_cm = -y_cm
        
        # 신뢰도 점수 (거리에 반비례, 표준편차에 반비례)
        # 가까운 물체일수록, 깊이 표준편차가 작을수록 신뢰도 높음
        confidence = max(0.1, min(1.0, 0.8 * (1.0 - depth_value)))
        
        return {
            "x": float(x_cm),
            "y": float(y_cm),
            "z": float(z_cm),
            "confidence": float(confidence)
        }
    
    def _depth_to_distance(self, depth_value: float) -> float:
        """
        상대적 깊이 값(0-1)을 실제 거리(cm)로 변환
        
        Args:
            depth_value: 0-1 사이의 깊이 값
            
        Returns:
            실제 거리 (cm)
        """
        # 스케일 인자가 설정된 경우 사용
        if self.depth_scale_factor is not None:
            return depth_value * self.depth_scale_factor
        
        # 기본 변환: min_depth_cm에서 max_depth_cm 사이로 선형 매핑
        distance_cm = self.min_depth_cm + depth_value * (self.max_depth_cm - self.min_depth_cm)
        return distance_cm
    
    def _smooth_position(self, object_id: str, position_3d: Dict[str, float]) -> Dict[str, float]:
        """
        이전 추정값들을 이용한 현재 좌표 평활화
        
        Args:
            object_id: 객체 식별자
            position_3d: 현재 프레임의 3D 좌표
            
        Returns:
            평활화된 좌표
        """
        # 좌표 히스토리가 없으면 현재 좌표 반환
        if object_id not in self.coordinate_history:
            self.coordinate_history[object_id] = []
        
        history = self.coordinate_history[object_id]
        
        # 히스토리가 비어있으면 현재 좌표 사용
        if not history:
            smoothed = position_3d.copy()
        else:
            # 이전 좌표 가져오기
            prev_position = history[-1]
            
            # 신뢰도 기반 가중치 적용 (신뢰도가 높을수록 현재 값에 더 높은 가중치)
            alpha = self.smoothing_alpha * position_3d.get("confidence", 0.8)
            
            # 좌표 평활화
            smoothed = {
                "x": alpha * position_3d["x"] + (1 - alpha) * prev_position["x"],
                "y": alpha * position_3d["y"] + (1 - alpha) * prev_position["y"],
                "z": alpha * position_3d["z"] + (1 - alpha) * prev_position["z"],
                "confidence": position_3d.get("confidence", 0.8)
            }
        
        # 히스토리 업데이트 (최대 10개 항목만 유지)
        history.append(smoothed)
        if len(history) > 10:
            history.pop(0)
        
        return smoothed
    
    def get_object_3d_position(self, detection: Dict[str, Any], depth_map: np.ndarray,
                              image_width: int, image_height: int) -> Dict[str, float]:
        """
        객체의 3D 위치 정보 계산
        
        Args:
            detection: 객체 감지 정보 (바운딩 박스 등)
            depth_map: 깊이 맵
            image_width, image_height: 이미지 크기
            
        Returns:
            객체의 3D 위치 정보
        """
        bbox = detection["bbox"]
        obj_id = detection.get("id", str(time.time())) # 객체 ID (추적/평활화용)
        
        # 1. 객체의 평균 깊이 정보 얻기 (개선된 방식 사용)
        depth_info = self.get_object_depth(bbox, depth_map)
        # avg_depth 사용 (0~1 사이 값)
        object_avg_depth_normalized = depth_info["avg_depth"] 
        
        if object_avg_depth_normalized <= 0:
            self.logger.warning(f"객체 ID {obj_id}의 평균 깊이 값이 유효하지 않음: {object_avg_depth_normalized}")
            return {}
        
        # 2. 객체 중심 픽셀 좌표 계산
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # 3. 중심 픽셀과 평균 깊이를 사용하여 3D 좌표 계산
        position_3d = self.pixel_to_3d(
            center_x, center_y, object_avg_depth_normalized, # 평균 깊이 사용
            image_width, image_height
        )
        
        if not position_3d:
            self.logger.warning(f"객체 ID {obj_id}의 3D 좌표 계산 실패")
            return {}
        
        # 4. (선택 사항) 좌표 평활화 적용
        smoothed_position = self._smooth_position(obj_id, position_3d)
        
        return smoothed_position
    
    def estimate_goal_point_3d(self, goal_point: Dict[str, float], depth_map: np.ndarray) -> Dict[str, float]:
        """
        목표 지점의 3D 좌표 추정
        
        Args:
            goal_point: 목표 지점 정보 (x, y 좌표 포함)
            depth_map: 깊이 맵
            
        Returns:
            목표 지점의 3D 좌표
        """
        # 목표 지점 좌표
        x, y = goal_point["x"], goal_point["y"]
        
        # 이미지 크기
        h, w = depth_map.shape
        
        # 좌표가 이미지 범위 내에 있는지 확인
        if x < 0 or x >= w or y < 0 or y >= h:
            self.logger.warning(f"목표 지점이 이미지 범위를 벗어남: ({x}, {y}), 이미지 크기: {w}x{h}")
            # 이미지 경계 내로 좌표 조정
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
        
        # 목표 지점 주변 영역 깊이 값 추출 (더 안정적인 추정을 위해)
        r = 5  # 반경
        x1, y1 = max(0, int(x-r)), max(0, int(y-r))
        x2, y2 = min(w-1, int(x+r)), min(h-1, int(y+r))
        
        if x1 >= x2 or y1 >= y2:
            # 영역을 추출할 수 없는 경우 단일 픽셀 사용
            x, y = int(x), int(y)
            depth_value = depth_map[y, x]
        else:
            # 영역 내 깊이 값의 중위수 사용 (이상치에 덜 민감)
            goal_area = depth_map[y1:y2, x1:x2]
            depth_value = float(np.median(goal_area))
        
        # 픽셀 좌표 및 깊이를 3D 좌표로 변환
        position_3d = self.pixel_to_3d(x, y, depth_value, w, h)
        
        # 목표 지점은 평활화하지 않음 (매번 새로운 목표 지점이므로)
        
        return position_3d
    
    def estimate_3d_coordinates(self, image: Union[np.ndarray, Image.Image], 
                              detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        이미지 내 감지된 객체들의 3D 좌표 추정
        
        Args:
            image: 입력 RGB 이미지
            detections: YOLO 감지 결과 목록
            
        Returns:
            3D 좌표가 추가된 감지 결과 목록
        """
        # 이미지 크기 확인
        if isinstance(image, np.ndarray):
            image_height, image_width = image.shape[:2]
        elif isinstance(image, Image.Image):
            image_width, image_height = image.size
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 깊이 맵 추정
        depth_map = self.estimate_depth_map(image)
        
        # 각 객체에 3D 좌표 추가
        for detection in detections:
            # 객체의 3D 위치 계산
            position_3d = self.get_object_3d_position(detection, depth_map, image_width, image_height)
            detection["position_3d"] = position_3d
            
            # 객체 ID 생성 (클래스와 바운딩 박스 위치 기반)
            bbox = detection["bbox"]
            object_id = f"{detection.get('class_name', 'object')}_{bbox[0]:.0f}_{bbox[1]:.0f}"
            
            # 좌표 평활화
            smoothed_position = self._smooth_position(object_id, position_3d)
            detection["position_3d_smoothed"] = smoothed_position
            
            # 추가 정보: 실제 거리(cm) (이미 cm 단위로 계산됨)
            distance_cm = smoothed_position["z"]
            detection["estimated_distance_cm"] = distance_cm
        
        return detections 