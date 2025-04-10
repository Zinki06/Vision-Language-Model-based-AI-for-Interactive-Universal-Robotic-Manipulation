"""
위치 히스토리 기반 객체 인식 개선 모듈

이 모듈은 객체가 간헐적으로 인식되는 문제를 해결하기 위한
확률 기반 객체 감지 보정 메커니즘을 제공합니다.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# 로깅 설정
logger = logging.getLogger("LLM2PF6.history_detector")
logger.setLevel(logging.INFO)

class HistoryBasedDetector:
    """위치 히스토리 기반 객체 인식 개선 클래스"""
    
    def __init__(self, history_size=10, detection_threshold=0.6, grid_size=(10, 10)):
        """
        Args:
            history_size: 기록할 프레임 수
            detection_threshold: 객체 존재 판단 임계값 (0~1)
            grid_size: 화면 분할 그리드 크기
        """
        self.history_size = history_size
        self.detection_threshold = detection_threshold
        self.grid_size = grid_size
        self.detection_history = []  # 프레임별 감지 이력
        self.probability_map = None  # 위치별 객체 존재 확률
        self.cell_width = 0
        self.cell_height = 0
        
        logger.info(f"히스토리 기반 감지기 초기화: 히스토리 크기={history_size}, "
                   f"임계값={detection_threshold}, 그리드 크기={grid_size}")
    
    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]):
        """새 프레임의 감지 결과로 히스토리 업데이트
        
        Args:
            detections: 현재 프레임의 감지 결과
            frame_shape: 프레임 크기 (h, w, c)
        """
        # 감지 결과가 None이면 빈 리스트로 초기화
        if detections is None:
            detections = []
            
        # 그리드 맵 초기화 (처음 호출 시)
        if self.probability_map is None:
            h, w = frame_shape[:2]
            cells_h, cells_w = self.grid_size
            self.cell_width = w // cells_w
            self.cell_height = h // cells_h
            self.probability_map = np.zeros(self.grid_size)
            logger.debug(f"그리드 초기화: 프레임 크기={frame_shape[:2]}, "
                       f"셀 크기=({self.cell_width}, {self.cell_height})")
        
        # 현재 프레임의 감지 위치 맵 생성
        current_map = np.zeros(self.grid_size)
        
        # 각 감지된 객체의 위치를 그리드에 매핑
        for det in detections:
            if "bbox" in det:
                try:
                    x1, y1, x2, y2 = det["bbox"]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 중심점의 그리드 셀 인덱스 계산
                    cell_x = min(int(center_x // self.cell_width), self.grid_size[1]-1)
                    cell_y = min(int(center_y // self.cell_height), self.grid_size[0]-1)
                    
                    # 범위 확인
                    if 0 <= cell_x < self.grid_size[1] and 0 <= cell_y < self.grid_size[0]:
                        # 해당 셀 활성화
                        current_map[cell_y, cell_x] = 1.0
                except Exception as e:
                    logger.error(f"객체 위치 매핑 중 오류: {str(e)}")
                    continue
        
        # 히스토리에 현재 맵 추가
        self.detection_history.append(current_map)
        
        # 히스토리 크기 제한
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # 확률 맵 업데이트 (최근 기록에 더 높은 가중치 부여)
        self.probability_map = np.zeros(self.grid_size)
        for i, hist_map in enumerate(self.detection_history):
            # 최근 기록일수록 높은 가중치
            weight = (i + 1) / len(self.detection_history)
            self.probability_map += hist_map * weight
        
        # 정규화 (0~1 범위로)
        if len(self.detection_history) > 0:
            self.probability_map /= len(self.detection_history)
            
        logger.debug(f"히스토리 업데이트 완료: 히스토리 크기={len(self.detection_history)}, "
                   f"높은 확률 셀={np.sum(self.probability_map >= self.detection_threshold)}")
    
    def enhance_detections(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]):
        """확률 맵을 기반으로 감지 결과 개선
        
        Args:
            detections: 현재 프레임의 감지 결과
            frame_shape: 프레임 크기
            
        Returns:
            개선된 감지 결과
        """
        # 감지 결과가 None이면 빈 리스트로 초기화
        if detections is None:
            detections = []
            
        # 히스토리가 충분하지 않으면 원본 반환
        if len(self.detection_history) < 3:
            logger.debug("히스토리가 부족하여 원본 감지 결과 반환")
            return detections
        
        # 감지 결과 복사
        enhanced_detections = list(detections)  # 깊은 복사 대신 새 리스트 생성
        
        try:
            # 확률 맵에서 임계값 이상인 위치 찾기
            high_prob_cells = np.where(self.probability_map >= self.detection_threshold)
            
            # 확률이 높은 셀에 현재 프레임에서 감지되지 않은 객체가 있는지 확인
            for y, x in zip(high_prob_cells[0], high_prob_cells[1]):
                # 셀 중심 좌표
                center_x = (x + 0.5) * self.cell_width
                center_y = (y + 0.5) * self.cell_height
                
                # 인접한 감지 결과가 있는지 확인
                has_nearby_detection = False
                for det in detections:
                    if "bbox" in det:
                        x1, y1, x2, y2 = det["bbox"]
                        det_center_x = (x1 + x2) / 2
                        det_center_y = (y1 + y2) / 2
                        
                        # 거리 계산
                        distance = np.sqrt((center_x - det_center_x)**2 + (center_y - det_center_y)**2)
                        
                        # 가까운 감지 결과가 있으면 스킵
                        if distance < (self.cell_width + self.cell_height) / 2:
                            has_nearby_detection = True
                            break
                
                # 가까운 감지 결과가 없고 확률이 높으면 새 가상 객체 추가
                if not has_nearby_detection and len(self.detection_history) >= self.history_size / 2:
                    # 이 위치에서 과거에 감지된 객체 정보 수집
                    hist_objects = []
                    
                    # 현재 감지된 모든 객체를 각 프레임에서 검사
                    for hist_idx, hist in enumerate(self.detection_history):
                        # 이 위치가 활성화된 히스토리 프레임 검사
                        if hist[y, x] > 0:
                            # 이전 프레임들의 감지 결과에서 이 위치에 있었던 객체 찾기
                            hist_frame_objects = []
                            for hist_det in detections:
                                if "bbox" in hist_det and "class_name" in hist_det:
                                    try:
                                        x1, y1, x2, y2 = hist_det["bbox"]
                                        hist_det_center_x = (x1 + x2) / 2
                                        hist_det_center_y = (y1 + y2) / 2
                                        
                                        hist_cell_x = min(int(hist_det_center_x // self.cell_width), self.grid_size[1]-1)
                                        hist_cell_y = min(int(hist_det_center_y // self.cell_height), self.grid_size[0]-1)
                                        
                                        # 같은 셀에 있는 객체 찾기
                                        if hist_cell_x == x and hist_cell_y == y:
                                            hist_frame_objects.append({
                                                "class_name": hist_det["class_name"],
                                                "confidence": hist_det.get("confidence", 0.0),
                                                "bbox": hist_det["bbox"].copy() if hasattr(hist_det["bbox"], "copy") else hist_det["bbox"]
                                            })
                                    except Exception as e:
                                        logger.error(f"히스토리 객체 정보 추출 중 오류: {str(e)}")
                                        continue
                            
                            # 이 프레임에서 찾은 객체들 추가
                            hist_objects.extend(hist_frame_objects)
                    
                    # 과거 객체 중 가장 자주 감지된 클래스와 평균 크기 찾기
                    if hist_objects:
                        # 가장 많이 감지된 클래스 찾기
                        class_counts = {}
                        for obj in hist_objects:
                            cls = obj["class_name"]
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        # 클래스 수가 0이면 건너뛰기
                        if not class_counts:
                            continue
                            
                        most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
                        
                        # 해당 클래스의 평균 바운딩 박스 크기 계산
                        bbox_sizes = []
                        for obj in hist_objects:
                            if obj["class_name"] == most_common_class:
                                try:
                                    x1, y1, x2, y2 = obj["bbox"]
                                    width, height = x2 - x1, y2 - y1
                                    if width > 0 and height > 0:  # 유효한 크기만 포함
                                        bbox_sizes.append((width, height))
                                except Exception as e:
                                    logger.error(f"바운딩 박스 크기 계산 중 오류: {str(e)}")
                                    continue
                        
                        # 크기 계산이 불가능하면 기본값 사용
                        avg_width = sum(w for w, h in bbox_sizes) / len(bbox_sizes) if bbox_sizes else 100
                        avg_height = sum(h for w, h in bbox_sizes) / len(bbox_sizes) if bbox_sizes else 100
                        
                        # 프레임 경계 확인
                        h, w = frame_shape[:2]
                        
                        # 경계 내로 제한
                        x1 = max(0, center_x - avg_width/2)
                        y1 = max(0, center_y - avg_height/2)
                        x2 = min(w-1, center_x + avg_width/2)
                        y2 = min(h-1, center_y + avg_height/2)
                        
                        # 가상 객체 생성
                        virtual_obj = {
                            "class_name": most_common_class,
                            "confidence": float(self.probability_map[y, x]),  # 확률을 신뢰도로 사용
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "width": float(x2 - x1),
                            "height": float(y2 - y1),
                            "center_x": float(center_x),
                            "center_y": float(center_y),
                            "is_virtual": True  # 가상 객체 표시
                        }
                        
                        enhanced_detections.append(virtual_obj)
                        logger.debug(f"가상 객체 추가: 클래스={most_common_class}, 위치=({center_x:.1f}, {center_y:.1f}), 확률={self.probability_map[y, x]:.2f}")
            
            # 가상 객체가 추가되었는지 로그
            virtual_count = sum(1 for det in enhanced_detections if det.get("is_virtual", False))
            if virtual_count > 0:
                logger.info(f"가상 객체 {virtual_count}개 추가됨 (총 {len(enhanced_detections)}개 객체)")
        
        except Exception as e:
            logger.error(f"감지 결과 개선 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 반환
            return detections
        
        return enhanced_detections
    
    def get_probability_map(self):
        """현재 확률 맵 반환"""
        return self.probability_map.copy() if self.probability_map is not None else None
    
    def get_stats(self):
        """상태 정보 반환"""
        if self.probability_map is None:
            return {
                "initialized": False,
                "history_size": 0,
                "high_prob_cells": 0
            }
            
        high_prob_cells = np.sum(self.probability_map >= self.detection_threshold)
        return {
            "initialized": True,
            "history_size": len(self.detection_history),
            "high_prob_cells": int(high_prob_cells),
            "max_probability": float(np.max(self.probability_map)) if self.probability_map.size > 0 else 0.0,
            "avg_probability": float(np.mean(self.probability_map)) if self.probability_map.size > 0 else 0.0
        } 