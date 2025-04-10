"""
목표지점 감지 모듈

특정 시각적 특성(색상, 형태 등)을 기반으로 목표지점을 감지하는 모듈입니다.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

class GoalDetector:
    """
    색상 및 형태 기반으로 목표지점을 감지하는 클래스
    """
    
    def __init__(
        self,
        color_ranges: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        shape_types: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        목표지점 감지기 초기화
        
        Args:
            color_ranges: 감지할 색상 범위 (색상명: (하한값, 상한값))
            shape_types: 감지할 형태 목록 ('circle', 'rectangle', 'triangle')
            logger: 로깅을 위한 로거 객체
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 기본 색상 범위 설정 (HSV 형식)
        self.color_ranges = color_ranges or {
            'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
            'green': (np.array([40, 100, 100]), np.array([80, 255, 255])),
            'blue': (np.array([100, 100, 100]), np.array([140, 255, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
            'purple': (np.array([140, 100, 100]), np.array([170, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
        }
        
        # 추가 빨간색 범위 (HSV 색상 공간에서 빨간색은 0도와 180도 부근에 위치)
        self.color_ranges['red2'] = (np.array([170, 100, 100]), np.array([180, 255, 255]))
        
        # 감지할 형태 설정
        self.shape_types = shape_types or ['circle', 'rectangle', 'triangle']
        
        # 신뢰도 임계값
        self.min_area = 500  # 최소 영역 크기
        self.min_confidence = 0.7  # 최소 신뢰도
        
        self.logger.info(f"GoalDetector 초기화 완료: {len(self.color_ranges)} 색상, {len(self.shape_types)} 형태")
    
    def detect_goal_points(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 목표지점 감지
        
        Args:
            image: 입력 이미지 (BGR 형식의 numpy 배열)
            
        Returns:
            감지된 목표지점 목록 (바운딩 박스, 색상, 형태, 신뢰도 포함)
        """
        result = []
        
        # 1. 색상 기반 감지
        color_detections = self._detect_by_color(image)
        result.extend(color_detections)
        
        # 2. 형태 기반 감지
        shape_detections = self._detect_by_shape(image)
        result.extend(shape_detections)
        
        # 3. 중복 제거
        result = self._remove_duplicates(result)
        
        self.logger.info(f"감지된 목표지점: {len(result)}개")
        return result
    
    def _detect_by_color(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        색상 기반으로 목표지점 감지
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            색상 기반으로 감지된 목표지점 목록
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        detections = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # 색상 마스크 생성
            mask = cv2.inRange(hsv, lower, upper)
            
            # 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 신뢰도 계산 (면적 기반)
                    confidence = min(area / 10000, 0.99)  # 적당한 크기에 맞추어 조정
                    
                    if confidence >= self.min_confidence:
                        # ID 생성
                        goal_id = f"goal_{color_name}_{len(detections)}"
                        
                        # 결과 추가
                        detections.append({
                            "id": goal_id,
                            "bbox": [x, y, x + w, y + h],
                            "class_name": "goal_point",
                            "color": color_name,
                            "shape": "unknown",
                            "confidence": float(confidence),
                            "detection_type": "color"
                        })
        
        return detections
    
    def _detect_by_shape(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        형태 기반으로 목표지점 감지
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            형태 기반으로 감지된 목표지점 목록
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 엣지 감지
        edges = cv2.Canny(blurred, 50, 150)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # 윤곽선 근사화
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 형태 결정
                shape = "unknown"
                if len(approx) == 3:
                    shape = "triangle"
                elif len(approx) == 4:
                    # 정사각형/직사각형 구분
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    shape = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
                elif len(approx) > 8:
                    shape = "circle"
                
                if shape in self.shape_types:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 신뢰도 계산
                    confidence = min(area / 8000, 0.99)
                    
                    if confidence >= self.min_confidence:
                        # 색상 결정
                        mask = np.zeros(gray.shape, np.uint8)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        mean_color = cv2.mean(image, mask=mask)[:3]  # BGR
                        color_name = self._get_color_name(mean_color)
                        
                        # ID 생성
                        goal_id = f"goal_{shape}_{len(detections)}"
                        
                        # 결과 추가
                        detections.append({
                            "id": goal_id,
                            "bbox": [x, y, x + w, y + h],
                            "class_name": "goal_point",
                            "color": color_name,
                            "shape": shape,
                            "confidence": float(confidence),
                            "detection_type": "shape"
                        })
        
        return detections
    
    def _get_color_name(self, bgr_color: Tuple[float, float, float]) -> str:
        """
        BGR 색상에 가장 가까운 색상명 반환
        
        Args:
            bgr_color: BGR 형식의 색상값
            
        Returns:
            가장 가까운 색상명
        """
        # BGR에서 HSV로 변환
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # 각 색상 범위와 비교
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name == 'red2':  # 특수 케이스 처리
                continue
                
            if np.all(lower <= hsv_color) and np.all(hsv_color <= upper):
                return color_name
            
            # 빨간색 특수 처리 (HSV 색상 공간에서 경계를 넘어감)
            if color_name == 'red':
                lower2, upper2 = self.color_ranges['red2']
                if np.all(lower2 <= hsv_color) and np.all(hsv_color <= upper2):
                    return 'red'
        
        # 기본값
        return "unknown"
    
    def _remove_duplicates(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        중복된 감지 결과 제거
        
        Args:
            detections: 모든 감지 결과 목록
            
        Returns:
            중복이 제거된 감지 결과 목록
        """
        if not detections:
            return []
            
        # IoU(Intersection over Union) 기반 중복 제거
        final_detections = []
        sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        while sorted_detections:
            # 가장 높은 신뢰도의 감지 선택
            best_detection = sorted_detections.pop(0)
            final_detections.append(best_detection)
            
            # 나머지 감지 결과와 IoU 계산하여 중복 제거
            bbox1 = best_detection["bbox"]
            remaining_detections = []
            
            for detection in sorted_detections:
                bbox2 = detection["bbox"]
                
                # IoU 계산
                iou = self._calculate_iou(bbox1, bbox2)
                
                # IoU가 임계값보다 작으면 보존
                if iou < 0.5:
                    remaining_detections.append(detection)
            
            sorted_detections = remaining_detections
        
        return final_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        두 바운딩 박스의 IoU(Intersection over Union) 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
            bbox2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            IoU 값 (0에서 1 사이)
        """
        # 교집합 계산
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 각 바운딩 박스 면적 계산
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 합집합 계산
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # IoU 계산
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
        
    def visualize_goal_points(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        감지된 목표지점을 시각화
        
        Args:
            image: 입력 이미지
            detections: 감지된 목표지점 목록
            
        Returns:
            시각화된 이미지
        """
        result_img = image.copy()
        
        for det in detections:
            # 바운딩 박스
            x1, y1, x2, y2 = det["bbox"]
            
            # 색상 결정 (BGR)
            color_name = det.get("color", "unknown")
            if color_name == "red":
                color = (0, 0, 255)
            elif color_name == "green":
                color = (0, 255, 0)
            elif color_name == "blue":
                color = (255, 0, 0)
            elif color_name == "yellow":
                color = (0, 255, 255)
            elif color_name == "purple":
                color = (255, 0, 255)
            elif color_name == "orange":
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)  # 기본값 (노란색)
            
            # 바운딩 박스 그리기
            cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 레이블 생성
            shape = det.get("shape", "")
            label = f"Goal: {color_name} {shape}"
            conf = det.get("confidence", 0.0)
            label = f"{label} {conf:.2f}"
            
            # 레이블 표시
            cv2.putText(
                result_img, 
                label, 
                (int(x1), int(y1 - 10) if y1 - 10 > 10 else int(y1 + 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return result_img 