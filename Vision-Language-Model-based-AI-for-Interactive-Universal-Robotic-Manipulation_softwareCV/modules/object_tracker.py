"""
객체 추적을 위한 IOU 기반 추적기 모듈

이 모듈은 YOLO 감지 결과를 프레임 간에 추적하여 더 안정적인 객체 인식을 제공합니다.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class ObjectTracker:
    """IOU(Intersection over Union) 기반 객체 추적 클래스
    
    감지된 객체를 프레임간에 추적하여 더 안정적인 객체 인식을 제공합니다.
    각 객체에 고유한 ID를 할당하고, 추적하며, 일정 시간동안 보이지 않으면 삭제합니다.
    """
    
    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3, 
                max_track_history: int = 20):
        """
        ObjectTracker 초기화
        
        Args:
            max_disappeared: 객체가 사라졌다고 판단하기 전까지 기다릴 최대 프레임 수
            iou_threshold: 동일 객체로 판단할 IOU 임계값
            max_track_history: 경로 추적을 위해 저장할 최대 위치 개수
        """
        self.logger = logging.getLogger("ObjectTracker")
        self.logger.setLevel(logging.WARNING)  # 로깅 레벨 설정
        
        # 현재 추적 중인 객체 목록 (ID → 객체 정보)
        self.tracked_objects = {}
        
        # 각 객체별 사라진 프레임 수 카운트 (ID → 카운트)
        self.disappeared = {}
        
        # 각 객체별 경로 추적 (ID → 중심점 좌표 목록)
        self.track_history = {}
        
        # 다음 객체 ID
        self.next_object_id = 0
        
        # 최대 사라짐 허용 프레임 수
        self.max_disappeared = max_disappeared
        
        # IOU 매칭 임계값
        self.iou_threshold = iou_threshold
        
        # 경로 추적을 위해 저장할 최대 위치 개수
        self.max_track_history = max_track_history
        
        self.logger.debug(f"객체 추적기 초기화: max_disappeared={max_disappeared}, iou_threshold={iou_threshold}")
    
    def register(self, detection: Dict[str, Any]) -> int:
        """
        새 객체 등록
        
        Args:
            detection: 감지된 객체 정보
            
        Returns:
            할당된 객체 ID
        """
        # 객체 ID 할당 및 등록
        object_id = self.next_object_id
        self.next_object_id += 1
        
        # 객체 정보에 고유 ID 추가
        detection_with_id = detection.copy()
        detection_with_id["track_id"] = object_id
        
        # 추적 목록에 추가
        self.tracked_objects[object_id] = detection_with_id
        self.disappeared[object_id] = 0
        
        # 경로 추적 초기화 (첫 위치 추가)
        center_x = detection.get("center_x", (detection["bbox"][0] + detection["bbox"][2]) / 2)
        center_y = detection.get("center_y", (detection["bbox"][1] + detection["bbox"][3]) / 2)
        self.track_history[object_id] = [(center_x, center_y)]
        
        # 객체 정보에 경로 정보 추가
        detection_with_id["track_history"] = self.track_history[object_id].copy()
        
        self.logger.debug(f"새 객체 등록: ID={object_id}, class={detection.get('class_name', 'unknown')}")
        return object_id
    
    def deregister(self, object_id: int) -> None:
        """
        객체 등록 취소 (추적 중단)
        
        Args:
            object_id: 취소할 객체 ID
        """
        # 추적 목록에서 제거
        if object_id in self.tracked_objects:
            self.logger.debug(f"객체 추적 중단: ID={object_id}, class={self.tracked_objects[object_id].get('class_name', 'unknown')}")
            del self.tracked_objects[object_id]
            del self.disappeared[object_id]
            
            # 경로 정보 제거
            if object_id in self.track_history:
                del self.track_history[object_id]
    
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        추적 객체 목록 업데이트
        
        Args:
            detections: 현재 프레임에서 감지된 객체 목록
            
        Returns:
            업데이트된 추적 객체 목록 (ID → 객체 정보)
        """
        # 감지된 객체가 없는 경우
        if len(detections) == 0:
            # 모든 추적 객체의 사라짐 카운트 증가
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # 최대 사라짐 프레임 수를 초과하면 삭제
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # 추적 중인 객체 반환
            return self.tracked_objects
        
        # 현재 추적 중인 객체가 없는 경우
        if len(self.tracked_objects) == 0:
            # 모든 감지 객체를 새로운 추적 객체로 등록
            for detection in detections:
                self.register(detection)
        else:
            # 기존 추적 객체와 새 감지 객체 매칭
            self._match_and_update(detections)
        
        return self.tracked_objects
    
    def _match_and_update(self, detections: List[Dict[str, Any]]) -> None:
        """
        기존 추적 객체와 새 감지 객체 매칭 및 업데이트
        
        Args:
            detections: 현재 프레임에서 감지된 객체 목록
        """
        # 현재 추적 중인 객체 ID 목록
        current_ids = list(self.tracked_objects.keys())
        
        # 매칭되지 않은 객체 (초기에는 모든 객체가 매칭되지 않음)
        unmatched_trackers = set(current_ids)
        unmatched_detections = set(range(len(detections)))
        
        # 각 추적 객체와 감지 객체 간 IOU 매트릭스 계산
        iou_matrix = np.zeros((len(current_ids), len(detections)))
        
        for i, object_id in enumerate(current_ids):
            tracked_obj = self.tracked_objects[object_id]
            tracked_bbox = tracked_obj["bbox"]
            
            for j, detection in enumerate(detections):
                detection_bbox = detection["bbox"]
                
                # IOU 계산
                iou = self._calculate_iou(tracked_bbox, detection_bbox)
                iou_matrix[i, j] = iou
        
        # 매칭 수행 (그리디 방식: IOU가 가장 높은 쌍부터 매칭)
        # numpy.argmax 사용하지 않고 직접 최대값 찾기
        while True:
            # 남은 가장 높은 IOU 값 찾기
            max_iou = 0
            max_i, max_j = -1, -1
            
            for i in range(len(current_ids)):
                if i not in unmatched_trackers:
                    continue
                    
                for j in range(len(detections)):
                    if j not in unmatched_detections:
                        continue
                        
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j
            
            # 더 이상 매칭할 쌍이 없거나 IOU가 임계값보다 작으면 종료
            if max_i == -1 or max_j == -1 or max_iou < self.iou_threshold:
                break
                
            # 매칭 처리
            object_id = current_ids[max_i]
            
            # 객체 정보 업데이트 (위치 등)
            self._update_object(object_id, detections[max_j])
            
            # 매칭된 객체는 목록에서 제거
            unmatched_trackers.remove(max_i)
            unmatched_detections.remove(max_j)
        
        # 매칭되지 않은 추적 객체: 사라짐 카운트 증가
        for idx in unmatched_trackers:
            object_id = current_ids[idx]
            self.disappeared[object_id] += 1
            
            # 최대 사라짐 프레임 수를 초과하면 삭제
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # 매칭되지 않은 감지 객체: 새로 등록
        for idx in unmatched_detections:
            self.register(detections[idx])
    
    def _update_object(self, object_id: int, detection: Dict[str, Any]) -> None:
        """
        추적 중인 객체 정보 업데이트
        
        Args:
            object_id: 업데이트할 객체 ID
            detection: 새로 감지된 객체 정보
        """
        # 기존 정보 유지하면서 위치 및 신뢰도 정보 업데이트
        tracked_obj = self.tracked_objects[object_id]
        
        # 신뢰도가 더 높은 경우에만 클래스 정보 업데이트
        if detection["confidence"] > tracked_obj["confidence"]:
            tracked_obj["class_id"] = detection["class_id"]
            tracked_obj["class_name"] = detection["class_name"]
            tracked_obj["confidence"] = detection["confidence"]
        
        # 위치 정보는 항상 업데이트
        tracked_obj["bbox"] = detection["bbox"]
        tracked_obj["width"] = detection["width"]
        tracked_obj["height"] = detection["height"]
        tracked_obj["center_x"] = detection["center_x"]
        tracked_obj["center_y"] = detection["center_y"]
        
        # 경로 정보 업데이트
        if object_id in self.track_history:
            center_x = detection["center_x"]
            center_y = detection["center_y"]
            
            # 새 위치 추가
            self.track_history[object_id].append((center_x, center_y))
            
            # 최대 개수 제한
            if len(self.track_history[object_id]) > self.max_track_history:
                self.track_history[object_id] = self.track_history[object_id][-self.max_track_history:]
            
            # 객체 정보에 경로 정보 갱신
            tracked_obj["track_history"] = self.track_history[object_id].copy()
        
        # 사라짐 카운트 초기화
        self.disappeared[object_id] = 0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        두 바운딩 박스의 IoU(Intersection over Union) 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
            bbox2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            IoU 값 (0에서 1 사이)
        """
        # 교집합 영역 계산
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # 교집합이 없는 경우
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # 교집합 영역 계산
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 각 박스 영역 계산
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 합집합 영역 계산
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # IOU 계산
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def get_tracked_objects_list(self) -> List[Dict[str, Any]]:
        """
        추적 중인 객체 목록을 리스트 형태로 반환
        
        Returns:
            추적 중인 객체 정보 리스트
        """
        return list(self.tracked_objects.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """
        추적기 상태 요약 정보 반환
        
        Returns:
            추적기 상태 요약 정보
        """
        # 클래스별 객체 수 계산
        class_counts = {}
        for obj in self.tracked_objects.values():
            class_name = obj.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "total_objects": len(self.tracked_objects),
            "next_id": self.next_object_id,
            "class_counts": class_counts
        } 