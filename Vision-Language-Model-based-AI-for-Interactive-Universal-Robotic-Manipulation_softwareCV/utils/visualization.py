import numpy as np
import cv2
import random
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# 로깅 객체 생성
logger = logging.getLogger("LLM2PF6.visualization")
logger.setLevel(logging.ERROR)  # WARNING에서 ERROR로 변경

def contains_korean(text: str) -> bool:
    """
    Check if text contains Korean characters
    
    Args:
        text: Text string to check
        
    Returns:
        True if text contains Korean characters, False otherwise
    """
    return any('\u3131' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' or '\u1100' <= c <= '\u11FF' for c in text)

def draw_bounding_boxes(image: np.ndarray, detections: List[Dict[str, Any]], 
                       color_map: Dict[str, Tuple[int, int, int]] = None,
                       show_yolo_class: bool = False) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image
    
    Args:
        image: Original image (numpy array)
        detections: List of detected objects
        color_map: Color mapping by class (random if not provided)
        show_yolo_class: Whether to also display the class name detected by YOLO
        
    Returns:
        Image with drawn bounding boxes
    """
    # Create a copy
    img_result = image.copy()
    
    # Set default color mapping if not provided
    if color_map is None:
        color_map = {
            "person": (0, 128, 255),    # Orange
            "bicycle": (0, 255, 0),     # Green
            "car": (0, 255, 255),       # Yellow
            "motorcycle": (255, 0, 0),  # Blue
            "bus": (255, 0, 255),       # Purple
            "truck": (255, 255, 0),     # Cyan
            "book": (0, 165, 255),      # Orange
            "keyboard": (180, 105, 255),# Pink
            "remote": (50, 205, 154),   # Light green
            "cell phone": (0, 0, 255),  # Red
            "laptop": (255, 191, 0),    # Sky blue
            "mouse": (0, 215, 255),     # Yellow-orange
            
            # Goal point with special color
            "goal_point": (128, 0, 128) # Dark purple
        }
    
    # Draw bounding box for each object
    for det in detections:
        # Bounding box coordinates
        if "bbox" not in det:
            continue
            
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Determine object name (Gemini result takes priority)
        if "name" in det:
            class_name = det["name"]
        else:
            class_name = det.get("class_name", "unknown")
        
        # Check for Korean text in class_name
        if contains_korean(class_name):
            display_class_name = "(Korean object)"
        else:
            display_class_name = class_name
        
        # Confidence
        confidence = det.get("confidence", 0.0)
        
        # 가상 객체 여부 확인
        is_virtual = det.get("is_virtual", False)
        
        # Assign unique color for the class (random if not in map)
        if class_name not in color_map:
            import random
            color_hash = hash(class_name) % 256
            random.seed(color_hash)
            color_map[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            random.seed(None)  # 난수 생성기 초기화 복원
        
        color = color_map[class_name]
        
        # Create label text
        if class_name == "goal_point":
            # Goal point shows more detailed information
            color_name = det.get("color", "")
            shape_name = det.get("shape", "")
            detection_type = det.get("detection_type", "")
            
            # Check for Korean in color or shape names
            if contains_korean(color_name) or contains_korean(shape_name):
                label = "Goal: (Korean description)"
            elif color_name and shape_name:
                label = f"Goal: {color_name} {shape_name}"
            elif color_name:
                label = f"Goal: {color_name}"
            else:
                label = "Goal Point"
                
            if confidence > 0:
                label = f"{label} {confidence:.2f}"
        else:
            # Regular object label
            if show_yolo_class and "yolo_class" in det and det["yolo_class"] != class_name:
                yolo_class = det["yolo_class"]
                if contains_korean(yolo_class):
                    label = f"{display_class_name} ((Korean class)) {confidence:.2f}"
                else:
                    label = f"{display_class_name} ({yolo_class}) {confidence:.2f}"
            else:
                label = f"{display_class_name} {confidence:.2f}"
                
                # 가상 객체 표시
                if is_virtual:
                    label = f"{label} (virtual)"
            
            # Show additional attributes (if available)
            if "color" in det and "state" in det:
                color_text = det["color"]
                state_text = det["state"]
                
                # Check for Korean in attributes
                if contains_korean(color_text) or contains_korean(state_text):
                    label = f"{display_class_name} ((Korean attributes)) {confidence:.2f}"
                else:
                    label = f"{display_class_name} ({color_text}, {state_text}) {confidence:.2f}"
        
        # Draw bounding box
        if class_name == "goal_point":
            # Goal point shows dashed pattern
            for i in range(x1, x2, 10):
                cv2.line(img_result, (i, y1), (i + 5, y1), color, 2)
                cv2.line(img_result, (i, y2), (i + 5, y2), color, 2)
            for i in range(y1, y2, 10):
                cv2.line(img_result, (x1, i), (x1, i + 5), color, 2)
                cv2.line(img_result, (x2, i), (x2, i + 5), color, 2)
        elif is_virtual:
            # 가상 객체는 대시 패턴으로 그림 (밝은 색상)
            bright_color = tuple(min(c + 80, 255) for c in color)
            
            # 대시 패턴으로 바운딩 박스 그리기
            for i in range(x1, x2, 10):
                cv2.line(img_result, (i, y1), (i + 5, y1), bright_color, 2)
                cv2.line(img_result, (i, y2), (i + 5, y2), bright_color, 2)
            
            for i in range(y1, y2, 10):
                cv2.line(img_result, (x1, i), (x1, i + 5), bright_color, 2)
                cv2.line(img_result, (x2, i), (x2, i + 5), bright_color, 2)
            
            color = bright_color  # 밝은 색상으로 레이블도 표시
        else:
            # Regular bounding box
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_result

def draw_comparison_image(image: np.ndarray, yolo_detections: List[Dict[str, Any]], 
                          gemini_detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    YOLO와 Gemini 결과를 나란히 비교하는 이미지 생성
    
    Args:
        image: 원본 이미지 (numpy 배열)
        yolo_detections: YOLO가 감지한 객체 목록
        gemini_detections: Gemini가 분류한 객체 목록
        
    Returns:
        비교 이미지 (좌: YOLO, 우: Gemini)
    """
    # 이미지 복사본 생성
    yolo_img = image.copy()
    gemini_img = image.copy()
    
    # YOLO 결과 그리기 (왼쪽)
    yolo_img = draw_bounding_boxes(yolo_img, yolo_detections)
    
    # Gemini 결과 그리기 (오른쪽)
    gemini_img = draw_bounding_boxes(gemini_img, gemini_detections, show_yolo_class=True)
    
    # 이미지에 제목 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(yolo_img, "YOLO Detection", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(gemini_img, "Gemini Classification", (10, 30), font, 1, (0, 255, 0), 2)
    
    # 수직으로 두 이미지 연결
    h, w = image.shape[:2]
    result = np.zeros((h * 2, w, 3), dtype=np.uint8)
    result[:h, :] = yolo_img
    result[h:, :] = gemini_img
    
    return result

def draw_side_by_side_comparison(image: np.ndarray, yolo_detections: List[Dict[str, Any]], 
                                gemini_detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    YOLO와 Gemini 결과를 좌우로 나란히 비교하는 이미지 생성
    
    Args:
        image: 원본 이미지 (numpy 배열)
        yolo_detections: YOLO가 감지한 객체 목록
        gemini_detections: Gemini가 분류한 객체 목록
        
    Returns:
        비교 이미지 (좌: YOLO, 우: Gemini)
    """
    # 이미지 복사본 생성
    yolo_img = image.copy()
    gemini_img = image.copy()
    
    # YOLO 결과 그리기 (왼쪽)
    yolo_img = draw_bounding_boxes(yolo_img, yolo_detections)
    
    # Gemini 결과 그리기 (오른쪽)
    gemini_img = draw_bounding_boxes(gemini_img, gemini_detections, show_yolo_class=True)
    
    # 이미지에 제목 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(yolo_img, "YOLO Detection", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(gemini_img, "Gemini Classification", (10, 30), font, 1, (0, 255, 0), 2)
    
    # 수평으로 두 이미지 연결
    h, w = image.shape[:2]
    result = np.zeros((h, w * 2, 3), dtype=np.uint8)
    result[:, :w] = yolo_img
    result[:, w:] = gemini_img
    
    return result

def draw_command_result(
    image: np.ndarray, 
    target_object: Dict[str, Any] = None, 
    reference_object: Dict[str, Any] = None,
    destination: Dict[str, Any] = None
) -> np.ndarray:
    """
    이미지에 명령어 결과(타겟, 레퍼런스, 목적지)를 시각화하는 함수
    
    Args:
        image: 원본 이미지 (numpy 배열)
        target_object: 타겟 객체 정보
        reference_object: 레퍼런스 객체 정보
        destination: 목적지 정보
    
    Returns:
        시각화된 이미지
    """
    # 이미지 복사본 생성
    result_img = image.copy()
    
    # 색상 정의
    TARGET_COLOR = (0, 255, 0)  # Green
    REFERENCE_COLOR = (255, 0, 0)  # Blue
    DESTINATION_COLOR = (0, 255, 255)  # Yellow
    
    # 1. Target object drawing
    if target_object and "bbox" in target_object:
        bbox = target_object["bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Thick bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), TARGET_COLOR, 3)
        
        # Target label
        name = target_object.get("name", "Target")
        label = f"TARGET: {name}"
        
        # Label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, (x1, y1 - 25), (x1 + text_size[0], y1), TARGET_COLOR, -1)
        
        # Label text
        cv2.putText(result_img, label, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 2. Reference object drawing
    if reference_object and "bbox" in reference_object:
        bbox = reference_object["bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Dashed pattern bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), REFERENCE_COLOR, 2)
        
        # Reference label
        name = reference_object.get("name", "Reference")
        label = f"REF: {name}"
        
        # Label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, (x1, y1 - 25), (x1 + text_size[0], y1), REFERENCE_COLOR, -1)
        
        # Label text
        cv2.putText(result_img, label, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 3. Destination drawing
    if destination and "bounding_box" in destination:
        bbox = destination["bounding_box"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Dashed line bounding box (dashed pattern)
        for i in range(x1, x2, 10):
            cv2.line(result_img, (i, y1), (i + 5, y1), DESTINATION_COLOR, 2)
            cv2.line(result_img, (i, y2), (i + 5, y2), DESTINATION_COLOR, 2)
        for i in range(y1, y2, 10):
            cv2.line(result_img, (x1, i), (x1, i + 5), DESTINATION_COLOR, 2)
            cv2.line(result_img, (x2, i), (x2, i + 5), DESTINATION_COLOR, 2)
        
        # Destination label
        relation = destination.get("relation", "").capitalize()
        label = f"DEST: {relation}"
        
        # Label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, (x1, y2 + 5), (x1 + text_size[0], y2 + 30), DESTINATION_COLOR, -1)
        
        # Label text
        cv2.putText(result_img, label, (x1, y2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Target->Destination arrow (when both target and destination exist)
        if target_object and "bbox" in target_object:
            t_bbox = target_object["bbox"]
            t_x = int((t_bbox[0] + t_bbox[2]) / 2)
            t_y = int((t_bbox[1] + t_bbox[3]) / 2)
            
            d_x = int((x1 + x2) / 2)
            d_y = int((y1 + y2) / 2)
            
            # Arrow drawing
            cv2.arrowedLine(result_img, (t_x, t_y), (d_x, d_y), DESTINATION_COLOR, 2, tipLength=0.03)
    
    return result_img

def draw_target_destination_image(image: np.ndarray, command_result: Dict[str, Any]) -> np.ndarray:
    """
    Visualize command understanding results on the image
    
    Args:
        image: Original image
        command_result: Command understanding results
        
    Returns:
        Visualized image
    """
    img_result = image.copy()
    
    # Set base colors
    target_color = (0, 0, 255)  # Target object: Red
    reference_color = (0, 255, 0)  # Reference object: Green
    destination_color = (255, 0, 0)  # Destination: Blue
    
    # Extract command
    command = command_result.get("command", "")
    
    # Visualize target object
    target_obj = command_result.get("target_object", {})
    if target_obj and target_obj.get("bbox"):
        bbox = target_obj["bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw bounding box (with thick line)
        cv2.rectangle(img_result, (x1, y1), (x2, y2), target_color, 3)
        
        # Label text
        label = "Target Object"
        if target_obj.get("name"):
            name = target_obj.get("name")
            if contains_korean(name):
                label += ": (Korean name)"
            else:
                label += f": {name}"
            
        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), target_color, -1)
        
        # Draw label text
        cv2.putText(img_result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Visualize reference object
    ref_obj = command_result.get("reference_object", {})
    if ref_obj and ref_obj.get("bbox"):
        bbox = ref_obj["bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw bounding box (with thick line)
        cv2.rectangle(img_result, (x1, y1), (x2, y2), reference_color, 3)
        
        # Label text
        label = "Reference Object"
        if ref_obj.get("name"):
            name = ref_obj.get("name")
            if contains_korean(name):
                label += ": (Korean name)"
            else:
                label += f": {name}"
            
        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), reference_color, -1)
        
        # Draw label text
        cv2.putText(img_result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Visualize destination
    dest = command_result.get("destination", {})
    if dest and dest.get("bounding_box"):
        bbox = dest["bounding_box"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw dashed line bounding box (dashed pattern)
        for i in range(x1, x2, 10):
            cv2.line(img_result, (i, y1), (i + 5, y1), destination_color, 2)
            cv2.line(img_result, (i, y2), (i + 5, y2), destination_color, 2)
        
        # Draw dashed line bounding box (dashed pattern)
        for i in range(y1, y2, 10):
            cv2.line(img_result, (x1, i), (x1, i + 5), destination_color, 2)
            cv2.line(img_result, (x2, i), (x2, i + 5), destination_color, 2)
        
        # Label text
        label = "Goal Point"
        if dest.get("description"):
            description = dest["description"]
            # 한국어 설명이 있는 경우 영어로 대체
            if contains_korean(description):
                label += ": (Korean description)"
            else:
                if len(description) > 20:
                    description = description[:20] + "..."
                label += f": {description}"
            
        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), destination_color, -1)
        
        # Draw label text
        cv2.putText(img_result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw command text (top)
    if command:
        # 한국어가 포함된 경우 영어로 대체
        if contains_korean(command):
            cv2.putText(img_result, "Command: (Korean command)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            cv2.putText(img_result, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img_result

def draw_tracked_objects(image: np.ndarray, tracked_objects: List[Dict[str, Any]],
                        color_map: Dict[str, Tuple[int, int, int]] = None,
                        show_track_id: bool = True) -> np.ndarray:
    """
    추적 중인 객체 시각화 (ID 포함)
    
    Args:
        image: 원본 이미지 (numpy 배열)
        tracked_objects: 추적 중인 객체 목록
        color_map: 클래스별 색상 매핑 (없으면 무작위 생성)
        show_track_id: 추적 ID 표시 여부
        
    Returns:
        시각화된 이미지
    """
    # 추적 객체가 없으면 원본 이미지 반환
    if not tracked_objects:
        return image.copy()
        
    # random 모듈 임포트
    import random
    
    # 이미지 복사본 생성
    result_img = image.copy()
    
    # 기본 색상 맵 설정
    if color_map is None:
        color_map = {
            "person": (0, 128, 255),    # Orange
            "bicycle": (0, 255, 0),     # Green
            "car": (0, 255, 255),       # Yellow
            "motorcycle": (255, 0, 0),  # Blue
            "bus": (255, 0, 255),       # Purple
            "truck": (255, 255, 0),     # Cyan
            "book": (0, 165, 255),      # Orange
            "keyboard": (180, 105, 255),# Pink
            "remote": (50, 205, 154),   # Light green
            "cell phone": (0, 0, 255),  # Red
            "laptop": (255, 191, 0),    # Sky blue
            "mouse": (0, 215, 255),     # Yellow-orange
        }
    
    # 개체별 바운딩 박스 및 레이블 그리기
    for obj in tracked_objects:
        try:
            # 바운딩 박스 좌표
            if "bbox" not in obj:
                continue
                
            # bbox 형식 확인
            if not isinstance(obj["bbox"], list) or len(obj["bbox"]) != 4:
                continue
                
            x1, y1, x2, y2 = obj["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 유효한 좌표인지 확인
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue
            
            # 이미지 경계 확인
            h, w = result_img.shape[:2]
            if x1 >= w or y1 >= h:
                continue
                
            # 좌표를 이미지 경계 내로 제한
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # 클래스 이름 및 ID
            class_name = obj.get("class_name", "unknown")
            track_id = obj.get("track_id", -1)
            
            # 신뢰도
            confidence = obj.get("confidence", 0.0)
            
            # 색상 할당 (클래스별 고유 색상)
            if class_name not in color_map:
                # 난수 시드를 클래스 이름 해시값으로 설정하여 항상 동일한 색상 생성
                color_hash = hash(class_name) % 256
                random.seed(color_hash)
                color_map[class_name] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                random.seed(None)  # 난수 시드 초기화
            
            color = color_map[class_name]
            
            # 레이블 텍스트 생성
            if show_track_id and track_id >= 0:
                label = f"#{track_id}: {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            # 바운딩 박스 그리기 (좀 더 두껍게)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # 바운딩 박스 상단에 객체 ID 표시를 위한 원 그리기
            if show_track_id and track_id >= 0:
                center_x = (x1 + x2) // 2
                cv2.circle(result_img, (center_x, y1-15), 15, color, -1)
                
                id_text = f"{track_id}"
                text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = y1 - 10
                cv2.putText(result_img, id_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 레이블 배경 그리기
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 레이블이 이미지 상단을 벗어나지 않도록 조정
            label_y1 = max(0, y1 - text_size[1] - 10)
            cv2.rectangle(result_img, (x1, label_y1), (x1 + text_size[0] + 10, y1), color, -1)
            
            # 레이블 텍스트 그리기
            cv2.putText(result_img, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 객체 이동 경로 그리기 (최근 위치 5개까지만)
            if "track_history" in obj and obj["track_history"] and len(obj["track_history"]) > 1:
                history = obj["track_history"][-5:]  # 최근 5개 위치
                
                # 경로 그리기
                for i in range(1, len(history)):
                    # 유효한 좌표인지 확인
                    if (isinstance(history[i-1], tuple) and len(history[i-1]) == 2 and 
                        isinstance(history[i], tuple) and len(history[i]) == 2):
                        
                        pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                        pt2 = (int(history[i][0]), int(history[i][1]))
                        
                        # 좌표가 이미지 내부에 있는지 확인
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                            0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(result_img, pt1, pt2, color, 2)
        
        except Exception as e:
            # 객체 시각화 중 오류 발생 시 해당 객체는 건너뛰고 계속 진행
            logger.error(f"객체 시각화 중 오류 발생: {str(e)}")
            continue
    
    return result_img 