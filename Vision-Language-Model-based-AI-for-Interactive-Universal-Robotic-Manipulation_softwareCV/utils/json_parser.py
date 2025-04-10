"""
JSON 파서 모듈

Gemini API 응답에서 JSON 추출 및 바운딩 박스 스키마 검증 기능 제공
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

# 로깅 설정
logger = logging.getLogger("JsonParser")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """텍스트에서 JSON 추출
    
    Args:
        text: JSON을 포함할 수 있는 텍스트
        
    Returns:
        추출된 JSON 딕셔너리, 실패 시 빈 딕셔너리
    """
    try:
        # 코드 블록에서 JSON 추출 시도
        json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # 중괄호로 둘러싸인 부분 추출 시도
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # 전체 텍스트가 JSON인지 확인
        if text.strip().startswith('{') and text.strip().endswith('}'):
            return json.loads(text.strip())
        
        # JSON을 찾지 못한 경우
        logger.warning("텍스트에서 JSON을 찾을 수 없습니다")
        return {}
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        return {}
    
    except Exception as e:
        logger.error(f"JSON 처리 중 오류 발생: {str(e)}")
        return {}

def rescale_bounding_box(bbox: List[List[int]], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> List[List[int]]:
    """바운딩 박스 좌표를 한 이미지 크기에서 다른 이미지 크기로 스케일링
    
    Args:
        bbox: 바운딩 박스 좌표 리스트 [[x1, y1], [x2, y2], ...]
        from_size: 원본 이미지 크기 (너비, 높이)
        to_size: 대상 이미지 크기 (너비, 높이)
        
    Returns:
        스케일링된 바운딩 박스 좌표 리스트
    """
    # 스케일 비율 계산
    fx = to_size[0] / from_size[0]
    fy = to_size[1] / from_size[1]
    
    # 좌표 변환
    scaled_bbox = []
    for point in bbox:
        if len(point) >= 2:
            scaled_x = int(point[0] * fx)
            scaled_y = int(point[1] * fy)
            scaled_bbox.append([scaled_x, scaled_y])
        else:
            # 유효하지 않은 좌표는 그대로 유지
            scaled_bbox.append(point)
    
    return scaled_bbox

def detect_coordinate_scale(bbox: List[List[int]], image_size: List[int]) -> float:
    """바운딩 박스 좌표의 스케일을 탐지
    
    Args:
        bbox: 바운딩 박스 좌표 리스트
        image_size: 이미지 크기 [너비, 높이]
        
    Returns:
        감지된 스케일 비율 (1.0이면 정상, >1.0이면 좌표가 더 큼)
    """
    width, height = image_size
    max_x = max(point[0] for point in bbox)
    
    # 이미지 너비를 초과하는 경우
    if max_x > width:
        return max_x / width
    
    return 1.0  # 정상 스케일

def validate_bbox_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """바운딩 박스 JSON 스키마 검증
    
    Args:
        data: 검증할 JSON 데이터
        
    Returns:
        검증 및 정규화된 JSON 데이터
    """
    # 빈 데이터 확인
    if not data:
        logger.error("검증할 데이터가 비어 있습니다")
        return {"error": "검증할 데이터가 비어 있습니다"}
    
    # 기본 스키마 확인
    required_keys = ["image_size", "objects"]
    for key in required_keys:
        if key not in data:
            logger.error(f"필수 키가 없습니다: {key}")
            return {"error": f"필수 키가 없습니다: {key}"}
    
    # 이미지 크기 검증
    image_size = data.get("image_size", [])
    if not isinstance(image_size, list) or len(image_size) != 2:
        logger.error("이미지 크기 형식이 잘못되었습니다")
        if isinstance(image_size, dict) and "width" in image_size and "height" in image_size:
            # 다른 형식으로 제공된 경우 변환
            data["image_size"] = [image_size["width"], image_size["height"]]
        else:
            return {"error": "이미지 크기 형식이 잘못되었습니다"}
    
    # 객체 리스트 검증
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        logger.error("객체 목록이 리스트 형식이 아닙니다")
        return {"error": "객체 목록이 리스트 형식이 아닙니다"}
    
    width, height = data["image_size"]
    
    # 각 객체 검증
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            logger.error(f"객체 {i}가 딕셔너리 형식이 아닙니다")
            continue
        
        # 필수 필드 확인
        if "label" not in obj:
            logger.warning(f"객체 {i}에 label 필드가 없습니다")
            obj["label"] = f"object_{i}"
        
        if "bounding_box" not in obj:
            logger.error(f"객체 {i}에 bounding_box 필드가 없습니다")
            continue
        
        # 바운딩 박스 좌표 검증
        bbox = obj["bounding_box"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            logger.error(f"객체 {i}의 바운딩 박스 형식이 잘못되었습니다")
            continue
        
        # 좌표 형식 검증 및 수정
        valid_bbox = True
        for j, point in enumerate(bbox):
            if not isinstance(point, list) or len(point) != 2:
                logger.error(f"객체 {i}의 좌표 {j}가 유효하지 않습니다")
                valid_bbox = False
                break
            
            # 좌표가 정수가 아닌 경우 정수로 변환
            if not all(isinstance(coord, int) for coord in point):
                bbox[j] = [int(float(coord)) for coord in point]
        
        if not valid_bbox:
            continue
        
        # 스케일 이슈 감지 및 조정
        scale = detect_coordinate_scale(bbox, data["image_size"])
        if scale > 1.1:  # 10% 이상 크기 차이가 있으면 스케일 조정
            logger.warning(f"객체 {i}의 좌표가 이미지 크기보다 {scale:.2f}배 큽니다. 스케일링을 수행합니다.")
            # 모델이 사용한 이미지 크기 추정 (좌표 최대값 기준)
            max_x = max(point[0] for point in bbox)
            max_y = max(point[1] for point in bbox)
            estimated_size = (max_x, max_y)
            # 추정된 크기에서 실제 이미지 크기로 스케일링
            obj["bounding_box"] = rescale_bounding_box(bbox, estimated_size, (width, height))
            bbox = obj["bounding_box"]  # 업데이트된 바운딩 박스
        
        # 이미지 범위를 벗어나는 좌표 보정 (클램핑)
        for j, [x, y] in enumerate(bbox):
            bbox[j] = [
                max(0, min(x, width)),
                max(0, min(y, height))
            ]
    
    # 바운딩 박스가 없거나 모두 유효하지 않은 경우
    valid_objects = [obj for obj in objects if "bounding_box" in obj]
    if not valid_objects:
        logger.error("유효한 바운딩 박스가 없습니다")
        return {"error": "유효한 바운딩 박스가 없습니다", **data}
    
    # 필터링된 객체 목록으로 업데이트
    data["objects"] = valid_objects
    
    return data

def format_bbox_response(data: Dict[str, Any], original_command: str) -> Dict[str, Any]:
    """바운딩 박스 응답 형식화
    
    Args:
        data: 형식화할 JSON 데이터
        original_command: 원본 명령어
        
    Returns:
        형식화된 JSON 데이터
    """
    # 원본 명령어 추가
    data["original_command"] = original_command
    
    return data
