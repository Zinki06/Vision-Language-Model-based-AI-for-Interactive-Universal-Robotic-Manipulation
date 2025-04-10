"""
데이터 처리 관련 유틸리티 모듈

이 모듈은 데이터 접근, 검증, 변환 등을 위한 함수 및 클래스를 제공합니다.
"""

from typing import Dict, List, Any, TypedDict
import logging

# 타입 정의
class DetectionType(TypedDict):
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

def safe_get(dictionary: Dict[str, Any], *keys, default: Any = None) -> Any:
    """딕셔너리에서 안전하게 값 가져오기
    
    Args:
        dictionary: 검색할 딕셔너리
        *keys: 검색할 여러 키 (순서대로 검색)
        default: 키를 찾지 못했을 때 반환할 기본값
        
    Returns:
        찾은 값 또는 기본값
    """
    if not isinstance(dictionary, dict):
        return default
    
    # 단일 키 처리
    if len(keys) == 1:
        return dictionary.get(keys[0], default)
    
    # 여러 키 순서대로 검색
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    
    # 모든 키를 찾지 못한 경우 기본값 반환
    return default

def validate_detection_format(detection: Dict[str, Any], logger=None) -> DetectionType:
    """감지 결과 형식 검증 및 변환"""
    try:
        # 필수 키 확인 및 타입 변환
        result = {
            "class_name": str(safe_get(detection, "class_name") or safe_get(detection, "class", "")),
            "confidence": float(safe_get(detection, "confidence", 0)),
            "box": [float(v) for v in safe_get(detection, "box", [0, 0, 0, 0])]
        }
        return result
    except Exception as e:
        # 기본값 반환하며 로깅
        if logger:
            logger.error(f"객체 검증 오류: {e}")
        else:
            logging.error(f"객체 검증 오류: {e}")
        return {"class_name": "unknown", "confidence": 0.0, "box": [0, 0, 0, 0]}

class DetectionCoordinator:
    """객체 감지 결과 조정 및 표준화"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def standardize_detections(self, detections: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """다양한 형식의 감지 결과를 표준 형식으로 변환"""
        if not detections or "objects" not in detections:
            return {"objects": []}
            
        standard_objects = []
        for idx, obj in enumerate(detections["objects"]):
            try:
                # 키 이름 불일치 방지를 위한 표준화
                standard_obj = {
                    "id": idx,
                    "class_name": safe_get(obj, "class_name") or safe_get(obj, "class", "unknown"),
                    "confidence": float(safe_get(obj, "confidence", 0)),
                    "box": safe_get(obj, "box", [0, 0, 0, 0])
                }
                # 추가 필드 복사
                for key, value in obj.items():
                    if key not in ["id", "class_name", "class", "confidence", "box"]:
                        standard_obj[key] = value
                        
                standard_objects.append(standard_obj)
            except Exception as e:
                self.logger.error(f"객체 표준화 오류: {e}")
                
        return {"objects": standard_objects}

def diagnose_data_structure(data, depth=0, max_depth=3, path="", logger=None):
    """재귀적으로 데이터 구조 분석 및 로깅"""
    log = logger or logging.getLogger()
    
    if depth > max_depth:
        return
        
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            log.debug(f"검사 경로: {current_path}, 타입: {type(value)}")
            diagnose_data_structure(value, depth+1, max_depth, current_path, log)
    elif isinstance(data, list) and data and depth < max_depth:
        log.debug(f"목록 검사: {path}[0]/{len(data)}개 항목")
        if data:  # 비어있지 않은 리스트만 첫 요소 검사
            diagnose_data_structure(data[0], depth+1, max_depth, f"{path}[0]", log) 