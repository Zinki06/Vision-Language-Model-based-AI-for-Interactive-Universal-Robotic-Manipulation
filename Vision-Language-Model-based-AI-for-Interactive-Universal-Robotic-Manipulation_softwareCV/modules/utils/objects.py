"""
객체 관리 관련 유틸리티 모듈

이 모듈은 객체 감지 결과를 표준화된 형식으로 관리하는 클래스를 제공합니다.
"""

from typing import Dict, List, Any, Optional
from .data import safe_get

class DetectedObject:
    """객체 감지 결과를 표준화된 형식으로 관리하는 클래스"""
    
    def __init__(self, id: int, class_name: str, confidence: float, box: List[float], **kwargs):
        """
        객체 초기화
        
        Args:
            id: 객체 ID
            class_name: 객체 클래스 이름
            confidence: 감지 신뢰도
            box: 바운딩 박스 [x1, y1, x2, y2]
            **kwargs: 추가 속성
        """
        self.id = id
        self.class_name = class_name
        self.confidence = confidence
        self.box = box
        
        # 추가 속성 처리
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], id: Optional[int] = None) -> 'DetectedObject':
        """딕셔너리에서 객체 생성"""
        # 필수 키가 없는 경우 대체값 사용
        object_id = id if id is not None else data.get("id", 0)
        class_name = safe_get(data, "class_name") or safe_get(data, "class", "unknown")
        confidence = float(safe_get(data, "confidence", 0.0))
        box = safe_get(data, "box", [0, 0, 0, 0])
        
        # 이미 처리한 필수 키를 제외한 나머지 키를 kwargs로 전달
        kwargs = {k: v for k, v in data.items() 
                  if k not in ["id", "class_name", "class", "confidence", "box"]}
        
        return cls(object_id, class_name, confidence, box, **kwargs) 