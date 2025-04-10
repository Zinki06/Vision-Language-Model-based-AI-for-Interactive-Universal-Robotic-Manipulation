"""
위치 추론기 모듈

이 모듈은 자연어 텍스트에서 방향 정보를 추출하고 물체의 위치를 추론하는 기능을 제공합니다.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .direction_factory import DirectionFactory
from .direction_strategy import DirectionStrategy

class PositionInferrer:
    """위치 추론기 클래스"""
    
    def __init__(self):
        """
        위치 추론기 초기화
        
        방향 팩토리 및 필요한 컴포넌트를 초기화합니다.
        """
        self.logger = logging.getLogger(__name__)
        self.direction_factory = DirectionFactory()
    
    def extract_directions(self, text: str) -> List[Tuple[str, DirectionStrategy]]:
        """
        텍스트에서 방향 정보 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[Tuple[str, DirectionStrategy]]: (키워드, 방향 전략) 튜플 목록
        """
        return self.direction_factory.extract_direction_from_text(text)
    
    def infer_position(
        self, 
        reference_obj: Dict[str, Any], 
        text: str, 
        default_distance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        참조 객체를 기준으로 위치 추론
        
        Args:
            reference_obj: 참조 객체 정보 (위치, 크기 등)
            text: 방향 정보를 포함한 텍스트
            default_distance: 기본 거리 (None인 경우 방향 전략 기본값 사용)
            
        Returns:
            Dict[str, Any]: 추론된 위치 및 관련 정보
        """
        # 참조 객체에서 필요한 정보 추출
        ref_pos = np.array(reference_obj.get('position', [0, 0, 0]))
        ref_dims = np.array(reference_obj.get('dimensions', [1, 1, 1]))
        
        # 텍스트에서 방향 정보 추출
        directions = self.extract_directions(text)
        
        if not directions:
            self.logger.warning(f"텍스트 '{text}'에서 방향 정보를 찾을 수 없습니다.")
            return {
                'success': False,
                'position': ref_pos.tolist(),
                'reason': "방향 정보를 찾을 수 없습니다."
            }
        
        # 가장 좋은 방향 전략 선택 (현재는 첫 번째 발견된 전략 사용)
        # TODO: 더 복잡한 방향 결정 로직 구현 (우선순위, 확률 등)
        keyword, direction_strategy = directions[0]
        
        # 방향 벡터 계산
        direction_vector = direction_strategy.get_direction_vector()
        
        # 거리 결정
        if default_distance is None:
            distance = direction_strategy.get_default_distance()
        else:
            distance = default_distance
        
        # 참조 객체 치수 기준 오프셋 계산
        offset = direction_strategy.calculate_offset(ref_dims, distance)
        
        # 새 위치 계산
        new_position = ref_pos + offset
        
        return {
            'success': True,
            'position': new_position.tolist(),
            'direction_keyword': keyword,
            'direction_vector': direction_vector.tolist(),
            'offset': offset.tolist(),
            'distance': distance,
            'reference_position': ref_pos.tolist(),
            'reference_dimensions': ref_dims.tolist()
        }
    
    def infer_relative_position(
        self,
        reference_objs: List[Dict[str, Any]],
        target_text: str,
        default_distance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        여러 참조 객체에 대한 상대적 위치 추론
        
        Args:
            reference_objs: 참조 객체 목록
            target_text: 대상 텍스트
            default_distance: 기본 거리
            
        Returns:
            Dict[str, Any]: 추론된 위치 정보
        """
        if not reference_objs:
            return {
                'success': False,
                'reason': "참조 객체가 없습니다."
            }
        
        # 현재는 첫 번째 참조 객체만 사용
        # TODO: 여러 참조 객체를 고려한 위치 추론 로직 구현
        primary_ref = reference_objs[0]
        
        return self.infer_position(primary_ref, target_text, default_distance)
    
    def get_supported_directions(self, lang: str = 'ko') -> Dict[str, List[str]]:
        """
        지원되는 방향 키워드 반환
        
        Args:
            lang: 언어 코드
            
        Returns:
            Dict[str, List[str]]: 방향 ID와 해당 키워드 매핑
        """
        return self.direction_factory.get_direction_keywords(lang) 