"""
방향 전략 팩토리 모듈

이 모듈은 다양한 방향 전략을 생성하는 팩토리 클래스를 제공합니다.
"""

from typing import Dict, Optional, Type, Any, Union
import numpy as np

# 지연 임포트를 위해 필요할 때만 direction_strategy 모듈을 임포트
# 이렇게 하면 순환 참조 문제를 방지할 수 있습니다


class DirectionStrategyFactory:
    """
    방향 전략 팩토리 클래스
    
    다양한 키워드나 지시에 따라 적절한 방향 전략 객체를 생성합니다.
    영어와 한국어 키워드를 모두 지원합니다.
    """
    
    def __init__(self):
        """
        방향 전략 팩토리 초기화
        
        영어와 한국어 방향 키워드에 대한 매핑을 설정합니다.
        """
        # 방향 전략 모듈 지연 임포트
        from modules.goal_inference_components.direction_strategy import (
            DirectionStrategy,
            DefaultDirectionStrategy,
            UpDirectionStrategy,
            DownDirectionStrategy,
            LeftDirectionStrategy,
            RightDirectionStrategy,
            ForwardDirectionStrategy,
            BackwardDirectionStrategy,
            RandomDirectionStrategy
        )
        
        # 방향 전략 클래스 매핑
        self._strategy_classes: Dict[str, Type[DirectionStrategy]] = {
            # 영어 키워드
            "up": UpDirectionStrategy,
            "down": DownDirectionStrategy,
            "left": LeftDirectionStrategy,
            "right": RightDirectionStrategy,
            "front": ForwardDirectionStrategy,
            "forward": ForwardDirectionStrategy,
            "back": BackwardDirectionStrategy,
            "backward": BackwardDirectionStrategy,
            "random": RandomDirectionStrategy,
            
            # 한국어 키워드
            "위": UpDirectionStrategy,
            "위쪽": UpDirectionStrategy,
            "아래": DownDirectionStrategy,
            "아래쪽": DownDirectionStrategy,
            "왼쪽": LeftDirectionStrategy,
            "오른쪽": RightDirectionStrategy,
            "앞": ForwardDirectionStrategy,
            "앞쪽": ForwardDirectionStrategy,
            "뒤": BackwardDirectionStrategy,
            "뒤쪽": BackwardDirectionStrategy,
            "랜덤": RandomDirectionStrategy,
            "무작위": RandomDirectionStrategy
        }
        
    def create_strategy(self, direction_keyword: str) -> Any:
        """
        주어진 방향 키워드에 따라 적절한 방향 전략 객체를 생성합니다.
        
        Args:
            direction_keyword: 방향 키워드(예: "up", "위", "left" 등)
        
        Returns:
            Any: 생성된 방향 전략 객체
        """
        # 소문자로 변환하여 매핑 확인
        direction_key = direction_keyword.lower()
        
        # 매핑된 전략 클래스 가져오기
        strategy_class = self._strategy_classes.get(direction_key)
        
        # 매핑된 클래스가 있으면 인스턴스 생성, 없으면 기본 전략 사용
        if strategy_class:
            return strategy_class()
        else:
            # 기본 전략 클래스 임포트 및 인스턴스 생성
            from modules.goal_inference_components.direction_strategy import DefaultDirectionStrategy
            return DefaultDirectionStrategy()
    
    def get_supported_keywords(self) -> Dict[str, str]:
        """
        지원되는 방향 키워드와 해당 방향 이름을 반환합니다.
        
        Returns:
            Dict[str, str]: 키워드와 방향 이름의 매핑
        """
        result = {}
        for keyword, strategy_class in self._strategy_classes.items():
            strategy = strategy_class()
            result[keyword] = strategy.get_direction_name()
        return result
    
    def create_composite_strategy(self, keywords: list) -> Optional[Any]:
        """
        여러 방향 키워드를 조합한 복합 방향 전략을 생성합니다.
        
        Args:
            keywords: 방향 키워드 목록
        
        Returns:
            Optional[Any]: 생성된 복합 방향 전략 또는 None
        """
        if not keywords:
            from modules.goal_inference_components.direction_strategy import DefaultDirectionStrategy
            return DefaultDirectionStrategy()
        
        combined_vector = np.array([0.0, 0.0, 0.0])
        strategies = []
        
        # 모든 키워드에 대한 전략 생성
        for keyword in keywords:
            strategy = self.create_strategy(keyword)
            
            # DefaultDirectionStrategy 여부 확인을 위해 클래스 이름으로 검사
            if strategy.__class__.__name__ != 'DefaultDirectionStrategy':
                strategies.append(strategy)
                combined_vector += strategy.get_direction_vector()
        
        # 결합된 벡터가 영벡터가 아니면 정규화
        if np.any(combined_vector):
            # 벡터 길이가 0이 아닌 경우에만 정규화
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                combined_vector = combined_vector / norm
                
        # 필요한 클래스와 인터페이스 지연 임포트
        from modules.goal_inference_components.direction_strategy import DirectionStrategy
        
        # 복합 전략을 위한 클래스 생성
        class CompositeDirectionStrategy(DirectionStrategy):
            def __init__(self, vector, strategy_names):
                self._vector = vector
                self._strategy_names = strategy_names
                
            def get_direction_vector(self) -> np.ndarray:
                return self._vector
                
            def get_direction_name(self) -> str:
                return "+".join([s.get_direction_name() for s in self._strategy_names]) if self._strategy_names else "default"
        
        return CompositeDirectionStrategy(combined_vector, strategies)

def adapt_direction(direction: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    기존 방향 문자열을 새 방향 객체 형식으로 변환합니다.
    이미 방향 객체 형식이라면 그대로 반환합니다.
    
    Args:
        direction: 방향 문자열 또는 방향 객체 딕셔너리
        
    Returns:
        Dict[str, Any]: 방향 객체 딕셔너리
    """
    if isinstance(direction, str):
        return {"type": "simple", "value": direction}
    return direction 