"""
방향 전략 모듈

이 모듈은 방향 이동을 처리하기 위한 전략 패턴 구현체들을 제공합니다.
각 방향 전략은 해당 방향에 대한 벡터를 반환합니다.
"""

import abc
from typing import Tuple
import numpy as np
import random

# 순환 임포트 제거
# from modules.goal_inference_components.direction_factory import DirectionStrategyFactory


class DirectionStrategy(abc.ABC):
    """
    방향 전략 인터페이스
    
    모든 방향 전략의 기본 인터페이스를 정의합니다.
    """
    
    @abc.abstractmethod
    def get_direction_vector(self) -> np.ndarray:
        """
        방향 벡터를 반환하는 추상 메소드
        
        Returns:
            np.ndarray: 방향 벡터 (3차원 좌표계)
        """
        pass
    
    @abc.abstractmethod
    def get_direction_name(self) -> str:
        """
        방향 이름을 반환하는 추상 메소드
        
        Returns:
            str: 방향 이름
        """
        pass


class DefaultDirectionStrategy(DirectionStrategy):
    """
    기본 방향 전략
    
    알 수 없거나 정의되지 않은 방향에 대한 기본 전략입니다.
    """
    
    def get_direction_vector(self) -> np.ndarray:
        """
        기본 방향 벡터(0, 0, 0)을 반환
        
        Returns:
            np.ndarray: 영벡터 (이동 없음)
        """
        return np.array([0, 0, 0])
    
    def get_direction_name(self) -> str:
        """
        기본 방향 이름 반환
        
        Returns:
            str: "default"
        """
        return "default"


class UpDirectionStrategy(DirectionStrategy):
    """위쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        위쪽 방향 벡터(0, 1, 0)을 반환
        
        Returns:
            np.ndarray: 위쪽 방향 벡터
        """
        return np.array([0, 1, 0])
    
    def get_direction_name(self) -> str:
        """
        위쪽 방향 이름 반환
        
        Returns:
            str: "up"
        """
        return "up"


class DownDirectionStrategy(DirectionStrategy):
    """아래쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        아래쪽 방향 벡터(0, -1, 0)을 반환
        
        Returns:
            np.ndarray: 아래쪽 방향 벡터
        """
        return np.array([0, -1, 0])
    
    def get_direction_name(self) -> str:
        """
        아래쪽 방향 이름 반환
        
        Returns:
            str: "down"
        """
        return "down"


class LeftDirectionStrategy(DirectionStrategy):
    """왼쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        왼쪽 방향 벡터(-1, 0, 0)을 반환
        
        Returns:
            np.ndarray: 왼쪽 방향 벡터
        """
        return np.array([-1, 0, 0])
    
    def get_direction_name(self) -> str:
        """
        왼쪽 방향 이름 반환
        
        Returns:
            str: "left"
        """
        return "left"


class RightDirectionStrategy(DirectionStrategy):
    """오른쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        오른쪽 방향 벡터(1, 0, 0)을 반환
        
        Returns:
            np.ndarray: 오른쪽 방향 벡터
        """
        return np.array([1, 0, 0])
    
    def get_direction_name(self) -> str:
        """
        오른쪽 방향 이름 반환
        
        Returns:
            str: "right"
        """
        return "right"


class ForwardDirectionStrategy(DirectionStrategy):
    """앞쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        앞쪽 방향 벡터(0, 0, 1)을 반환
        
        Returns:
            np.ndarray: 앞쪽 방향 벡터
        """
        return np.array([0, 0, 1])
    
    def get_direction_name(self) -> str:
        """
        앞쪽 방향 이름 반환
        
        Returns:
            str: "forward"
        """
        return "forward"


class BackwardDirectionStrategy(DirectionStrategy):
    """뒤쪽 방향 전략"""
    
    def get_direction_vector(self) -> np.ndarray:
        """
        뒤쪽 방향 벡터(0, 0, -1)을 반환
        
        Returns:
            np.ndarray: 뒤쪽 방향 벡터
        """
        return np.array([0, 0, -1])
    
    def get_direction_name(self) -> str:
        """
        뒤쪽 방향 이름 반환
        
        Returns:
            str: "backward"
        """
        return "backward"


class CompositeDirectionStrategy(DirectionStrategy):
    """
    복합 방향 전략
    
    여러 방향 전략을 조합하여 하나의 방향 전략으로 만듭니다.
    여러 방향의 벡터 합을 정규화하여 최종 방향 벡터를 제공합니다.
    """
    
    def __init__(self, strategies: list[DirectionStrategy], weights: list[float] = None):
        """
        복합 방향 전략 생성자
        
        Args:
            strategies: 조합할 방향 전략 목록
            weights: 각 전략에 적용할 가중치 목록 (None인 경우 모두 1로 설정)
        """
        if not strategies:
            raise ValueError("최소 하나 이상의 전략이 필요합니다")
        
        self.strategies = strategies
        
        if weights is None:
            self.weights = [1.0] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("전략과 가중치의 개수가 일치해야 합니다")
            self.weights = weights
    
    def get_direction_vector(self) -> np.ndarray:
        """
        조합된 방향 벡터 반환
        
        모든 전략의 방향 벡터를 가중치를 적용하여 합산하고 정규화합니다.
        
        Returns:
            np.ndarray: 정규화된 방향 벡터
        """
        result_vector = np.zeros(3)
        
        for strategy, weight in zip(self.strategies, self.weights):
            result_vector += strategy.get_direction_vector() * weight
        
        # 벡터 정규화 (길이가 0인 경우 그대로 반환)
        norm = np.linalg.norm(result_vector)
        if norm > 0:
            result_vector = result_vector / norm
        
        return result_vector
    
    def get_direction_name(self) -> str:
        """
        조합된 방향 이름 반환
        
        Returns:
            str: 조합된 방향 이름
        """
        return "+".join([strategy.get_direction_name() for strategy in self.strategies])


class RandomDirectionStrategy(DirectionStrategy):
    """
    무작위 방향 전략
    
    매 호출마다 무작위 방향 벡터를 생성합니다.
    옵션과 가중치를 지정하면 특정 방향들 중에서 무작위로 선택합니다.
    옵션이 지정되지 않으면 완전 무작위 3D 방향 벡터를 생성합니다.
    """
    
    def __init__(self, options=None):
        """
        무작위 방향 전략 생성자
        
        Args:
            options (dict, optional): 방향 옵션 설정
                {
                    "options": ["left", "right", ...],  # 선택 가능한 방향 목록
                    "weights": [0.3, 0.7, ...]          # 각 방향의 선택 가중치 (옵션)
                }
        """
        self.selected_direction = None  # 선택된 방향 저장 변수
        
        # 기본값 설정
        self.direction_options = []
        self.direction_weights = []
        
        # 옵션이 제공된 경우 파싱
        if options and isinstance(options, dict):
            # 방향 옵션 목록 가져오기
            if "options" in options and isinstance(options["options"], list):
                self.direction_options = options["options"]
                
                # 가중치 가져오기 (제공된 경우)
                if "weights" in options and isinstance(options["weights"], list):
                    # 가중치와 옵션 길이가 일치하는지 확인
                    if len(options["weights"]) == len(self.direction_options):
                        self.direction_weights = options["weights"]
                    else:
                        # 길이가 일치하지 않으면 균등한 가중치 생성
                        self.direction_weights = [1.0/len(self.direction_options)] * len(self.direction_options)
                else:
                    # 가중치가 제공되지 않은 경우 균등한 가중치 생성
                    self.direction_weights = [1.0/len(self.direction_options)] * len(self.direction_options)
    
    def get_direction_vector(self) -> np.ndarray:
        """
        무작위 방향 벡터를 반환합니다.
        
        옵션이 설정된 경우, 지정된 방향 목록에서 가중치에 따라 무작위로 선택합니다.
        옵션이 없는 경우, 완전 무작위 방향 벡터를 생성하고 정규화합니다.
        
        Returns:
            np.ndarray: 정규화된 무작위 방향 벡터 (길이 1)
        """
        # 방향 옵션이 있는 경우
        if self.direction_options:
            # 가중치를 기반으로 방향 선택
            self.selected_direction = random.choices(
                self.direction_options, 
                weights=self.direction_weights, 
                k=1
            )[0]
            
            # 팩토리 대신 방향에 따른 벡터 직접 생성
            direction_vectors = {
                "up": np.array([0, 1, 0]),
                "down": np.array([0, -1, 0]),
                "left": np.array([-1, 0, 0]),
                "right": np.array([1, 0, 0]),
                "front": np.array([0, 0, 1]),
                "forward": np.array([0, 0, 1]),
                "back": np.array([0, 0, -1]),
                "backward": np.array([0, 0, -1]),
                # 한국어 키워드
                "위": np.array([0, 1, 0]),
                "위쪽": np.array([0, 1, 0]),
                "아래": np.array([0, -1, 0]),
                "아래쪽": np.array([0, -1, 0]),
                "왼쪽": np.array([-1, 0, 0]),
                "오른쪽": np.array([1, 0, 0]),
                "앞": np.array([0, 0, 1]),
                "앞쪽": np.array([0, 0, 1]),
                "뒤": np.array([0, 0, -1]),
                "뒤쪽": np.array([0, 0, -1])
            }
            
            # 선택된 방향에 대한 벡터 반환
            return direction_vectors.get(self.selected_direction, np.array([0, 0, 0]))
        
        # 방향 옵션이 없는 경우 - 완전 무작위 3D 벡터 생성
        
        # 성능 최적화: 직접 벡터 생성 (numpy 난수 함수 사용)
        # 범위 [-1, 1]에서 3개의 난수 생성
        random_vector = np.random.uniform(-1, 1, 3)
        
        # 벡터 정규화 (길이가 0인 경우를 대비)
        norm = np.linalg.norm(random_vector)
        if norm > 1e-10:  # 수치 안정성을 위한 임계값 사용
            return random_vector / norm
        else:
            # 영벡터 생성된 경우 (매우 희박한 확률) 기본 벡터 반환
            return np.array([1.0, 0.0, 0.0])
    
    def get_direction_name(self) -> str:
        """
        무작위 방향 이름 반환
        
        선택된 방향이 있는 경우 "(random: 선택된 방향)" 형식으로 반환합니다.
        
        Returns:
            str: 방향 이름
        """
        if self.selected_direction:
            return f"random:{self.selected_direction}"
        return "random"
    
    def get_selected_direction(self) -> str:
        """
        가장 최근에 선택된 방향을 반환합니다.
        
        Returns:
            str: 선택된 방향 이름 또는 None
        """
        return self.selected_direction 