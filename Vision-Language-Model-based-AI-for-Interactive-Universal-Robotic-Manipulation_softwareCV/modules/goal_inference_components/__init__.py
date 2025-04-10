"""
목표 지점 추론 컴포넌트 패키지

이 패키지는 GoalPointInferenceEngine의 책임을 분리한 개별 컴포넌트들을 포함합니다.
각 컴포넌트는 단일 책임 원칙(SRP)에 따라 특정 기능에 집중합니다.

Components:
- DetectionManager: 객체 감지 관련 로직
- DepthProcessor: 깊이 맵 처리 및 3D 좌표 계산
- GestureHandler: 제스처 인식 및 해석
- LLMOrchestrator: LLM 상호작용 관리
- PlacementCalculator: 목표 위치 계산
- ResultAggregator: 결과 데이터 통합 및 생성
- DirectionStrategy: 방향 벡터 생성을 위한 전략 패턴 클래스들
- DirectionStrategyFactory: 방향 전략 객체를 생성하는 팩토리
"""

from .detection_manager import DetectionManager
from .depth_processor import DepthProcessor
from .gesture_handler import GestureHandler
from .llm_orchestrator import LLMOrchestrator
from .placement_calculator import PlacementCalculator
from .result_aggregator import ResultAggregator

# 방향 관련 클래스들은 순환 참조 방지를 위해 직접 임포트하지 않음
# 필요한 경우 직접 하위 모듈을 임포트하여 사용

__all__ = [
    'DetectionManager',
    'DepthProcessor',
    'GestureHandler',
    'LLMOrchestrator',
    'PlacementCalculator',
    'ResultAggregator',
] 