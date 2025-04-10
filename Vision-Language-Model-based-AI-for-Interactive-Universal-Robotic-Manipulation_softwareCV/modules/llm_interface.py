from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import logging
import json
import re

class LLMInterface(ABC):
    """
    LLM API 호출을 위한 추상 인터페이스 클래스
    현재 구현체는 OpenAI GPT-4o만 지원합니다.
    """
    
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        추상 LLM 인터페이스 초기화
        
        Args:
            model_name: 사용할 LLM 모델 이름 (예: gpt-4o)
            api_key: API 키 (없으면 환경 변수에서 가져옴)
            timeout: API 요청 타임아웃 (초)
            temperature: 생성 다양성 조절 파라미터 (낮을수록 결정적 응답)
            logger: 로깅을 위한 logger 객체
        """
        pass
    
    @abstractmethod
    def classify_objects(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        YOLO로 감지된 객체들에 대해 LLM을 사용하여 더 정확한 분류 및 설명 생성
        
        Args:
            image: 원본 이미지 (numpy 배열)
            detections: YOLO에서 감지한 객체 목록
        
        Returns:
            LLM이 식별 및 설명한 객체 목록
        """
        pass
    
    @abstractmethod
    def analyze_specific_object(self, image: np.ndarray, bbox: List[float], query: str) -> Dict[str, Any]:
        """
        특정 바운딩 박스 내 객체에 대한 상세 분석 수행
        
        Args:
            image: 원본 이미지 (numpy 배열)
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            query: 질의 내용 (예: "이 물체는 무엇인가요?")
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        pass
    
    @abstractmethod
    def select_objects_from_command(self, image: np.ndarray, command: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        이미지와 사용자 명령어를 바탕으로 작업에 필요한 객체들을 선택
        
        Args:
            image: 이미지 (numpy 배열)
            command: 사용자 명령어 (예: "빨간 공을 파란 상자 위에 올려줘")
            detections: 이미지에서 감지된 객체 목록
            
        Returns:
            선택된 객체 정보와 목표 위치를 포함한 딕셔너리
        """
        pass
    
    @abstractmethod
    def detect_goal_points(self, image: np.ndarray) -> Dict[str, Any]:
        """
        LLM을 사용하여 이미지에서 목표지점(Goal Point)의 바운딩 박스 좌표를 추출
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            목표지점 좌표 및 메타 정보를 담은 딕셔너리
        """
        pass
    
    @abstractmethod
    def infer_goal_point(self, target_object: Dict[str, Any], reference_object: Dict[str, Any], user_prompt: str = None) -> Dict[str, Any]:
        """
        타겟 객체와 레퍼런스 객체의 정보를 기반으로 목표지점을 추론합니다.
        
        Args:
            target_object: 타겟 객체 정보 (바운딩 박스, 클래스, 깊이 통계 포함)
            reference_object: 레퍼런스 객체 정보 (바운딩 박스, 클래스, 깊이 통계 포함)
            user_prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
            
        Returns:
            Dict: 목표지점 좌표와 추론 근거를 포함하는 딕셔너리
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.") 
        
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 부분을 추출하고 파싱
        
        Args:
            response_text: LLM의 텍스트 응답
            
        Returns:
            파싱된 JSON 데이터 (딕셔너리)
            
        Raises:
            json.JSONDecodeError: JSON 파싱 실패 시 발생
        """
        if not response_text or not isinstance(response_text, str):
            raise ValueError("응답 텍스트가 비어있거나, 문자열이 아닙니다.")
            
        # 1. markdown 코드 블록 제거
        clean_text = response_text.strip()
        if "```json" in clean_text:
            # JSON 코드 블록 추출
            pattern = r"```json\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, clean_text)
            if matches:
                clean_text = matches[0].strip()
        elif "```" in clean_text:
            # 일반 코드 블록 추출
            pattern = r"```\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, clean_text)
            if matches:
                clean_text = matches[0].strip()
                
        # 2. 첫 번째 { 와 마지막 } 사이의 내용 추출 (일부 응답에서 JSON 앞뒤에 텍스트가 있는 경우)
        if '{' in clean_text and '}' in clean_text:
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1
            clean_text = clean_text[start_idx:end_idx]
                
        # 3. JSON 파싱
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"JSON 파싱 오류: {e}, 원본 텍스트: {clean_text[:100]}...")
            raise 