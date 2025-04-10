import os
import logging
from typing import Optional

from modules.llm_interface import LLMInterface
from modules.gpt4o_interface import GPT4oInterface

class GPT4oBuilder:
    """
    GPT-4o 인터페이스 객체를 생성하는 빌더 클래스
    GPT-4o API를 쉽게 사용할 수 있도록 인스턴스 생성을 담당함
    """
    
    @staticmethod
    def create_gpt4o(
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ) -> LLMInterface:
        """
        GPT-4o 인터페이스 객체 생성
        
        Args:
            model_name: 사용할 모델 이름 (기본값: gpt-4o)
            api_key: API 키 (None이면 환경 변수에서 가져옴)
            temperature: 생성 다양성 조절 파라미터 (낮을수록 결정적 응답)
            logger: 로깅을 위한 logger 객체
            
        Returns:
            GPT4oInterface 객체
        """
        logger = logger or logging.getLogger(__name__)
        
        # API 키 확인
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        logger.info(f"GPT-4o 인터페이스 생성: {model_name}")
        return GPT4oInterface(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            logger=logger
        ) 