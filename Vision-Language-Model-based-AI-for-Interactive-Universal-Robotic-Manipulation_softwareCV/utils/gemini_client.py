"""
Google Gemini API 클라이언트

Gemini API에 이미지 분석 요청을 전송하고 결과를 처리하는 기능 제공
"""

import os
import base64
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import google.generativeai as genai
from PIL import Image
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeminiClient")

class GeminiClient:
    """Google Gemini API와 통신하는 클라이언트 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Gemini API 클라이언트 초기화
        
        Args:
            api_key: Google API 키, None일 경우 환경변수에서 로드
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. 환경변수를 확인하세요.")
            return
            
        # Gemini API 초기화
        genai.configure(api_key=self.api_key)
        
        # 기본 설정
        self.config = {
            "model": "gemini-pro-vision",
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048
        }
        
        logger.info("Gemini API 클라이언트가 초기화되었습니다.")
    
    def _encode_image(self, image: Image.Image) -> str:
        """PIL 이미지를 base64 인코딩된 문자열로 변환
        
        Args:
            image: 인코딩할 PIL 이미지
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def analyze_image(
        self, 
        image: Image.Image, 
        prompt: str,
        temperature: float = None,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """이미지를 Gemini API로 분석
        
        Args:
            image: 분석할 PIL 이미지
            prompt: 분석 프롬프트
            temperature: 생성 랜덤성 (0.0-1.0)
            output_format: 출력 형식 (기본 'json')
            
        Returns:
            분석 결과 딕셔너리
        """
        if not self.api_key:
            return {"error": "API 키가 설정되지 않았습니다"}
        
        # 온도 설정 
        temp = temperature if temperature is not None else self.config["temperature"]
        
        try:
            # JSON 출력 지시를 프롬프트에 추가
            if output_format == "json":
                format_instruction = (
                    "\n\n당신의 응답은 파싱 가능한 JSON 형식으로 제공해야 합니다. "
                    "응답은 JSON 오브젝트여야 하며, 마크다운이나 텍스트 설명 없이 "
                    "직접 파싱 가능한 JSON이어야 합니다. 다음 구조를 따르세요:\n"
                    "{\n"
                    '  "objects": [\n'
                    '    {"label": "객체명", "confidence": 0.95, "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]},\n'
                    "    ...\n"
                    "  ],\n"
                    '  "summary": "이미지에 대한 간략한 설명"\n'
                    "}"
                )
                prompt = prompt + format_instruction
            
            # 모델명
            model = self.config["model"]
            
            # API 요청 전송
            logger.info("Gemini API 요청 전송 중...")
            
            response = genai.generate_content(
                model=model,
                contents=[prompt, image],
                generation_config={
                    "temperature": temp,
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "max_output_tokens": self.config["max_tokens"]
                }
            )
            
            # 응답 처리
            if response and hasattr(response, 'text'):
                text_response = response.text
                
                # JSON 파싱 시도
                if output_format == "json":
                    try:
                        # JSON 문자열 추출 (코드 블록에서)
                        import re
                        json_match = re.search(r'```(?:json)?(.*?)```', text_response, re.DOTALL)
                        
                        if json_match:
                            import json
                            json_str = json_match.group(1).strip()
                            result = json.loads(json_str)
                        else:
                            # 코드 블록이 없으면 직접 파싱
                            import json
                            result = json.loads(text_response)
                        
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 파싱 오류: {str(e)}")
                        return {
                            "error": "JSON 파싱 오류",
                            "raw_text": text_response
                        }
                else:
                    return {"text": text_response}
            else:
                logger.error("API 응답 처리 실패")
                return {"error": "API 응답 처리 실패"}
                
        except Exception as e:
            logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
            return {"error": str(e)}

    def detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """이미지에서 객체 감지
        
        Args:
            image: 분석할 PIL 이미지
            
        Returns:
            감지된 객체 정보가 포함된 딕셔너리
        """
        prompt = (
            "이 이미지에서 주요 객체를 감지하고 각 객체의 바운딩 박스 좌표를 제공해주세요. "
            "좌표는 이미지의 왼쪽 상단을 원점(0,0)으로 하는 정규화된 좌표로 제공해주세요. "
            "각 바운딩 박스는 왼쪽 상단, 오른쪽 상단, 오른쪽 하단, 왼쪽 하단 순서로 4개의 점으로 표현해주세요."
        )
        return self.analyze_image(image, prompt)
    
    def analyze_scene(self, image: Image.Image) -> Dict[str, Any]:
        """이미지의 장면 분석
        
        Args:
            image: 분석할 PIL 이미지
            
        Returns:
            장면 분석 결과 딕셔너리
        """
        prompt = (
            "이 이미지를 자세히 분석하고 설명해주세요. 이미지에 포함된 주요 객체, 활동, 배경 등을 포함하여 "
            "이미지에서 주목할만한 모든 내용을 설명해주세요."
        )
        return self.analyze_image(image, prompt)
    
    def analyze_with_prompt(self, image: Image.Image, text_prompt: str) -> Dict[str, Any]:
        """텍스트 프롬프트와 함께 이미지 분석
        
        Args:
            image: 분석할 PIL 이미지
            text_prompt: 분석 프롬프트
            
        Returns:
            분석 결과 딕셔너리
        """
        if not self.api_key:
            return {"error": "API 키가 설정되지 않았습니다"}
            
        try:
            # API 요청 전송
            logger.info("Gemini API 요청 전송 중...")
            
            response = genai.generate_content(
                model=self.config["model"],
                contents=[text_prompt, image],
                generation_config={
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "max_output_tokens": self.config["max_tokens"]
                }
            )
            
            # 응답 처리
            if response and hasattr(response, 'text'):
                return {"text": response.text}
            else:
                logger.error("API 응답 처리 실패")
                return {"error": "API 응답 처리 실패"}
                
        except Exception as e:
            logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
            return {"error": str(e)} 