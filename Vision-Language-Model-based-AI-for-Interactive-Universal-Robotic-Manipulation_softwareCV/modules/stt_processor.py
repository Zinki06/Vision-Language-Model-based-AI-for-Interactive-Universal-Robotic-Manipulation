"""
OpenAI Whisper API를 활용한 음성-텍스트 변환 모듈

이 모듈은 녹음된 MP3 파일을 OpenAI Whisper API를 통해 텍스트로 변환하는 기능을 제공합니다.
"""
import os
import logging
import tempfile
from typing import Optional, Dict, Any, Union
from pathlib import Path

# 의존성 확인
from .audio_dependencies import get_stt_support_status

# 의존성 체크
STT_SUPPORT = get_stt_support_status()
if STT_SUPPORT:
    try:
        # openai 1.0.0 이상 버전 사용
        from openai import OpenAI
        OPENAI_NEW_API = True
    except ImportError:
        # 이전 버전 호환성 유지
        import openai
        OPENAI_NEW_API = False

logger = logging.getLogger("STTProcessor")

class STTProcessor:
    """
    음성-텍스트 변환 클래스
    
    OpenAI Whisper API를 이용하여 MP3 파일을 텍스트로 변환합니다.
    """
    
    def __init__(self, language: str = "ko", api_key: Optional[str] = None):
        """
        STTProcessor 초기화
        
        Args:
            language (str, optional): 인식할 언어 (기본값: "ko")
            api_key (Optional[str], optional): OpenAI API 키 (기본값: 환경 변수에서 로드)
        
        Raises:
            ImportError: OpenAI 패키지가 설치되지 않은 경우
            ValueError: API 키가 설정되지 않은 경우
        """
        if not STT_SUPPORT:
            logger.error("OpenAI 패키지가 설치되지 않았습니다. pip install openai를 실행하세요.")
            raise ImportError("OpenAI 패키지가 설치되지 않았습니다.")
        
        self.language = language
        
        # API 키 설정
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            error_msg = "OpenAI API 키가 설정되지 않았습니다. API 키를 인자로 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # OpenAI 클라이언트 초기화 (새 API 방식)
        if OPENAI_NEW_API:
            self.client = OpenAI(api_key=self.api_key)
        else:
            # 이전 방식 호환성 유지
            openai.api_key = self.api_key
        
        logger.info(f"STTProcessor 초기화 완료: 언어={language}")
    
    def transcribe(self, audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        음성 파일을 텍스트로 변환
        
        Args:
            audio_file_path (str): 변환할 음성 파일 경로
            language (Optional[str], optional): 인식할 언어 (기본값: 초기화 시 설정한 언어)
        
        Returns:
            Dict[str, Any]: 변환 결과
                {
                    "success": bool,
                    "text": str,
                    "error": Optional[str]
                }
        """
        if not os.path.exists(audio_file_path):
            error_msg = f"파일이 존재하지 않습니다: {audio_file_path}"
            logger.error(error_msg)
            return {"success": False, "text": "", "error": error_msg}
        
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            error_msg = "빈 오디오 파일입니다."
            logger.error(error_msg)
            return {"success": False, "text": "", "error": error_msg}
        
        # 파일 크기 로깅 (MB 단위)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"음성 파일 변환 시작: {audio_file_path} ({file_size_mb:.2f} MB)")
        
        # 사용할 언어 설정
        use_language = language or self.language
        
        try:
            # OpenAI API 호출
            with open(audio_file_path, "rb") as audio_file:
                if OPENAI_NEW_API:
                    # 새로운 API 방식 (openai >= 1.0.0)
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language=use_language
                    )
                    # 새 API는 직접 텍스트를 반환
                    transcribed_text = transcript
                else:
                    # 이전 API 방식 (openai < 1.0.0)
                    response = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        language=use_language
                    )
                    # 이전 API는 딕셔너리를 반환
                    transcribed_text = response.get("text", "")
            
            # 응답에서 텍스트 추출
            if not transcribed_text:
                logger.warning("API 응답에 텍스트가 없습니다.")
                return {"success": True, "text": "", "error": None}
            
            logger.info(f"음성 변환 완료: 길이={len(transcribed_text)} 문자")
            return {"success": True, "text": transcribed_text, "error": None}
            
        except Exception as e:
            error_msg = f"음성 변환 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "text": "", "error": error_msg}
    
    def transcribe_with_retry(self, audio_file_path: str, max_retries: int = 3, 
                            language: Optional[str] = None) -> Dict[str, Any]:
        """
        재시도 메커니즘이 포함된 음성-텍스트 변환
        
        Args:
            audio_file_path (str): 변환할 음성 파일 경로
            max_retries (int, optional): 최대 재시도 횟수 (기본값: 3)
            language (Optional[str], optional): 인식할 언어
        
        Returns:
            Dict[str, Any]: 변환 결과
        """
        result = {"success": False, "text": "", "error": "최대 재시도 횟수 초과"}
        
        for retry in range(max_retries):
            try:
                result = self.transcribe(audio_file_path, language)
                if result["success"]:
                    return result
                
                logger.warning(f"변환 실패 (시도 {retry+1}/{max_retries}): {result['error']}")
                # 네트워크 오류 등의 경우 잠시 대기
                import time
                time.sleep(1 * (retry + 1))  # 지수 백오프
                
            except Exception as e:
                logger.error(f"재시도 중 예외 발생 (시도 {retry+1}/{max_retries}): {str(e)}")
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        현재 설정 정보 반환
        
        Returns:
            Dict[str, Any]: 설정 정보
        """
        return {
            "language": self.language,
            "api_key_set": bool(self.api_key),
        } 