"""
음성 녹음 및 STT 관련 의존성 관리 모듈

이 모듈은 음성 녹음 및 STT 기능에 필요한 의존성을 확인하고 관리합니다.
"""
import logging

logger = logging.getLogger("AudioDependencies")

# 음성 지원 여부 판단 변수
AUDIO_SUPPORT = True
STT_SUPPORT = True
OPENAI_NEW_API = False
OPENAI_VERSION = None

# PyAudio, wave, pydub 등 녹음 관련 의존성 확인
try:
    import pyaudio
    import wave
    from pydub import AudioSegment
    logger.info("음성 녹음 의존성 확인 완료: PyAudio, wave, pydub")
except ImportError as e:
    AUDIO_SUPPORT = False
    error_module = str(e).split("'")[1] if "'" in str(e) else "알 수 없는 모듈"
    logger.warning(f"음성 지원 라이브러리({error_module})가 설치되지 않았습니다. 음성 녹음 기능이 비활성화됩니다.")
    logger.warning("필요한 패키지를 설치하려면: pip install pyaudio pydub")

# OpenAI API 의존성 확인 (신규 API 형식 우선 시도)
try:
    try:
        # 새로운 OpenAI 버전 확인 (1.0.0 이상)
        from openai import OpenAI
        import openai
        OPENAI_VERSION = openai.__version__
        OPENAI_NEW_API = True
        STT_SUPPORT = True
        logger.info(f"STT 의존성 확인 완료: OpenAI API (새 버전 {OPENAI_VERSION})")
    except ImportError:
        # 이전 버전 확인
        import openai
        OPENAI_VERSION = openai.__version__
        OPENAI_NEW_API = False
        STT_SUPPORT = True
        logger.info(f"STT 의존성 확인 완료: OpenAI API (이전 버전 {OPENAI_VERSION})")
except ImportError:
    STT_SUPPORT = False
    logger.warning("OpenAI 패키지가 설치되지 않았습니다. STT 기능이 비활성화됩니다.")
    logger.warning("필요한 패키지를 설치하려면: pip install openai")

def get_audio_support_status():
    """
    음성 녹음 지원 여부 반환
    
    Returns:
        bool: 음성 녹음 지원 여부
    """
    return AUDIO_SUPPORT

def get_stt_support_status():
    """
    STT 지원 여부 반환
    
    Returns:
        bool: STT 지원 여부
    """
    return STT_SUPPORT

def get_openai_api_version():
    """
    OpenAI API 버전 정보 및 새 API 사용 여부 반환
    
    Returns:
        dict: OpenAI API 버전 정보
    """
    return {
        "version": OPENAI_VERSION,
        "is_new_api": OPENAI_NEW_API
    }

def check_dependencies():
    """
    모든 의존성 확인 및 상태 반환
    
    Returns:
        dict: 의존성 상태 정보
    """
    return {
        "audio_support": AUDIO_SUPPORT,
        "stt_support": STT_SUPPORT,
        "pyaudio_version": pyaudio.__version__ if AUDIO_SUPPORT else None,
        "openai_version": OPENAI_VERSION,
        "openai_new_api": OPENAI_NEW_API
    } 