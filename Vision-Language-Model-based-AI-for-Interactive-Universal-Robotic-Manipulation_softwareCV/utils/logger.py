import logging
import os
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
        
    Returns:
        설정된 로거 객체
    """
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 핸들러가 이미 존재하면 추가하지 않음
    if not logger.handlers:
        # 파일 핸들러
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger 