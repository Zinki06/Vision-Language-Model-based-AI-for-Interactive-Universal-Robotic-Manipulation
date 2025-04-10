"""
JSON 유틸리티 모듈

텍스트에서 JSON 추출 및 검증 기능 제공
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Union

# 로깅 설정
logger = logging.getLogger("JsonUtils")

def extract_json_from_text(text: str) -> Optional[str]:
    """텍스트에서 JSON 문자열 추출
    
    Args:
        text: JSON을 포함할 수 있는 텍스트
        
    Returns:
        추출된 JSON 문자열 또는 None
    """
    # 코드 블록에서 JSON 추출 시도
    json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # 중괄호로 둘러싸인 부분 추출 시도
    json_match = re.search(r'({.*})', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # 전체 텍스트가 JSON인지 확인
    if text.strip().startswith('{') and text.strip().endswith('}'):
        return text.strip()
    
    return None

def validate_json_structure(
    json_data: Dict[str, Any],
    required_keys: List[str] = None
) -> bool:
    """JSON 구조 유효성 검사
    
    Args:
        json_data: 검증할 JSON 데이터
        required_keys: 필수 키 목록
        
    Returns:
        유효성 여부
    """
    if not isinstance(json_data, dict):
        logger.error("JSON 데이터가 딕셔너리 형식이 아닙니다.")
        return False
    
    if required_keys:
        missing_keys = [key for key in required_keys if key not in json_data]
        if missing_keys:
            logger.error(f"JSON에 필수 키가 없습니다: {', '.join(missing_keys)}")
            return False
    
    return True

def parse_json(text: str, required_keys: List[str] = None) -> Optional[Dict[str, Any]]:
    """텍스트에서 JSON 파싱
    
    Args:
        text: JSON을 포함하는 텍스트
        required_keys: 필수 키 목록
        
    Returns:
        파싱된 JSON 또는 None
    """
    try:
        # JSON 문자열 추출
        json_str = extract_json_from_text(text)
        if not json_str:
            logger.warning("텍스트에서 JSON을 찾을 수 없습니다.")
            return None
        
        # JSON 파싱
        json_data = json.loads(json_str)
        
        # 구조 검증
        if required_keys and not validate_json_structure(json_data, required_keys):
            return None
        
        return json_data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        return None
    
    except Exception as e:
        logger.error(f"JSON 처리 중 오류 발생: {str(e)}")
        return None

def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """JSON 데이터를 파일로 저장
    
    Args:
        data: 저장할 JSON 데이터
        filepath: 저장할 파일 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"JSON 저장 완료: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"JSON 저장 실패: {str(e)}")
        return False

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """파일에서 JSON 데이터 로드
    
    Args:
        filepath: 로드할 파일 경로
        
    Returns:
        로드된 JSON 데이터 또는 None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"JSON 로드 완료: {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"JSON 로드 실패: {str(e)}")
        return None 