"""
파일 입출력 관련 유틸리티 모듈

이 모듈은 파일 읽기/쓰기, 특히 JSON 데이터 처리와 관련된 함수를 제공합니다.
"""

import os
import json

def save_json(data, file_path):
    """JSON 데이터를 파일로 저장
    
    Args:
        data (dict): 저장할 데이터
        file_path (str): 저장 경로
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path):
    """JSON 파일 로드
    
    Args:
        file_path (str): 파일 경로
        
    Returns:
        dict: 로드된 데이터
    """
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f) 