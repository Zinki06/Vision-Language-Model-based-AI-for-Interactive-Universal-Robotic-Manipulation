#!/usr/bin/env python3
"""
설정 관리자 테스트 유틸리티

이 스크립트는 설정 관리자를 사용하여 다양한 환경의 설정을 로드하고 표시합니다.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_manager import ConfigManager

def main():
    """테스트 메인 함수"""
    parser = argparse.ArgumentParser(description='설정 관리자 테스트 유틸리티')
    parser.add_argument('--config-dir', type=str, default='config',
                        help='설정 디렉토리 경로 (기본값: config)')
    parser.add_argument('--env', type=str, choices=['development', 'testing', 'production'], 
                        default='development',
                        help='실행 환경 (기본값: development)')
    parser.add_argument('--key', type=str,
                        help='특정 설정 키 (예: webcam.fps)')
    args = parser.parse_args()
    
    # 설정 관리자 초기화
    config_manager = ConfigManager(
        config_dir=args.config_dir,
        environment=args.env
    )
    
    # 설정 로드
    config = config_manager.get_config()
    
    # 특정 키를 요청한 경우
    if args.key:
        value = config_manager.get(args.key)
        print(f"\n설정 키 '{args.key}'의 값:")
        print(f" - {value}")
        return
    
    # 전체 설정 출력
    print(f"\n환경 '{args.env}'의 로드된 설정:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
