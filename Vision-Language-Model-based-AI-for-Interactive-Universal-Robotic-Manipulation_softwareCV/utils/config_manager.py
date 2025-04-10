"""
설정 관리 유틸리티

다양한 환경(개발, 테스트, 프로덕션)에 따른 설정을 관리하고
설정 파일을 로드하는 기능을 제공합니다.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """설정 관리 클래스
    
    환경(개발, 테스트, 프로덕션)에 따른 설정을 관리하고
    설정 파일을 로드하는 기능을 제공합니다.
    
    Attributes:
        config_dir (Path): 설정 파일 디렉토리
        default_config (Dict): 기본 설정
        environment (str): 현재 환경 ('development', 'testing', 'production')
        config (Dict): 최종 설정 (기본 + 환경별 설정)
    """
    
    def __init__(self, config_dir: Union[str, Path] = 'config', 
                environment: str = None):
        """ConfigManager 초기화
        
        Args:
            config_dir (Union[str, Path]): 설정 파일 디렉토리 (기본값: 'config')
            environment (str, optional): 환경 이름 (기본값: None, 환경변수 참조)
                'development', 'testing', 'production' 중 하나
        """
        # 설정 디렉토리 경로
        self.config_dir = Path(config_dir)
        
        # 환경 설정 (환경변수 또는 기본값)
        self.environment = environment or os.environ.get('APP_ENV', 'development')
        
        # 기본 설정 로드
        self.default_config = self._load_yaml_file('default.yaml') or {}
        
        # 환경별 설정 로드 및 병합
        self.config = self._load_config()
        
        logger.info(f"설정 관리자 초기화: 환경={self.environment}")
    
    def _load_yaml_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """YAML 파일 로드
        
        Args:
            filename (str): 파일명
            
        Returns:
            Optional[Dict[str, Any]]: 로드된 설정 또는 None
        """
        file_path = self.config_dir / filename
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"설정 파일을 찾을 수 없음: {file_path}")
                return None
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {file_path}, {e}")
            return None
    
    def _load_config(self) -> Dict[str, Any]:
        """환경에 따른 설정 로드 및 병합
        
        Returns:
            Dict[str, Any]: 병합된 설정
        """
        # 기본 설정 복사
        config = self.default_config.copy()
        
        # 환경별 설정 파일명
        env_config_file = f"{self.environment}.yaml"
        
        # 환경별 설정 로드
        env_config = self._load_yaml_file(env_config_file) or {}
        
        # 환경별 설정 병합 (재귀적으로)
        self._merge_configs(config, env_config)
        
        # 분석 전용 설정 병합 (존재하는 경우)
        analysis_config = self._load_yaml_file('analysis_config.yaml') or {}
        self._merge_configs(config, analysis_config)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """설정 병합 (재귀적)
        
        base 딕셔너리에 override 딕셔너리의 값을 병합합니다.
        딕셔너리 내부 값도 재귀적으로 병합됩니다.
        
        Args:
            base (Dict[str, Any]): 기본 설정
            override (Dict[str, Any]): 오버라이드 설정
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # 양쪽 모두 딕셔너리인 경우 재귀적으로 병합
                self._merge_configs(base[key], value)
            else:
                # 그 외의 경우 값 교체
                base[key] = value
    
    def get_config(self) -> Dict[str, Any]:
        """전체 설정 반환
        
        Returns:
            Dict[str, Any]: 전체 설정
        """
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """특정 설정값 반환
        
        중첩된 키는 점(.)으로 구분하여 지정 (예: 'webcam.fps')
        
        Args:
            key (str): 설정 키
            default (Any, optional): 기본값
            
        Returns:
            Any: 설정값 또는 기본값
        """
        # 점(.)으로 구분된 키 처리
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """특정 설정값 설정
        
        중첩된 키는 점(.)으로 구분하여 지정 (예: 'webcam.fps')
        중간 딕셔너리가 없는 경우 자동으로 생성합니다.
        
        Args:
            key (str): 설정 키
            value (Any): 설정값
        """
        keys = key.split('.')
        config = self.config
        
        # 마지막 키 이전까지 딕셔너리 경로 생성
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # 마지막 키에 값 설정
        config[keys[-1]] = value
