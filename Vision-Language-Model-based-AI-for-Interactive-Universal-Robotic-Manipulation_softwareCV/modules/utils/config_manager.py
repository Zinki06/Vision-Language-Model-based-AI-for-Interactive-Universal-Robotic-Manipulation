"""
통합 설정 관리 모듈

환경(개발, 테스트, 프로덕션)에 따른 설정 관리 및 명령행 인자, 환경 변수를 통한
애플리케이션 설정을 제공합니다.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigManager:
    """통합 설정 관리 클래스
    
    환경(개발, 테스트, 프로덕션)에 따른 설정을 관리하고 명령행 인자, 환경 변수를 통합합니다.
    
    Attributes:
        config_dir (Path): 설정 파일 디렉토리
        default_config (Dict): 기본 설정
        environment (str): 현재 환경 ('development', 'testing', 'production')
        config (Dict): 최종 설정 (기본 + 환경별 설정)
    """
    
    # 시스템 기본 설정
    DEFAULT_CONFIG = {
        "detection": {
            "yolo_model": "yolov8l.pt",
            "conf_threshold": 0.15,
            "iou_threshold": 0.45,
            "input_size": [640, 640],
            "class_key": "class_name",
            "confidence_key": "confidence",
            "box_key": "bbox"
        },
        "llm": {
            "model": "gpt4o",
            "temperature": 0.2,
            "max_tokens": 1000
        },
        "gesture": {
            "model_path": "models/hand_landmarker.task",
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        },
        "camera": {
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "storage": {
            "output_dir": "output",
            "session_dir_format": "session_%Y%m%d_%H%M%S",
            "snapshot_format": "snapshot_%Y%m%d_%H%M%S.jpg",
            "result_format": "result_%Y%m%d_%H%M%S.jpg",
            "json_format": "result_%Y%m%d_%H%M%S.json"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    def __init__(self, 
                config_dir: Union[str, Path] = 'config', 
                environment: str = None,
                cmd_args: Dict[str, Any] = None):
        """ConfigManager 초기화
        
        Args:
            config_dir (Union[str, Path]): 설정 파일 디렉토리 (기본값: 'config')
            environment (str, optional): 환경 이름 (기본값: None, 환경변수 참조)
            cmd_args (Dict[str, Any], optional): 명령행 인자로 전달된 설정
        """
        # 설정 디렉토리 경로
        self.config_dir = Path(config_dir)
        
        # 환경 설정 (환경변수 또는 기본값)
        self.environment = environment or os.environ.get('APP_ENV', 'development')
        
        # 기본 설정 초기화
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 파일 기반 설정 로드 및 병합
        self._load_file_configs()
        
        # 환경 변수 기반 설정 로드 및 병합
        self._load_env_vars()
        
        # 명령행 인자 기반 설정 로드 및 병합
        if cmd_args:
            self._load_command_args(cmd_args)
        
        logger.info(f"설정 관리자 초기화 완료: 환경={self.environment}")
    
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
                logger.debug(f"설정 파일을 찾을 수 없음: {file_path}")
                return None
        except Exception as e:
            logger.warning(f"설정 파일 로드 오류: {file_path}, {e}")
            return None
    
    def _load_file_configs(self) -> None:
        """파일 기반 설정 로드 및 병합"""
        # 기본 설정 파일 로드
        default_config = self._load_yaml_file('default.yaml') or {}
        self._deep_merge(self.config, default_config)
        
        # 환경별 설정 파일 로드
        env_config_file = f"{self.environment}.yaml"
        env_config = self._load_yaml_file(env_config_file) or {}
        self._deep_merge(self.config, env_config)
        
        # 분석 전용 설정 병합 (존재하는 경우)
        analysis_config = self._load_yaml_file('analysis_config.yaml') or {}
        self._deep_merge(self.config, analysis_config)
    
    def _load_env_vars(self) -> None:
        """환경 변수 기반 설정 로드 및 병합
        
        APP_ 접두사를 가진 환경 변수들을 설정에 반영합니다.
        형식: APP_SECTION__KEY=VALUE (예: APP_DETECTION__CONF_THRESHOLD=0.25)
        """
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                parts = key[4:].lower().split('__')
                if len(parts) == 2:
                    section, option = parts
                    # 값 타입 변환 시도
                    try:
                        # 숫자 변환 시도
                        if value.replace('.', '', 1).isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        # 불리언 변환 시도
                        elif value.lower() in ('true', 'yes', '1'):
                            value = True
                        elif value.lower() in ('false', 'no', '0'):
                            value = False
                    except (ValueError, TypeError):
                        pass  # 변환 실패 시 문자열 그대로 사용
                    
                    # 설정 업데이트
                    if section in self.config:
                        self.config[section][option] = value
                    else:
                        self.config[section] = {option: value}
    
    def _load_command_args(self, args: Dict[str, Any]) -> None:
        """명령행 인자 기반 설정 로드 및 병합
        
        Args:
            args (Dict[str, Any]): 명령행 인자 딕셔너리
        """
        # 명령행 인자 매핑
        mapping = {
            'yolo': ('detection', 'yolo_model'),
            'llm': ('llm', 'model'),
            'conf': ('detection', 'conf_threshold'),
            'iou': ('detection', 'iou_threshold'),
            'gesture_model': ('gesture', 'model_path'),
            'output_dir': ('storage', 'output_dir')
        }
        
        for arg_name, value in args.items():
            if arg_name in mapping and value is not None:
                section, option = mapping[arg_name]
                self.config[section][option] = value
            else:
                # 직접 매핑되지 않은 인자는 디버그 정보로 기록
                logger.debug(f"매핑되지 않은 명령행 인자: {arg_name}={value}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """딕셔너리 깊은 병합
        
        base 딕셔너리에 override 딕셔너리의 값을 재귀적으로 병합합니다.
        
        Args:
            base (Dict[str, Any]): 기본 딕셔너리
            override (Dict[str, Any]): 덮어쓸 딕셔너리
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # 양쪽 모두 딕셔너리인 경우 재귀적으로 병합
                self._deep_merge(base[key], value)
            else:
                # 그 외의 경우 값 교체
                base[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """전체 설정 반환
        
        Returns:
            Dict[str, Any]: 전체 설정 딕셔너리
        """
        return self.config
    
    def get(self, path: str, default: Any = None) -> Any:
        """설정값 조회
        
        점 표기법을 사용하여 중첩된 설정 접근 (예: 'detection.yolo_model')
        
        Args:
            path (str): 설정 경로
            default (Any, optional): 기본값
            
        Returns:
            Any: 설정값 또는 기본값
        """
        parts = path.split('.')
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """섹션 전체 설정 반환
        
        Args:
            section (str): 섹션 이름
            
        Returns:
            Dict[str, Any]: 섹션 설정 또는 빈 딕셔너리
        """
        return self.config.get(section, {})
    
    def set(self, path: str, value: Any) -> None:
        """설정값 설정
        
        점 표기법을 사용하여 중첩된 설정 접근 (예: 'detection.yolo_model')
        경로의 중간 딕셔너리가 없는 경우 자동으로 생성
        
        Args:
            path (str): 설정 경로
            value (Any): 설정값
        """
        parts = path.split('.')
        config = self.config
        
        # 마지막 키 이전까지 경로 생성
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
        
        # 마지막 키에 값 설정
        config[parts[-1]] = value
    
    def save_to_file(self, filename: str = None) -> bool:
        """현재 설정을 파일로 저장
        
        Args:
            filename (str, optional): 파일명 (기본값: <environment>.yaml)
            
        Returns:
            bool: 저장 성공 여부
        """
        if not filename:
            filename = f"{self.environment}.yaml"
        
        file_path = self.config_dir / filename
        try:
            # 디렉토리가 없는 경우 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"설정이 저장됨: {file_path}")
            return True
        except Exception as e:
            logger.error(f"설정 저장 오류: {file_path}, {e}")
            return False


# 싱글톤 인스턴스
_instance = None

def get_config_manager(config_dir: Union[str, Path] = 'config', 
                      environment: str = None,
                      cmd_args: Dict[str, Any] = None) -> ConfigManager:
    """ConfigManager 싱글톤 인스턴스 반환
    
    이미 초기화된 인스턴스가 있으면 재사용하고, 없으면 새로 생성합니다.
    
    Args:
        config_dir (Union[str, Path], optional): 설정 파일 디렉토리
        environment (str, optional): 환경 이름
        cmd_args (Dict[str, Any], optional): 명령행 인자
        
    Returns:
        ConfigManager: 설정 관리자 인스턴스
    """
    global _instance
    if _instance is None:
        _instance = ConfigManager(config_dir, environment, cmd_args)
    return _instance 