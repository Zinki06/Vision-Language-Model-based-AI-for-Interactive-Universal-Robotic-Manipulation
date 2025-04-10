"""
설정 관리 관련 유틸리티 모듈

이 모듈은 시스템 설정과 관련된 클래스 및 함수를 제공합니다.
"""

class SystemConfig:
    """시스템 설정 중앙화"""
    
    DEFAULT_CONFIG = {
        "detection_keys": {
            "class": "class_name",  # 표준 키 이름
            "confidence": "confidence",
            "box": "box"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    def __init__(self, custom_config=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            self._update_config(custom_config)
    
    def _update_config(self, custom_config):
        """설정 업데이트"""
        for section, values in custom_config.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
            else:
                self.config[section] = values
    
    def get_key_name(self, module, key):
        """모듈별 키 이름 조회"""
        module_config = self.config.get(module, {})
        return module_config.get(key, self.config["detection_keys"].get(key)) 