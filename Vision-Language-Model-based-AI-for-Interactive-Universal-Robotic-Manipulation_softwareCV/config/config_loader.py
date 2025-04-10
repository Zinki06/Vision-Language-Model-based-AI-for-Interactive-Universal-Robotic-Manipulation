import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(env: str = "default") -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        env: 환경 이름 (default, development, testing)
        
    Returns:
        설정 정보가 담긴 딕셔너리
    """
    # 설정 파일 경로
    config_dir = Path("config")
    config_file = config_dir / f"{env}.yaml"
    
    # 기본 설정 파일 존재 여부 확인
    if not config_file.exists():
        # 설정 파일이 없으면 기본 설정 생성
        default_config = {
            "yolo": {
                "model_name": "yolov8n.pt",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45
            },
            "gemini": {
                "api_key_env": "GEMINI_API_KEY",
                "model": "gemini-pro-vision",
                "timeout": 30
            },
            "webcam": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "demo": {
                "run_benchmark": False,
                "save_results": True
            }
        }
        
        # 설정 디렉토리 생성
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정 파일 저장
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    # 설정 파일 로드
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 