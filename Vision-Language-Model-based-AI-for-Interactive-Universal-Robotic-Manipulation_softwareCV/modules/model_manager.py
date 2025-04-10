"""
모델 다운로드 및 관리 모듈
"""
import os
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

logger = logging.getLogger("ModelManager")

class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self, model_dir="models"):
        """
        모델 관리자 초기화
        
        Args:
            model_dir: 모델 저장 경로
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        
    def get_model(self, model_name):
        """
        모델 로드 또는 다운로드
        
        Args:
            model_name: 모델 이름 (예: 'yolov8l')
            
        Returns:
            모델 경로
        """
        if model_name in self.models:
            return self.models[model_name]
            
        model_path = self.model_dir / f"{model_name}.pt"
        
        # 모델이 이미 다운로드되어 있는지 확인
        if not model_path.exists():
            logger.info(f"모델 다운로드 중: {model_name}")
            # YOLO 모델 다운로드 (ultralytics에서 자동 처리)
            _ = YOLO(model_name)
            
        self.models[model_name] = str(model_path)
        return str(model_path)
        
    def check_device_compatibility(self):
        """
        장치 호환성 확인
        
        Returns:
            최적 장치 (str): "cuda", "mps", "cpu" 중 하나
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu" 