"""
YOLO 모델 설정 관리 모듈
"""

YOLO_MODELS = {
    "yolov8n": {
        "path": "yolov8n.pt",
        "input_size": (1280, 1280),
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "description": "가벼운 모델, 빠른 추론 속도"
    },
    "yolov8l": {
        "path": "yolov8l.pt",
        "input_size": (1280, 1280),
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "description": "고정확도 모델, 중간 추론 속도"
    }
}

def get_model_config(model_name):
    """모델 설정 반환"""
    return YOLO_MODELS.get(model_name, YOLO_MODELS["yolov8n"]) 