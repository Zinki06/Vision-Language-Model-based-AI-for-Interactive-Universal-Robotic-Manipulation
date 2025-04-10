import os
import time
import logging
from typing import Dict, List, Union, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import cv2

from config.yolo_config import get_model_config

class YOLODetector:
    """YOLOv8 객체 탐지를 위한 클래스.
    
    YOLOv8 모델을 로드하고 이미지에서 객체를 탐지하는 기능을 제공합니다.
    
    Attributes:
        model_name (str): 사용 중인 YOLO 모델 이름
        conf_threshold (float): 객체 감지 신뢰도 임계값
        iou_threshold (float): NMS IoU 임계값 
        input_size (Tuple[int, int]): 입력 이미지 크기
        device (str): 추론에 사용할 장치 (cpu, cuda, mps)
        model (YOLO): 로드된 YOLO 모델 인스턴스
        class_names (Dict[int, str]): 클래스 ID와 이름 매핑
        logger (logging.Logger): 로거 인스턴스
    """
    
    def __init__(
        self, 
        model_name: str = "yolov8n.pt", 
        conf_threshold: Optional[float] = None, 
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None
    ) -> None:
        """YOLODetector 초기화.
        
        Args:
            model_name: 사용할 YOLOv8 모델 이름 또는 경로. 기본값은 "yolov8n.pt".
            conf_threshold: 객체 감지 신뢰도 임계값. None이면 모델 설정에서 가져옴. 
            iou_threshold: NMS IoU 임계값. None이면 모델 설정에서 가져옴.
            device: 추론에 사용할 장치 (None일 경우 자동 선택).
        
        Raises:
            Exception: 모델 로드 중 오류가 발생한 경우.
        """
        self.logger = logging.getLogger("YOLODetector")
        # 로깅 레벨을 WARNING으로 설정하여 INFO 로그 출력 중단
        self.logger.setLevel(logging.WARNING)
        
        # 모델 이름 표준화
        self.model_name = self._standardize_model_name(model_name)
        
        # 모델 설정 로드
        model_name_base = self.model_name.replace('.pt', '')
        model_config = get_model_config(model_name_base)
        
        # 파라미터 설정 (인자로 제공된 값 우선)
        self.conf_threshold = conf_threshold if conf_threshold is not None else model_config["conf_threshold"]
        self.iou_threshold = iou_threshold if iou_threshold is not None else model_config["iou_threshold"]
        self.input_size = model_config["input_size"]
        
        # 장치 설정
        self.device = self._determine_device(device)
        self.logger.debug(f"장치 선택: {self.device}")
        
        # 모델 로드
        self._load_model()
        
        # 클래스 매핑 로드
        self.class_names = self.model.names
        self.logger.debug(f"클래스 수: {len(self.class_names)}")
    
    def _standardize_model_name(self, model_name: str) -> str:
        """모델 이름 표준화.
        
        Args:
            model_name: 모델 이름 또는 경로
            
        Returns:
            표준화된 모델 이름 (.pt 확장자 포함)
        """
        if not model_name.endswith('.pt'):
            return f"{model_name}.pt"
        return model_name
    
    def _determine_device(self, device: Optional[str]) -> str:
        """추론에 사용할 장치 결정.
        
        Args:
            device: 사용자가 지정한 장치 (None이면 자동 선택)
            
        Returns:
            사용할 장치 문자열 ('mps', 'cuda', 또는 'cpu')
        """
        if device is not None:
            return device
            
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """YOLO 모델 로드.
        
        모델을 로드하고 기본 정보를 로깅합니다.
        
        Raises:
            Exception: 모델 로드 실패시 예외 발생
        """
        try:
            self.model = YOLO(self.model_name)
            self.logger.debug(f"YOLOv8 모델 로드 완료: {self.model_name}")
            
            # 추가 모델 정보 로깅
            self.logger.debug(f"모델 설정: conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}")
            self.logger.debug(f"추론 장치: {self.device}")
            if hasattr(self.model, 'task') and self.model.task is not None:
                self.logger.debug(f"모델 타입: {self.model.task}")
            
            # 클래스 목록 중 일부만 로깅 (전체는 너무 길 수 있음)
            sample_classes = list(self.model.names.items())[:10]
            self.logger.debug(f"객체 클래스 샘플: {sample_classes}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """이미지 전처리.
        
        다양한 형식의 입력 이미지를 YOLO 모델에 적합한 형식으로 변환합니다.
        
        Args:
            image: 입력 이미지 (경로, numpy 배열 또는 PIL 이미지)
            
        Returns:
            전처리된 이미지 (numpy 배열)
            
        Raises:
            FileNotFoundError: 이미지 파일을 찾을 수 없는 경우
            ValueError: 지원되지 않는 이미지 타입인 경우
        """
        if isinstance(image, str):
            # 이미지 경로인 경우
            if not os.path.exists(image):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {image}")
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            # BGR → RGB 변환 (OpenCV 이미지인 경우)
            if image.shape[2] == 3:  # 3채널 이미지인 경우만
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_rgb)
            else:
                img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 원본 해상도 유지 (리사이징 없음)
        img_array = np.array(img)
        
        # 픽셀 값 범위 로깅 (디버깅용)
        min_val, max_val = np.min(img_array), np.max(img_array)
        self.logger.debug(f"전처리 후 이미지 픽셀값 범위: min={min_val}, max={max_val}")
        
        return img_array
    
    def detect(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """이미지에서 객체 감지 수행.
        
        Args:
            image: 입력 이미지 (파일 경로, numpy 배열 또는 PIL 이미지)
            
        Returns:
            감지된 객체 목록 (각 객체는 클래스, 바운딩 박스, 신뢰도 등 포함)
        """
        # 입력 이미지 처리 및 정보 로깅
        self._log_input_image_info(image)
        
        # 이미지 전처리 (원본 해상도 유지)
        processed_image = self.preprocess_image(image)
        
        # 이미지 크기 정보 추출
        height, width = processed_image.shape[:2]
        
        # YOLO 추론 수행
        detections = self._perform_inference(processed_image)
        
        # 수직 위치 기반 가중치 적용
        detections = self.process_detections(detections, height)
        
        # 결과 요약 로깅
        self._log_detection_summary(detections)
        
        return detections
    
    def _log_input_image_info(self, image: Union[str, np.ndarray, Image.Image]) -> None:
        """입력 이미지 정보 로깅.
        
        Args:
            image: 입력 이미지 (파일 경로, numpy 배열 또는 PIL 이미지)
        """
        image_type = type(image).__name__
        
        if isinstance(image, str):
            # 이미지 경로인 경우
            self.logger.debug(f"이미지 파일 경로로부터 감지 수행: {image}")
        elif isinstance(image, np.ndarray):
            # Numpy 배열인 경우
            self.logger.debug(f"Numpy 배열 이미지로부터 감지 수행: 형태={image.shape}, 타입={image.dtype}")
        elif isinstance(image, Image.Image):
            # PIL 이미지인 경우
            self.logger.debug(f"PIL 이미지로부터 감지 수행: 크기={image.size}, 모드={image.mode}")
        else:
            self.logger.warning(f"알 수 없는 이미지 타입: {image_type}")
    
    def _perform_inference(self, processed_image: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO 모델로 추론 수행.
        
        Args:
            processed_image: 전처리된 이미지
            
        Returns:
            감지된 객체 목록
        """
        self.logger.debug(f"YOLO 추론 시작: conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}")
        
        try:
            results = self.model.predict(
                source=processed_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                imgsz=self.input_size,
                verbose=False  # 로그 출력 비활성화
            )
            self.logger.debug(f"YOLO 추론 완료: {len(results)} 결과")
        except Exception as e:
            self.logger.error(f"YOLO 추론 중 오류 발생: {str(e)}")
            return []  # 오류 발생 시 빈 목록 반환
        
        # 결과를 표준 형식으로 변환
        detections = self._process_detection_results(results)
        
        return detections
    
    def _process_detection_results(self, results) -> List[Dict[str, Any]]:
        """YOLO 결과를 표준 형식으로 변환.
        
        Args:
            results: YOLO 모델의 추론 결과
            
        Returns:
            표준화된 감지 결과 목록
        """
        detections = []
        
        for result_idx, result in enumerate(results):
            boxes = result.boxes
            
            # 각 결과별 감지된 객체 수 로깅
            self.logger.debug(f"결과 {result_idx+1}/{len(results)}: {len(boxes)} 객체 감지됨")
            
            # 감지된 객체가 없을 경우 로깅
            if len(boxes) == 0:
                self.logger.debug(f"결과 {result_idx+1}에서 감지된 객체 없음")
                continue
            
            for i, box in enumerate(boxes):
                try:
                    detection = self._extract_detection_info(i, box)
                    if detection:
                        detections.append(detection)
                except Exception as e:
                    self.logger.error(f"객체 감지 정보 처리 중 오류 발생: {str(e)}")
                    continue  # 오류 발생 시 다음 박스로 진행
        
        return detections
    
    def _extract_detection_info(self, idx: int, box) -> Optional[Dict[str, Any]]:
        """개별 감지 객체 정보 추출.
        
        Args:
            idx: 객체 인덱스
            box: YOLO 결과의 박스 객체
            
        Returns:
            감지 객체 정보 딕셔너리 또는 None (유효하지 않은 경우)
        """
        try:
            # 바운딩 박스 좌표 (x1, y1, x2, y2)
            if box.xyxy.size(0) > 0:  # 박스 좌표가 존재하는지 확인
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            else:
                self.logger.warning(f"유효하지 않은 바운딩 박스 좌표: idx={idx}")
                return None
            
            # 신뢰도 및 클래스
            if box.conf.size(0) > 0 and box.cls.size(0) > 0:  # 신뢰도와 클래스 ID가 존재하는지 확인
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # 유효한 클래스 ID인지 확인
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                else:
                    self.logger.warning(f"알 수 없는 클래스 ID: {class_id}, 기본값 사용")
                    class_name = "unknown"
            else:
                self.logger.warning(f"유효하지 않은 신뢰도 또는 클래스 정보: idx={idx}")
                return None
            
            # 객체 정보 딕셔너리
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2)
            }
            
            # 객체 정보 로깅 (debug 레벨)
            self.logger.debug(
                f"감지된 객체 {idx+1}: 클래스={class_name}, 신뢰도={confidence:.4f}, " +
                f"바운딩 박스=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )
            
            return detection
        except IndexError as e:
            self.logger.error(f"바운딩 박스 처리 중 인덱스 오류 발생: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"객체 정보 추출 중 오류 발생: {str(e)}")
            return None
    
    def _log_detection_summary(self, detections: List[Dict[str, Any]]) -> None:
        """감지 결과 요약 로깅.
        
        Args:
            detections: 감지된 객체 목록
        """
        if detections:
            class_counts = {}
            for det in detections:
                class_name = det["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            class_summary = ", ".join([f"{cls}({count})" for cls, count in class_counts.items()])
            self.logger.debug(f"감지 결과 요약: 총 {len(detections)}개 객체 - {class_summary}")
        else:
            self.logger.debug("감지된 객체가 없습니다")
    
    def benchmark(self, num_runs: int = 10) -> Dict[str, float]:
        """모델 성능 벤치마크 실행.
        
        Args:
            num_runs: 실행할 횟수, 기본값은 10
            
        Returns:
            벤치마크 결과 (평균 FPS, 추론 시간 등)
        """
        # 테스트용 이미지 생성 (640x640 크기)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 메모리 사용량 측정을 위한 초기 메모리
        initial_memory = self._get_initial_memory_usage()
        
        # 추론 시간 측정
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.detect(test_img)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        # 결과 계산
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_fps = 1.0 / avg_inference_time
        
        # 메모리 사용량 측정
        memory_usage = self._get_memory_usage_delta(initial_memory)
        
        return {
            "avg_fps": avg_fps,
            "avg_inference_time": avg_inference_time,
            "memory_usage": memory_usage
        }
    
    def _get_initial_memory_usage(self) -> float:
        """초기 메모리 사용량 측정.
        
        Returns:
            현재 메모리 사용량 (MB)
        """
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def _get_memory_usage_delta(self, initial_memory: float) -> float:
        """메모리 사용량 변화 측정.
        
        Args:
            initial_memory: 초기 메모리 사용량 (MB)
            
        Returns:
            메모리 사용량 변화 (MB)
        """
        if torch.cuda.is_available() and self.device == "cuda":
            final_memory = torch.cuda.memory_allocated() / 1024 / 1024
            return final_memory - initial_memory
        
        # CPU나 MPS는 정확한 메모리 측정이 어려움
        return 0
    
    def process_detections(self, detections: List[Dict[str, Any]], image_height: int) -> List[Dict[str, Any]]:
        """
        감지된 객체에 수직 위치 기반 가중치를 적용하여 상단 객체에 우선권 부여
        
        Args:
            detections: YOLO에서 감지된 객체 목록
            image_height: 원본 이미지 높이
        
        Returns:
            우선순위가 조정된 객체 목록
        """
        if not detections:
            return detections
        
        self.logger.debug(f"수직 위치 기반 가중치 적용 전: {len(detections)}개 객체")
        
        # 각 객체에 대해 수직 위치 기반 신뢰도 조정
        for detection in detections:
            # 중심점 y좌표 (상단일수록 값이 작음)
            center_y = detection['center_y']
            
            # 상단에 위치할수록 높은 가중치 부여 (0.5~1.5 범위)
            # (1 - y/h) : 상단=1, 하단=0 으로 정규화
            vertical_weight = 1.0 + 0.5 * (1 - center_y / image_height)
            
            # 원래 신뢰도에 가중치 적용
            original_confidence = detection['confidence']
            detection['original_confidence'] = original_confidence  # 원래 값 보존
            detection['confidence'] = min(original_confidence * vertical_weight, 1.0)
            
            # 추가 정보 로깅을 위한 필드
            detection['vertical_weight'] = vertical_weight
            
            self.logger.debug(
                f"객체 {detection['class_name']}: 원래 신뢰도={original_confidence:.4f}, " +
                f"수직 위치 가중치={vertical_weight:.4f}, 조정 신뢰도={detection['confidence']:.4f}"
            )
        
        # 조정된 신뢰도로 다시 정렬
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        self.logger.debug(f"수직 위치 기반 가중치 적용 후: 상위 객체 클래스={sorted_detections[0]['class_name'] if sorted_detections else 'none'}")
        
        return sorted_detections 