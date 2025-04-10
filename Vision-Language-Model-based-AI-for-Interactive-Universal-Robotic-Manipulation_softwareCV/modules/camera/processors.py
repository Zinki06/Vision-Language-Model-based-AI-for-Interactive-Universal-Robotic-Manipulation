"""
기본 프레임 프로세서 구현

카메라 프레임 처리를 위한 기본 프로세서 구현을 제공합니다.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
import time

from modules.camera.frame_processor import FrameProcessor


class BaseFrameProcessor(FrameProcessor):
    """기본 프레임 프로세서 클래스
    
    모든 프로세서에 공통으로 필요한 기능을 제공하는 베이스 클래스입니다.
    """
    
    def __init__(self, processor_id: Optional[str] = None, enabled: bool = True):
        """초기화
        
        Args:
            processor_id: 프로세서 ID (기본값: None, 자동 생성)
            enabled: 활성화 여부 (기본값: True)
        """
        super().__init__(processor_id, enabled)
        self._parameters = {}
    
    def set_parameter(self, key: str, value: Any) -> None:
        """매개변수 설정
        
        Args:
            key: 매개변수 키
            value: 매개변수 값
        """
        self._parameters[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """매개변수 가져오기
        
        Args:
            key: 매개변수 키
            default: 기본값 (기본값: None)
            
        Returns:
            Any: 매개변수 값
        """
        return self._parameters.get(key, default)
    
    def get_parameters(self) -> Dict[str, Any]:
        """모든 매개변수 가져오기
        
        Returns:
            Dict[str, Any]: 모든 매개변수
        """
        return self._parameters.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """프로세서 메타데이터 가져오기
        
        Returns:
            Dict[str, Any]: 프로세서 메타데이터
        """
        metadata = super().get_metadata()
        metadata["parameters"] = self.get_parameters()
        return metadata


class ResizeProcessor(BaseFrameProcessor):
    """프레임 크기 조정 프로세서
    
    입력 프레임의 크기를 지정된 크기로 조정합니다.
    """
    
    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_LINEAR, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            width: 조정할 너비
            height: 조정할 높이
            interpolation: 보간 방법 (기본값: cv2.INTER_LINEAR)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("width", width)
        self.set_parameter("height", height)
        self.set_parameter("interpolation", interpolation)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임 크기 조정
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 크기 조정된 프레임과 메타데이터
        """
        width = self.get_parameter("width")
        height = self.get_parameter("height")
        interpolation = self.get_parameter("interpolation")
        
        # 원본 크기 저장
        original_height, original_width = frame.shape[:2]
        
        # 크기 조정
        resized_frame = cv2.resize(frame, (width, height), interpolation=interpolation)
        
        return resized_frame, {
            "original_size": (original_width, original_height),
            "new_size": (width, height)
        }


class GrayscaleProcessor(BaseFrameProcessor):
    """그레이스케일 변환 프로세서
    
    컬러 프레임을 그레이스케일로 변환합니다.
    """
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임을 그레이스케일로 변환
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 그레이스케일 프레임과 메타데이터
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 차원 유지를 위해 확장
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            return gray_frame, {"converted_to_grayscale": True}
        
        return frame, {"converted_to_grayscale": False}


class BlurProcessor(BaseFrameProcessor):
    """블러 프로세서
    
    프레임에 블러 효과를 적용합니다.
    """
    
    def __init__(self, kernel_size: int = 5, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            kernel_size: 커널 크기 (기본값: 5)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("kernel_size", kernel_size)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임에 블러 적용
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 블러 처리된 프레임과 메타데이터
        """
        kernel_size = self.get_parameter("kernel_size")
        
        # 홀수 커널 크기 보장
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        return blurred_frame, {"kernel_size": kernel_size}


class EdgeDetectionProcessor(BaseFrameProcessor):
    """엣지 검출 프로세서
    
    프레임에서 엣지를 검출합니다.
    """
    
    def __init__(self, threshold1: float = 100, threshold2: float = 200, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            threshold1: 첫 번째 임계값 (기본값: 100)
            threshold2: 두 번째 임계값 (기본값: 200)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("threshold1", threshold1)
        self.set_parameter("threshold2", threshold2)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임에서 엣지 검출
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 엣지 검출된 프레임과 메타데이터
        """
        threshold1 = self.get_parameter("threshold1")
        threshold2 = self.get_parameter("threshold2")
        
        # 그레이스케일 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 엣지 검출
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # 3채널 프레임으로 변환
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_bgr, {
            "threshold1": threshold1, 
            "threshold2": threshold2
        }


class RotationProcessor(BaseFrameProcessor):
    """회전 프로세서
    
    프레임을 지정된 각도로 회전합니다.
    """
    
    def __init__(self, angle: float = 90, scale: float = 1.0, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            angle: 회전 각도 (도 단위, 기본값: 90)
            scale: 확대/축소 비율 (기본값: 1.0)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("angle", angle)
        self.set_parameter("scale", scale)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임 회전
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 회전된 프레임과 메타데이터
        """
        angle = self.get_parameter("angle")
        scale = self.get_parameter("scale")
        
        height, width = frame.shape[:2]
        center = (width / 2, height / 2)
        
        # 회전 행렬 계산
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 회전 적용
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        
        return rotated_frame, {
            "angle": angle, 
            "scale": scale,
            "center": center
        }


class FlipProcessor(BaseFrameProcessor):
    """뒤집기 프로세서
    
    프레임을 수평, 수직 또는 양방향으로 뒤집습니다.
    """
    
    def __init__(self, flip_code: int = 1, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            flip_code: 뒤집기 방향 (0: 수직, 1: 수평, -1: 양방향, 기본값: 1)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("flip_code", flip_code)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임 뒤집기
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 뒤집힌 프레임과 메타데이터
        """
        flip_code = self.get_parameter("flip_code")
        
        # 뒤집기 적용
        flipped_frame = cv2.flip(frame, flip_code)
        
        # 뒤집기 방향 텍스트 설정
        flip_direction = {
            0: "vertical",
            1: "horizontal",
            -1: "both"
        }.get(flip_code, "unknown")
        
        return flipped_frame, {"flip_direction": flip_direction}


class BrightnessContrastProcessor(BaseFrameProcessor):
    """밝기/대비 조정 프로세서
    
    프레임의 밝기와 대비를 조정합니다.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0, processor_id: Optional[str] = None):
        """초기화
        
        Args:
            alpha: 대비 (1.0 = 원본, 기본값: 1.0)
            beta: 밝기 (0 = 원본, 기본값: 0)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("alpha", alpha)
        self.set_parameter("beta", beta)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임의 밝기/대비 조정
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 조정된 프레임과 메타데이터
        """
        alpha = self.get_parameter("alpha")
        beta = self.get_parameter("beta")
        
        # 밝기/대비 조정
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return adjusted_frame, {
            "alpha": alpha, 
            "beta": beta
        }


class OverlayTextProcessor(BaseFrameProcessor):
    """텍스트 오버레이 프로세서
    
    프레임에 텍스트를 오버레이합니다.
    """
    
    def __init__(self, text: str = "", position: Tuple[int, int] = (10, 30), 
                 font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2, font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
                 processor_id: Optional[str] = None):
        """초기화
        
        Args:
            text: 표시할 텍스트 (기본값: "")
            position: 텍스트 위치 (x, y) (기본값: (10, 30))
            font_scale: 폰트 크기 (기본값: 1.0)
            color: 텍스트 색상 (B, G, R) (기본값: (0, 255, 0))
            thickness: 선 두께 (기본값: 2)
            font_face: 폰트 유형 (기본값: cv2.FONT_HERSHEY_SIMPLEX)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("text", text)
        self.set_parameter("position", position)
        self.set_parameter("font_scale", font_scale)
        self.set_parameter("color", color)
        self.set_parameter("thickness", thickness)
        self.set_parameter("font_face", font_face)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임에 텍스트 오버레이
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 텍스트가 오버레이된 프레임과 메타데이터
        """
        text = self.get_parameter("text")
        position = self.get_parameter("position")
        font_scale = self.get_parameter("font_scale")
        color = self.get_parameter("color")
        thickness = self.get_parameter("thickness")
        font_face = self.get_parameter("font_face")
        
        # 프레임 복사
        result_frame = frame.copy()
        
        # 텍스트가 있는 경우에만 오버레이
        if text:
            cv2.putText(result_frame, text, position, font_face, font_scale, color, thickness)
        
        return result_frame, {
            "text": text,
            "position": position
        }


class TimestampProcessor(BaseFrameProcessor):
    """타임스탬프 오버레이 프로세서
    
    프레임에 현재 시간을 오버레이합니다.
    """
    
    def __init__(self, format_string: str = "%Y-%m-%d %H:%M:%S", 
                 position: Tuple[int, int] = (10, 30),
                 font_scale: float = 1.0, 
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2, 
                 processor_id: Optional[str] = None):
        """초기화
        
        Args:
            format_string: 시간 형식 (기본값: "%Y-%m-%d %H:%M:%S")
            position: 텍스트 위치 (기본값: (10, 30))
            font_scale: 폰트 크기 (기본값: 1.0)
            color: 텍스트 색상 (기본값: (0, 255, 0))
            thickness: 선 두께 (기본값: 2)
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.set_parameter("format_string", format_string)
        self.set_parameter("position", position)
        self.set_parameter("font_scale", font_scale)
        self.set_parameter("color", color)
        self.set_parameter("thickness", thickness)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임에 타임스탬프 오버레이
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 타임스탬프가 오버레이된 프레임과 메타데이터
        """
        format_string = self.get_parameter("format_string")
        position = self.get_parameter("position")
        font_scale = self.get_parameter("font_scale")
        color = self.get_parameter("color")
        thickness = self.get_parameter("thickness")
        
        # 현재 시간 가져오기
        current_time = time.strftime(format_string)
        
        # 프레임 복사
        result_frame = frame.copy()
        
        # 타임스탬프 오버레이
        cv2.putText(result_frame, current_time, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        
        return result_frame, {
            "timestamp": current_time,
            "format": format_string
        }


class CustomProcessor(BaseFrameProcessor):
    """사용자 정의 프로세서
    
    사용자 정의 함수를 사용하여 프레임을 처리합니다.
    """
    
    def __init__(self, process_function: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]], 
                 processor_id: Optional[str] = None):
        """초기화
        
        Args:
            process_function: 프레임 처리 함수 (인자: np.ndarray, 반환: Tuple[np.ndarray, Dict[str, Any]])
            processor_id: 프로세서 ID (기본값: 자동 생성)
        """
        super().__init__(processor_id=processor_id)
        self.process_function = process_function
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """사용자 정의 함수를 사용하여 프레임 처리
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 처리된 프레임과 메타데이터
        """
        if self.process_function:
            return self.process_function(frame)
        
        return frame, {"error": "No process function defined"} 