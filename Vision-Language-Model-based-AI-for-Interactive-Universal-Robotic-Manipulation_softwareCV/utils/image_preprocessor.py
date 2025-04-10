"""
이미지 전처리 모듈

이미지 처리, 변환 및 시각화 기능 제공
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 로깅 설정
logger = logging.getLogger("ImagePreprocessor")

def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = None,
    normalize: bool = False,
    rgb: bool = True
) -> Image.Image:
    """이미지 전처리
    
    Args:
        image: 이미지 경로 또는 이미지 객체
        target_size: 목표 크기 (너비, 높이)
        normalize: 정규화 여부 (0-1 범위로)
        rgb: RGB 변환 여부
        
    Returns:
        전처리된 PIL 이미지
    """
    # 이미지 로드
    if isinstance(image, str):
        # 이미지 경로인 경우
        try:
            img = Image.open(image)
        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            raise ValueError(f"이미지 로드 실패: {str(e)}")
    elif isinstance(image, np.ndarray):
        # NumPy 배열인 경우 (OpenCV 이미지)
        if rgb and image.shape[2] == 3:
            # BGR에서 RGB로 변환 (OpenCV는 BGR 형식 사용)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, Image.Image):
        # 이미 PIL Image인 경우
        img = image
    else:
        raise TypeError("지원되지 않는 이미지 타입입니다")
    
    # RGB 변환
    if rgb and img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 크기 조정
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # 정규화 (NumPy 배열로 변환 후 다시 PIL Image로)
    if normalize:
        img_array = np.array(img).astype(np.float32) / 255.0
        img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    return img

def scale_bounding_boxes(
    objects: List[Dict[str, Any]],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """바운딩 박스 좌표를 소스 크기에서 타겟 크기로 스케일링
    
    Args:
        objects: 바운딩 박스 객체 리스트
            [{"label": "obj1", "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}, ...]
        source_size: 원본 이미지 크기 (너비, 높이)
        target_size: 타겟 이미지 크기 (너비, 높이)
        
    Returns:
        스케일링된 바운딩 박스 객체 리스트
    """
    if not objects:
        return []
    
    # json_parser 모듈 가져오기
    from utils.json_parser import rescale_bounding_box
    
    source_width, source_height = source_size
    target_width, target_height = target_size
    
    # 스케일 비율 계산 - 로그 추가
    width_ratio = target_width / source_width
    height_ratio = target_height / source_height
    
    logger.info(f"바운딩 박스 스케일링: 원본 크기 {source_size}, 대상 크기 {target_size}")
    logger.info(f"스케일 비율: 너비 {width_ratio:.4f}, 높이 {height_ratio:.4f}")
    
    scaled_objects = []
    
    for i, obj in enumerate(objects):
        if "bounding_box" not in obj:
            # 바운딩 박스가 없는 경우 그대로 복사
            scaled_objects.append(obj.copy())
            continue
        
        # 객체 복사
        scaled_obj = obj.copy()
        bbox = obj["bounding_box"]
        
        # 새 함수 사용
        scaled_bbox = rescale_bounding_box(bbox, source_size, target_size)
        
        # 로그 추가
        logger.info(f"객체 {i} ({obj.get('label', '알 수 없음')}): 원본 바운딩 박스 {bbox} -> 스케일링된 바운딩 박스 {scaled_bbox}")
        
        # 스케일링된 바운딩 박스 적용
        scaled_obj["bounding_box"] = scaled_bbox
        scaled_objects.append(scaled_obj)
    
    return scaled_objects

def add_bounding_boxes(
    image: Union[str, Image.Image, np.ndarray],
    objects: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    text_color: Tuple[int, int, int] = (255, 255, 255),  # White
    line_width: int = 2,
    font_size: int = 16
) -> Image.Image:
    """이미지에 바운딩 박스 추가
    
    Args:
        image: 이미지 경로 또는 이미지 객체
        objects: 바운딩 박스 객체 리스트
            [{"label": "obj1", "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}, ...]
        box_color: 박스 색상 (R, G, B)
        text_color: 텍스트 색상 (R, G, B)
        line_width: 선 두께
        font_size: 폰트 크기
        
    Returns:
        바운딩 박스가 추가된 PIL 이미지
    """
    # 이미지 전처리
    img = preprocess_image(image, rgb=True)
    
    # ImageDraw 생성
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정 (시스템에 폰트가 없을 경우 기본 폰트 사용)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # 각 객체에 대해 바운딩 박스 그리기
    for obj in objects:
        if "bounding_box" not in obj or "label" not in obj:
            continue
        
        bbox = obj["bounding_box"]
        label = obj["label"]
        
        # 박스 그리기
        if len(bbox) == 4:
            # 바운딩 박스가 4개의 점으로 구성된 경우
            for i in range(4):
                start_point = tuple(bbox[i])
                end_point = tuple(bbox[(i + 1) % 4])
                draw.line([start_point, end_point], fill=box_color, width=line_width)
            
            # 레이블 그리기
            label_position = (bbox[0][0], bbox[0][1] - font_size - 4)
            
            # 레이블 배경 그리기
            label_width = font.getbbox(label)[2] if hasattr(font, 'getbbox') else font.getsize(label)[0]
            label_bg = [
                label_position[0], label_position[1],
                label_position[0] + label_width + 4, label_position[1] + font_size + 4
            ]
            draw.rectangle(label_bg, fill=box_color)
            
            # 레이블 텍스트 그리기
            draw.text(
                (label_position[0] + 2, label_position[1] + 2),
                label,
                fill=text_color,
                font=font
            )
    
    return img

def resize_with_aspect_ratio(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int]
) -> Image.Image:
    """이미지를 가로세로 비율을 유지하며 리사이징
    
    Args:
        image: 이미지 경로 또는 이미지 객체
        target_size: 목표 크기 (너비, 높이)
        
    Returns:
        리사이징된 PIL 이미지
    """
    # 이미지 전처리
    img = preprocess_image(image, rgb=True)
    
    # 현재 크기
    current_width, current_height = img.size
    target_width, target_height = target_size
    
    # 가로세로 비율 계산
    ratio = min(target_width / current_width, target_height / current_height)
    new_width = int(current_width * ratio)
    new_height = int(current_height * ratio)
    
    # 리사이징
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

def create_grid_image(
    images: List[Union[str, Image.Image, np.ndarray]],
    grid_size: Tuple[int, int] = None
) -> Image.Image:
    """여러 이미지를 그리드로 배치
    
    Args:
        images: 이미지 목록
        grid_size: 그리드 크기 (행, 열), None이면 자동 계산
        
    Returns:
        그리드 이미지
    """
    if not images:
        raise ValueError("이미지 목록이 비어 있습니다")
    
    # 이미지 로드
    pil_images = [preprocess_image(img, rgb=True) for img in images]
    
    # 그리드 크기 결정
    if grid_size:
        rows, cols = grid_size
    else:
        import math
        count = len(pil_images)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
    
    # 이미지 크기 결정 (가장 큰 이미지에 맞춤)
    max_width = max(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)
    
    # 그리드 이미지 생성
    grid_width = max_width * cols
    grid_height = max_height * rows
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # 이미지 배치
    for idx, img in enumerate(pil_images):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        # 이미지 위치 계산
        x = col * max_width
        y = row * max_height
        
        # 이미지 붙이기
        grid_img.paste(img, (x, y))
    
    return grid_img
