import os
import time
import logging
import base64
import io
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from modules.llm_interface import LLMInterface

class GPT4oInterface(LLMInterface):
    """
    OpenAI의 GPT-4o API를 사용하여 이미지를 분석하고 
    객체 식별, 명령어 이해, 목표지점 탐지 등의 기능을 제공하는 클래스
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        timeout: int = 30,
        temperature: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        GPT-4o 인터페이스 초기화
        
        Args:
            model_name: 사용할 OpenAI 모델 이름
            api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
            timeout: API 요청 타임아웃 (초)
            temperature: 생성 다양성 조절 파라미터 (낮을수록 결정적 응답)
            logger: 로깅을 위한 logger 객체
        """
        # .env 파일 로드
        load_dotenv()
        
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)
        
        # API 키 설정
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # OpenAI 클라이언트 초기화
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info(f"GPT4oInterface 초기화 완료: {model_name}")
        except Exception as e:
            self.logger.error(f"OpenAI API 초기화 실패: {e}")
            raise
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        PIL 이미지를 base64 문자열로 인코딩
        
        Args:
            image: PIL 이미지
        
        Returns:
            base64로 인코딩된 이미지 문자열
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _clamp_bbox(self, bbox: List[float], w: int, h: int) -> List[float]:
        """
        바운딩 박스가 이미지 경계 내에 있도록 조정
        
        Args:
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            w: 이미지 너비
            h: 이미지 높이
            
        Returns:
            조정된 바운딩 박스 좌표 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        
        # 경계 내로 좌표 제한
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        # x1 > x2 또는 y1 > y2 경우 수정
        if x1 >= x2:
            x2 = min(w, x1 + 1)  # 최소 1 픽셀 너비 보장
        if y1 >= y2:
            y2 = min(h, y1 + 1)  # 최소 1 픽셀 높이 보장
            
        return [x1, y1, x2, y2]
    
    def _validate_bbox(self, bbox: List[float], w: int, h: int) -> Tuple[List[float], bool]:
        """
        바운딩 박스 유효성 검사 및 필요시 조정
        
        Args:
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            w: 이미지 너비
            h: 이미지 높이
            
        Returns:
            (조정된 바운딩 박스, 조정 여부) 튜플
        """
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            # 유효하지 않은 바운딩 박스 - 기본값 사용
            self.logger.warning(f"유효하지 않은 바운딩 박스 형식: {bbox}")
            return [0, 0, min(w, 10), min(h, 10)], True
            
        x1, y1, x2, y2 = bbox
        
        # 좌표가 숫자인지 확인
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            self.logger.warning(f"바운딩 박스에 숫자가 아닌 좌표 포함: {bbox}")
            # 가능한 부분만 변환 시도
            try:
                x1 = float(x1) if isinstance(x1, (int, float, str)) else 0
                y1 = float(y1) if isinstance(y1, (int, float, str)) else 0
                x2 = float(x2) if isinstance(x2, (int, float, str)) else w
                y2 = float(y2) if isinstance(y2, (int, float, str)) else h
            except:
                return [0, 0, min(w, 10), min(h, 10)], True
        
        # 경계 밖 또는 잘못된 순서 확인
        needs_adjustment = (x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2)
        
        if needs_adjustment:
            return self._clamp_bbox([x1, y1, x2, y2], w, h), True
            
        return bbox, False
    
    def _crop_object_from_image(self, image: np.ndarray, bbox: List[float]) -> Image.Image:
        """
        이미지에서 바운딩 박스 부분을 잘라내어 PIL 이미지로 반환
        
        Args:
            image: 원본 이미지 (numpy 배열)
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
        
        Returns:
            바운딩 박스로 잘라낸 PIL 이미지
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 이미지 범위를 벗어나지 않도록 좌표 보정
        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # 바운딩 박스 영역 잘라내기
        cropped_img = image[y1:y2, x1:x2]
        
        # PIL 이미지로 변환
        return Image.fromarray(cropped_img)
    
    def classify_objects(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        YOLO로 감지된 객체들에 대해 GPT-4o를 사용하여 더 정확한 분류 및 설명 생성
        
        Args:
            image: 원본 이미지 (numpy 배열)
            detections: YOLO에서 감지한 객체 목록
        
        Returns:
            GPT-4o가 식별 및 설명한 객체 목록
        """
        if not detections:
            self.logger.info("감지된 객체가 없습니다.")
            return []
        
        self.logger.info(f"GPT-4o를 사용하여 {len(detections)} 개의 객체 분류 시작")
        
        # 전체 이미지를 PIl로 변환
        full_image = Image.fromarray(image)
        
        # 이미지 base64 인코딩
        base64_image = self._encode_image_to_base64(full_image)
        self.logger.info("이미지 base64 인코딩 완료")
        
        # 객체 바운딩 박스 정보 구성
        objects_info = []
        for i, obj in enumerate(detections):
            # 객체 ID, 바운딩 박스, 원래 YOLO가 인식한 클래스 이름 (참고용)
            obj_info = {
                "id": f"obj_{i}",
                "bbox": obj["bbox"],
                "yolo_class": obj["class_name"],
                "confidence": obj["confidence"]
            }
            objects_info.append(obj_info)
        
        # GPT-4o에 전달할 프롬프트 구성
        prompt = f"""
        다음 이미지에서 바운딩 박스로 표시된 {len(detections)}개의 객체를 자세히 분석해주세요.
        각 객체에 대해 다음 정보를 JSON 형식으로 제공해주세요:
        
        1. 객체 ID: 주어진 ID 그대로 사용
        2. 객체 명칭: 정확한 물체 이름
        3. 설명: 물체의 간단한 설명 (최대 30자)
        4. 색상: 주요 색상
        5. 상태: 물체의 상태나 특성
        
        다음은 감지된 객체들입니다:
        {objects_info}
        
        JSON 형식으로 답변해주세요. 다음과 같은 형식이어야 합니다:
        [
          {{"id": "obj_0", "name": "객체명", "description": "설명", "color": "색상", "state": "상태"}},
          ...
        ]
        
        응답에는 반드시 이 JSON 형식만 포함되어야 합니다. 다른 설명이나 텍스트는 제외해주세요.
        """
        
        try:
            # OpenAI API 호출 시작
            start_time = time.time()
            
            # API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful vision assistant that analyzes objects in images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=self.temperature
            )
            
            # 응답 처리
            text = response.choices[0].message.content
            if not text:
                raise ValueError("OpenAI API에서 응답이 없거나 빈 응답이 반환되었습니다.")
            
            # JSON 응답 추출
            text = text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # JSON 파싱
            classified_objects = json.loads(text)
            
            # YOLO 감지 결과와 GPT-4o 식별 결과 병합
            for i, (yolo_obj, gpt_obj) in enumerate(zip(detections, classified_objects)):
                # 원래 YOLO의 바운딩 박스와 신뢰도 정보 유지
                gpt_obj["bbox"] = yolo_obj["bbox"]
                gpt_obj["confidence"] = yolo_obj["confidence"]
                gpt_obj["yolo_class"] = yolo_obj["class_name"]
                
                # 원본 ID 유지 (만약 ID가 달라졌다면)
                if gpt_obj["id"] != yolo_obj.get("id", f"obj_{i}"):
                    gpt_obj["id"] = yolo_obj.get("id", f"obj_{i}")
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            self.logger.info(f"GPT-4o 객체 분류 완료: {len(classified_objects)}개, 소요 시간: {processing_time:.2f}초")
            
            return classified_objects
            
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
            # 오류 발생 시 원본 YOLO 결과 반환하되 형식 통일
            return [{
                "id": obj.get("id", f"obj_{i}"),
                "name": obj["class_name"],  
                "description": obj["class_name"],
                "color": "unknown",
                "state": "unknown",
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "yolo_class": obj["class_name"]
            } for i, obj in enumerate(detections)]
    
    def analyze_specific_object(self, image: np.ndarray, bbox: List[float], query: str) -> Dict[str, Any]:
        """
        특정 바운딩 박스 내 객체에 대한 상세 분석 수행
        
        Args:
            image: 원본 이미지 (numpy 배열)
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            query: 질의 내용 (예: "이 물체는 무엇인가요?")
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        # 바운딩 박스 영역 잘라내기
        cropped_img = self._crop_object_from_image(image, bbox)
        
        # 이미지 base64 인코딩
        base64_image = self._encode_image_to_base64(cropped_img)
        self.logger.info("이미지 base64 인코딩 완료")
        
        # 프롬프트 구성
        prompt = f"""
        다음 이미지에 보이는 물체에 대해 질문에 답해주세요.
        
        질문: {query}
        
        상세하고 정확하게 답변해주세요.
        """
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful vision assistant that analyzes objects in images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=self.temperature
            )
            
            if not response.choices[0].message.content:
                raise ValueError("OpenAI API에서 응답이 없거나 빈 응답이 반환되었습니다.")
            
            return {
                "query": query,
                "answer": response.choices[0].message.content.strip(),
                "bbox": bbox
            }
            
        except Exception as e:
            self.logger.error(f"특정 객체 분석 중 오류 발생: {e}")
            return {
                "query": query,
                "answer": f"분석 중 오류 발생: {str(e)}",
                "bbox": bbox
            }
    
    def select_objects_from_command(self, image: np.ndarray, command: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        이미지와 사용자 명령어를 바탕으로 작업에 필요한 객체들을 선택
        
        Args:
            image: 이미지 (numpy 배열)
            command: 사용자 명령어 (예: "빨간 공을 파란 상자 위에 올려줘")
            detections: 이미지에서 감지된 객체 목록
            
        Returns:
            선택된 객체 정보와 목표 위치를 포함한 딕셔너리
        """
        if not detections:
            self.logger.warning("객체가 감지되지 않았습니다. 명령어 이해 및 객체 선택을 진행할 수 없습니다.")
            return None
            
        self.logger.info(f"명령어 이해 및 객체 선택 시작: '{command}'")
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(image)
        
        # 이미지 base64 인코딩
        base64_image = self._encode_image_to_base64(pil_img)
        self.logger.info("이미지 base64 인코딩 완료")
        
        # 객체 ID와 바운딩 박스를 저장할 매핑 생성
        object_id_to_bbox = {}
        
        # 객체 목록 생성
        objects_text = ""
        for i, obj in enumerate(detections):
            obj_id = obj.get("id", f"obj_{i}")
            obj_class = obj.get("class_name", "unknown")
            obj_bbox = obj.get("bbox", [0, 0, 0, 0])
            
            # ID와 바운딩 박스 매핑 저장
            object_id_to_bbox[obj_id] = obj_bbox
            
            # 바운딩 박스 좌표 정수화
            bbox_str = ", ".join([f"{int(coord)}" for coord in obj_bbox])
            
            # 객체 색상 정보 추가 (있는 경우)
            obj_color = obj.get("color", "")
            color_info = f", 색상: {obj_color}" if obj_color else ""
            
            # 객체 텍스트 추가
            objects_text += f"- {obj_id}: 클래스={obj_class}{color_info}, 바운딩박스=[{bbox_str}]\n"
        
        # 이미지 크기 계산
        h, w = image.shape[:2]
        
        # 프롬프트 구성
        prompt = f"""
# 작업

이미지와 사용자 명령어를 정확히 분석하여 다음을 식별하세요:
1. 움직여야 할 타겟 객체 - YOLO에서 감지된 객체 중 선택
2. 기준이 되는 레퍼런스 객체 - YOLO에서 감지된 객체 중 선택
3. 타겟 객체의 목표 위치 - 새로 계산
4. 방향 키워드 - "앞", "뒤", "위", "아래", "왼쪽", "오른쪽", "안" 중에서 선택

# 명령어

"{command}"

# 감지된 객체 목록

이미지에서 감지된 객체 목록 (ID, 클래스, 바운딩 박스 [x1, y1, x2, y2]):
{objects_text}

# 이미지 크기

너비: {w}, 높이: {h}

# 중요 지침

1. YOLO가 제공한 클래스 이름(예: laptop, cell phone)은 실제 물체와 다를 수 있으므로 이를 절대적으로 신뢰하지 마세요.
2. 이미지를 직접 보고 물체의 실제 특성(크기, 형태, 색상, 위치)을 기반으로 식별하세요.
3. 명령어와 이미지를 세심하게 분석하여 실제로 어떤 물체를 타겟과 레퍼런스로 지정해야 하는지 판단하세요.
4. 타겟 객체와 레퍼런스 객체는 YOLO가 감지한 객체 중에서만 선택하고, 해당 객체의 ID를 반환해야 합니다.
5. 타겟 객체와 레퍼런스 객체의 바운딩 박스는 YOLO가 제공한 것을 그대로 사용해야 합니다.
6. 목표 위치만 자체적으로 계산하여 새로운 바운딩 박스를 생성하세요.
7. 목표 위치는 명령어에서 표현한 관계(위, 옆, 앞 등)를 고려하여 타겟 물체의 정확한 목적지를 계산하세요.
8. 목표 위치의 바운딩 박스는 실제로 물체를 놓을 수 있는 유효한 영역이어야 합니다.
9. 방향 키워드는 "앞", "뒤", "위", "아래", "왼쪽", "오른쪽", "안" 중에서 명령어에 가장 적합한 방향을 선택하세요.

# 출력 형식

다음 JSON 형식으로만 출력하세요:

{{{{
  "command": "원본 명령어",
  "target_object": {{{{
    "id": "객체 ID",
    "name": "객체 이름",
    "description": "상세 설명",
    "bbox": [x1, y1, x2, y2]
  }}}},
  "reference_object": {{{{
    "id": "객체 ID",
    "name": "객체 이름",
    "description": "상세 설명",
    "bbox": [x1, y1, x2, y2]
  }}}},
  "relation": "위치 관계 (on, in, next_to, in_front_of 등)",
  "direction": "방향 키워드",
  "destination": {{{{
    "description": "목표 위치 설명",
    "relation": "위치 관계",
    "bounding_box": [x1, y1, x2, y2]
  }}}},
  "explanation": "선택 이유에 대한 상세한 설명"
}}}}

타겟 또는 레퍼런스 객체가 식별되지 않으면 해당 필드를 null로 설정하세요.
JSON 형식 외에 다른 텍스트는 제외하고 응답하세요.
"""
        
        try:
            # 시작 시간 기록
            start_time = time.time()
            self.logger.info("OpenAI API 호출 시작")
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful vision assistant that analyzes objects in images and understands commands."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=self.temperature,
                timeout=self.timeout
            )
            self.logger.info(f"OpenAI API 응답 수신 완료 - 상태: {'성공'}")
            
            # 응답 처리
            if not response.choices[0].message.content:
                self.logger.error("OpenAI API에서 빈 응답이 반환되었습니다.")
                raise ValueError("OpenAI API에서 응답이 없거나 빈 응답이 반환되었습니다.")
            
            # JSON 응답 추출
            text = response.choices[0].message.content.strip()
            self.logger.debug(f"API 응답 원본: {text[:500]}...")  # 응답 첫 500자만 로깅
            
            # 마크다운 코드 블록 제거
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # 처리 시간 계산
            gpt_time = time.time() - start_time
            
            try:
                # JSON 파싱
                self.logger.info("JSON 응답 파싱 시작")
                result = json.loads(text)
                self.logger.info(f"명령어 처리 완료: {gpt_time:.4f}초")
                self.logger.debug(f"파싱된 결과 키: {list(result.keys())}")
                
                # 필수 필드 확인
                required_fields = ["target_object", "reference_object", "relation", "destination"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    self.logger.warning(f"응답에 필수 필드가 누락됨: {missing_fields}")
                
                # direction 필드 확인 (새로 추가된 필드)
                if "direction" not in result:
                    self.logger.warning("응답에 direction 필드가 없습니다")
                    # 기본값으로 'front' 설정 (기본 방향을 '앞'으로 가정)
                    result["direction"] = "앞"
                
                # 타겟 객체 정보 처리
                target_obj = result.get("target_object")
                if target_obj and "id" in target_obj:
                    # 원본 YOLO 바운딩 박스로 교체
                    obj_id = target_obj["id"]
                    if obj_id in object_id_to_bbox:
                        target_obj["bbox"] = object_id_to_bbox[obj_id]
                        self.logger.info(f"타겟 객체의 바운딩 박스를 YOLO 원본으로 교체: {obj_id}")
                
                # 레퍼런스 객체 정보 처리
                ref_obj = result.get("reference_object")
                if ref_obj and "id" in ref_obj:
                    # 원본 YOLO 바운딩 박스로 교체
                    obj_id = ref_obj["id"]
                    if obj_id in object_id_to_bbox:
                        ref_obj["bbox"] = object_id_to_bbox[obj_id]
                        self.logger.info(f"레퍼런스 객체의 바운딩 박스를 YOLO 원본으로 교체: {obj_id}")
                
                # 목표 위치 유효성 검사
                destination = result.get("destination", {})
                bbox = destination.get("bounding_box")
                if bbox and isinstance(bbox, list):
                    # 이미지 경계 안에 있는지 확인
                    original_bbox = bbox.copy()  # 원본 바운딩 박스 저장
                    bbox, adjusted = self._validate_bbox(bbox, w, h)
                    
                    if adjusted:
                        self.logger.warning(f"목표지점 바운딩 박스가 조정됨: {original_bbox} -> {bbox}")
                        destination["bounding_box"] = bbox
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {e}")
                self.logger.error(f"파싱 실패한 텍스트: {text[:200]}...")  # 처음 200자만 로깅
                
                # JSON 형식을 강제로 정리해보기
                try:
                    # JSON 형식에 맞는 부분만 추출 시도
                    json_pattern = r'\{[\s\S]*\}'
                    json_match = re.search(json_pattern, text)
                    if json_match:
                        cleaned_json = json_match.group(0)
                        self.logger.info("정규식으로 JSON 부분 추출 시도")
                        result = json.loads(cleaned_json)
                        self.logger.info("정규식으로 추출한 JSON 파싱 성공")
                        return result
                except Exception as e2:
                    self.logger.error(f"JSON 복구 시도 실패: {e2}")
                
                # 기본 응답 구조 반환
                return {
                    "command": command,
                    "target_object": None,
                    "reference_object": None,
                    "relation": None,
                    "direction": "앞",  # 기본 방향
                    "destination": {"description": "처리 실패", "bounding_box": [0, 0, 10, 10]},
                    "explanation": f"JSON 파싱 오류: {str(e)}"
                }
            
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
            self.logger.error(traceback.format_exc())  # 전체 스택 트레이스 로깅
            
            # 기본 응답 구조 반환
            return {
                "command": command,
                "target_object": None,
                "reference_object": None,
                "relation": None,
                "direction": "앞",  # 기본 방향
                "destination": {"description": "API 호출 오류", "bounding_box": [0, 0, 10, 10]},
                "explanation": f"오류: {str(e)}"
            }
    
    def detect_goal_points(self, image: np.ndarray) -> Dict[str, Any]:
        """
        GPT-4o API를 사용하여 이미지에서 목표지점(Goal Point)의 바운딩 박스 좌표를 추출
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            목표지점 좌표 및 메타 정보를 담은 딕셔너리
        """
        self.logger.info("GPT-4o API를 사용하여 목표지점 감지 시작")
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(image)
        
        # 이미지 base64 인코딩
        base64_image = self._encode_image_to_base64(pil_img)
        self.logger.info("이미지 base64 인코딩 완료")
        
        # 프롬프트 구성
        prompt = """
        이미지에서 로봇 작업의 목표지점(Goal Point)을 찾아 정확한 바운딩 박스 좌표를 추출해주세요.
        
        목표지점은 다음과 같은 특성을 갖습니다:
        1. 타겟 물체를 놓아야 할 명확한 빈 공간이나 영역
        2. 특별히 표시된 영역 (마커, 테이프, 지정된 공간 등)
        3. 물체 간 상호작용이 필요한 위치
        4. 사용자 명령어에서 '~로', '~에', '~위에' 등으로 지정된 위치
        5. 대개 평평하고 물체를 놓을 수 있는 표면을 가지고 있음
        
        중요 지침:
        1. 반드시 물체를 놓을 수 있는 실제 표면이나 위치를 선택하세요.
        2. 바운딩 박스는 목표지점의 실제 크기와 최대한 일치하는 최소한의 크기로 지정하세요.
        3. 목표지점이 이미지에 없거나 명확하지 않으면 null을 반환하세요.
        4. 이미지에서 가장 적합한 목표지점이 무엇인지 주의 깊게 분석하세요.
        5. 타겟 물체가 이동해야 할 가장 논리적인 위치를 선택하세요.
        6. 여러 가능한 위치가 있다면 가장 명확하고 논리적인 하나만 선택하세요.
        
        딱 하나의 가장 유력한 목표지점의 바운딩 박스 좌표를 다음 JSON 형식으로 반환해주세요:
        
        {{{{
          "goal_point": {{{{
            "x": 340,  // 좌상단 x 좌표
            "y": 220,  // 좌상단 y 좌표
            "width": 80,  // 너비
            "height": 60,  // 높이
            "description": "목표지점에 대한 상세한 설명"
          }}}}
        }}}}
        
        좌표는 정수값으로 반환해주세요. 이미지에서 명확한 목표지점이 없다면 null을 반환해주세요.
        """
        
        try:
            # OpenAI API 호출
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful vision assistant that detects goal points in images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=self.temperature
            )
            
            if not response.choices[0].message.content:
                raise ValueError("OpenAI API에서 응답이 없거나 빈 응답이 반환되었습니다.")
            
            # JSON 응답 추출
            text = response.choices[0].message.content.strip()
            
            # 마크다운 코드 블록 제거
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            try:
                # JSON 파싱
                result = json.loads(text)
                
                # 목표지점 정보 추출
                goal_point = result.get("goal_point")
                
                if not goal_point:
                    self.logger.info("목표지점이 감지되지 않았습니다.")
                    return None
                
                # 바운딩 박스 형식 변환
                x = goal_point.get("x", 0)
                y = goal_point.get("y", 0)
                width = goal_point.get("width", 0)
                height = goal_point.get("height", 0)
                
                # x1, y1, x2, y2 형식의 바운딩 박스
                bbox = [x, y, x + width, y + height]
                
                # 신뢰도 계산 (이 예시에서는 고정값 사용)
                confidence = 0.95
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                self.logger.info(f"GPT-4o 목표지점 감지 완료: {bbox}, 소요 시간: {execution_time:.2f}초")
                
                # 이미지 크기 내에 있는지 확인
                h, w = image.shape[:2]
                original_bbox = bbox.copy()  # 원본 바운딩 박스 저장
                bbox, adjusted = self._validate_bbox(bbox, w, h)
                
                if adjusted:
                    self.logger.warning(f"목표지점 바운딩 박스가 조정됨: {original_bbox} -> {bbox}")
                
                # 목표지점 정보 반환
                goal_point_info = {
                    "goal_point": goal_point,
                    "bbox": bbox,
                    "id": "goal_point_gpt4o",
                    "class_name": "goal_point",
                    "confidence": confidence,
                    "description": goal_point.get("description", "GPT-4o로 감지된 목표지점")
                }
                
                return goal_point_info
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {e}")
                return None
            except Exception as e:
                self.logger.error(f"목표지점 정보 처리 중 오류 발생: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"목표지점 감지 중 오류 발생: {e}")
            return None
    
    def _validate_bbox_simple(self, bbox: List[float]) -> List[float]:
        """
        바운딩 박스 좌표 간단 검증 (이미지 크기 없이)
        
        Args:
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            
        Returns:
            검증된 바운딩 박스 좌표
        """
        # 최소한의 유효성 검사
        if len(bbox) != 4:
            self.logger.warning(f"유효하지 않은 바운딩 박스 형식: {bbox}, 기본값 [0,0,100,100] 사용")
            return [0, 0, 100, 100]
        
        # x1, y1이 x2, y2보다 작은지 확인
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # 음수 값 처리
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        # 너무 작은 바운딩 박스 처리
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.logger.warning(f"너무 작은 바운딩 박스: {bbox}, 기본 크기로 확장")
            x2 = max(x1 + 10, x2)
            y2 = max(y1 + 10, y2)
            
        return [x1, y1, x2, y2]

    def infer_goal_point(self, target_object: Dict[str, Any], reference_object: Dict[str, Any], user_prompt: str = None) -> Dict[str, Any]:
        """
        타겟 객체와 레퍼런스 객체의 정보를 기반으로 목표지점을 추론합니다.
        
        Args:
            target_object: 타겟 객체 정보 (바운딩 박스, 클래스, 깊이 통계 포함)
            reference_object: 레퍼런스 객체 정보 (바운딩 박스, 클래스, 깊이 통계 포함)
            user_prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
            
        Returns:
            Dict: 목표지점 좌표와 추론 근거를 포함하는 딕셔너리
        """
        # 객체가 없는 경우 처리
        if target_object is None or reference_object is None:
            self.logger.error("타겟 객체 또는 레퍼런스 객체가 None입니다.")
            return {
                "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                "confidence": 0.0,
                "reasoning": "객체 정보가 없어 추론할 수 없습니다.",
                "direction_type": "simple",
                "direction": "front"
            }

        # 1. 한국어 방향어 직접 추출 (최우선 적용)
        explicit_direction = None
        if user_prompt:
            explicit_direction = self._extract_explicit_korean_direction(user_prompt)
            if explicit_direction:
                self.logger.info(f"사용자 프롬프트에서 명시적 방향 감지: '{explicit_direction}' - 직접 반환")
                
                # 방향어에 따른 신뢰도 설정
                confidence = 0.95  # 기본 높은 신뢰도
                
                # side_random 처리
                if explicit_direction == 'side_random':
                    return {
                        "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                        "confidence": confidence,
                        "reasoning": f"사용자가 '{explicit_direction}' 방향을 지정했습니다. 좌우 무작위 방향으로 배치합니다.",
                        "direction_type": "random",
                        "options": ["left", "right"],
                        "weights": [0.5, 0.5]
                    }
                
                # 명시적 방향 감지 시 즉시 반환
                return {
                    "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                    "confidence": confidence,
                    "reasoning": f"사용자가 명시적으로 '{explicit_direction}' 방향을 지정했습니다.",
                    "direction_type": "simple",
                    "direction": explicit_direction
                }
        
        # 2. 사용자의 랜덤 방향 의도 분석 (두 번째 우선순위)
        random_direction = self._analyze_random_direction_intent(user_prompt) if user_prompt else None
        if random_direction:
            self.logger.info(f"사용자 프롬프트에서 랜덤 방향 의도 감지: {random_direction}")
            # 기본 목표 지점 설정
            return {
                "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                "confidence": random_direction.get("confidence", 0.8),
                "reasoning": f"사용자가 요청한 {random_direction.get('options', [])} 중 랜덤 방향으로 배치합니다.",
                "direction_type": "random",
                "options": random_direction.get("options", ["left", "right"]),
                "weights": random_direction.get("weights", [0.5, 0.5])
            }
        
        # 3. 객체 정보 추출 및 LLM 기반 추론 (마지막 우선순위)
        target_bbox = target_object.get("bbox", [0, 0, 10, 10])
        ref_bbox = reference_object.get("bbox", [0, 0, 10, 10])
        
        # 클래스명 추출
        target_class = target_object.get("class_name", "알 수 없는 객체")
        ref_class = reference_object.get("class_name", "알 수 없는 객체")
        
        # 3D 정보 확인
        has_depth = "depth_stats" in target_object and "depth_stats" in reference_object
        
        # 공간 관계 정보 구성
        spatial_info = ""
        if has_depth:
            target_depth = target_object["depth_stats"].get("mean", 0.5)
            ref_depth = reference_object["depth_stats"].get("mean", 0.5)
            depth_diff = abs(target_depth - ref_depth)
            spatial_info = f"""
3D 깊이 정보:
- 타겟 객체 평균 깊이: {target_depth:.4f}
- 레퍼런스 객체 평균 깊이: {ref_depth:.4f}
- 깊이 차이: {depth_diff:.4f}
"""
        
        # 프롬프트 구성
        system_prompt = f"""당신은 3D 공간 관계를 분석하여 목표 지점을 추론하는 전문가입니다.

사용자의 명령을 분석하고, 타겟 객체가 레퍼런스 객체를 기준으로 어느 위치에 놓여야 하는지 추론해주세요.
가능한 방향: 'front'(앞), 'back'(뒤), 'left'(왼쪽), 'right'(오른쪽), 'above'(위), 'below'(아래)

결과는 다음 JSON 형식으로 응답해야 합니다:
```json
{{
  "goal_point": {{
    "x": 0.5,  // 이미지 내 x 좌표 (0.0~1.0)
    "y": 0.5,  // 이미지 내 y 좌표 (0.0~1.0)
    "z": 0.5   // 깊이 값 (0.0~1.0, 작을수록 카메라에 가까움)
  }},
  "confidence": 0.9,       // 추론 신뢰도 (0.0~1.0)
  "reasoning": "타겟을 레퍼런스의 오른쪽에 배치함", // 추론 근거
  "direction_type": "simple", // "simple" 또는 "random"
  "direction": "front",        // 단일 방향인 경우 방향값
  "options": ["left", "right"],// 랜덤 방향인 경우 선택지 목록
  "weights": [0.5, 0.5]        // 각 선택지의 확률 가중치 (합이 1이 되도록)
}}
```

시각적 및 공간적 문맥을 분석하여 타겟 객체가 레퍼런스 객체를 기준으로 어디에 위치해야 하는지 추론해주세요.
현재 타겟과 레퍼런스 객체 정보는 다음과 같습니다:

타겟 객체: {target_class}
레퍼런스 객체: {ref_class}
{spatial_info}

사용자 요청: "{user_prompt}"
"""

        # 시스템 프롬프트 개선
        system_prompt = self._enhance_direction_inference_prompt(system_prompt, user_prompt)

        try:
            # GPT 요청
            self.logger.info(f"목표 위치 추론 요청 중... (타겟: {target_class}, 레퍼런스: {ref_class})")
            response_text = self.generate_text(system_prompt, user_prompt)
            
            # JSON 추출 및 파싱
            try:
                json_data = self._extract_json_from_response(response_text)
                
                # 필수 필드 확인
                if "goal_point" not in json_data or "x" not in json_data["goal_point"] or "y" not in json_data["goal_point"]:
                    self.logger.warning("응답에 필수 필드가 없습니다. 기본값 사용")
                    json_data = {
                        "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                        "confidence": 0.3,
                        "reasoning": "구조화된 응답에 실패했습니다. 기본 목표 지점을 사용합니다.",
                        "direction_type": "simple",
                        "direction": "front" 
                    }
                
                # 방향 타입 필드가 없으면 기본값 추가
                if "direction_type" not in json_data:
                    json_data["direction_type"] = "simple"
                    
                # 단순 방향 호환성 유지
                if "direction" not in json_data:
                    json_data["direction"] = "front"
                
                # 랜덤 방향인 경우 옵션 필드 확인
                if json_data["direction_type"] == "random" and "options" not in json_data:
                    json_data["options"] = ["left", "right"]
                    json_data["weights"] = [0.5, 0.5]
                
                self.logger.info(f"목표지점 추론 결과: {json_data}")
                return json_data
                
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON 파싱 오류: {je}")
                return {
                    "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                    "confidence": 0.0,
                    "reasoning": f"JSON 파싱 오류: {str(je)}. 기본 목표 지점을 사용합니다.",
                    "direction_type": "simple",
                    "direction": "front" 
                }
                
        except Exception as e:
            self.logger.error(f"목표 위치 추론 중 오류: {e}")
            return {
                "goal_point": {"x": 0.5, "y": 0.5, "z": 0.5},
                "confidence": 0.0,
                "reasoning": f"오류: {str(e)}. 기본 목표 지점을 사용합니다.",
                "direction_type": "simple",
                "direction": "front"
            }

    def _analyze_random_direction_intent(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        사용자 프롬프트에서 랜덤 방향 의도 분석
        
        Args:
            user_prompt: 사용자 입력 프롬프트
            
        Returns:
            Optional[Dict[str, Any]]: 랜덤 방향 정보 또는 None
        """
        if not user_prompt:
            return None
            
        # 랜덤/무작위 키워드 확인
        random_keywords = ["랜덤", "무작위", "random", "randomly", "either", "any", "아무", "어느", "어디든"]
        has_random = any(keyword in user_prompt for keyword in random_keywords)
        
        if not has_random:
            return None
        
        # 축 방향 확인
        lr_keywords = ["좌우", "좌 우", "left right", "왼쪽 오른쪽", "왼쪽이나 오른쪽"]
        fb_keywords = ["앞뒤", "앞 뒤", "front back", "앞쪽 뒤쪽", "앞쪽이나 뒤쪽"]
        
        if any(keyword in user_prompt for keyword in lr_keywords):
            return {
                "type": "random",
                "options": ["left", "right"],
                "weights": [0.5, 0.5],
                "confidence": 0.9
            }
        elif any(keyword in user_prompt for keyword in fb_keywords):
            return {
                "type": "random", 
                "options": ["front", "back"],
                "weights": [0.5, 0.5],
                "confidence": 0.9
            }
        
        # 일반 랜덤 (방향 미지정)
        return {
            "type": "random",
            "options": ["front", "back", "left", "right"],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "confidence": 0.8
        }

    def _extract_explicit_korean_direction(self, prompt: str) -> Optional[str]:
        """
        사용자 프롬프트에서 명시적인 한국어 방향 표현을 추출합니다.
        
        Args:
            prompt: 사용자 입력 프롬프트
            
        Returns:
            Optional[str]: 감지된 방향 ('right', 'left', 'front', 'back', 'above', 'below', 'side_random') 또는 None
        """
        if not prompt:
            return None
            
        # 기본 방향어 및 동의어 매핑
        direction_mapping = {
            # 오른쪽
            '오른쪽': 'right', '우측': 'right', '우': 'right', '오른편': 'right', '오른': 'right',
            # 왼쪽
            '왼쪽': 'left', '좌측': 'left', '좌': 'left', '왼편': 'left', '왼': 'left',
            # 앞
            '앞': 'front', '앞쪽': 'front', '전방': 'front', '정면': 'front', '앞부분': 'front',
            # 뒤
            '뒤': 'back', '뒤쪽': 'back', '후방': 'back', '뒷편': 'back', '뒷부분': 'back',
            # 위
            '위': 'above', '위쪽': 'above', '상단': 'above', '꼭대기': 'above', '윗쪽': 'above', '위에': 'above',
            # 아래
            '아래': 'below', '아래쪽': 'below', '하단': 'below', '밑': 'below', '밑으로': 'below', '아래에': 'below'
        }
        
        import re
        
        # 디버깅 정보 로깅
        self.logger.debug(f"방향어 추출 시작: '{prompt}'")
        
        # 1. 단순 방향어 검색 (우선순위: 오른쪽 > 왼쪽 > 앞 > 뒤 > 위 > 아래)
        priority_directions = [
            ('오른', 'right'), ('우측', 'right'), ('우', 'right'), 
            ('왼', 'left'), ('좌측', 'left'), ('좌', 'left'),
            ('앞', 'front'), ('전방', 'front'), ('정면', 'front'),
            ('뒤', 'back'), ('후방', 'back'),
            ('위', 'above'), ('상단', 'above'),
            ('아래', 'below'), ('하단', 'below'), ('밑', 'below')
        ]
        
        for kr_key, en_value in priority_directions:
            if kr_key in prompt:
                self.logger.info(f"방향어 감지: '{kr_key}' -> '{en_value}'")
                return en_value
        
        # 2. 모호한 방향어 (옆/사이드) 처리
        ambiguous_directions = ['옆', '옆쪽', '사이드', '측면', '측면으로']
        for direction in ambiguous_directions:
            if direction in prompt:
                self.logger.info(f"모호한 방향어 감지: '{direction}' -> 'side_random'")
                return 'side_random'
        
        # 방향어를 찾지 못한 경우
        self.logger.debug("명시적 방향어를 찾지 못했습니다")
        return None

    def _enhance_direction_inference_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        방향 추론을 위한 시스템 프롬프트를 향상시킵니다.
        
        한국어 방향 표현에 대한 구체적인 가이드라인을 제공하고, 
        사용자 프롬프트에서 명시적인 방향 정보를 추출하여 시스템 프롬프트에 추가합니다.
        
        Args:
            system_prompt: 원본 시스템 프롬프트
            user_prompt: 사용자 입력 프롬프트
            
        Returns:
            str: 향상된 시스템 프롬프트
        """
        if not user_prompt:
            return system_prompt
            
        # 사용자 프롬프트에서 명시적인 방향 추출
        explicit_direction = self._extract_explicit_korean_direction(user_prompt)
        
        # 한국어 방향 표현에 대한 가이드라인 추가
        kr_direction_guide = """
## 한국어 방향 표현 가이드
한국어 방향 표현을 영어로 변환할 때 다음 지침을 따르세요:

1. 기본 방향어:
   - 오른쪽/우측/우/오른편/오른 → right
   - 왼쪽/좌측/좌/왼편/왼 → left
   - 앞/앞쪽/전방/정면/앞부분 → front
   - 뒤/뒤쪽/후방/뒷편/뒷부분 → back
   - 위/위쪽/상단/꼭대기/윗쪽/위에 → above
   - 아래/아래쪽/하단/밑/밑으로/아래에 → below

2. 복합 패턴:
   - "이거/이것 [물체] 오른쪽으로" → right
   - "왼쪽에 [물체]" → left
   - "위에 놓아줘" → above
   - "[물체]를 오른쪽으로" → right

3. 모호한 표현 (문맥 고려):
   - "옆에/옆으로/사이드" → 문맥에 따라 left/right 결정
   - "이쪽/저쪽/그쪽" → 문맥에 따라 방향 결정

4. 복합 표현의 방향어 우선순위:
   - "이거 에어팟 오른쪽으로 옮겨줘" → 방향어 "오른쪽"을 우선 고려

사용자 명령의 의도를 정확히 분석하고, 명시적인 방향어가 있으면 우선적으로 고려하세요.
"""
        # 추출된 명시적 방향 정보 추가
        explicit_direction_guide = ""
        if explicit_direction:
            explicit_direction_guide = f"""
## 명시적 방향 정보
사용자 프롬프트에서 명시적인 방향 표현이 감지되었습니다: '{explicit_direction}'
이 명시적 방향 정보를 최우선으로 고려하여 의사결정에 반영하세요.

이 경우, 다른 모든 방향 관련 힌트보다 이 명시적 방향을 우선시해야 합니다.
특히 다음 명령:
"{user_prompt}"
에서는 '{explicit_direction}' 방향이 명확하게 지정되었음을 유의하세요.
"""
        
        # 복합 패턴에 대한 분석 추가
        complex_pattern_guide = ""
        if "이거" in user_prompt or "저거" in user_prompt or "그거" in user_prompt:
            complex_pattern_guide = f"""
## 복합 패턴 분석
사용자 명령 "{user_prompt}"에는 지시대명사(이거/저거/그거)가 포함되어 있습니다.
이 패턴에서는:
1. 지시대명사는 주로 대상 객체를 가리킵니다.
2. 명시적인 방향어가 있으면 해당 방향을 반영하세요.
3. 객체명이 언급된 경우 지시대명사의 실제 대상을 확인하세요.
"""
        
        # 향상된 프롬프트 구성
        enhanced_prompt = system_prompt + kr_direction_guide + explicit_direction_guide + complex_pattern_guide
        
        self.logger.debug(f"방향 추론 프롬프트가 향상되었습니다 (명시적 방향: {explicit_direction or '없음'})")
        return enhanced_prompt

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """
        LLM 응답에서 JSON 추출 및 예외 처리
        
        Args:
            response_text: LLM 응답 텍스트
            
        Returns:
            Dict: 파싱된 JSON 또는 기본값
        """
        try:
            # JSON 블록 추출 정규식
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, response_text)
            
            if match:
                json_str = match.group(1)
            else:
                # JSON 블록이 없으면 전체 응답을 JSON으로 간주
                json_str = response_text
            
            # 불필요한 문자 제거 및 파싱
            json_str = json_str.strip()
            result = json.loads(json_str)
            return result
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {e}, 응답: {response_text}")
            
            # 방향 정보 추출 실패 시 기본값
            if "direction" in response_text.lower():
                direction_match = re.search(r'direction["\s:]+([a-z_]+)', response_text, re.IGNORECASE)
                direction = direction_match.group(1).lower() if direction_match else "front"
                
                return {
                    "direction": direction,
                    "confidence": 0.5,
                    "keywords": []
                }
            
            # 관계 정보 추출 실패 시 기본값
            return {
                "direction": "front",
                "confidence": 0.3,
                "target_object": {"id": -1, "name": "unknown", "confidence": 0.1},
                "reference_object": {"id": -1, "name": "unknown", "confidence": 0.1},
                "reasoning": "파싱 실패"
            }

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        """
        GPT-4o 모델에 텍스트 프롬프트를 전송하고 응답을 반환합니다.
        
        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            temperature: 응답 다양성 조절 파라미터 (None이면 객체 초기화 시 설정값 사용)
            
        Returns:
            str: GPT-4o 모델의 응답 텍스트
        """
        try:
            self.logger.info("GPT-4o API 텍스트 생성 요청 시작")
            
            # 온도 설정 (기본값은 클래스 초기화 시 설정한 값)
            temp = temperature if temperature is not None else self.temperature
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                timeout=self.timeout
            )
            
            # 응답 추출
            if not response.choices[0].message.content:
                self.logger.error("OpenAI API에서 빈 응답이 반환되었습니다.")
                raise ValueError("OpenAI API에서 응답이 없거나 빈 응답이 반환되었습니다.")
            
            response_text = response.choices[0].message.content.strip()
            self.logger.info("GPT-4o API 텍스트 생성 응답 수신 완료")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"GPT-4o API 텍스트 생성 요청 실패: {e}")
            raise 