import re
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

class GestureAnalyzer:
    """
    손 제스처와 지시대명사를 연결하여 사용자의 의도를 해석하는 클래스
    
    손가락 방향과 객체 매핑, 지시대명사 추출 및 제스처 연결 기능 제공
    """
    
    def __init__(self, 
                 spatial_analyzer,
                 max_pointing_distance_cm: float = 300,
                 ray_margin_degrees: float = 15):
        """
        GestureAnalyzer 초기화
        
        Args:
            spatial_analyzer: 3D 공간 분석 객체 (SpatialAnalyzer 인스턴스)
            max_pointing_distance_cm: 최대 포인팅 거리 (cm)
            ray_margin_degrees: 레이 교차 검사 시 허용 각도 마진 (도)
        """
        self.spatial_analyzer = spatial_analyzer
        self.max_pointing_distance_cm = max_pointing_distance_cm
        self.ray_margin_degrees = ray_margin_degrees
        
        # 한글 지시대명사 패턴
        self.deictic_patterns = {
            'ko': r'(이거|저거|그거|이것|저것|그것|여기|저기|거기|이|저|그)',
            'en': r'\b(this|that|these|those|here|there)\b'
        }
        
        # 지시어-제스처 연결 최대 시간 차이 (초)
        self.max_time_diff = 3.0
        
        # 최근 지시어 이력 [{'word': str, 'timestamp': datetime}, ...]
        self.deictic_history = []
        
        # 최근 제스처 이력 [{'gesture': Dict, 'timestamp': datetime}, ...]
        self.gesture_history = []
        
        # 이력 저장 최대 길이
        self.history_max_len = 10
        
        print("GestureAnalyzer 초기화됨")
    
    def link_gesture_to_prompt(self, 
                              gesture_data: Dict[str, Any], 
                              prompt: str, 
                              detected_objects: List[Dict], 
                              depth_map: np.ndarray,
                              timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        제스처와 프롬프트 연결 분석
        
        Args:
            gesture_data: HandDetector에서 반환된 제스처 정보
            prompt: 사용자 프롬프트 텍스트
            detected_objects: 감지된 객체 목록
            depth_map: 깊이 맵
            timestamp: 현재 시간 (None인 경우 현재 시간 사용)
            
        Returns:
            Dict: {
                "target_object": Dict,  # 지정된 타겟 객체
                "confidence": float,    # 매칭 신뢰도
                "deictic_words": List,  # 감지된 지시대명사 목록
                "gesture_type": str     # 제스처 유형
            }
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 결과 초기화
        result = {
            "target_object": None,
            "confidence": 0.0,
            "deictic_words": [],
            "gesture_type": "none"
        }
        
        # 프롬프트에서 지시대명사 추출
        deictic_words = self.extract_deictic_words(prompt)
        result["deictic_words"] = deictic_words
        
        # 지시어 이력 업데이트
        for word in deictic_words:
            self._update_deictic_history(word, timestamp)
        
        # 제스처 이력 업데이트
        self._update_gesture_history(gesture_data, timestamp)
        
        # 제스처 유형 저장
        result["gesture_type"] = gesture_data.get("gesture_type", "none")
        
        # 포인팅 제스처 처리
        if gesture_data.get("gesture_type") == "pointing" and gesture_data.get("pointing_vector"):
            # 감지된 객체가 없으면 반환
            if not detected_objects:
                return result
                
            # 포인팅 방향과 객체 교차 계산
            origin, direction = gesture_data["pointing_vector"]
            intersections = self._find_intersecting_objects(origin, direction, detected_objects)
            
            # 교차하는 객체가 있으면 거리가 가장 가까운 객체 선택
            if intersections:
                target_obj = intersections[0]["object"]
                angle = intersections[0]["angle"]
                distance = intersections[0]["distance"]
                
                # 신뢰도 계산 (각도와 거리 기반)
                angle_confidence = 1.0 - (angle / self.ray_margin_degrees)
                distance_confidence = 1.0 - min(1.0, distance / 500)  # 500px 기준으로 정규화
                gesture_confidence = angle_confidence * 0.7 + distance_confidence * 0.3
                
                result["target_object"] = target_obj
                result["confidence"] = gesture_confidence
                return result
        
        # 쥐기 제스처 처리
        elif gesture_data.get("gesture_type") == "grabbing" and gesture_data.get("hands"):
            # 주요 손 정보 가져오기
            hand_info = gesture_data["hands"][0] if gesture_data["hands"] else None
            
            if hand_info and "joints" in hand_info:
                # 손 중심 위치 계산
                wrist_pos = hand_info["joints"].get(0, (0, 0, 0))
                hand_center_x, hand_center_y = wrist_pos[0], wrist_pos[1]
                
                # 손 위치와 가장 가까운 객체 찾기
                closest_obj = None
                min_distance = float('inf')
                
                for obj in detected_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    obj_center_x = (x1 + x2) / 2
                    obj_center_y = (y1 + y2) / 2
                    
                    # 거리 계산
                    distance = math.sqrt((hand_center_x - obj_center_x)**2 + (hand_center_y - obj_center_y)**2)
                    
                    # 객체가 손 위치와 가까울 경우
                    if distance < min_distance:
                        min_distance = distance
                        closest_obj = obj
                
                # 일정 거리 이내에 객체가 있으면 선택
                if closest_obj and min_distance < 100:  # 100px 임계값
                    # 거리 기반 신뢰도 계산
                    grab_confidence = 1.0 - min(1.0, min_distance / 100)
                    
                    result["target_object"] = closest_obj
                    result["confidence"] = grab_confidence
                    return result
        
        # 지시대명사만 있는 경우, 최근 제스처와 연결 시도
        if deictic_words and not result["target_object"]:
            # 가장 최근의 제스처 가져오기
            recent_gestures = self._get_recent_gestures(timestamp, self.max_time_diff)
            
            if recent_gestures:
                # 신뢰도가 가장 높은 제스처 선택
                best_gesture = max(recent_gestures, key=lambda g: g["gesture"].get("confidence", 0))
                
                # 다시 포인팅/쥐기 계산 수행
                result_from_history = self.link_gesture_to_prompt(
                    best_gesture["gesture"],
                    prompt,
                    detected_objects,
                    depth_map,
                    best_gesture["timestamp"]
                )
                
                # 결과가 있으면 사용
                if result_from_history["target_object"]:
                    # 시간 지연에 따른 신뢰도 감소
                    time_diff = (timestamp - best_gesture["timestamp"]).total_seconds()
                    time_penalty = min(0.5, time_diff / self.max_time_diff)
                    
                    result = result_from_history
                    result["confidence"] *= (1.0 - time_penalty)
        
        return result
    
    def extract_deictic_words(self, text: str, lang: str = 'ko') -> List[str]:
        """
        텍스트에서 지시대명사 추출
        
        Args:
            text: 분석할 텍스트
            lang: 언어 코드 ('ko', 'en')
            
        Returns:
            List[str]: 추출된 지시대명사 목록
        """
        if not text:
            return []
            
        pattern = self.deictic_patterns.get(lang, self.deictic_patterns['ko'])
        matches = re.findall(pattern, text)
        
        return list(set(matches))  # 중복 제거
    
    def analyze_deictic_context(self, text: str, deictic_word: str) -> Dict[str, Any]:
        """
        지시대명사의 문맥 분석
        
        Args:
            text: 전체 텍스트
            deictic_word: 분석할 지시대명사
            
        Returns:
            Dict: {
                "pre_context": str,  # 지시어 앞 문맥
                "post_context": str, # 지시어 뒤 문맥
                "is_subject": bool,  # 주어 여부
                "is_object": bool,   # 목적어 여부
                "distance_type": str # 거리 유형 (near, far, mid)
            }
        """
        # 지시대명사 위치 찾기
        pattern = r'\b' + re.escape(deictic_word) + r'\b'
        match = re.search(pattern, text)
        
        if not match:
            return {
                "pre_context": "",
                "post_context": "",
                "is_subject": False,
                "is_object": False,
                "distance_type": "mid"
            }
        
        # 문맥 추출
        start, end = match.span()
        pre_context = text[:start].strip()
        post_context = text[end:].strip()
        
        # 주어 여부 확인 (문장 시작 또는 조사 뒤)
        is_subject = len(pre_context) == 0 or pre_context.endswith(("이", "가", "은", "는"))
        
        # 목적어 여부 확인 (조사 앞)
        is_object = post_context.startswith(("을", "를", "이", "가"))
        
        # 거리 유형 파악
        distance_type = "mid"
        if deictic_word in ["이거", "이것", "여기", "이"]:
            distance_type = "near"
        elif deictic_word in ["저거", "저것", "저기", "저"]:
            distance_type = "far"
        
        return {
            "pre_context": pre_context,
            "post_context": post_context,
            "is_subject": is_subject,
            "is_object": is_object,
            "distance_type": distance_type
        }
    
    def _find_intersecting_objects(self, 
                                 ray_origin: Tuple[float, float], 
                                 ray_direction: Tuple[float, float], 
                                 objects: List[Dict]) -> List[Dict]:
        """
        레이 방향과 교차하는 객체 찾기
        
        Args:
            ray_origin: 레이 시작점
            ray_direction: 레이 방향 벡터
            objects: 객체 목록
            
        Returns:
            List: 교차하는 객체 목록 (거리순 정렬)
        """
        intersections = []
        margin_rad = math.radians(self.ray_margin_degrees)
        
        for obj in objects:
            # 객체 중심점 계산
            x1, y1, x2, y2 = obj.get('bbox', [0, 0, 0, 0])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 레이 원점에서 객체 중심까지의 벡터
            to_obj_x = center_x - ray_origin[0]
            to_obj_y = center_y - ray_origin[1]
            
            # 벡터 크기 계산
            magnitude = math.sqrt(to_obj_x**2 + to_obj_y**2)
            
            # 객체까지의 거리가 0이면 건너뜀
            if magnitude < 1e-6:
                continue
                
            # 벡터 정규화
            to_obj_x /= magnitude
            to_obj_y /= magnitude
            
            # 두 벡터 사이의 각도 계산 (코사인 유사도)
            dot_product = ray_direction[0] * to_obj_x + ray_direction[1] * to_obj_y
            angle = math.acos(max(-1.0, min(1.0, dot_product)))
            
            # 허용 각도 내에 있는 경우
            if angle <= margin_rad:
                # 교차점까지의 거리 및 각도 저장
                intersections.append({
                    'object': obj,
                    'distance': magnitude,
                    'angle': math.degrees(angle)
                })
        
        # 거리순으로 정렬
        intersections.sort(key=lambda x: x['distance'])
        
        return intersections
    
    def _update_deictic_history(self, word: str, timestamp: datetime) -> None:
        """
        지시대명사 이력 업데이트
        
        Args:
            word: 지시대명사
            timestamp: 시간
        """
        self.deictic_history.append({
            'word': word,
            'timestamp': timestamp
        })
        
        # 최대 길이 제한
        if len(self.deictic_history) > self.history_max_len:
            self.deictic_history.pop(0)
    
    def _update_gesture_history(self, gesture: Dict[str, Any], timestamp: datetime) -> None:
        """
        제스처 이력 업데이트
        
        Args:
            gesture: 제스처 정보
            timestamp: 시간
        """
        # 의미 있는 제스처만 저장
        if gesture.get("gesture_type", "none") != "none":
            self.gesture_history.append({
                'gesture': gesture,
                'timestamp': timestamp
            })
            
            # 최대 길이 제한
            if len(self.gesture_history) > self.history_max_len:
                self.gesture_history.pop(0)
    
    def _get_recent_gestures(self, current_time: datetime, max_seconds: float) -> List[Dict]:
        """
        최근 제스처 목록 가져오기
        
        Args:
            current_time: 현재 시간
            max_seconds: 최대 시간 차이 (초)
            
        Returns:
            List: 최근 제스처 목록
        """
        time_threshold = current_time - timedelta(seconds=max_seconds)
        return [g for g in self.gesture_history if g['timestamp'] >= time_threshold]
    
    def _get_recent_deictic_words(self, current_time: datetime, max_seconds: float) -> List[Dict]:
        """
        최근 지시대명사 목록 가져오기
        
        Args:
            current_time: 현재 시간
            max_seconds: 최대 시간 차이 (초)
            
        Returns:
            List: 최근 지시대명사 목록
        """
        time_threshold = current_time - timedelta(seconds=max_seconds)
        return [d for d in self.deictic_history if d['timestamp'] >= time_threshold]
    
    def create_llm_prompt_extension(self, 
                                  gesture_data: Dict[str, Any], 
                                  prompt: str, 
                                  target_info: Dict[str, Any]) -> str:
        """
        LLM 프롬프트 확장 생성
        
        Args:
            gesture_data: 제스처 정보
            prompt: 원본 프롬프트
            target_info: 타겟 객체 정보
            
        Returns:
            str: LLM 입력용 확장 프롬프트
        """
        # 지시대명사 추출
        deictic_words = self.extract_deictic_words(prompt)
        deictic_text = ", ".join(deictic_words) if deictic_words else "없음"
        
        # 제스처 정보 형식화
        gesture_type = gesture_data.get("gesture_type", "none")
        gesture_text = {
            "pointing": "가리키기",
            "grabbing": "쥐기",
            "none": "없음"
        }.get(gesture_type, "없음")
        
        # 대상 객체 정보 형식화
        target_obj_text = "없음"
        target_confidence = 0.0
        
        if target_info and target_info.get("target_object"):
            obj = target_info["target_object"]
            target_obj_text = f"{obj.get('name', '알 수 없는 객체')} (신뢰도: {target_info.get('confidence', 0.0):.2f})"
            target_confidence = target_info.get("confidence", 0.0)
        
        # 확장 프롬프트 생성
        extension = f"""
사용자의 제스처 정보:
- 제스처 유형: {gesture_text}
- 지시대명사: {deictic_text}
- 가리키는 객체: {target_obj_text}
- 제스처-객체 매칭 신뢰도: {target_confidence:.2f}

사용자의 발화:
"{prompt}"

위 정보를 바탕으로 사용자가 지시하는 타겟 객체와 목표 위치를 결정하세요.
지시대명사가 있고 제스처가 인식된 경우, 두 정보를 통합하여 분석하세요.
지시대명사와 제스처 중 더 명확한 정보에 더 높은 가중치를 두세요.
        """
        
        return extension 