import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Optional, Tuple, List, Dict, Any

class HandDetector:
    """
    MediaPipe Hands를 활용한 손 감지 및 제스처 인식 클래스
    
    손가락 관절 감지, 포인팅/쥐기 제스처 인식, 방향 벡터 계산 기능 제공
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5, 
                 max_num_hands: int = 2,
                 detection_threshold: float = 0.5,
                 static_image_mode: bool = False,
                 min_tracking_confidence: float = 0.5):
        """
        HandDetector 초기화
        
        Args:
            confidence_threshold: 손 감지 신뢰도 임계값
            max_num_hands: 최대 감지할 손 개수
            detection_threshold: 제스처 감지 신뢰도 임계값
            static_image_mode: 정적 이미지 모드 여부 (True: 매 프레임 감지, False: 추적 사용)
            min_tracking_confidence: 손 추적 최소 신뢰도
        """
        self.confidence_threshold = confidence_threshold
        self.max_num_hands = max_num_hands
        self.detection_threshold = detection_threshold
        
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hands 모델 설정
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 제스처 유형 정의
        self.GESTURE_POINTING = "pointing"
        self.GESTURE_GRABBING = "grabbing"
        self.GESTURE_NONE = "none"
        
        # 손가락 관절 인덱스 정의
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6  # 검지 중간 관절
        self.INDEX_MCP = 5  # 검지 시작 관절
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # 최근 제스처 이력 (안정성 향상을 위해 사용)
        self.gesture_history = []
        self.history_max_len = 5
        
        print("HandDetector 초기화됨")
    
    def detect_hands(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        프레임에서 손을 감지하고 제스처를 분석
        
        Args:
            frame: 분석할 비디오 프레임 (BGR)
            
        Returns:
            Dict: {
                "hands": List[Dict],  # 감지된 각 손의 정보
                "pointing_vector": Optional[Tuple],  # 가리키는 방향 벡터 (있을 경우)
                "gesture_type": str,  # "pointing", "grabbing", "none" 중 하나
                "confidence": float   # 제스처 신뢰도
            }
        """
        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        # 손 감지 처리
        results = self.hands.process(frame_rgb)
        
        # 결과 초기화
        response = {
            "hands": [],
            "pointing_vector": None,
            "gesture_type": self.GESTURE_NONE,
            "confidence": 0.0
        }
        
        # 손이 감지되지 않은 경우
        if not results.multi_hand_landmarks:
            return response
        
        # 각 손에 대한 정보 처리
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= self.max_num_hands:
                break
                
            # 손 정보 구조화
            hand_info = self._extract_hand_info(hand_landmarks, frame_width, frame_height)
            
            # 손 정보 추가
            response["hands"].append(hand_info)
            
            # 제스처 분석
            gesture_type, confidence = self._classify_gesture(hand_landmarks)
            hand_info["gesture_type"] = gesture_type
            hand_info["confidence"] = confidence
            
            # 가장 높은 신뢰도를 가진 제스처 선택
            if confidence > response["confidence"]:
                response["gesture_type"] = gesture_type
                response["confidence"] = confidence
                
                # 포인팅 제스처인 경우 방향 벡터 계산
                if gesture_type == self.GESTURE_POINTING:
                    origin, direction = self.get_pointing_ray(
                        hand_landmarks, (frame_width, frame_height)
                    )
                    response["pointing_vector"] = (origin, direction)
                    hand_info["pointing_vector"] = (origin, direction)
        
        # 제스처 이력 업데이트 (안정성 향상)
        self._update_gesture_history(response["gesture_type"])
        response["gesture_type"] = self._get_stable_gesture()
        
        return response
    
    def visualize_hands(self, 
                       frame: np.ndarray, 
                       hand_data: Dict[str, Any]) -> np.ndarray:
        """
        감지된 손과 제스처 시각화
        
        Args:
            frame: 원본 비디오 프레임
            hand_data: detect_hands()에서 반환된 손 데이터
            
        Returns:
            np.ndarray: 시각화된 프레임
        """
        vis_frame = frame.copy()
        
        # 손이 감지되지 않은 경우
        if not hand_data["hands"]:
            return vis_frame
        
        # MediaPipe 시각화 도구를 사용해 손 랜드마크 표시
        for hand_info in hand_data["hands"]:
            landmarks = hand_info["landmarks_mp"]
            
            # 랜드마크 그리기
            self.mp_drawing.draw_landmarks(
                vis_frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 제스처 유형 표시
            gesture_text = f"{hand_info['gesture_type']} ({hand_info['confidence']:.2f})"
            wrist_pos = hand_info["joints"][self.WRIST]
            cv2.putText(
                vis_frame, 
                gesture_text, 
                (int(wrist_pos[0]), int(wrist_pos[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # 포인팅 방향 벡터 시각화
            if hand_info["gesture_type"] == self.GESTURE_POINTING and "pointing_vector" in hand_info:
                origin, direction = hand_info["pointing_vector"]
                end_point = (
                    int(origin[0] + direction[0] * 150),
                    int(origin[1] + direction[1] * 150)
                )
                cv2.line(vis_frame, 
                         (int(origin[0]), int(origin[1])), 
                         end_point, 
                         (0, 0, 255), 2)
                cv2.circle(vis_frame, end_point, 5, (0, 0, 255), -1)
        
        return vis_frame
    
    def get_pointing_ray(self, 
                        hand_landmarks, 
                        image_shape: Tuple[int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        손가락 포인팅 방향에서 3D 레이(ray) 계산
        
        Args:
            hand_landmarks: MediaPipe 손 랜드마크
            image_shape: 이미지 크기 (width, height)
        
        Returns:
            Tuple: (origin_point, direction_vector)
                - origin_point: (x, y) 형태의 시작점 (검지 끝)
                - direction_vector: 정규화된 (dx, dy) 방향 벡터
        """
        # 이미지 크기
        width, height = image_shape
        
        # 관절 위치 추출
        index_tip = (hand_landmarks.landmark[self.INDEX_TIP].x * width, 
                    hand_landmarks.landmark[self.INDEX_TIP].y * height)
        
        index_pip = (hand_landmarks.landmark[self.INDEX_PIP].x * width, 
                    hand_landmarks.landmark[self.INDEX_PIP].y * height)
                    
        # 방향 벡터 계산 (PIP -> TIP 방향)
        dx = index_tip[0] - index_pip[0]
        dy = index_tip[1] - index_pip[1]
        
        # 벡터 정규화
        magnitude = math.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
            
        return (index_tip, (dx, dy))
    
    def _extract_hand_info(self, 
                          hand_landmarks, 
                          frame_width: int, 
                          frame_height: int) -> Dict[str, Any]:
        """
        손 랜드마크에서 관절 정보 추출
        
        Args:
            hand_landmarks: MediaPipe 손 랜드마크
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            
        Returns:
            Dict: 손 관절 및 상태 정보
        """
        joints = {}
        
        # 각 관절의 좌표 변환 (상대좌표 -> 절대좌표)
        for i, landmark in enumerate(hand_landmarks.landmark):
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            z = landmark.z
            joints[i] = (x, y, z)
        
        return {
            "joints": joints,
            "landmarks_mp": hand_landmarks,
            "gesture_type": self.GESTURE_NONE,
            "confidence": 0.0
        }
    
    def _classify_gesture(self, hand_landmarks) -> Tuple[str, float]:
        """
        손 랜드마크 기반 제스처 분류
        
        Args:
            hand_landmarks: MediaPipe 손 랜드마크
            
        Returns:
            Tuple[str, float]: (제스처 유형, 신뢰도)
        """
        # 각 손가락 굽힘 상태 확인
        finger_states = self._check_finger_states(hand_landmarks)
        
        # 가리키기(Pointing) 제스처 확인: 검지만 펴고 나머지는 접힌 상태
        if (finger_states[1] and not finger_states[0] and 
            not finger_states[2] and not finger_states[3] and 
            not finger_states[4]):
            return self.GESTURE_POINTING, 0.9
        
        # 쥐기(Grabbing) 제스처 확인: 모든 손가락이 접힌 상태
        elif (not finger_states[0] and not finger_states[1] and 
              not finger_states[2] and not finger_states[3] and 
              not finger_states[4]):
            return self.GESTURE_GRABBING, 0.8
        
        # 기타 제스처
        else:
            return self.GESTURE_NONE, 0.5
    
    def _check_finger_states(self, hand_landmarks) -> List[bool]:
        """
        각 손가락의 펴짐/접힘 상태 확인
        
        Args:
            hand_landmarks: MediaPipe 손 랜드마크
            
        Returns:
            List[bool]: 5개 손가락의 펴짐 상태 (True: 펴짐, False: 접힘)
        """
        fingers_open = [False] * 5
        
        # 엄지 상태 확인 (복잡한 규칙 필요, 단순화를 위해 CMC-MCP-IP 각도 사용)
        thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        thumb_cmc = hand_landmarks.landmark[1]
        
        # 엄지가 다른 손가락에 비해 x 좌표가 바깥쪽에 있으면 펴진 것으로 판단
        # (실제로는 더 복잡한 각도 계산이 필요할 수 있음)
        fingers_open[0] = thumb_tip.x < thumb_ip.x
        
        # 나머지 손가락 상태 확인 (MCP보다 PIP가 위에 있으면 펴진 것으로 판단)
        for finger_id in range(1, 5):
            pip_id = finger_id * 4
            mcp_id = pip_id - 2
            
            if hand_landmarks.landmark[pip_id].y < hand_landmarks.landmark[mcp_id].y:
                fingers_open[finger_id] = True
        
        return fingers_open
    
    def _update_gesture_history(self, gesture: str) -> None:
        """
        제스처 이력 업데이트 (안정적인 제스처 인식을 위해)
        
        Args:
            gesture: 현재 감지된 제스처
        """
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_max_len:
            self.gesture_history.pop(0)
    
    def _get_stable_gesture(self) -> str:
        """
        이력 기반 안정적인 제스처 판단
        
        Returns:
            str: 안정적인 제스처 유형
        """
        if not self.gesture_history:
            return self.GESTURE_NONE
            
        # 가장 많이 나타난 제스처 선택
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
        return max(gesture_counts.items(), key=lambda x: x[1])[0]
    
    def ray_object_intersection(self, 
                               ray_origin: Tuple[float, float], 
                               ray_direction: Tuple[float, float], 
                               objects_2d: List[Dict],
                               margin_degrees: float = 15) -> List[Dict]:
        """
        방향 벡터와 2D 객체 간 교차 계산
        
        Args:
            ray_origin: 레이 시작점 (x, y)
            ray_direction: 레이 방향 벡터 (dx, dy)
            objects_2d: 2D 객체 목록 (각 객체는 'bbox' 키를 포함해야 함)
            margin_degrees: 허용 각도 마진 (도)
            
        Returns:
            List: 교차하는 객체 목록 (거리 순 정렬)
        """
        intersections = []
        margin_rad = math.radians(margin_degrees)
        
        for obj in objects_2d:
            # 객체 중심점 계산
            x1, y1, x2, y2 = obj['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 레이 원점에서 객체 중심까지의 벡터
            to_obj_x = center_x - ray_origin[0]
            to_obj_y = center_y - ray_origin[1]
            
            # 벡터 크기 계산
            to_obj_magnitude = math.sqrt(to_obj_x**2 + to_obj_y**2)
            
            # 객체까지의 거리가 0이면 건너뜀
            if to_obj_magnitude < 1e-6:
                continue
                
            # 두 벡터 사이의 각도 계산 (코사인 유사도)
            dot_product = ray_direction[0] * to_obj_x / to_obj_magnitude + ray_direction[1] * to_obj_y / to_obj_magnitude
            angle = math.acos(max(-1.0, min(1.0, dot_product)))
            
            # 허용 각도 내에 있는 경우
            if angle <= margin_rad:
                # 교차점까지의 거리 및 각도 저장
                intersections.append({
                    'object': obj,
                    'distance': to_obj_magnitude,
                    'angle': math.degrees(angle)
                })
        
        # 거리순으로 정렬
        intersections.sort(key=lambda x: x['distance'])
        
        return intersections 