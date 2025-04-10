import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class GestureType(Enum):
    POINTING = "POINTING"
    GRABBING = "GRABBING"
    UNKNOWN = "UNKNOWN"

class GestureRecognizer:
    """
    MediaPipe Hand Landmarker를 사용하여 손 감지 및 제스처 인식을 수행합니다.
    """

    def __init__(self, model_path='hand_landmarker.task', num_hands=1, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        GestureRecognizer 초기화

        Args:
            model_path (str): MediaPipe Hand Landmarker 모델 파일 경로.
            num_hands (int): 감지할 최대 손 개수.
            min_hand_detection_confidence (float): 손 감지 최소 신뢰도.
            min_tracking_confidence (float): 손 추적 최소 신뢰도.
        """
        logger.info(f"GestureRecognizer 초기화 시작 (모델: {model_path})")
        self.num_hands = num_hands
        
        # 랜드마크 인덱스 (MediaPipe 공식 문서 참조)
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.INDEX_FINGER_PIP = 6
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE, # 스냅샷 처리용
            num_hands=self.num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        try:
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_landmarker_options)
            logger.info("MediaPipe Hand Landmarker 생성 성공")
        except Exception as e:
            logger.exception(f"MediaPipe Hand Landmarker 생성 실패: {e}")
            raise

    def process_frame(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        단일 이미지 프레임에서 손 감지 및 제스처 인식을 수행합니다.

        Args:
            image (np.ndarray): 처리할 이미지 (BGR 형식).

        Returns:
            List[Dict[str, Any]]: 감지된 각 손에 대한 정보 딕셔너리 리스트.
                각 딕셔너리는 다음 키를 포함합니다:
                - 'landmarks_2d': 2D 랜드마크 좌표 리스트 (각각 {'x': float, 'y': float}).
                - 'landmarks_3d': 3D 월드 랜드마크 좌표 리스트 (각각 {'x': float, 'y': float, 'z': float}).
                - 'gesture': 인식된 제스처 유형 (GestureType Enum).
                - 'pointing_vector': 가리키는 방향 벡터 (3D, POINTING 제스처일 경우, 없으면 None).
                - 'pointing_points': 손목과 검지손가락 끝의 2D 좌표 (POINTING 제스처일 경우).
                - 'pointing_points_3d': 손목과 검지손가락 끝의 3D 좌표 (POINTING 제스처일 경우).
                - 'hand_center_3d': 손바닥 중심 추정 3D 좌표 (GRABBING 제스처일 경우, 없으면 None).
                - 'handedness': 감지된 손 (왼손/오른손).
                - 'all_points_2d': 모든 손 랜드마크의 2D 좌표 리스트 [[x, y], ...] (convex hull 계산용).
        """
        # MediaPipe는 RGB 이미지를 요구하므로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        try:
            detection_result = self.landmarker.detect(mp_image)
        except Exception as e:
            logger.exception(f"MediaPipe 감지 중 오류 발생: {e}")
            return []

        results = []
        if detection_result and detection_result.hand_landmarks:
            for i, landmarks in enumerate(detection_result.hand_landmarks):
                world_landmarks = detection_result.hand_world_landmarks[i]
                handedness = detection_result.handedness[i][0].category_name # 'Left' or 'Right'

                landmarks_2d = [{'x': lm.x, 'y': lm.y} for lm in landmarks]
                landmarks_3d = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in world_landmarks]

                # 2D 랜드마크 좌표를 [x, y] 리스트로 변환 (convex hull 계산용)
                all_points_2d = [[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in landmarks]
                
                # 제스처 분류 및 관련 정보 계산
                gesture_type, pointing_vector, hand_center_3d, pointing_points, pointing_points_3d = self._classify_gesture(
                    landmarks_2d, landmarks_3d, image.shape[1], image.shape[0]
                )
                
                results.append({
                    'landmarks_2d': landmarks_2d,
                    'landmarks_3d': landmarks_3d,
                    'gesture': gesture_type,
                    'pointing_vector': pointing_vector,
                    'pointing_points': pointing_points,  # 손목과 검지 끝 2D 좌표
                    'pointing_points_3d': pointing_points_3d,  # 손목과 검지 끝 3D 좌표
                    'hand_center_3d': hand_center_3d,
                    'handedness': handedness,
                    'all_points_2d': all_points_2d  # convex hull 계산용 2D 좌표
                })
        else:
            logger.debug("프레임에서 손 감지되지 않음")

        return results

    def _classify_gesture(self, landmarks_2d: List[Dict[str, float]], landmarks_3d: List[Dict[str, float]], 
                           img_width: int, img_height: int) -> Tuple[GestureType, 
                                                                     Optional[np.ndarray], 
                                                                     Optional[np.ndarray],
                                                                     Optional[List[List[float]]],
                                                                     Optional[List[List[float]]]]:
        """
        3D 랜드마크를 기반으로 제스처를 분류합니다. (규칙 기반)

        Args:
            landmarks_2d: 2D 랜드마크 좌표 리스트.
            landmarks_3d: 3D 랜드마크 좌표 리스트.
            img_width: 이미지 너비.
            img_height: 이미지 높이.

        Returns:
            Tuple[GestureType, Optional[np.ndarray], Optional[np.ndarray], Optional[List[List[float]]], Optional[List[List[float]]]]:
                (제스처 유형, 가리키는 방향 벡터, 손 중심 좌표, 손목-검지 2D 좌표, 손목-검지 3D 좌표)
        """
        if not landmarks_3d or len(landmarks_3d) < 21 or not landmarks_2d or len(landmarks_2d) < 21:
            return GestureType.UNKNOWN, None, None, None, None
        
        # 3D 좌표 기반 계산
        # 손가락 끝 좌표
        finger_tips = {
            'thumb': np.array([landmarks_3d[self.THUMB_TIP]['x'], landmarks_3d[self.THUMB_TIP]['y'], landmarks_3d[self.THUMB_TIP]['z']]),
            'index': np.array([landmarks_3d[self.INDEX_FINGER_TIP]['x'], landmarks_3d[self.INDEX_FINGER_TIP]['y'], landmarks_3d[self.INDEX_FINGER_TIP]['z']]),
            'middle': np.array([landmarks_3d[self.MIDDLE_FINGER_TIP]['x'], landmarks_3d[self.MIDDLE_FINGER_TIP]['y'], landmarks_3d[self.MIDDLE_FINGER_TIP]['z']]),
            'ring': np.array([landmarks_3d[self.RING_FINGER_TIP]['x'], landmarks_3d[self.RING_FINGER_TIP]['y'], landmarks_3d[self.RING_FINGER_TIP]['z']]),
            'pinky': np.array([landmarks_3d[self.PINKY_TIP]['x'], landmarks_3d[self.PINKY_TIP]['y'], landmarks_3d[self.PINKY_TIP]['z']]),
        }
        
        # 손목 좌표
        wrist = np.array([landmarks_3d[self.WRIST]['x'], landmarks_3d[self.WRIST]['y'], landmarks_3d[self.WRIST]['z']])
        
        # 검지 중간 관절 좌표 (방향 계산용)
        index_pip = np.array([landmarks_3d[self.INDEX_FINGER_PIP]['x'], landmarks_3d[self.INDEX_FINGER_PIP]['y'], landmarks_3d[self.INDEX_FINGER_PIP]['z']])

        # 2D 좌표 추출 (이미지 좌표계로 변환)
        wrist_2d = [landmarks_2d[self.WRIST]['x'] * img_width, landmarks_2d[self.WRIST]['y'] * img_height]
        index_tip_2d = [landmarks_2d[self.INDEX_FINGER_TIP]['x'] * img_width, landmarks_2d[self.INDEX_FINGER_TIP]['y'] * img_height]
        
        # 3D 좌표 리스트 형태로 준비
        wrist_3d = [wrist[0], wrist[1], wrist[2]]
        index_tip_3d = [finger_tips['index'][0], finger_tips['index'][1], finger_tips['index'][2]]

        # 손가락 구부러짐 여부 판단 (단순화: 손목과의 거리 비교)
        dist_wrist_index = np.linalg.norm(finger_tips['index'] - wrist)
        dist_wrist_middle = np.linalg.norm(finger_tips['middle'] - wrist)
        dist_wrist_ring = np.linalg.norm(finger_tips['ring'] - wrist)
        dist_wrist_pinky = np.linalg.norm(finger_tips['pinky'] - wrist)
        
        # 검지가 다른 손가락보다 상대적으로 펴져 있는지 확인 (임계값은 실험적으로 조정 필요)
        is_index_extended = (dist_wrist_index > dist_wrist_middle * 0.8 and
                             dist_wrist_index > dist_wrist_ring * 0.8 and
                             dist_wrist_index > dist_wrist_pinky * 0.8)

        # 다른 손가락(중지, 약지, 새끼)이 구부러져 있는지 확인
        # (손목과의 거리가 검지보다 확연히 짧은지) - 임계값 조정 필요
        are_others_flexed = (dist_wrist_middle < dist_wrist_index * 0.7 and
                             dist_wrist_ring < dist_wrist_index * 0.7 and
                             dist_wrist_pinky < dist_wrist_index * 0.7)

        pointing_vector = None
        hand_center_3d = None
        pointing_points = None
        pointing_points_3d = None

        # 가리키기(POINTING) 조건: 검지는 펴고, 나머지 손가락은 구부림
        if is_index_extended and are_others_flexed:
            gesture = GestureType.POINTING
            # 방향 벡터 계산 (검지 끝 - 검지 중간 관절) - 좀 더 안정적인 방향 제공 가능
            pointing_vector = finger_tips['index'] - index_pip
            norm = np.linalg.norm(pointing_vector)
            if norm > 1e-6: # ZeroDivisionError 방지
                pointing_vector = pointing_vector / norm
            else:
                pointing_vector = None # 벡터 계산 불가
            
            # 2D 및 3D 포인팅 좌표 설정
            pointing_points = [wrist_2d, index_tip_2d]
            pointing_points_3d = [wrist_3d, index_tip_3d]
            
            logger.debug(f"제스처 분류: POINTING, 방향 벡터: {pointing_vector}")
            logger.debug(f"손목 2D: {wrist_2d}, 검지 끝 2D: {index_tip_2d}")

        # 쥐기(GRABBING) 조건: 모든 손가락 끝이 손목에 가까움 (단순화)
        # 임계값 조정 필요
        elif (dist_wrist_index < dist_wrist_middle * 1.2 and # 모든 손가락이 비슷한 수준으로 구부러짐
              dist_wrist_middle < dist_wrist_ring * 1.2 and
              dist_wrist_ring < dist_wrist_pinky * 1.2 and
              dist_wrist_index < np.mean([dist_wrist_middle, dist_wrist_ring, dist_wrist_pinky]) * 1.5): # 평균적으로 구부러짐 확인
            gesture = GestureType.GRABBING
            # 손 중심 계산 (손목과 손가락 끝점들의 평균)
            all_tips = np.array([finger_tips[f] for f in finger_tips])
            hand_center_3d = np.mean(np.vstack([wrist, all_tips]), axis=0)
            logger.debug(f"제스처 분류: GRABBING, 손 중심: {hand_center_3d}")
        else:
            gesture = GestureType.UNKNOWN
            logger.debug("제스처 분류: UNKNOWN")

        return gesture, pointing_vector, hand_center_3d, pointing_points, pointing_points_3d

    def close(self):
        """Hand Landmarker 리소스를 해제합니다."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
            logger.info("MediaPipe Hand Landmarker 리소스 해제됨")

# 사용 예시 (테스트용)
if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 모델 파일 경로 설정 (실제 경로로 변경 필요)
    # mediapipe 웹사이트에서 hand_landmarker.task 파일 다운로드 필요
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models
    model_file = 'hand_landmarker.task' 
    
    try:
        # 이미지 로드 (테스트용 이미지 경로 설정)
        image_path = 'path/to/your/test_image_with_hand.jpg' 
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"테스트 이미지를 로드할 수 없습니다: {image_path}")
            exit()

        # GestureRecognizer 인스턴스 생성
        recognizer = GestureRecognizer(model_path=model_file)
        
        # 제스처 인식 수행
        results = recognizer.process_frame(img)

        if results:
            logger.info(f"{len(results)}개의 손 감지됨:")
            for i, hand_info in enumerate(results):
                logger.info(f"--- 손 {i+1} ({hand_info['handedness']}) ---")
                logger.info(f"  제스처: {hand_info['gesture']}")
                if hand_info['gesture'] == GestureType.POINTING:
                    logger.info(f"  가리키는 방향 벡터: {hand_info['pointing_vector']}")
                    logger.info(f"  손목 2D: {hand_info['pointing_points'][0]}, 검지 끝 2D: {hand_info['pointing_points'][1]}")
                    logger.info(f"  손목 3D: {hand_info['pointing_points_3d'][0]}, 검지 끝 3D: {hand_info['pointing_points_3d'][1]}")
                elif hand_info['gesture'] == GestureType.GRABBING:
                    logger.info(f"  손 중심 좌표 (3D): {hand_info['hand_center_3d']}")
                # logger.debug(f"  2D 랜드마크: {hand_info['landmarks_2d']}") # 너무 길어서 주석 처리
                # logger.debug(f"  3D 랜드마크: {hand_info['landmarks_3d']}") # 너무 길어서 주석 처리
        else:
            logger.info("이미지에서 손이 감지되지 않았습니다.")
            
        # 리소스 해제
        recognizer.close()

    except FileNotFoundError:
        logger.error(f"Hand Landmarker 모델 파일을 찾을 수 없습니다: {model_file}")
        logger.error("MediaPipe 웹사이트에서 모델을 다운로드하고 경로를 올바르게 설정하세요.")
    except ImportError:
        logger.error("MediaPipe 라이브러리가 설치되지 않았습니다. 'pip install mediapipe' 명령어로 설치하세요.")
    except Exception as e:
        logger.exception(f"테스트 중 예외 발생: {e}") 