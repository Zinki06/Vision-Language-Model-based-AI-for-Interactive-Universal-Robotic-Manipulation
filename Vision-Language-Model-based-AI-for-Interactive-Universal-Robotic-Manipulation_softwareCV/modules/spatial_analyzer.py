from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import logging
import re

class SpatialAnalyzer:
    """객체 간 공간 관계 분석 및 목표 위치 계산 클래스"""
    
    def __init__(self, logger=None):
        """
        SpatialAnalyzer 초기화
        
        Args:
            logger: 로깅을 위한 로거 객체. None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger("SpatialAnalyzer")
        
        # 방향 관련 상수
        self.DIRECTION_FRONT = "front"
        self.DIRECTION_BACK = "back"
        self.DIRECTION_LEFT = "left"
        self.DIRECTION_RIGHT = "right"
        self.DIRECTION_ABOVE = "above"
        self.DIRECTION_BELOW = "below"
        
        # 방향 키워드 매핑
        self.direction_keywords = {
            # 한국어 키워드
            "앞": self.DIRECTION_FRONT,
            "앞쪽": self.DIRECTION_FRONT,
            "앞으로": self.DIRECTION_FRONT,
            "앞에": self.DIRECTION_FRONT,
            "전방": self.DIRECTION_FRONT,
            
            "뒤": self.DIRECTION_BACK,
            "뒤쪽": self.DIRECTION_BACK,
            "뒤로": self.DIRECTION_BACK,
            "뒤에": self.DIRECTION_BACK,
            "후방": self.DIRECTION_BACK,
            
            "왼쪽": self.DIRECTION_LEFT,
            "좌측": self.DIRECTION_LEFT,
            "왼": self.DIRECTION_LEFT,
            "왼쪽으로": self.DIRECTION_LEFT,
            "왼편": self.DIRECTION_LEFT,
            
            "오른쪽": self.DIRECTION_RIGHT,
            "우측": self.DIRECTION_RIGHT,
            "오른": self.DIRECTION_RIGHT,
            "오른쪽으로": self.DIRECTION_RIGHT,
            "오른편": self.DIRECTION_RIGHT,
            
            "위": self.DIRECTION_ABOVE,
            "위쪽": self.DIRECTION_ABOVE,
            "위로": self.DIRECTION_ABOVE,
            "위에": self.DIRECTION_ABOVE,
            "상단": self.DIRECTION_ABOVE,
            
            "아래": self.DIRECTION_BELOW,
            "아래쪽": self.DIRECTION_BELOW,
            "아래로": self.DIRECTION_BELOW,
            "아래에": self.DIRECTION_BELOW,
            "하단": self.DIRECTION_BELOW,
            
            # 영어 키워드
            "front": self.DIRECTION_FRONT,
            "forward": self.DIRECTION_FRONT,
            "in front of": self.DIRECTION_FRONT,
            
            "back": self.DIRECTION_BACK,
            "backward": self.DIRECTION_BACK,
            "behind": self.DIRECTION_BACK,
            
            "left": self.DIRECTION_LEFT,
            "to the left": self.DIRECTION_LEFT,
            "on the left": self.DIRECTION_LEFT,
            
            "right": self.DIRECTION_RIGHT,
            "to the right": self.DIRECTION_RIGHT,
            "on the right": self.DIRECTION_RIGHT,
            
            "above": self.DIRECTION_ABOVE,
            "over": self.DIRECTION_ABOVE,
            "on top of": self.DIRECTION_ABOVE,
            "up": self.DIRECTION_ABOVE,
            
            "below": self.DIRECTION_BELOW,
            "under": self.DIRECTION_BELOW,
            "underneath": self.DIRECTION_BELOW,
            "down": self.DIRECTION_BELOW,
        }
        
        # 방향별 계산 거리 비율 설정
        self.distance_ratio = {
            self.DIRECTION_FRONT: 0.5,    # 앞쪽 방향으로 객체 크기의 50% 거리에 위치
            self.DIRECTION_BACK: 0.5,     # 뒤쪽 방향으로 객체 크기의 50% 거리에 위치
            self.DIRECTION_LEFT: 0.3,     # 왼쪽 방향으로 객체 크기의 30% 거리에 위치
            self.DIRECTION_RIGHT: 0.3,    # 오른쪽 방향으로 객체 크기의 30% 거리에 위치
            self.DIRECTION_ABOVE: 0.2,    # 위쪽 방향으로 객체 크기의 20% 거리에 위치
            self.DIRECTION_BELOW: 0.2,    # 아래쪽 방향으로 객체 크기의 20% 거리에 위치
        }
        
        # 프롬프트 분석을 위한 패턴 정의
        # "A를 B 방향으로 옮겨줘" 형식의 패턴
        self.move_pattern_ko = re.compile(r'(.+?)(?:을|를)\s+(.+?)\s+(.+?)(?:으로|로|에)\s+(?:옮겨|이동|보내)', re.IGNORECASE)
        
        # "Move A to the direction of B" 형식의 패턴
        self.move_pattern_en = re.compile(r'(?:move|put|place)\s+(.+?)\s+(?:to|in|on|at)\s+(?:the)?\s*(.+?)\s+(?:of|from)?\s+(.+)', re.IGNORECASE)
    
    def analyze_spatial_relation(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Dict[str, Any]:
        """
        두 객체 간의 공간 관계 분석
        
        Args:
            obj1: 첫 번째 객체 정보
            obj2: 두 번째 객체 정보
            
        Returns:
            두 객체 간의 공간 관계 정보를 담은 딕셔너리
        """
        # 바운딩 박스 추출
        bbox1 = self._extract_bbox(obj1)
        bbox2 = self._extract_bbox(obj2)
        
        if bbox1 is None or bbox2 is None:
            self.logger.error("유효하지 않은 바운딩 박스 정보")
            return {}
        
        # 중심점 계산
        center1 = self._calculate_center(bbox1)
        center2 = self._calculate_center(bbox2)
        
        # 깊이 정보 추출
        depth1 = self._extract_depth(obj1)
        depth2 = self._extract_depth(obj2)
        
        # 수평 관계 (왼쪽/오른쪽)
        x_diff = center2[0] - center1[0]
        horizontal_relation = self.DIRECTION_LEFT if x_diff > 0 else self.DIRECTION_RIGHT
        horizontal_diff_px = abs(x_diff)
        
        # 수직 관계 (위/아래)
        y_diff = center2[1] - center1[1]
        vertical_relation = self.DIRECTION_ABOVE if y_diff < 0 else self.DIRECTION_BELOW
        vertical_diff_px = abs(y_diff)
        
        # 깊이 관계 (앞/뒤)
        depth_diff = depth2 - depth1
        depth_threshold = 0.05  # 깊이 차이 임계값
        
        if depth_diff > depth_threshold:
            depth_relation = self.DIRECTION_BACK
        elif depth_diff < -depth_threshold:
            depth_relation = self.DIRECTION_FRONT
        else:
            depth_relation = "same"  # 같은 깊이
        
        depth_diff_cm = abs(depth_diff * 100)  # 대략적인 cm 단위 변환 (실제 스케일은 다를 수 있음)
        
        # 결과 반환
        return {
            "horizontal": {
                "relation": horizontal_relation,
                "difference_px": float(horizontal_diff_px)
            },
            "vertical": {
                "relation": vertical_relation,
                "difference_px": float(vertical_diff_px)
            },
            "depth": {
                "relation": depth_relation,
                "difference": float(depth_diff),
                "difference_cm": float(depth_diff_cm)
            },
            "center_distance_px": float(np.sqrt(horizontal_diff_px**2 + vertical_diff_px**2)),
            "obj1_center": center1,
            "obj2_center": center2
        }
    
    def extract_direction_from_prompt(self, prompt: str) -> Optional[str]:
        """
        사용자 프롬프트에서 방향 키워드 추출
        
        Args:
            prompt: 사용자 프롬프트
            
        Returns:
            추출된 방향 상수 또는 None
        """
        if not prompt:
            return None
        
        # 소문자로 변환
        prompt_lower = prompt.lower()
        
        # 각 방향 키워드 검색
        for keyword, direction in self.direction_keywords.items():
            if keyword in prompt_lower:
                self.logger.info(f"프롬프트에서 방향 키워드 발견: '{keyword}' -> '{direction}'")
                return direction
        
        # 기본값은 None (방향 키워드 없음)
        self.logger.info("프롬프트에서 방향 키워드를 찾을 수 없음")
        return None
    
    def parse_prompt(self, prompt: str, available_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        사용자 프롬프트를 분석하여 타겟 객체, 레퍼런스 객체, 방향을 추출
        
        Args:
            prompt: 사용자 프롬프트
            available_objects: 감지된 객체 목록
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        if not prompt:
            return {
                "success": False,
                "error": "프롬프트가 없습니다."
            }
            
        # 객체 클래스 이름 목록 추출
        object_classes = {}
        for i, obj in enumerate(available_objects):
            class_name = obj.get("class_name", "unknown").lower()
            object_classes[class_name] = i
            
            # 추가 이름들(에어팟 = airpods 등)
            if class_name == "cell phone":
                object_classes["에어팟"] = i
                object_classes["airpods"] = i
                object_classes["이어폰"] = i
                object_classes["earbuds"] = i
                object_classes["earphone"] = i
            elif class_name == "baseball bat":
                object_classes["립밤"] = i
                object_classes["lipbalm"] = i
                object_classes["립스틱"] = i
                object_classes["lipstick"] = i
        
        # 프롬프트 정규화
        prompt_lower = prompt.lower()
        
        # 패턴 매칭 시도 - 한국어
        match = self.move_pattern_ko.search(prompt_lower)
        if match:
            target_name, reference_name, direction_text = match.groups()
            
            # 방향 추출
            direction = None
            for keyword, dir_const in self.direction_keywords.items():
                if keyword in direction_text:
                    direction = dir_const
                    break
            
            # 타겟 객체와 레퍼런스 객체 인덱스 찾기
            target_idx = self._find_object_by_name(target_name, object_classes)
            reference_idx = self._find_object_by_name(reference_name, object_classes)
            
            if target_idx is not None and reference_idx is not None:
                self.logger.info(f"프롬프트 분석 결과: 타겟='{target_name}'({target_idx}), "
                               f"레퍼런스='{reference_name}'({reference_idx}), 방향={direction}")
                return {
                    "success": True,
                    "target_idx": target_idx,
                    "reference_idx": reference_idx,
                    "direction": direction,
                    "target_name": target_name,
                    "reference_name": reference_name
                }
            else:
                missing = []
                if target_idx is None:
                    missing.append(f"타겟 객체('{target_name}')")
                if reference_idx is None:
                    missing.append(f"레퍼런스 객체('{reference_name}')")
                
                return {
                    "success": False,
                    "error": f"프롬프트에서 추출한 {', '.join(missing)}를 찾을 수 없습니다."
                }
        
        # 패턴 매칭 시도 - 영어
        match = self.move_pattern_en.search(prompt_lower)
        if match:
            target_name, direction_text, reference_name = match.groups()
            
            # 방향 추출
            direction = None
            for keyword, dir_const in self.direction_keywords.items():
                if keyword in direction_text:
                    direction = dir_const
                    break
            
            # 타겟 객체와 레퍼런스 객체 인덱스 찾기
            target_idx = self._find_object_by_name(target_name, object_classes)
            reference_idx = self._find_object_by_name(reference_name, object_classes)
            
            if target_idx is not None and reference_idx is not None:
                self.logger.info(f"프롬프트 분석 결과: 타겟='{target_name}'({target_idx}), "
                               f"레퍼런스='{reference_name}'({reference_idx}), 방향={direction}")
                return {
                    "success": True,
                    "target_idx": target_idx,
                    "reference_idx": reference_idx,
                    "direction": direction,
                    "target_name": target_name,
                    "reference_name": reference_name
                }
            else:
                missing = []
                if target_idx is None:
                    missing.append(f"타겟 객체('{target_name}')")
                if reference_idx is None:
                    missing.append(f"레퍼런스 객체('{reference_name}')")
                
                return {
                    "success": False,
                    "error": f"프롬프트에서 추출한 {', '.join(missing)}를 찾을 수 없습니다."
                }
        
        # 패턴 매칭 실패
        return {
            "success": False,
            "error": "프롬프트에서 타겟 객체, 레퍼런스 객체, 방향을 추출할 수 없습니다."
        }
    
    def _find_object_by_name(self, name: str, object_classes: Dict[str, int]) -> Optional[int]:
        """객체 이름으로 인덱스 찾기"""
        name = name.strip().lower()
        
        # 정확한 매칭
        if name in object_classes:
            return object_classes[name]
        
        # 포함 관계 매칭
        for class_name, idx in object_classes.items():
            if name in class_name or class_name in name:
                return idx
                
        return None

    def calculate_goal_point(self, reference_object: Dict[str, Any], 
                           direction: str, 
                           target_object: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        주어진 방향과 기준 객체를 기반으로 목표 지점 계산
        
        Args:
            reference_object: 기준 객체 정보
            direction: 방향 상수 (self.DIRECTION_XXX)
            target_object: 타겟 객체 정보 (객체 크기 계산용, 없으면 기준 객체 크기 사용)
            
        Returns:
            목표 지점 정보를 담은 딕셔너리
        """
        # --- 계산 기준 명확화 ---
        # 목표 지점의 위치는 주로 reference_object_3d (레퍼런스 객체의 3D 좌표)와
        # direction (방향 벡터)에 의해 결정됩니다.
        # target_object는 현재 목표 지점의 크기(bounding box)를 결정하는 데 사용될 수 있으나,
        # 위치 자체는 reference_object로부터 주어진 direction만큼 떨어진 곳으로 계산됩니다.
        # 추후 타겟 객체의 크기나 형태를 고려하여 위치를 조정하는 로직 추가 가능
        # (예: target_object의 크기를 고려하여 reference_object로부터의 거리 조정)
        # -------------------------

        # 객체 정보 유효성 검사
        ref_coords = reference_object.get('3d_coords')

        # 기준 객체 바운딩 박스 및 중심점
        ref_bbox = self._extract_bbox(reference_object)
        ref_center = self._calculate_center(ref_bbox)
        
        # 객체 크기
        ref_width = ref_bbox[2] - ref_bbox[0]
        ref_height = ref_bbox[3] - ref_bbox[1]
        
        # 타겟 객체 크기 (없으면 기준 객체 크기 사용)
        if target_object:
            target_bbox = self._extract_bbox(target_object)
            target_width = target_bbox[2] - target_bbox[0]
            target_height = target_bbox[3] - target_bbox[1]
        else:
            target_width = ref_width
            target_height = ref_height
        
        # 방향에 따른 거리 계산 비율
        ratio = self.distance_ratio.get(direction, 0.3)
        
        # 깊이 정보 추출
        ref_depth = self._extract_depth(reference_object)
        
        # 방향에 따른 목표 좌표 계산
        goal_point = {"width": target_width, "height": target_height}
        
        if direction == self.DIRECTION_FRONT:
            # 앞: 카메라 시점에서 전방(이미지에서는 아래쪽)
            goal_point["x"] = ref_center[0] - target_width/2
            goal_point["y"] = ref_bbox[3] + ref_height * ratio  # 아래쪽 방향(y 증가)
            goal_point["z"] = max(0.0, ref_depth - 0.15)  # 깊이 감소
            
            reasoning = f"기준 객체 앞쪽에 위치시키기 위해 깊이를 줄이고 화면에서 아래쪽으로 배치"
            
        elif direction == self.DIRECTION_BACK:
            # 뒤: 카메라 시점에서 후방(이미지에서는 위쪽)
            goal_point["x"] = ref_center[0] - target_width/2
            goal_point["y"] = ref_bbox[1] - target_height - ref_height * ratio  # 위쪽 방향(y 감소)
            goal_point["z"] = min(1.0, ref_depth + 0.15)  # 깊이 증가
            
            reasoning = f"기준 객체 뒤쪽에 위치시키기 위해 깊이를 늘리고 화면에서 위쪽으로 배치"
            
        elif direction == self.DIRECTION_LEFT:
            # 왼쪽: x 감소, y/깊이는 동일
            goal_point["x"] = ref_bbox[0] - target_width - ref_width * ratio
            goal_point["y"] = ref_center[1] - target_height/2
            goal_point["z"] = ref_depth
            
            reasoning = f"기준 객체 왼쪽에 위치시키기 위해 x좌표를 감소"
            
        elif direction == self.DIRECTION_RIGHT:
            # 오른쪽: x 증가, y/깊이는 동일
            goal_point["x"] = ref_bbox[2] + ref_width * ratio
            goal_point["y"] = ref_center[1] - target_height/2
            goal_point["z"] = ref_depth
            
            reasoning = f"기준 객체 오른쪽에 위치시키기 위해 x좌표를 증가"
            
        elif direction == self.DIRECTION_ABOVE:
            # 위: y 감소, x/깊이는 동일
            goal_point["x"] = ref_center[0] - target_width/2
            goal_point["y"] = ref_bbox[1] - target_height - ref_height * ratio
            goal_point["z"] = ref_depth
            
            reasoning = f"기준 객체 위쪽에 위치시키기 위해 y좌표를 감소"
            
        elif direction == self.DIRECTION_BELOW:
            # 아래: y 증가, x/깊이는 동일
            goal_point["x"] = ref_center[0] - target_width/2
            goal_point["y"] = ref_bbox[3] + ref_height * ratio
            goal_point["z"] = ref_depth
            
            reasoning = f"기준 객체 아래쪽에 위치시키기 위해 y좌표를 증가"
            
        else:
            # 기본값: 객체 오른쪽
            goal_point["x"] = ref_bbox[2] + ref_width * 0.3
            goal_point["y"] = ref_center[1] - target_height/2
            goal_point["z"] = ref_depth
            
            reasoning = f"방향이 명확하지 않아 기본값(오른쪽)으로 배치"
        
        # 음수 좌표 방지
        goal_point["x"] = max(0, goal_point["x"])
        goal_point["y"] = max(0, goal_point["y"])
        
        self.logger.info(f"{direction} 방향으로 목표 지점 계산: {goal_point}")
        
        return {
            "goal_point": goal_point,
            "reasoning": reasoning
        }
    
    def calculate_goal_point_3d(self, target_object: Dict[str, Any], 
                                reference_object: Dict[str, Any],
                                direction: str = None) -> Dict[str, Any]:
        """
        타겟 객체와 레퍼런스 객체로부터 3D 공간상의 목표 지점을 계산합니다.
        
        Args:
            target_object: 타겟 객체 정보 (이동할 객체)
            reference_object: 레퍼런스 객체 정보 (기준 객체)
            direction: 방향 (self.DIRECTION_XXX). None이면 프롬프트에서 추출 시도
            
        Returns:
            Dict: 목표 지점 정보가 포함된 딕셔너리
                - goal_point: 3D 좌표가 포함된 목표 지점 정보
                - direction: 사용된 방향
                - confidence: 계산 신뢰도
                - method: 사용된 방법
        """
        self.logger.info("3D 목표 지점 계산 시작...")
        
        # 객체가 없으면 기본 오류 응답 반환
        if target_object is None or reference_object is None:
            self.logger.error("타겟 또는 레퍼런스 객체가 없습니다.")
            return {
                "error": "타겟 또는 레퍼런스 객체가 없습니다.",
                "confidence": 0.0
            }
        
        # 방향이 지정되지 않았으면 기본값 사용
        if direction is None or direction not in self.direction_keywords.values():
            direction = self.DIRECTION_FRONT
            self.logger.info(f"방향이 명시되지 않아 기본값 사용: {direction}")
        
        try:
            # 2D 목표 지점 계산 (기존 함수 활용)
            goal_2d_result = self.calculate_goal_point(
                reference_object=reference_object,
                direction=direction,
                target_object=target_object
            )
            
            if "error" in goal_2d_result:
                return goal_2d_result
            
            # 2D 목표 지점에서 필요한 정보 추출
            goal_point_2d = goal_2d_result["goal_point"]
            
            # 3D 좌표 계산을 위한 기본 정보 수집
            ref_3d = reference_object.get("3d_coords", {})
            target_3d = target_object.get("3d_coords", {})
            
            # 기본 좌표 설정
            x_cm = ref_3d.get("x_cm", 0.0)
            y_cm = ref_3d.get("y_cm", 0.0)
            z_cm = ref_3d.get("z_cm", 100.0)  # 기본값 1미터
            
            # 방향에 따른 좌표 계산
            offset_cm = 20.0  # 기본 오프셋 (20cm)
            
            if direction == self.DIRECTION_FRONT:
                z_cm -= offset_cm
            elif direction == self.DIRECTION_BACK:
                z_cm += offset_cm
            elif direction == self.DIRECTION_LEFT:
                x_cm -= offset_cm
            elif direction == self.DIRECTION_RIGHT:
                x_cm += offset_cm
            elif direction == self.DIRECTION_ABOVE:
                y_cm -= offset_cm
            elif direction == self.DIRECTION_BELOW:
                y_cm += offset_cm
            
            # 3D 좌표를 포함한 최종 결과
            goal_3d = {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "z_cm": z_cm,
                "direction": direction,
                # 2D 정보도 포함
                "screen_x": goal_point_2d.get("x", 0),
                "screen_y": goal_point_2d.get("y", 0),
                "depth": goal_point_2d.get("z", 0.5),
            }
            
            self.logger.info(f"목표 지점 3D 계산 완료: [{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f}] cm, 방향: {direction}")
            
            return {
                "goal_point": {
                    "3d_coords": goal_3d
                },
                "direction": direction,
                "confidence": 0.8,
                "method": "spatial_3d_calculation"
            }
            
        except Exception as e:
            self.logger.error(f"3D 목표 지점 계산 중 오류: {e}", exc_info=True)
            return {
                "error": f"계산 중 오류: {str(e)}",
                "confidence": 0.0
            }
    
    def _extract_bbox(self, obj: Dict[str, Any]) -> Optional[List[float]]:
        """객체에서 바운딩 박스 추출"""
        # 다양한 키 이름 대응
        if "bbox" in obj:
            return obj["bbox"]
        elif "box" in obj:
            return obj["box"]
        else:
            self.logger.warning(f"바운딩 박스 정보를 찾을 수 없음: {list(obj.keys())}")
            return None
    
    def _calculate_center(self, bbox: List[float]) -> Tuple[float, float]:
        """바운딩 박스 중심점 계산"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _extract_depth(self, obj: Dict[str, Any]) -> float:
        """객체에서 깊이 정보 추출"""
        # 다양한 키 이름 대응
        if "depth" in obj and isinstance(obj["depth"], dict):
            depth_dict = obj["depth"]
            # 중심 깊이 우선 사용
            if "center_depth" in depth_dict:
                return float(depth_dict["center_depth"])
            # 평균 깊이 차선책
            elif "avg_depth" in depth_dict:
                return float(depth_dict["avg_depth"])
            # 중위수 깊이 차선책
            elif "median_depth" in depth_dict:
                return float(depth_dict["median_depth"])
            # 그 외 깊이 정보
            else:
                for key, value in depth_dict.items():
                    if isinstance(value, (float, int)) and "depth" in key:
                        return float(value)
        
        # 깊이 정보가 없는 경우 기본값 0.5 반환
        self.logger.warning(f"깊이 정보를 찾을 수 없음: {list(obj.keys() if isinstance(obj, dict) else [])}")
        return 0.5 