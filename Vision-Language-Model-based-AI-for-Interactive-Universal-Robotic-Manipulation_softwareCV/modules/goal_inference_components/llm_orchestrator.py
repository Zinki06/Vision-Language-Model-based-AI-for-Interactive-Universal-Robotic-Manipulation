"""
LLM 상호작용 관리 모듈

이 모듈은 LLM 상호작용 관리를 담당하는 LLMOrchestrator 클래스를 포함합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from ..gpt4o_builder import GPT4oBuilder
from ..enhanced_prompting_pipeline import EnhancedPromptingPipeline
from ..target_inference_enhancer import TargetInferenceEnhancer

class LLMOrchestrator:
    """
    LLM 상호작용 관리 클래스
    
    이 클래스는 다음과 같은 LLM 관련 기능을 담당합니다:
    - LLM 인터페이스 초기화 및 관리
    - 향상된 프롬프팅 파이프라인 관리
    - 타겟 추론 관련 LLM 요청 처리
    """
    
    def __init__(self, 
                llm_type: str = "gpt4o",
                temperature: float = 0.1,
                logger: Optional[logging.Logger] = None):
        """
        LLMOrchestrator 초기화
        
        Args:
            llm_type: 사용할 LLM 유형 ('gpt4o')
            temperature: LLM 응답 다양성 조절 파라미터
            logger: 로깅을 위한 로거 객체, None이면 새로 생성
        """
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # LLM 인터페이스 초기화
        self.logger.info(f"LLM: {llm_type} 인터페이스 초기화 중...")
        try:
            self.llm = GPT4oBuilder.create_gpt4o(
                temperature=temperature,
                logger=self.logger
            )
            self.logger.info(f"LLM: {llm_type} 인터페이스 초기화 완료")
        except Exception as e:
            self.logger.error(f"LLM 인터페이스 초기화 실패: {e}")
            self.llm = None
        
        # 향상된 프롬프팅 파이프라인 초기화
        if self.llm:
            self.logger.info("EnhancedPromptingPipeline 초기화 중...")
            try:
                self.prompting_pipeline = EnhancedPromptingPipeline(
                    llm_interface=self.llm,
                    logger=self.logger
                )
                self.logger.info("EnhancedPromptingPipeline 초기화 완료")
            except Exception as e:
                self.logger.error(f"EnhancedPromptingPipeline 초기화 실패: {e}")
                self.prompting_pipeline = None
        else:
            self.prompting_pipeline = None
            
        # 타깃 추론 강화 엔진 초기화
        if self.llm:
            self.logger.info("TargetInferenceEnhancer 초기화 중...")
            try:
                self.target_enhancer = TargetInferenceEnhancer(
                    llm=self.llm,
                    logger=self.logger
                )
                self.logger.info("TargetInferenceEnhancer 초기화 완료")
            except Exception as e:
                self.logger.error(f"TargetInferenceEnhancer 초기화 실패: {e}")
                self.target_enhancer = None
        else:
            self.target_enhancer = None
    
    def process_user_prompt(self, 
                          user_prompt: str, 
                          detections: List[Dict[str, Any]],
                          gesture_results: Optional[List[Dict[str, Any]]] = None,
                          depth_map: Optional[Any] = None) -> Dict[str, Any]:
        """
        사용자 프롬프트 처리
        
        Args:
            user_prompt: 사용자 명령 텍스트
            detections: 감지된 객체 목록
            gesture_results: 인식된 제스처 정보 (선택적)
            depth_map: 깊이 맵 데이터 (선택적)
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.prompting_pipeline:
            self.logger.error("프롬프팅 파이프라인이 초기화되지 않았습니다.")
            return {
                "success": False,
                "error": "프롬프팅 파이프라인이 초기화되지 않았습니다."
            }
            
        try:
            result = self.prompting_pipeline.process(
                user_prompt=user_prompt,
                detections=detections,
                gesture_results=gesture_results,
                depth_data=depth_map
            )
            self.logger.info(f"프롬프트 처리 완료: 타겟={result.get('final_target_id')}, 레퍼런스={result.get('final_reference_id')}")
            
            # 결과에 성공 플래그 추가
            result["success"] = True
            
            return result
        except Exception as e:
            self.logger.error(f"프롬프트 처리 중 오류: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def confirm_reference(self, 
                         user_command: str,
                         target_object: Dict[str, Any],
                         reference_object: Dict[str, Any]) -> Dict[str, Any]:
        """
        레퍼런스 객체 유효성 확인
        
        Args:
            user_command: 사용자 명령 텍스트
            target_object: 타겟 객체 정보
            reference_object: 레퍼런스 객체 정보
            
        Returns:
            Dict[str, Any]: 확인 결과
        """
        if not self.prompting_pipeline:
            self.logger.error("프롬프팅 파이프라인이 초기화되지 않았습니다.")
            return {
                "is_valid": True,  # 파이프라인 없으면 기본적으로 유효하다고 가정
                "confidence": 0.5,
                "explanation": "프롬프팅 파이프라인이 없어 자동 확인됨"
            }
            
        try:
            result = self.prompting_pipeline.confirm_reference(
                user_command=user_command,
                target_object=target_object,
                reference_object=reference_object
            )
            self.logger.info(f"레퍼런스 확인 완료: 유효성={result.get('is_valid')}, 신뢰도={result.get('confidence', 0.0):.2f}")
            return result
        except Exception as e:
            self.logger.error(f"레퍼런스 확인 중 오류: {e}")
            return {
                "is_valid": True,  # 오류 시 기본적으로 유효하다고 가정
                "confidence": 0.5,
                "explanation": f"확인 중 오류 발생: {str(e)}"
            }
    
    def infer_goal_bounding_box(self, 
                               user_command: str,
                               target_object: Dict[str, Any],
                               reference_object: Dict[str, Any],
                               img_width: int,
                               img_height: int,
                               direction: str = "front") -> Dict[str, Any]:
        """
        목표 바운딩 박스 추론
        
        Args:
            user_command: 사용자 명령 텍스트
            target_object: 타겟 객체 정보
            reference_object: 레퍼런스 객체 정보
            img_width: 이미지 너비
            img_height: 이미지 높이
            direction: 기본 방향 (기본값: "front")
            
        Returns:
            Dict[str, Any]: 추론 결과
        """
        if not self.prompting_pipeline:
            self.logger.error("프롬프팅 파이프라인이 초기화되지 않았습니다.")
            return {
                "parse_success": False,
                "error": "프롬프팅 파이프라인이 초기화되지 않았습니다.",
                "direction": direction
            }
            
        try:
            result = self.prompting_pipeline.infer_goal_bounding_box(
                user_command=user_command,
                target_object=target_object,
                reference_object=reference_object,
                img_width=img_width,
                img_height=img_height,
                direction=direction
            )
            if result["parse_success"]:
                inferred_direction = result.get("direction", direction)
                self.logger.info(f"목표 바운딩 박스 추론 완료: 방향={inferred_direction}")
            else:
                self.logger.warning(f"목표 바운딩 박스 추론 실패: {result.get('error')}")
            
            return result
        except Exception as e:
            self.logger.error(f"목표 바운딩 박스 추론 중 오류: {e}")
            return {
                "parse_success": False,
                "error": str(e),
                "direction": direction
            } 