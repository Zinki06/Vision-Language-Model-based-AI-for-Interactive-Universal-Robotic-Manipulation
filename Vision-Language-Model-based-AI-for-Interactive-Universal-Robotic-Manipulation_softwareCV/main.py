#!/usr/bin/env python
"""
LLM 기반 목표지점 추론 시스템

이 스크립트는 웹캠에서 실시간 영상을 받아서 'd' 키를 누르면 
스냅샷을 찍고 YOLO, MiDaS, LLM을 이용해 분석한 후 결과를 보여줍니다.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
import traceback
import threading
import json

from modules.goal_inference import GoalPointInferenceEngine
from modules.utils.config_manager import get_config_manager
from modules.camera.camera_manager import CameraManager
# 음성 녹음 및 STT 모듈 임포트
from modules.audio_recorder import AudioRecorder, AUDIO_SUPPORT
from modules.stt_processor import STTProcessor, STT_SUPPORT

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

class InteractiveGoalInferenceApp:
    """
    대화형 목표지점 추론 애플리케이션
    
    웹캠에서 실시간 영상을 보여주고, 'd' 키를 누르면 스냅샷을 찍어서
    YOLO, MiDaS, LLM을 이용해 분석한 후 결과를 보여줍니다.
    
    Attributes:
        config_manager: 설정 관리자
        engine: 목표지점 추론 엔진
        output_dir (str): 결과 저장 디렉토리
        coordinate_dir (str): 좌표 저장 디렉토리
        window_name (str): OpenCV 창 이름
        is_analyzing (bool): 분석 진행 중 여부
        awaiting_prompt (bool): 사용자 프롬프트 입력 대기 상태
        user_prompt (str): 사용자 입력 프롬프트
        snapshot: 캡처된 스냅샷 이미지
        result_image: 분석 결과 이미지
        is_entering_prompt (bool): 프롬프트 입력 모드 활성화 여부
        gesture_mode_active (bool): 제스처 인식 모드 활성화 여부
        is_recording (bool): 녹음 진행 중 여부
        audio_recorder (AudioRecorder): 오디오 녹음 객체
        mp3_path (str): 최근 녹음된 MP3 파일 경로
    """
    
    def __init__(self, config_manager=None):
        """
        애플리케이션 초기화
        
        Args:
            config_manager: 설정 관리자 (None인 경우 기본값 사용)
        """
        # 설정 관리자 초기화
        self.config_manager = config_manager or get_config_manager()
        
        # 결과 저장 디렉토리 설정
        output_dir = self.config_manager.get('storage.output_dir', 'output')
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"출력 디렉토리: {self.output_dir}")
        
        # 좌표 저장 디렉토리 설정
        self.coordinate_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordinate")
        os.makedirs(self.coordinate_dir, exist_ok=True)
        logger.info(f"좌표 디렉토리: {self.coordinate_dir}")
        
        # 목표지점 추론 엔진 초기화
        try:
            self._initialize_engine()
            logger.info("GoalPointInferenceEngine 초기화 완료")
        except FileNotFoundError as e:
            logger.error(f"초기화 중 모델 파일({e.filename})을 찾을 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"GoalPointInferenceEngine 초기화 중 예상치 못한 오류 발생: {e}")
            sys.exit(1)
        
        # 상태 변수 초기화
        self.is_analyzing = False
        self.awaiting_prompt = False
        self.user_prompt = ""
        self.snapshot = None
        self.result_image = None
        self.is_entering_prompt = False
        self.gesture_mode_active = False
        
        # 음성 녹음 관련 상태 변수 초기화
        self.is_recording = False
        self.audio_recorder = None
        self.mp3_path = None
        
        # 음성 및 STT 기능 활성화 설정 로드
        audio_config = self.config_manager.get_section('audio')
        self.audio_enabled = audio_config.get('enabled', False) and AUDIO_SUPPORT
        
        # 음성 녹음 기능 초기화
        if self.audio_enabled:
            try:
                self._initialize_audio_recorder()
            except Exception as e:
                logger.error(f"오디오 레코더 초기화 중 오류 발생: {e}")
                self.audio_enabled = False
        
        # 카메라 매니저 초기화
        self.camera_manager = CameraManager()
        
        # 창 설정
        self.window_name = "Goal Point Inference"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def _initialize_engine(self):
        """목표지점 추론 엔진 초기화"""
        # 설정에서 값 가져오기
        detection_config = self.config_manager.get_section('detection')
        llm_config = self.config_manager.get_section('llm')
        gesture_config = self.config_manager.get_section('gesture')
        
        # 로그 출력
        logger.info(
            f"GoalPointInferenceEngine 초기화 중... "
            f"(YOLO: {detection_config['yolo_model']}, "
            f"LLM: {llm_config['model']}, "
            f"Gesture: {gesture_config['model_path']}, "
            f"신뢰도: {detection_config['conf_threshold']})"
        )
        
        self.engine = GoalPointInferenceEngine(
            yolo_model=detection_config['yolo_model'],
            llm_type=llm_config['model'],
            conf_threshold=detection_config['conf_threshold'],
            iou_threshold=detection_config['iou_threshold'],
            gesture_model_path=gesture_config['model_path'],
            logger=logger
        )
    
    def _initialize_camera(self):
        """카메라 초기화"""
        # 카메라 설정 로드
        camera_config = self.config_manager.get_section('camera')
        logger.info(f"카메라 설정: {camera_config}")
        
        # 카메라 매니저 초기화
        self.camera_manager.initialize(camera_config)
        
        # 시스템에서 사용 가능한 모든 카메라 검색
        available_cameras = self.camera_manager.get_available_cameras()
        if available_cameras:
            logger.info(f"사용 가능한 카메라: {len(available_cameras)}개")
            for i, camera in enumerate(available_cameras):
                logger.info(f"  [{i}] {camera['name']} - {camera['resolution'][0]}x{camera['resolution'][1]}@{camera['fps']}fps")
        else:
            logger.warning("사용 가능한 카메라가 없습니다.")
        
        # 카메라 설정에서 카메라 생성
        camera = self.camera_manager.create_camera_from_config()
        
        if camera is None:
            logger.error("카메라를 생성할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        # 활성 카메라로 설정
        if not self.camera_manager.set_active_camera(camera):
            logger.error("카메라를 활성화할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        logger.info("카메라 초기화 완료")
        return camera
    
    def start(self):
        """애플리케이션 시작"""
        # 카메라 초기화
        camera = self._initialize_camera()
        active_camera = self.camera_manager.get_active_camera()
        
        logger.info(f"활성 카메라 정보: {active_camera.get_camera_info()}")
        
        logger.info("애플리케이션 시작")
        logger.info("키 안내:")
        logger.info("  d: 스냅샷 찍기 + 녹음 시작/종료")
        logger.info("  a: 스냅샷 분석 시작 (텍스트 입력 후)")
        logger.info("  p: 프롬프트 입력 (GUI)")
        logger.info("  g: 제스처 인식 모드 토글")
        logger.info("  c: 카메라 전환 (사용 가능한 다음 카메라로)")
        logger.info("  r: 결과 이미지 다시 보기")
        logger.info("  s: 결과 저장")
        logger.info("  q: 종료")
        
        # 오디오 기능 상태 표시
        if self.audio_enabled:
            logger.info("음성 녹음 기능이 활성화되었습니다.")
            logger.info("  'd' 키를 한 번 누르면 스냅샷 촬영 및 녹음이 시작됩니다.")
            logger.info("  녹음 중 'd' 키를 다시 누르면 녹음이 종료되고 자동으로 텍스트 변환 및 분석이 시작됩니다.")
        else:
            logger.info("음성 녹음 기능이 비활성화되었습니다.")
        
        # 앱 상태 표시 텍스트
        status_text = "준비 (d: 스냅샷, p: 프롬프트, g: 제스처, q: 종료)"
        prompt_text = "프롬프트: 없음"
        gesture_status_text = "Gesture: OFF"
        
        read_fail_count = 0  # 프레임 읽기 실패 카운터
        max_read_fails = 10  # 최대 연속 실패 허용 횟수
        
        try:
            while True:
                # 프레임 캡처
                if active_camera and active_camera.is_open():
                    frame = active_camera.capture_frame()
                    ret = (frame is not None)
                else:
                    ret = False
                    frame = None
                
                # 프레임 읽기 실패 처리
                if not ret:
                    read_fail_count += 1
                    logger.warning(f"프레임 캡처 실패 ({read_fail_count}/{max_read_fails})")
                    if read_fail_count >= max_read_fails:
                        logger.error("연속된 프레임 캡처 실패로 루프를 종료합니다. 카메라 연결 또는 권한을 확인하세요.")
                        break
                    time.sleep(0.1)  # 짧은 지연 후 재시도
                    continue  # 루프의 다음 반복으로 건너뛰기
                
                # 성공 시 실패 카운터 리셋
                read_fail_count = 0 
                
                # 실시간 제스처 인식 (gesture_mode_active가 True일 때만)
                live_gesture_results = self._process_live_gestures(frame)
                
                # 현재 표시할 이미지 결정
                display_image = self._determine_display_image(frame, status_text)
                
                # 실시간 제스처 시각화
                self._visualize_live_gestures(display_image, live_gesture_results)
                
                # 상태 및 프롬프트 정보 표시
                self._draw_status_info(display_image, status_text, prompt_text, gesture_status_text)
                
                # 이미지 표시
                cv2.imshow(self.window_name, display_image)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                # 키 입력에 따른 상태 및 텍스트 업데이트
                status_text, prompt_text, gesture_status_text = self._handle_key_input(key, frame)
                
                # 종료 조건
                if key == ord('q'):
                    logger.info("종료")
                    break
                
        finally:
            # 리소스 해제
            self._cleanup_resources()
    
    def _process_live_gestures(self, frame):
        """실시간 제스처 인식 처리"""
        if not self.gesture_mode_active or not self.engine or not self.engine.gesture_recognizer:
            return None
            
        try:
            logger.debug(f"Gesture Input Frame: Shape={frame.shape}, Dtype={frame.dtype}")
            live_gesture_results = self.engine.gesture_recognizer.process_frame(frame)
            logger.debug(f"Gesture Results: {live_gesture_results}")
            return live_gesture_results
        except Exception as e:
            logger.error(f"실시간 제스처 인식 중 오류: {e}", exc_info=True)
            return None
    
    def _determine_display_image(self, frame, status_text):
        """현재 표시할 이미지 결정"""
        if self.snapshot is not None and not self.is_analyzing and self.result_image is None:
            # 스냅샷이 있고 분석 중이 아니며 결과 이미지가 없으면, 스냅샷 표시
            display_image = self.snapshot.copy()
            if self.is_recording and self.audio_enabled:
                status_text = "스냅샷 촬영 및 녹음 중... ('d' 키로 녹음 종료)"
            else:
                status_text = "스냅샷 준비됨 (프롬프트 입력 후 'a' 키로 분석)"
        elif self.is_analyzing:
            # 분석 중이면 실시간 영상 위에 분석 중임을 표시
            display_image = frame.copy()
            status_text = "분석 중... 잠시만 기다려주세요."
        elif self.result_image is not None:
            # 결과 이미지가 있으면 결과 이미지 표시
            display_image = self.result_image.copy()
            status_text = "분석 완료 (d: 새 스냅샷, r: 결과 보기, s: 저장, q: 종료)"
        else:
            # 기본 상태는 실시간 영상 표시
            display_image = frame.copy()
            if self.is_recording and self.audio_enabled:
                status_text = "녹음 중... ('d' 키로 녹음 종료)"
            else:
                status_text = "준비 (d: 스냅샷 + 녹음, p: 프롬프트, g: 제스처, q: 종료)"
            
        return display_image
    
    def _visualize_live_gestures(self, display_image, live_gesture_results):
        """실시간 제스처 시각화"""
        if self.gesture_mode_active and self.result_image is None and live_gesture_results and self.engine.visualization:
            try:
                logger.debug(f"Drawing Gestures: Results={live_gesture_results}, Target Image Shape={display_image.shape}, Dtype={display_image.dtype}")
                self.engine.visualization._draw_hand_gestures(display_image, live_gesture_results)
            except Exception as e:
                logger.error(f"실시간 제스처 시각화 중 오류: {e}", exc_info=True)
    
    def _draw_status_info(self, display_image, status_text, prompt_text, gesture_status_text):
        """상태 및 프롬프트 정보 표시"""
        # 상태 텍스트
        cv2.putText(display_image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 프롬프트 텍스트
        prompt_display = f"프롬프트: {self.user_prompt}" if self.user_prompt else "프롬프트: 없음"
        cv2.putText(display_image, prompt_display, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 제스처 상태
        gesture_color = (0, 255, 0) if self.gesture_mode_active else (0, 0, 255)
        gesture_status = f"Gesture: {'ON' if self.gesture_mode_active else 'OFF'}"
        cv2.putText(display_image, gesture_status, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        
        # 녹음 상태 표시
        if self.audio_enabled:
            recording_color = (0, 0, 255)  # 빨간색 (기본값)
            recording_status = "Record: OFF"
            
            if self.is_recording:
                # 깜빡이는 효과 (현재 시간 기반)
                if int(time.time() * 2) % 2 == 0:
                    recording_color = (0, 0, 255)  # 빨간색
                else:
                    recording_color = (0, 69, 255)  # 주황색
                recording_status = "Record: REC"
                
                # 녹음 시간 표시 (있는 경우)
                if self.audio_recorder:
                    status = self.audio_recorder.get_recording_status()
                    duration = status.get("duration_seconds", 0)
                    if duration > 0:
                        recording_status += f" ({duration:.1f}s)"
            
            cv2.putText(display_image, recording_status, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        
        # 프롬프트 입력 모드 안내
        y_offset = 150 if self.audio_enabled else 120
        if self.is_entering_prompt:
            cv2.putText(display_image, f"프롬프트 입력 중: {self.user_prompt}_", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_image, "Enter: 확인, Esc: 취소", 
                        (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 프롬프트 입력 대기 안내
        if self.awaiting_prompt:
            cv2.putText(display_image, "터미널에서 프롬프트를 입력하세요", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_image, "입력 후 'a' 키를 눌러 분석 시작", 
                        (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _handle_key_input(self, key, frame):
        """키 입력 처리"""
        status_text = "준비 (d: 스냅샷, p: 프롬프트, g: 제스처, q: 종료)"
        prompt_text = f"프롬프트: {self.user_prompt}" if self.user_prompt else "프롬프트: 없음"
        gesture_status_text = f"Gesture: {'ON' if self.gesture_mode_active else 'OFF'}"
        
        # 프롬프트 입력 모드
        if self.is_entering_prompt:
            self._handle_prompt_input_mode(key)
        elif self.awaiting_prompt:
            # 터미널 프롬프트 입력 대기 중에는 'a' 키로 분석 시작만 처리
            if key == ord('a'):
                self._start_analysis()
            # 녹음 중에 'd' 키를 눌렀을 때 녹음 종료 처리 추가
            elif key == ord('d') and self.is_recording and self.audio_enabled:
                self._toggle_audio_recording()  # 녹음 종료 및 STT 처리
        else:
            # 일반 모드 키 처리
            self._handle_normal_mode_keys(key, frame)
            
        return status_text, prompt_text, gesture_status_text
    
    def _handle_prompt_input_mode(self, key):
        """프롬프트 입력 모드 키 처리"""
        if key == 13:  # Enter 키
            self.is_entering_prompt = False
            logger.info(f"프롬프트 설정: {self.user_prompt}")
        elif key == 27:  # Esc 키
            self.is_entering_prompt = False
            self.user_prompt = ""
            logger.info("프롬프트 입력 취소")
        elif key == 8:  # Backspace 키
            self.user_prompt = self.user_prompt[:-1] if self.user_prompt else ""
        elif 32 <= key <= 126:  # 일반 ASCII 문자
            self.user_prompt += chr(key)
    
    def _handle_normal_mode_keys(self, key, frame):
        """일반 모드 키 처리"""
        if key == ord('d') and not self.is_analyzing:
            # 녹음 중이면 녹음 종료 + STT 처리
            if self.is_recording and self.audio_enabled:
                self._toggle_audio_recording()  # 녹음 종료 및 STT 처리
            else:
                # 첫 번째 'd' 키: 스냅샷 촬영 + 녹음 시작
                self._take_snapshot(frame)
                # 오디오 기능이 활성화되어 있으면 녹음 시작
                if self.audio_enabled:
                    self._toggle_audio_recording()
        elif key == ord('p') and not self.is_analyzing:
            # 프롬프트 입력 모드 시작 (GUI)
            self.is_entering_prompt = True
            logger.info("프롬프트 입력 모드 시작")
        elif key == ord('g'):
            self._toggle_gesture_mode()
        elif key == ord('r') and self.result_image is not None:
            # 결과 이미지 다시 보기
            logger.info("결과 이미지 다시 표시")
        elif key == ord('s') and self.result_image is not None:
            self._save_results()
        elif key == ord('c'):
            # 카메라 전환 (사용 가능한 다음 카메라로)
            self._toggle_camera()
    
    def _toggle_camera(self):
        """사용 가능한 다음 카메라로 전환"""
        available_cameras = self.camera_manager.get_available_cameras()
        if not available_cameras:
            logger.warning("전환 가능한 카메라가 없습니다.")
            return
            
        # 현재 활성 카메라 정보 가져오기
        current_camera = self.camera_manager.get_active_camera()
        if not current_camera:
            # 활성 카메라가 없으면 첫 번째 카메라 사용
            next_camera_info = available_cameras[0]
        else:
            current_info = current_camera.get_camera_info()
            current_index = current_info.get("index", 0)
            current_backend = current_info.get("backend", "default")
            
            # 다음 카메라 검색
            found_current = False
            next_camera_info = None
            
            for cam_info in available_cameras:
                if found_current:
                    next_camera_info = cam_info
                    break
                
                if cam_info["index"] == current_index and cam_info["backend"] == current_backend:
                    found_current = True
            
            # 리스트의 끝에 도달한 경우 처음으로 돌아감
            if not next_camera_info and found_current:
                next_camera_info = available_cameras[0]
            # 현재 카메라를 찾지 못한 경우 첫 번째 카메라 사용
            elif not next_camera_info:
                next_camera_info = available_cameras[0]
        
        # 카메라 전환
        if next_camera_info:
            logger.info(f"카메라 전환 중: {next_camera_info['name']}")
            index = next_camera_info["index"]
            backend = next_camera_info["backend"]
            
            success = self.camera_manager.set_active_camera_by_index(index, backend)
            if success:
                logger.info(f"카메라 전환 성공: {next_camera_info['name']}")
            else:
                logger.error(f"카메라 전환 실패: {next_camera_info['name']}")
    
    def _take_snapshot(self, frame):
        """스냅샷 촬영"""
        self.snapshot = frame.copy()
        self.result_image = None  # 결과 이미지 초기화
        logger.info("스냅샷 찍음 - 터미널에서 프롬프트를 입력하세요")
        
        # 사용자 프롬프트 입력 대기 모드로 전환
        self.awaiting_prompt = True
        
        # 터미널에서 프롬프트 입력 받기
        thread = threading.Thread(target=self.get_terminal_prompt)
        thread.daemon = True
        thread.start()
    
    def _start_analysis(self):
        """분석 시작"""
        self.awaiting_prompt = False
        self.is_analyzing = True
        logger.info(f"프롬프트 '{self.user_prompt}'로 분석 시작")
        
        # 별도 스레드에서 분석을 시작
        thread = threading.Thread(target=self.analyze_snapshot)
        thread.daemon = True
        thread.start()
    
    def _toggle_gesture_mode(self):
        """제스처 인식 모드 토글"""
        self.gesture_mode_active = not self.gesture_mode_active
        mode_status = "활성화됨" if self.gesture_mode_active else "비활성화됨"
        logger.info(f"제스처 인식 모드 {mode_status}")
        
        # GoalPointInferenceEngine에 제스처 모드 상태 전달
        if hasattr(self, 'engine') and self.engine:
            self.engine.set_gesture_mode(self.gesture_mode_active)
    
    def _save_results(self):
        """결과 저장"""
        try:
            # 저장 형식 설정 로드
            storage_config = self.config_manager.get_section('storage')
            snapshot_format = storage_config.get('snapshot_format', 'snapshot_%Y%m%d_%H%M%S.jpg')
            result_format = storage_config.get('result_format', 'result_%Y%m%d_%H%M%S.jpg')
            
            # 타임스탬프 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 결과 이미지 저장
            result_filename = time.strftime(result_format)
            result_path = os.path.join(self.output_dir, result_filename)
            cv2.imwrite(result_path, self.result_image)
            logger.info(f"결과 이미지 저장됨: {result_path}")
            
            # 스냅샷 저장
            if self.snapshot is not None:
                snapshot_filename = time.strftime(snapshot_format)
                snapshot_path = os.path.join(self.output_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, self.snapshot)
                logger.info(f"스냅샷 저장됨: {snapshot_path}")
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
    
    def _cleanup_resources(self):
        """리소스 해제"""
        try:
            # 녹음 중이면 녹음 종료
            if self.is_recording and self.audio_recorder:
                logger.info("녹음 종료 중...")
                self.audio_recorder.stop_recording()
                self.is_recording = False
            
            # AudioRecorder 리소스 해제
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                logger.info("AudioRecorder 리소스 해제...")
                # AudioRecorder의 __del__ 메서드에서 리소스 정리
                self.audio_recorder = None
            
            # GestureRecognizer 리소스 해제
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'gesture_recognizer') and self.engine.gesture_recognizer:
                logger.info("GestureRecognizer 리소스 해제 시도...")
                self.engine.gesture_recognizer.close()
                
            # 카메라 리소스 해제
            if hasattr(self, 'camera_manager'):
                logger.info("카메라 리소스 해제...")
                self.camera_manager.release_all()
                
            # OpenCV 창 닫기
            cv2.destroyAllWindows()
            
            logger.info("모든 리소스가 해제되었습니다.")
        except Exception as e:
            logger.error(f"리소스 해제 중 오류 발생: {e}")
    
    def get_terminal_prompt(self):
        """터미널에서 사용자 프롬프트 입력 받기"""
        try:
            logger.info("\n프롬프트를 입력하세요 (빈 줄 입력 시 프롬프트 없음으로 처리):")
            # 사용자 입력 받기
            user_input = input().strip()
            
            # 입력 처리
            if user_input:
                self.user_prompt = user_input
                logger.info(f"프롬프트 입력 완료: '{self.user_prompt}' - 'a' 키를 눌러 분석 시작")
            else:
                self.user_prompt = ""
                logger.info("빈 프롬프트 입력됨 - 'a' 키를 눌러 분석 시작")
        except Exception as e:
            logger.error(f"프롬프트 입력 중 오류 발생: {e}")
            self.user_prompt = ""
            self.awaiting_prompt = False
    
    def analyze_snapshot(self):
        """스냅샷을 분석하고 결과 이미지를 생성"""
        try:
            if self.snapshot is None:
                logger.error("스냅샷이 없습니다.")
                self.is_analyzing = False
                return
            
            # 이미지 처리
            logger.info("이미지 분석 중...")
            results = self.engine.process_image(self.snapshot, user_prompt=self.user_prompt)
            
            # 분석 결과 처리
            self._process_analysis_results(results)
            
            # 분석 완료
            self.is_analyzing = False
            
        except Exception as e:
            logger.exception(f"분석 중 오류: {str(e)}")
            self.is_analyzing = False
            
            # 오류 시 결과 이미지 생성
            self._create_error_image(str(e))
    
    def _process_analysis_results(self, results):
        """분석 결과 처리"""
        # 감지된 객체 수
        detected_objects_count = len(results.get("detections", []))
        
        # 타겟/레퍼런스 객체 추출
        target_inference = results.get("target_inference", {})
        target_idx = target_inference.get("target_idx", -1)
        
        # 타겟 객체 추출
        target_object = None
        if 0 <= target_idx < detected_objects_count:
            target_object = results.get("detections", [])[target_idx]
        
        # 레퍼런스 객체 추출
        reference_idx = target_inference.get("reference_idx", -1)
        reference_object = None
        if 0 <= reference_idx < detected_objects_count:
            reference_object = results.get("detections", [])[reference_idx]
        
        # 타겟 추론 실패 여부 판단
        target_inference_failed = target_inference.get("confidence", 0.0) < 0.5
        
        # 결과 저장
        self.result_image = results["visualization"]
        
        # 저장 경로 표시
        if "saved_paths" in results:
            saved_paths = results["saved_paths"]
            if "json" in saved_paths:
                session_dir = os.path.dirname(saved_paths["json"])
                logger.info(f"결과가 저장된 경로: {session_dir}")
        
        # 결과 정보 패널 추가
        self._add_info_panel_to_result(
            detected_objects_count, 
            target_object, 
            reference_object, 
            target_inference_failed,
            results
        )
        
        # 로그 출력
        logger.info(f"분석 완료: {detected_objects_count}개 객체 감지됨")
        if target_inference_failed:
            logger.warning("타겟 추론 신뢰도가 낮습니다. 결과가 정확하지 않을 수 있습니다.")
        
        # 타겟 물체와 목표 지점의 중심 좌표 계산 및 저장
        goal_point_data = results.get("goal_point", {})
        self._save_coordinate_results(target_object, goal_point_data)
    
    def _add_info_panel_to_result(self, detected_objects_count, target_object, reference_object, target_inference_failed, results):
        """결과 이미지에 정보 패널 추가"""
        if self.result_image is None:
            return
            
        h, w = self.result_image.shape[:2]
        info_background = np.zeros((150, w, 3), dtype=np.uint8)
        
        # 분석 요약 텍스트 및 색상 결정
        summary_text, text_color = self._get_summary_text(
            detected_objects_count, 
            target_object, 
            reference_object, 
            target_inference_failed,
            results
        )
        
        # 프롬프트 정보 텍스트
        prompt_text = f"사용자 입력: \"{self.user_prompt}\"" if self.user_prompt else "사용자 입력: 없음 (기본 위치 추론)"
        
        # 추론 방향 정보
        direction_text = self._get_direction_text(results)
        
        # 텍스트 그리기
        cv2.putText(info_background, summary_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(info_background, prompt_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_background, direction_text, (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 정보 패널을 이미지 하단에 추가
        self.result_image = np.vstack([self.result_image, info_background])
    
    def _get_summary_text(self, detected_objects_count, target_object, reference_object, target_inference_failed, results):
        """결과 요약 텍스트 생성"""
        if detected_objects_count == 0:
            return "분석 결과: 객체가 감지되지 않았습니다. 다른 각도에서 시도해보세요.", (0, 0, 255)  # 빨간색 경고
        elif detected_objects_count == 1:
            obj_name = results["detections"][0].get("class_name", "알 수 없는 객체")
            return f"분석 결과: 단일 객체 '{obj_name}' 감지. 화면 중심 기반 위치 추론.", (0, 255, 255)  # 노란색 경고
        elif target_inference_failed or target_object is None or reference_object is None:
            target_name = target_object.get("class_name", "알 수 없는 객체") if target_object else "알 수 없는 객체"
            ref_name = reference_object.get("class_name", "알 수 없는 객체") if reference_object else "알 수 없는 객체"
            return f"분석 결과: 타겟 추론 신뢰도 낮음. '{target_name}'와 '{ref_name}' 간 위치 추론.", (0, 165, 255)  # 주황색 경고
        else:
            target_name = target_object.get("class_name", "알 수 없는 객체")
            ref_name = reference_object.get("class_name", "알 수 없는 객체")
            inference_method = results.get("target_inference", {}).get("method", "unknown")
            inference_confidence = results.get("target_inference", {}).get("confidence", 0.0)
            return f"분석 결과: 타겟 '{target_name}'와 레퍼런스 '{ref_name}' 사이의 목표 위치 추론. (방법: {inference_method}, 신뢰도: {inference_confidence:.2f})", (0, 255, 0)  # 초록색 성공
    
    def _get_direction_text(self, results):
        """방향 정보 텍스트 생성"""
        direction_text = "추론 방향: "
        
        if "goal_point" in results and results["goal_point"]:
            # 방향 정보 추출
            direction = None
            
            # goal_point에서 direction 필드 찾기
            if "direction" in results["goal_point"]:
                direction = results["goal_point"]["direction"]
            elif "goal_point" in results["goal_point"] and "direction" in results["goal_point"]["goal_point"]:
                direction = results["goal_point"]["goal_point"]["direction"]
            
            # 방향 객체 형식 처리
            if direction is not None:
                if isinstance(direction, dict):
                    dir_type = direction.get("type", "simple")
                    
                    if dir_type == "simple":
                        # 단일 방향 처리
                        direction_value = direction.get("value", "front")
                        direction_desc = {
                            "front": "앞쪽", "back": "뒤쪽", 
                            "left": "왼쪽", "right": "오른쪽",
                            "above": "위쪽", "below": "아래쪽"
                        }
                        direction_text += direction_desc.get(direction_value, direction_value)
                    
                    elif dir_type == "random":
                        # 랜덤 방향 처리
                        options = direction.get("options", [])
                        if options:
                            # 방향 옵션 텍스트 변환
                            direction_desc = {
                                "front": "앞쪽", "back": "뒤쪽", 
                                "left": "왼쪽", "right": "오른쪽",
                                "above": "위쪽", "below": "아래쪽"
                            }
                            option_texts = [direction_desc.get(opt, opt) for opt in options]
                            
                            # 선택된 방향 확인 (결과에 direction_name이 있는 경우)
                            if "direction_name" in results["goal_point"]:
                                selected = results["goal_point"]["direction_name"]
                                selected_text = direction_desc.get(selected, selected)
                                direction_text += f"랜덤 ({'/'.join(option_texts)}) → {selected_text} 선택됨"
                            else:
                                direction_text += f"랜덤 ({'/'.join(option_texts)})"
                    else:
                        direction_text += "알 수 없는 방향 타입"
                else:
                    # 문자열 방향 처리 (기존 로직)
                    direction_desc = {
                        "front": "앞쪽", "back": "뒤쪽", 
                        "left": "왼쪽", "right": "오른쪽",
                        "above": "위쪽", "below": "아래쪽"
                    }
                    direction_text += direction_desc.get(direction, direction)
            else:
                direction_text += "명시적 방향 없음"
        else:
            direction_text += "추론 불가"
            
        return direction_text
    
    def _create_error_image(self, error_message):
        """오류 발생 시 에러 메시지가 표시된 이미지 생성"""
        if self.snapshot is None:
            return
            
        error_img = self.snapshot.copy()
        h, w = error_img.shape[:2]
        
        # 반투명 오버레이
        overlay = error_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        error_img = cv2.addWeighted(overlay, 0.5, error_img, 0.5, 0)
        
        # 오류 메시지
        cv2.putText(error_img, "분석 중 오류 발생", (int(w/2) - 150, int(h/2) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_img, error_message[:50], (int(w/2) - 200, int(h/2) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(error_img, "다시 시도하려면 'd'를 누르세요", (int(w/2) - 180, int(h/2) + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.result_image = error_img
        
        # 오류 발생 시에도 결과 저장 시도 (디버깅 목적)
        try:
            if hasattr(self.engine, 'result_storage'):
                error_results = {
                    "error": error_message,
                    "traceback": traceback.format_exc(),
                    "user_prompt": self.user_prompt
                }
                self.engine.result_storage.save_analysis_results(
                    results_data=error_results,
                    snapshot_image=self.snapshot,
                    result_image=error_img,
                    depth_map=None
                )
                logger.info("오류 정보가 저장되었습니다.")
        except Exception as save_error:
            logger.error(f"오류 정보 저장 실패: {save_error}")

    def _initialize_audio_recorder(self):
        """
        오디오 레코더 초기화
        설정 파일에서 오디오 관련 설정을 로드하여 AudioRecorder 객체 생성
        """
        # 오디오 지원 확인
        if not AUDIO_SUPPORT:
            logger.warning("오디오 지원 라이브러리가 설치되지 않아 음성 녹음 기능을 사용할 수 없습니다.")
            self.audio_enabled = False
            return
        
        # 설정 로드
        audio_config = self.config_manager.get_section('audio')
        sample_rate = audio_config.get('sample_rate', 44100)
        channels = audio_config.get('channels', 1)
        chunk_size = audio_config.get('chunk_size', 1024)
        
        # AudioRecorder 객체 생성
        try:
            import pyaudio
            format_value = getattr(pyaudio, audio_config.get('format', 'paInt16'))
            self.audio_recorder = AudioRecorder(
                output_dir=self.output_dir,
                format=format_value,
                channels=channels,
                rate=sample_rate,
                chunk=chunk_size
            )
            logger.info(f"오디오 레코더 초기화 완료: 샘플링 레이트={sample_rate}, 채널={channels}")
        except Exception as e:
            logger.exception(f"오디오 레코더 초기화 실패: {e}")
            self.audio_enabled = False

    def _toggle_audio_recording(self):
        """
        녹음 시작/종료 토글
        현재 녹음 상태에 따라 녹음을 시작하거나 종료합니다.
        """
        if not self.audio_enabled or not self.audio_recorder:
            logger.warning("오디오 레코더가 초기화되지 않아 녹음 기능을 사용할 수 없습니다.")
            return
        
        # 녹음 중이면 종료, 아니면 시작
        if self.is_recording:
            # 녹음 종료
            logger.info("녹음 종료 중...")
            self.mp3_path = self.audio_recorder.stop_recording()
            self.is_recording = False
            
            if self.mp3_path:
                logger.info(f"녹음 완료: {self.mp3_path}")
                # 자동으로 텍스트 변환 시작
                self._process_audio_to_text()
            else:
                logger.error("녹음 파일 생성에 실패했습니다.")
        else:
            # 녹음 시작
            logger.info("녹음 시작 중...")
            if self.audio_recorder.start_recording():
                self.is_recording = True
                logger.info("녹음이 시작되었습니다.")
            else:
                logger.error("녹음 시작에 실패했습니다.")

    def _process_audio_to_text(self):
        """
        녹음된 MP3 파일을 텍스트로 변환
        STT API를 이용하여 음성을 텍스트로 변환한 후 프롬프트로 설정합니다.
        """
        if not self.mp3_path or not os.path.exists(self.mp3_path):
            logger.error("변환할 오디오 파일이 없습니다.")
            return
        
        # STT 지원 확인
        if not STT_SUPPORT:
            logger.warning("STT 지원 라이브러리가 설치되지 않아 음성 인식 기능을 사용할 수 없습니다.")
            return
        
        # 설정 로드
        stt_config = self.config_manager.get_section('stt')
        language = stt_config.get('language', 'ko')
        max_retries = stt_config.get('max_retries', 3)
        auto_analyze = stt_config.get('auto_analyze', True)
        
        try:
            # STT 프로세서 생성
            logger.info(f"음성 파일 변환 시작: {self.mp3_path}")
            stt_processor = STTProcessor(language=language)
            
            # 음성 -> 텍스트 변환 (재시도 옵션 사용)
            result = stt_processor.transcribe_with_retry(
                audio_file_path=self.mp3_path, 
                max_retries=max_retries,
                language=language
            )
            
            # 변환 결과 처리
            if result["success"] and result["text"]:
                # 프롬프트로 설정
                self.user_prompt = result["text"]
                logger.info(f"음성 인식 결과: '{self.user_prompt}'")
                
                # 자동 분석 옵션이 켜져있으면 분석 시작
                if auto_analyze:
                    logger.info("음성 인식 결과로 자동 분석을 시작합니다.")
                    self._start_analysis()
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                logger.error(f"음성 인식 실패: {error_msg}")
        
        except Exception as e:
            logger.exception(f"음성 변환 중 오류 발생: {e}")

    def _save_coordinate_results(self, target_object, goal_point_data):
        """타겟 물체와 목표 지점의 중심 좌표 계산 및 저장"""
        # 1. 타겟 객체 확인
        if target_object is None:
            logger.warning("타겟 객체가 없어 좌표를 저장할 수 없습니다.")
            return
        
        # 타겟 물체의 중심 좌표 계산
        target_center = self._calculate_center(target_object)
        if target_center is None:
            logger.warning("타겟 물체의 중심 좌표를 계산할 수 없습니다.")
            return
        
        # 2. 목표 지점 확인 
        if goal_point_data is None:
            logger.warning("목표 지점 데이터가 없어 좌표를 저장할 수 없습니다.")
            return
        
        # 목표 지점의 중심 좌표 계산
        goal_center = self._calculate_center(goal_point_data)
        if goal_center is None:
            logger.warning("목표 지점의 중심 좌표를 계산할 수 없습니다.")
            return
        
        # 3. 좌표 검증
        if not all(isinstance(coord, (int, float)) for coord in target_center + goal_center):
            logger.warning(f"유효하지 않은 좌표값: 타겟={target_center}, 목표={goal_center}")
            return
        
        # 4. 유효한 좌표 저장
        logger.info(f"타겟 중심 좌표: {target_center}, 목표 지점 중심 좌표: {goal_center}")
        self._save_coordinates(target_center, goal_center)

    def _calculate_center(self, object_data):
        """객체의 중심 좌표 계산"""
        if not object_data:
            return None
        
        # 타겟 객체 (바운딩 박스 [x1, y1, x2, y2] 형식)
        if "bbox" in object_data:
            bbox = object_data["bbox"]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            logger.info(f"타겟 객체 중심점 계산: ({center_x}, {center_y})")
            return (center_x, center_y)
        
        # 목표 지점 (goal_point 구조 처리)
        elif "goal_point" in object_data:
            # goal_point 내부의 calculation_details 확인 (가장 정확한 정보)
            if "calculation_details" in object_data and "bbox" in object_data["calculation_details"]:
                bbox = object_data["calculation_details"]["bbox"]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                logger.info(f"목표 지점 중심점 계산 (calculation_details): ({center_x}, {center_y})")
                return (center_x, center_y)
            
            # goal_point 내부의 goal_point.2d_bbox 확인
            elif "goal_point" in object_data and isinstance(object_data["goal_point"], dict):
                if "2d_bbox" in object_data["goal_point"]:
                    bbox = object_data["goal_point"]["2d_bbox"]
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    logger.info(f"목표 지점 중심점 계산 (2d_bbox): ({center_x}, {center_y})")
                    return (center_x, center_y)
                # 3D 좌표에서 screen_x, screen_y 확인
                elif "3d_coords" in object_data["goal_point"]:
                    coords = object_data["goal_point"]["3d_coords"]
                    if "screen_x" in coords and "screen_y" in coords:
                        return (int(coords["screen_x"]), int(coords["screen_y"]))
                    elif "x_cm" in coords and "y_cm" in coords:
                        return (int(coords["x_cm"]), int(coords["y_cm"]))
        
        # 그 외 다른 형태의 중심점 데이터 처리 시도
        # center_x, center_y 직접 제공되는 경우
        if "center_x" in object_data and "center_y" in object_data:
            return (int(object_data["center_x"]), int(object_data["center_y"]))
        
        # x, y, width, height 형식
        elif "x" in object_data and "y" in object_data:
            x = object_data["x"]
            y = object_data["y"]
            width = object_data.get("width", 0)
            height = object_data.get("height", 0)
            return (int(x + width/2), int(y + height/2))
        
        # 로그에 알 수 없는 형식 기록
        if isinstance(object_data, dict):
            logger.warning(f"좌표 계산 실패: 알 수 없는 객체 형식 {object_data.keys()}")
        else:
            logger.warning(f"좌표 계산 실패: 알 수 없는 객체 형식 {type(object_data)}")
        
        return None

    def _save_coordinates(self, target_center, goal_center):
        """중심 좌표를 JSON 형식으로 저장"""
        if target_center is None or goal_center is None:
            logger.warning("유효한 좌표가 없어 저장하지 않습니다.")
            return
        
        # 1. 폴더 내 모든 JSON 파일 제거
        for file in os.listdir(self.coordinate_dir):
            if file.endswith('.json'):
                try:
                    os.remove(os.path.join(self.coordinate_dir, file))
                except Exception as e:
                    logger.error(f"파일 삭제 중 오류: {e}")
        
        # 2. 좌표 데이터 JSON 형식으로 저장
        coordinates = {
            "target_center": list(target_center),  # 튜플을 리스트로 변환
            "goal_center": list(goal_center)       # 튜플을 리스트로 변환
        }
        
        # 항상 동일한 파일명 사용 (덮어쓰기)
        filename = "coordinates.json"
        filepath = os.path.join(self.coordinate_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(coordinates, f, indent=2)
            logger.info(f"좌표가 {filepath}에 저장되었습니다: {coordinates}")
        except Exception as e:
            logger.error(f"좌표 저장 중 오류: {e}")

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="대화형 목표지점 추론 시스템")
    parser.add_argument("--yolo", default=None, help="사용할 YOLOv8 모델 (기본값: 설정 파일 참조)")
    parser.add_argument("--llm", choices=["gpt4o", "gemini"], 
                        default=None, help="사용할 LLM 유형 (기본값: 설정 파일 참조)")
    parser.add_argument("--conf", type=float, default=None, 
                        help="YOLO 객체 감지 신뢰도 임계값 (기본값: 설정 파일 참조)")
    parser.add_argument("--iou", type=float, default=None, 
                        help="YOLO 비최대 억제(NMS) IoU 임계값 (기본값: 설정 파일 참조)")
    parser.add_argument("--gesture_model", default=None,
                        help="MediaPipe Hand Landmarker 모델 파일 경로 (기본값: 설정 파일 참조)")
    parser.add_argument("--output_dir", default=None,
                        help="결과 저장 디렉토리 (기본값: 설정 파일 참조)")
    parser.add_argument("--env", default=None,
                        help="실행 환경 (development, testing, production) (기본값: 환경변수 또는 development)")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    # 환경 변수 로드
    load_dotenv()
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 명령행 인자를 딕셔너리로 변환
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # 설정 관리자 초기화
    config_manager = get_config_manager(
        environment=args.env,
        cmd_args=cmd_args
    )
    
    # 애플리케이션 시작
    app = InteractiveGoalInferenceApp(config_manager=config_manager)
    app.start()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.exception(f"오류 발생: {str(e)}")
