"""
카메라 매니저 모듈

다양한 카메라 인스턴스를 생성, 관리하고 전환하는 클래스입니다.
"""

import cv2
import time
import logging
import platform
from typing import Dict, List, Optional, Any
from modules.camera.camera_factory import CameraFactory
from modules.camera.camera_interface import CameraInterface
from modules.camera.webcam_camera import WebcamCamera

logger = logging.getLogger(__name__)

class CameraManager:
    """카메라 매니저 클래스
    
    시스템에서 사용 가능한 카메라를 검색하고 관리하는 싱글톤 클래스입니다.
    """
    
    _instance = None
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(CameraManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """CameraManager 초기화"""
        if self._initialized:
            return
            
        self._initialized = True
        self._cameras = {}  # 카메라 ID: 카메라 인스턴스
        self._active_camera = None  # 현재 활성화된 카메라
        self._available_cameras = []  # 사용 가능한 카메라 목록
        
    def initialize(self, config=None):
        """카메라 매니저 초기화
        
        Args:
            config: 카메라 설정 딕셔너리 (기본값=None)
        """
        self._config = config or {}
        
        # 카메라 스캔 건너뛰기 옵션 확인
        if self._config.get('skip_camera_scan', False):
            logger.info("카메라 스캔 건너뛰기 옵션이 활성화되어 지정된 카메라만 사용합니다.")
            # 지정된 device_id로 가상의 카메라 정보 생성
            device_id = self._config.get('device_id', 0)
            width = self._config.get('width', 640)
            height = self._config.get('height', 480)
            fps = self._config.get('fps', 30)
            
            self._available_cameras = [{
                'id': f"webcam_{device_id}",
                'name': f"카메라 {device_id}",
                'type': 'webcam',
                'index': device_id,
                'backend': 'default',
                'resolution': (width, height),
                'fps': fps
            }]
        else:
            # 일반적인 카메라 스캔 진행
            self._scan_available_cameras()
        
    def _scan_available_cameras(self):
        """시스템에서 사용 가능한 카메라 검색"""
        logger.info("카메라 검색 중...")
        
        # 이전 검색 결과 초기화
        self._available_cameras = []
        
        # 플랫폼 확인 (MacOS에서는 AVFoundation 백엔드 지원 여부 확인)
        system = platform.system()
        has_avfoundation = False
        
        if system == 'Darwin':  # macOS
            has_avfoundation = True
            logger.info("macOS 환경 감지됨: AVFoundation 백엔드 사용 가능")
        
        # 웹캠 카메라 검색
        try:
            # 최대 확인할 카메라 인덱스 (필요에 따라 조정)
            max_cameras = 10
            
            for i in range(max_cameras):
                # 기본 백엔드로 시도
                cap = None
                try:
                    logger.debug(f"카메라 인덱스 {i} 확인 중 (기본 백엔드)...")
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        self._available_cameras.append({
                            'id': f"webcam_{i}",
                            'name': f"카메라 {i}",
                            'type': 'webcam',
                            'index': i,
                            'backend': 'default',
                            'resolution': (width, height),
                            'fps': fps
                        })
                        logger.info(f"카메라 발견: 인덱스 {i}, 해상도 {width}x{height}")
                finally:
                    if cap is not None:
                        cap.release()
                
                # macOS에서 AVFoundation 백엔드로 시도
                if has_avfoundation:
                    cap = None
                    try:
                        logger.debug(f"카메라 인덱스 {i} 확인 중 (AVFoundation)...")
                        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            # 이미 기본 백엔드로 발견된 경우 중복 방지
                            if not any(c['index'] == i and c['backend'] == 'default' for c in self._available_cameras):
                                self._available_cameras.append({
                                    'id': f"avfoundation_{i}",
                                    'name': f"카메라 {i} (AVFoundation)",
                                    'type': 'avfoundation',
                                    'index': i,
                                    'backend': 'avfoundation',
                                    'resolution': (width, height),
                                    'fps': fps
                                })
                                logger.info(f"카메라 발견: 인덱스 {i} (AVFoundation), 해상도 {width}x{height}")
                    finally:
                        if cap is not None:
                            cap.release()
            
            logger.info(f"카메라 검색 완료: {len(self._available_cameras)}개 발견됨")
            
        except Exception as e:
            logger.error(f"카메라 검색 중 오류 발생: {e}")
    
    def get_available_cameras(self) -> List[Dict[str, Any]]:
        """사용 가능한 카메라 목록 반환
        
        Returns:
            List[Dict[str, Any]]: 사용 가능한 카메라 정보 목록
        """
        if not self._available_cameras:
            self._scan_available_cameras()
        return self._available_cameras
    
    def create_camera_from_config(self, config: Dict[str, Any] = None) -> Optional[CameraInterface]:
        """설정 기반으로 카메라 생성
        
        Args:
            config: 카메라 설정 (기본값=None, 기본 설정 사용)
            
        Returns:
            Optional[CameraInterface]: 생성된 카메라 또는 None
        """
        config = config or self._config
        
        # 설정에서 카메라 정보 가져오기
        device_id = config.get('device_id', 0)
        device_type = config.get('device_type', 'auto')
        width = config.get('width', 640)
        height = config.get('height', 480)
        fps = config.get('fps', 30)
        
        # 카메라 스캔 건너뛰기 옵션이 활성화된 경우 직접 지정된 카메라 생성
        if config.get('skip_camera_scan', False):
            logger.info(f"지정된 카메라 인덱스 {device_id}로 직접 카메라 생성")
            return self._create_default_camera(device_id, width, height, fps)
        
        # 자동 감지 모드
        if device_type == 'auto':
            # 사용 가능한 카메라 확인
            available = self.get_available_cameras()
            
            # 카메라가 없으면 기본 웹캠으로 시도
            if not available:
                logger.warning("사용 가능한 카메라가 없습니다. 기본 웹캠으로 시도합니다.")
                return self._create_default_camera(device_id, width, height, fps)
            
            # 요청한 인덱스의 카메라 찾기
            for camera_info in available:
                if camera_info['index'] == device_id:
                    logger.info(f"인덱스 {device_id}의 카메라를 사용합니다: {camera_info['name']}")
                    
                    # 해당 인덱스가 AVFoundation으로 사용 가능하고 macOS인 경우 우선 사용
                    if platform.system() == 'Darwin' and any(c['index'] == device_id and c['backend'] == 'avfoundation' for c in available):
                        return self._create_camera_with_backend(device_id, 'avfoundation', width, height, fps)
                    
                    # 그 외에는 기본 백엔드 사용
                    return self._create_camera_with_backend(device_id, 'default', width, height, fps)
            
            # 요청한 인덱스의 카메라가 없으면 첫 번째 사용 가능한 카메라 사용
            logger.warning(f"인덱스 {device_id}의 카메라를 찾을 수 없습니다. 첫 번째 사용 가능한 카메라를 사용합니다.")
            camera_info = available[0]
            return self._create_camera_with_backend(camera_info['index'], camera_info.get('backend', 'default'), width, height, fps)
        
        # 특정 타입 지정 모드
        elif device_type == 'avfoundation' and platform.system() == 'Darwin':
            return self._create_camera_with_backend(device_id, 'avfoundation', width, height, fps)
        
        # 기본 웹캠 모드 (또는 지원하지 않는 타입)
        else:
            return self._create_default_camera(device_id, width, height, fps)
    
    def _create_default_camera(self, device_id, width, height, fps):
        """기본 백엔드로 카메라 생성
        
        Args:
            device_id: 카메라 인덱스
            width: 화면 너비
            height: 화면 높이
            fps: 초당 프레임 수
            
        Returns:
            Optional[CameraInterface]: 생성된 카메라 또는 None
        """
        camera_id = f"webcam_{device_id}"
        logger.info(f"기본 백엔드로 카메라 생성 중: 인덱스 {device_id}")
        
        params = {
            'index': device_id,
            'resolution': (width, height),
            'fps': fps
        }
        
        try:
            # 고급 설정이 있으면 추가
            if 'advanced' in self._config:
                advanced = self._config['advanced']
                if 'flip_horizontal' in advanced:
                    params['flip_horizontal'] = advanced['flip_horizontal']
                if 'flip_vertical' in advanced:
                    params['flip_vertical'] = advanced['flip_vertical']
            
            camera = CameraFactory.create_camera('webcam', camera_id, f"카메라 {device_id}", **params)
            if camera:
                self._cameras[camera_id] = camera
                return camera
            else:
                logger.error(f"카메라 생성 실패: 인덱스 {device_id}")
                return None
                
        except Exception as e:
            logger.error(f"카메라 생성 중 오류 발생: {e}")
            return None
    
    def _create_camera_with_backend(self, device_id, backend, width, height, fps):
        """특정 백엔드로 카메라 생성
        
        Args:
            device_id: 카메라 인덱스
            backend: 백엔드 (default 또는 avfoundation)
            width: 화면 너비
            height: 화면 높이
            fps: 초당 프레임 수
            
        Returns:
            Optional[CameraInterface]: 생성된 카메라 또는 None
        """
        camera_id = f"{backend}_{device_id}" if backend != 'default' else f"webcam_{device_id}"
        logger.info(f"{backend} 백엔드로 카메라 생성 중: 인덱스 {device_id}")
        
        params = {
            'index': device_id,
            'resolution': (width, height),
            'fps': fps,
            'backend': backend
        }
        
        try:
            # 고급 설정이 있으면 추가
            if 'advanced' in self._config:
                advanced = self._config['advanced']
                if 'flip_horizontal' in advanced:
                    params['flip_horizontal'] = advanced['flip_horizontal']
                if 'flip_vertical' in advanced:
                    params['flip_vertical'] = advanced['flip_vertical']
            
            camera = CameraFactory.create_camera('webcam', camera_id, f"카메라 {device_id} ({backend})", **params)
            if camera:
                self._cameras[camera_id] = camera
                return camera
            else:
                logger.error(f"카메라 생성 실패: 인덱스 {device_id}, 백엔드 {backend}")
                return None
                
        except Exception as e:
            logger.error(f"카메라 생성 중 오류 발생: {e}")
            return None
    
    def get_camera(self, camera_id: str) -> Optional[CameraInterface]:
        """ID로 카메라 가져오기
        
        Args:
            camera_id: 카메라 ID
            
        Returns:
            Optional[CameraInterface]: 카메라 객체 또는 None
        """
        return self._cameras.get(camera_id)
    
    def get_camera_by_index(self, index: int, backend: str = 'default') -> Optional[CameraInterface]:
        """인덱스로 카메라 가져오기
        
        Args:
            index: 카메라 인덱스
            backend: 백엔드 (기본값: 'default')
            
        Returns:
            Optional[CameraInterface]: 카메라 객체 또는 None
        """
        camera_id = f"{backend}_{index}" if backend != 'default' else f"webcam_{index}"
        return self.get_camera(camera_id)
    
    def get_active_camera(self) -> Optional[CameraInterface]:
        """현재 활성화된 카메라 가져오기
        
        Returns:
            Optional[CameraInterface]: 활성화된 카메라 또는 None
        """
        return self._active_camera
    
    def set_active_camera(self, camera: CameraInterface) -> bool:
        """활성 카메라 설정
        
        Args:
            camera: 카메라 객체
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 현재 활성 카메라가 있으면 중지
            if self._active_camera and self._active_camera.is_open():
                self._active_camera.close()
            
            # 새 카메라 활성화
            if not camera.is_open():
                camera.open()
            
            self._active_camera = camera
            return True
            
        except Exception as e:
            logger.error(f"활성 카메라 설정 중 오류 발생: {e}")
            return False
    
    def set_active_camera_by_id(self, camera_id: str) -> bool:
        """ID로 활성 카메라 설정
        
        Args:
            camera_id: 카메라 ID
            
        Returns:
            bool: 성공 여부
        """
        camera = self.get_camera(camera_id)
        if camera:
            return self.set_active_camera(camera)
        else:
            logger.error(f"카메라를 찾을 수 없음: {camera_id}")
            return False
    
    def set_active_camera_by_index(self, index: int, backend: str = 'default') -> bool:
        """인덱스로 활성 카메라 설정
        
        Args:
            index: 카메라 인덱스
            backend: 백엔드 (기본값: 'default')
            
        Returns:
            bool: 성공 여부
        """
        camera_id = f"{backend}_{index}" if backend != 'default' else f"webcam_{index}"
        
        # 이미 생성된 카메라가 있는지 확인
        camera = self.get_camera(camera_id)
        
        # 없으면 새로 생성
        if not camera:
            width = self._config.get('width', 640)
            height = self._config.get('height', 480)
            fps = self._config.get('fps', 30)
            
            if backend == 'avfoundation':
                camera = self._create_camera_with_backend(index, 'avfoundation', width, height, fps)
            else:
                camera = self._create_default_camera(index, width, height, fps)
                
            if not camera:
                return False
        
        return self.set_active_camera(camera)
        
    def release_all(self):
        """모든 카메라 해제"""
        try:
            # 현재 활성 카메라 해제
            if self._active_camera:
                self._active_camera.close()
                self._active_camera = None
            
            # 모든 카메라 해제
            for camera_id, camera in self._cameras.items():
                if camera.is_open():
                    camera.close()
            
            self._cameras = {}
            logger.info("모든 카메라가 해제되었습니다.")
            
        except Exception as e:
            logger.error(f"카메라 해제 중 오류 발생: {e}") 