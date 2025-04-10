"""
프레임 저장 및 메타데이터 관리 모듈

캡처된 프레임을 효율적으로 저장하고 관리하는 기능을 제공합니다.
시간/세션별 폴더 구조, 메타데이터 저장, 디스크 공간 관리 기능을 포함합니다.
"""

import os
import json
import time
import shutil
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image

# 로거 설정
logger = logging.getLogger(__name__)

class FrameStorage:
    """프레임 저장 및 관리 클래스
    
    캡처된 프레임을 구조화된 폴더에 저장하고 메타데이터를 관리합니다.
    디스크 공간을 모니터링하고 필요시 오래된 파일을 정리합니다.
    
    Attributes:
        base_path (Path): 기본 저장 경로
        session_id (str): 현재 세션 ID
        compression_quality (int): JPEG 압축 품질 (0-100)
        max_disk_usage_gb (float): 최대 디스크 사용량 (GB)
        cleanup_threshold_gb (float): 정리 임계값 (GB)
        metadata_format (str): 메타데이터 저장 형식 ('json', 'exif')
        current_session_path (Path): 현재 세션 저장 경로
        image_counter (int): 이미지 카운터
    """
    
    def __init__(self, 
                 base_path: str = "captures", 
                 session_id: Optional[str] = None,
                 compression_quality: int = 85,
                 max_disk_usage_gb: float = 5.0,
                 cleanup_threshold_gb: float = 4.0,
                 metadata_format: str = "json"):
        """FrameStorage 초기화
        
        Args:
            base_path (str): 기본 저장 경로 (기본값: "captures")
            session_id (Optional[str]): 세션 ID (기본값: None, 자동 생성)
            compression_quality (int): JPEG 압축 품질 (0-100) (기본값: 85)
            max_disk_usage_gb (float): 최대 디스크 사용량 (GB) (기본값: 5.0)
            cleanup_threshold_gb (float): 정리 임계값 (GB) (기본값: 4.0)
            metadata_format (str): 메타데이터 저장 형식 ('json' 또는 'exif') (기본값: "json")
        
        Raises:
            ValueError: 잘못된 압축 품질, 디스크 사용량 또는 메타데이터 형식이 지정된 경우
        """
        # 인자 검증
        if not 0 <= compression_quality <= 100:
            raise ValueError("압축 품질은 0에서 100 사이여야 합니다.")
        
        if max_disk_usage_gb <= 0 or cleanup_threshold_gb <= 0:
            raise ValueError("디스크 사용량은 양수여야 합니다.")
            
        if cleanup_threshold_gb >= max_disk_usage_gb:
            raise ValueError("정리 임계값은 최대 디스크 사용량보다 작아야 합니다.")
            
        if metadata_format not in ["json", "exif"]:
            raise ValueError("메타데이터 형식은 'json' 또는 'exif'여야 합니다.")
        
        # 속성 초기화
        self.base_path = Path(base_path)
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.compression_quality = compression_quality
        self.max_disk_usage_gb = max_disk_usage_gb
        self.cleanup_threshold_gb = cleanup_threshold_gb
        self.metadata_format = metadata_format
        self.image_counter = 0
        
        # 세션 경로 생성
        self.current_session_path = self.base_path / self.session_id
        os.makedirs(self.current_session_path, exist_ok=True)
        
        # 메타데이터 경로
        if self.metadata_format == "json":
            self.metadata_path = self.current_session_path / "metadata"
            os.makedirs(self.metadata_path, exist_ok=True)
        
        # 로그
        logger.info(f"프레임 저장소 초기화: session_id={self.session_id}, "
                   f"base_path={self.base_path}, quality={self.compression_quality}")
    
    def save_frame(self, 
                  frame: np.ndarray, 
                  metadata: Dict[str, Any], 
                  filename: Optional[str] = None,
                  subfolder: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """프레임 저장
        
        Args:
            frame (np.ndarray): 저장할 프레임
            metadata (Dict[str, Any]): 메타데이터 딕셔너리
            filename (Optional[str]): 파일 이름 (기본값: None, 자동 생성)
            subfolder (Optional[str]): 하위 폴더 (기본값: None)
            
        Returns:
            Tuple[str, Dict[str, Any]]: (저장된 파일 경로, 확장된 메타데이터)
            
        Raises:
            RuntimeError: 프레임 저장 실패 시 발생
        """
        # 디스크 공간 확인 및 정리
        self._check_disk_space()
        
        # 저장 경로 결정
        save_path = self.current_session_path
        if subfolder:
            save_path = save_path / subfolder
            os.makedirs(save_path, exist_ok=True)
        
        # 파일명 결정
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self.image_counter += 1
            filename = f"frame_{self.image_counter:04d}_{timestamp}.jpg"
        
        # 파일 경로 생성
        file_path = save_path / filename
        
        # 메타데이터 확장
        extended_metadata = self._extend_metadata(metadata, filename)
        
        try:
            # 프레임 저장 방식 결정
            if self.metadata_format == "exif":
                # EXIF 메타데이터와 함께 저장
                self._save_with_exif(frame, file_path, extended_metadata)
            else:
                # 일반 저장 후 별도 JSON 파일에 메타데이터 저장
                self._save_frame_and_metadata(frame, file_path, extended_metadata)
            
            logger.info(f"프레임 저장 완료: {file_path}")
            return str(file_path), extended_metadata
            
        except Exception as e:
            error_msg = f"프레임 저장 실패: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _save_frame_and_metadata(self, 
                                frame: np.ndarray, 
                                file_path: Path, 
                                metadata: Dict[str, Any]) -> None:
        """프레임과 별도 메타데이터 파일 저장
        
        Args:
            frame (np.ndarray): 저장할 프레임
            file_path (Path): 이미지 파일 경로
            metadata (Dict[str, Any]): 메타데이터 딕셔너리
        """
        # 이미지 저장
        save_params = [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality]
        cv2.imwrite(str(file_path), frame, save_params)
        
        # 메타데이터 저장 (JSON)
        metadata_file = self.metadata_path / f"{file_path.stem}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_with_exif(self, 
                       frame: np.ndarray, 
                       file_path: Path, 
                       metadata: Dict[str, Any]) -> None:
        """EXIF 메타데이터와 함께 이미지 저장
        
        Args:
            frame (np.ndarray): 저장할 프레임
            file_path (Path): 파일 경로
            metadata (Dict[str, Any]): 메타데이터 딕셔너리
        """
        # BGR -> RGB 변환 (OpenCV -> PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(rgb_frame)
        
        # EXIF 데이터 생성
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # 사용자 주석에 JSON 메타데이터 저장
        user_comment = json.dumps(metadata).encode('utf-8')
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        
        # 기본 EXIF 정보 추가
        exif_dict["0th"][piexif.ImageIFD.Make] = "LLM2PF6".encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.Software] = "FrameStorage".encode('utf-8')
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode('utf-8')
        
        # EXIF 데이터 직렬화
        exif_bytes = piexif.dump(exif_dict)
        
        # 이미지 저장
        pil_image.save(file_path, "JPEG", quality=self.compression_quality, exif=exif_bytes)
    
    def _extend_metadata(self, metadata: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """메타데이터 확장
        
        기본 메타데이터에 추가 정보를 보강합니다.
        
        Args:
            metadata (Dict[str, Any]): 원본 메타데이터
            filename (str): 파일 이름
            
        Returns:
            Dict[str, Any]: 확장된 메타데이터
        """
        # 메타데이터 복사
        extended = metadata.copy()
        
        # 시간 정보
        current_time = time.time()
        extended.update({
            'capture_time': current_time,
            'capture_time_iso': datetime.datetime.fromtimestamp(current_time).isoformat(),
            'session_id': self.session_id,
            'filename': filename,
            'storage_info': {
                'compression_quality': self.compression_quality,
                'base_path': str(self.base_path),
                'session_path': str(self.current_session_path),
                'metadata_format': self.metadata_format
            }
        })
        
        return extended
    
    def _check_disk_space(self) -> None:
        """디스크 공간 확인 및 필요시 정리
        
        설정된 임계값에 도달하면 오래된 파일부터 정리합니다.
        """
        # 현재 디스크 사용량 계산
        current_usage_gb = self._get_folder_size_gb(self.base_path)
        
        # 임계값 초과 시 정리
        if current_usage_gb >= self.cleanup_threshold_gb:
            logger.warning(f"디스크 사용량 임계값 초과: {current_usage_gb:.2f}GB/{self.cleanup_threshold_gb:.2f}GB")
            self._cleanup_old_files()
            
            # 정리 후 사용량 다시 확인
            after_cleanup_gb = self._get_folder_size_gb(self.base_path)
            logger.info(f"정리 후 디스크 사용량: {after_cleanup_gb:.2f}GB")
    
    def _get_folder_size_gb(self, folder_path: Path) -> float:
        """폴더 크기 계산 (GB)
        
        Args:
            folder_path (Path): 크기를 계산할 폴더 경로
            
        Returns:
            float: 폴더 크기 (GB)
        """
        if not folder_path.exists():
            return 0.0
            
        total_size = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
                
        # 바이트를 GB로 변환
        return total_size / (1024 ** 3)
    
    def _cleanup_old_files(self) -> None:
        """오래된 파일 정리
        
        가장 오래된 세션부터 삭제하여 디스크 공간을 확보합니다.
        현재 세션은 보존합니다.
        """
        # 모든 세션 폴더 가져오기
        sessions = [d for d in self.base_path.iterdir() if d.is_dir() and d != self.current_session_path]
        
        # 생성 시간 기준으로 정렬
        sessions.sort(key=lambda d: d.stat().st_mtime)
        
        # 사용량이 목표 이하가 될 때까지 정리
        for session_path in sessions:
            # 현재 세션은 건너뛰기
            if session_path == self.current_session_path:
                continue
                
            logger.info(f"오래된 세션 정리 중: {session_path}")
            
            try:
                # 폴더 삭제
                shutil.rmtree(session_path)
                
                # 현재 사용량 다시 확인
                current_usage_gb = self._get_folder_size_gb(self.base_path)
                if current_usage_gb < self.cleanup_threshold_gb * 0.8:
                    # 20% 여유 공간 확보되면 중단
                    logger.info(f"충분한 공간 확보됨: {current_usage_gb:.2f}GB")
                    break
                    
            except Exception as e:
                logger.error(f"세션 정리 중 오류 발생: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """현재 세션 정보 반환
        
        Returns:
            Dict[str, Any]: 세션 정보
        """
        # 세션 내 파일 수 계산
        image_count = len(list(self.current_session_path.glob("**/*.jpg")))
        session_size_gb = self._get_folder_size_gb(self.current_session_path)
        
        return {
            'session_id': self.session_id,
            'session_path': str(self.current_session_path),
            'started_at': os.path.getctime(self.current_session_path),
            'image_count': image_count,
            'session_size_gb': session_size_gb,
            'total_size_gb': self._get_folder_size_gb(self.base_path),
            'compression_quality': self.compression_quality,
            'metadata_format': self.metadata_format
        }
    
    def list_session_files(self, limit: Optional[int] = None, 
                         sort_by_time: bool = True) -> List[Dict[str, Any]]:
        """세션 내 파일 목록 반환
        
        Args:
            limit (Optional[int]): 반환할 최대 파일 수 (기본값: None, 전체 반환)
            sort_by_time (bool): 시간순 정렬 여부 (기본값: True)
            
        Returns:
            List[Dict[str, Any]]: 파일 정보 목록
        """
        # 모든 JPG 파일 찾기
        image_files = list(self.current_session_path.glob("**/*.jpg"))
        
        # 정렬
        if sort_by_time:
            image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
        # 제한
        if limit:
            image_files = image_files[:limit]
            
        # 파일 정보 수집
        file_info = []
        for img_path in image_files:
            # 메타데이터 경로
            metadata = None
            if self.metadata_format == "json":
                metadata_path = self.metadata_path / f"{img_path.stem}.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"메타데이터 로드 실패: {e}")
            
            # 기본 파일 정보
            info = {
                'filename': img_path.name,
                'path': str(img_path),
                'size_bytes': img_path.stat().st_size,
                'created_at': img_path.stat().st_ctime,
                'metadata': metadata
            }
            
            file_info.append(info)
            
        return file_info
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 저장소 통계
        """
        # 전체 사용량
        total_size_gb = self._get_folder_size_gb(self.base_path)
        
        # 세션 개수
        session_count = len([d for d in self.base_path.iterdir() if d.is_dir()])
        
        # 세션별 크기
        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                session_info = {
                    'session_id': session_dir.name,
                    'size_gb': self._get_folder_size_gb(session_dir),
                    'file_count': len(list(session_dir.glob("**/*.jpg"))),
                    'created_at': os.path.getctime(session_dir)
                }
                sessions.append(session_info)
        
        # 세션을 크기 기준으로 정렬
        sessions.sort(key=lambda s: s['size_gb'], reverse=True)
        
        return {
            'total_size_gb': total_size_gb,
            'session_count': session_count,
            'max_allowed_gb': self.max_disk_usage_gb,
            'cleanup_threshold_gb': self.cleanup_threshold_gb,
            'usage_percent': (total_size_gb / self.max_disk_usage_gb) * 100 if self.max_disk_usage_gb > 0 else 0,
            'sessions': sessions
        }
    
    def load_frame(self, filepath: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """저장된 프레임 로드
        
        Args:
            filepath (Union[str, Path]): 로드할 파일 경로
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]: (프레임, 메타데이터) 또는 로드 실패 시 (None, None)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"파일이 존재하지 않음: {filepath}")
            return None, None
        
        try:
            # 이미지 로드
            frame = cv2.imread(str(filepath))
            
            # 메타데이터 로드
            metadata = None
            if self.metadata_format == "json":
                # 별도 JSON 파일에서 로드
                metadata_path = self.metadata_path / f"{filepath.stem}.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
            else:
                # EXIF에서 로드
                try:
                    pil_image = Image.open(filepath)
                    exif_data = pil_image.getexif()
                    
                    if exif_data:
                        # EXIF 데이터가 있으면 메타데이터 추출
                        exif_dict = piexif.load(pil_image.info.get('exif', b''))
                        if piexif.ExifIFD.UserComment in exif_dict.get('Exif', {}):
                            user_comment = exif_dict['Exif'][piexif.ExifIFD.UserComment]
                            metadata = json.loads(user_comment.decode('utf-8'))
                except Exception as e:
                    logger.error(f"EXIF 메타데이터 로드 실패: {e}")
            
            return frame, metadata
            
        except Exception as e:
            logger.error(f"이미지 로드 실패: {e}")
            return None, None
