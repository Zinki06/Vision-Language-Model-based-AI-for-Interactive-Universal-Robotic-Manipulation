"""
음성 녹음 및 MP3 변환 모듈

이 모듈은 시스템 마이크로부터 오디오를 녹음하고 MP3 형식으로 저장하는 기능을 제공합니다.
"""
import os
import time
import wave
import tempfile
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .audio_dependencies import get_audio_support_status

# 의존성 체크
AUDIO_SUPPORT = get_audio_support_status()
if AUDIO_SUPPORT:
    import pyaudio
    from pydub import AudioSegment

logger = logging.getLogger("AudioRecorder")

class AudioRecorder:
    """
    음성 녹음 및 MP3 변환 클래스
    
    사용자의 음성을 녹음하고 MP3 형식으로 저장하는 기능을 제공합니다.
    """
    
    def __init__(self, output_dir: str, 
                 format: int = None,  # pyaudio.paInt16
                 channels: int = 1, 
                 rate: int = 44100, 
                 chunk: int = 1024):
        """
        AudioRecorder 초기화
        
        Args:
            output_dir (str): 녹음 파일 저장 디렉토리
            format (int, optional): 오디오 형식 (기본값: pyaudio.paInt16)
            channels (int, optional): 채널 수 (기본값: 1)
            rate (int, optional): 샘플링 레이트 (기본값: 44100)
            chunk (int, optional): 청크 크기 (기본값: 1024)
        """
        if not AUDIO_SUPPORT:
            logger.error("음성 지원 라이브러리가 설치되지 않았습니다. pip install pyaudio pydub를 실행하세요.")
            raise ImportError("필요한 음성 라이브러리가 설치되지 않았습니다.")
        
        self.format = format if format is not None else pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.output_dir = output_dir
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.record_thread = None
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"AudioRecorder 초기화 완료: 출력 디렉토리={self.output_dir}")
    
    def start_recording(self) -> bool:
        """
        녹음 시작
        
        Returns:
            bool: 녹음 시작 성공 여부
        """
        # 이미 녹음 중이면 중복 시작 방지
        if self.recording:
            logger.warning("이미 녹음 중입니다.")
            return False
        
        try:
            self.frames = []
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.recording = True
            logger.info("녹음을 시작합니다.")
            
            # 별도 스레드에서 녹음 진행
            self.record_thread = threading.Thread(target=self._record)
            self.record_thread.daemon = True
            self.record_thread.start()
            return True
            
        except Exception as e:
            logger.error(f"녹음 시작 중 오류 발생: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None
            return False
    
    def _record(self):
        """녹음 스레드 함수"""
        try:
            while self.recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    logger.error(f"녹음 중 오류: {e}")
                    break
        except Exception as e:
            logger.error(f"녹음 스레드 오류: {e}")
            self.recording = False
    
    def stop_recording(self) -> Optional[str]:
        """
        녹음 종료 및 MP3 파일 저장
        
        Returns:
            Optional[str]: 저장된 MP3 파일 경로, 실패 시 None
        """
        if not self.recording:
            logger.warning("녹음 중이 아닙니다.")
            return None
        
        logger.info("녹음을 종료합니다.")
        self.recording = False
        
        # 스레드가 종료될 때까지 대기
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
            if self.record_thread.is_alive():
                logger.warning("녹음 스레드가 정상적으로 종료되지 않았습니다.")
        
        # 스트림 정리
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.error(f"스트림 정리 중 오류: {e}")
        
        # 프레임이 없으면 오류 리턴
        if not self.frames:
            logger.error("녹음된 오디오 프레임이 없습니다.")
            return None
        
        # 파일 경로 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_wav_path = os.path.join(tempfile.gettempdir(), f"recording_{timestamp}.wav")
        mp3_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp3")
        
        try:
            # WAV 파일로 먼저 저장
            logger.info(f"임시 WAV 파일 생성 중: {temp_wav_path}")
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            
            # MP3로 변환
            logger.info(f"WAV를 MP3로 변환 중: {mp3_path}")
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(mp3_path, format="mp3")
            
            # 임시 WAV 파일 삭제
            os.remove(temp_wav_path)
            logger.info(f"MP3 파일 저장 완료: {mp3_path}")
            
            return mp3_path
        
        except Exception as e:
            logger.error(f"오디오 파일 저장 중 오류: {e}")
            # 임시 파일 정리
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except:
                    pass
            return None
    
    def get_recording_status(self) -> Dict[str, Any]:
        """
        녹음 상태 정보 반환
        
        Returns:
            Dict[str, Any]: 녹음 상태 정보
        """
        return {
            "is_recording": self.recording,
            "channels": self.channels,
            "rate": self.rate,
            "format": self.format,
            "frames_count": len(self.frames),
            "duration_seconds": len(self.frames) * self.chunk / self.rate if self.frames else 0
        }
    
    def __del__(self):
        """소멸자: 자원 정리"""
        logger.info("AudioRecorder 자원 정리 중...")
        # 녹음 중이면 종료
        if self.recording:
            self.recording = False
            if self.record_thread and self.record_thread.is_alive():
                try:
                    self.record_thread.join(timeout=1.0)
                except:
                    pass
        
        # 스트림 정리
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        # PyAudio 객체 정리
        if hasattr(self, 'audio') and self.audio:
            try:
                self.audio.terminate()
            except:
                pass
        
        logger.info("AudioRecorder 자원 정리 완료") 