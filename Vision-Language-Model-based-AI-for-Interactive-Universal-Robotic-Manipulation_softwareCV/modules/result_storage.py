#!/usr/bin/env python
"""
결과 저장 모듈

이 모듈은 분석 또는 추론 결과를 storage/ 디렉토리에 저장하는 기능을 제공합니다.
시간 기반 폴더 구조를 만들고 다양한 결과 파일을 저장합니다.
"""

import os
import time
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import shutil
from enum import Enum

# 사용자 정의 JSON 인코더
class EnumEncoder(json.JSONEncoder):
    """
    Enum 클래스와 NumPy 배열을 JSON으로 직렬화할 수 있는 사용자 정의 인코더
    """
    def default(self, obj):
        if isinstance(obj, Enum):
            # Enum은 이름으로 직렬화
            return obj.name
        if isinstance(obj, np.ndarray):
            # NumPy 배열은 리스트로 변환
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            # 객체는 __dict__로 변환 (클래스 속성)
            return obj.__dict__
        # 다른 타입은 기본 JSON 인코더가 처리하도록 에러 발생
        return super().default(obj)

class ResultStorage:
    """
    결과 저장 클래스
    
    분석 또는 추론 결과를 저장하고 관리하는 기능을 제공합니다.
    시간 기반 폴더 구조를 생성하고 다양한 결과 파일(이미지, JSON 등)을 저장합니다.
    """
    
    def __init__(self, storage_dir: str = "results", 
                 max_results: int = 1000,
                 logger: Optional[logging.Logger] = None):
        """
        ResultStorage 클래스 초기화
        
        Args:
            storage_dir (str): 결과를 저장할 디렉토리 경로
            max_results (int): 메모리에 유지할 최대 결과 수
            logger (Optional[logging.Logger]): 로깅에 사용할 로거 객체
        """
        self.storage_dir = storage_dir
        self.max_results = max_results
        self.results = []
        self.logger = logger or logging.getLogger(__name__)
        
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        # 기존 결과 로드
        self._load_existing_results()
        
    def _load_existing_results(self) -> None:
        """
        저장된 모든 결과 파일을 로드
        """
        if not os.path.exists(self.storage_dir):
            return
            
        self.results = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.storage_dir, filename), 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        self.results.append(result)
                except Exception as e:
                    self.logger.error(f"파일 로드 중 오류 발생: {filename} - {str(e)}")
        
        # 타임스탬프 기준으로 정렬
        self.results = sorted(self.results, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # 최대 개수 제한
        if len(self.results) > self.max_results:
            self.results = self.results[:self.max_results]
            
    def store_result(self, result: Dict[str, Any]) -> str:
        """
        새 결과를 저장하고 파일 이름 반환
        
        Args:
            result (Dict[str, Any]): 저장할 결과 데이터
            
        Returns:
            str: 저장된 결과 파일 이름
        """
        # 타임스탬프 추가 (없는 경우)
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
            
        # 결과를 메모리에 저장
        self.results.insert(0, result)
        
        # 최대 개수 제한
        if len(self.results) > self.max_results:
            self.results = self.results[:self.max_results]
            
        # 파일명 생성 (타임스탬프 기반)
        timestamp = result.get('timestamp', datetime.now().isoformat())
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
            
        method = result.get('method', 'unknown')
        filename = f"{timestamp.replace(':', '-')}_{method}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # 파일에 저장 (EnumEncoder 사용)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, cls=EnumEncoder)
            self.logger.debug(f"결과 저장 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {str(e)}")
            
        return filename
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        최근 결과를 가져옴
        
        Args:
            count (int): 가져올 결과 개수
            
        Returns:
            List[Dict[str, Any]]: 최근 결과 목록
        """
        return self.results[:min(count, len(self.results))]
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """
        모든 결과를 가져옴
        
        Returns:
            List[Dict[str, Any]]: 모든 결과 목록
        """
        return self.results
    
    def get_results_by_success(self, success: bool = True) -> List[Dict[str, Any]]:
        """
        성공/실패 여부에 따라 결과 필터링
        
        Args:
            success (bool): 성공 여부
            
        Returns:
            List[Dict[str, Any]]: 필터링된 결과 목록
        """
        return [r for r in self.results if r.get('success') == success]
    
    def get_results_by_method(self, method: str) -> List[Dict[str, Any]]:
        """
        메서드 기준으로 결과 필터링
        
        Args:
            method (str): 메서드 이름
            
        Returns:
            List[Dict[str, Any]]: 필터링된 결과 목록
        """
        return [r for r in self.results if r.get('method') == method]
    
    def get_results_by_timerange(self, 
                               start_time: Optional[Union[str, datetime]] = None,
                               end_time: Optional[Union[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        시간 범위로 결과 필터링
        
        Args:
            start_time (Optional[Union[str, datetime]]): 시작 시간
            end_time (Optional[Union[str, datetime]]): 종료 시간
            
        Returns:
            List[Dict[str, Any]]: 필터링된 결과 목록
        """
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
            
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        filtered_results = self.results
        
        if start_time:
            filtered_results = [r for r in filtered_results if r.get('timestamp', '') >= start_time]
            
        if end_time:
            filtered_results = [r for r in filtered_results if r.get('timestamp', '') <= end_time]
            
        return filtered_results
    
    def clear_results(self) -> None:
        """
        메모리에 있는 모든 결과 삭제
        """
        self.results = []
        
    def delete_files_older_than(self, days: int) -> int:
        """
        특정 기간보다 오래된 결과 파일 삭제
        
        Args:
            days (int): 보관할 일수
            
        Returns:
            int: 삭제된 파일 개수
        """
        if not os.path.exists(self.storage_dir):
            return 0
            
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.storage_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            if file_time < cutoff_date:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    self.logger.debug(f"오래된 파일 삭제: {filepath}")
                except Exception as e:
                    self.logger.error(f"파일 삭제 중 오류 발생: {filepath} - {str(e)}")
                    
        # 메모리에 있는 결과도 다시 로드
        self._load_existing_results()
        
        return deleted_count
    
    def delete_file(self, filename: str) -> bool:
        """
        특정 결과 파일 삭제
        
        Args:
            filename (str): 삭제할 파일 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        filepath = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"삭제할 파일이 존재하지 않음: {filepath}")
            return False
            
        try:
            os.remove(filepath)
            
            # 메모리에서도 해당 결과 제거
            self.results = [r for r in self.results if r.get('filename') != filename]
            
            self.logger.debug(f"파일 삭제 완료: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"파일 삭제 중 오류 발생: {filepath} - {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        저장된 결과에 대한 통계 정보 제공
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        total_count = len(self.results)
        success_count = len([r for r in self.results if r.get('success') == True])
        
        # 메서드별 카운트
        method_counts = {}
        for result in self.results:
            method = result.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
        # 메서드별 성공률
        method_success_rates = {}
        for method in method_counts:
            success_count_method = len([r for r in self.results 
                                      if r.get('method') == method and r.get('success') == True])
            method_success_rates[method] = (success_count_method / method_counts[method]) * 100 if method_counts[method] > 0 else 0
            
        return {
            'total_count': total_count,
            'success_count': success_count,
            'success_rate': (success_count / total_count) * 100 if total_count > 0 else 0,
            'method_counts': method_counts,
            'method_success_rates': method_success_rates
        }
    
    def create_session(self) -> str:
        """
        새로운 세션 디렉토리 생성
        
        Returns:
            생성된 세션 디렉토리 경로
        """
        # 현재 시간 기반 폴더명 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(self.storage_dir, timestamp)
        
        # 디렉토리 생성
        os.makedirs(session_dir, exist_ok=True)
        self.current_session_dir = session_dir
        self.logger.info(f"새 세션 디렉토리 생성: {session_dir}")
        
        return session_dir
    
    def save_image(self, 
                  image: np.ndarray, 
                  filename: str = "result.jpg", 
                  session_dir: Optional[str] = None) -> str:
        """
        이미지 파일 저장
        
        Args:
            image: 저장할 이미지 (numpy 배열)
            filename: 저장할 파일명 (기본값: "result.jpg")
            session_dir: 세션 디렉토리 (기본값: None, 현재 세션 디렉토리 사용)
            
        Returns:
            저장된 파일의 전체 경로
        """
        # 세션 디렉토리 확인
        if session_dir is None:
            if self.current_session_dir is None:
                session_dir = self.create_session()
            else:
                session_dir = self.current_session_dir
        
        # 파일 경로 생성
        file_path = os.path.join(session_dir, filename)
        
        # 이미지 저장
        try:
            cv2.imwrite(file_path, image)
            self.logger.info(f"이미지 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"이미지 저장 실패: {e}")
            return ""
    
    def save_json(self, 
                 data: Dict[str, Any], 
                 filename: str = "results.json", 
                 session_dir: Optional[str] = None) -> str:
        """
        JSON 데이터 저장
        
        Args:
            data: 저장할 데이터 (딕셔너리)
            filename: 저장할 파일명 (기본값: "results.json")
            session_dir: 세션 디렉토리 (기본값: None, 현재 세션 디렉토리 사용)
            
        Returns:
            저장된 파일의 전체 경로
        """
        # 세션 디렉토리 확인
        if session_dir is None:
            if self.current_session_dir is None:
                session_dir = self.create_session()
            else:
                session_dir = self.current_session_dir
        
        # 파일 경로 생성
        file_path = os.path.join(session_dir, filename)
        
        # JSON 데이터 저장 (EnumEncoder 사용)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=EnumEncoder)
            self.logger.info(f"JSON 데이터 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"JSON 데이터 저장 실패: {e}")
            return ""
    
    def save_text(self, 
                 text: str, 
                 filename: str = "log.txt", 
                 session_dir: Optional[str] = None) -> str:
        """
        텍스트 파일 저장
        
        Args:
            text: 저장할 텍스트
            filename: 저장할 파일명 (기본값: "log.txt")
            session_dir: 세션 디렉토리 (기본값: None, 현재 세션 디렉토리 사용)
            
        Returns:
            저장된 파일의 전체 경로
        """
        # 세션 디렉토리 확인
        if session_dir is None:
            if self.current_session_dir is None:
                session_dir = self.create_session()
            else:
                session_dir = self.current_session_dir
        
        # 파일 경로 생성
        file_path = os.path.join(session_dir, filename)
        
        # 텍스트 저장
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"텍스트 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"텍스트 저장 실패: {e}")
            return ""
    
    def save_analysis_results(self, 
                             results_data: Dict[str, Any], 
                             snapshot_image: Optional[np.ndarray] = None, 
                             result_image: Optional[np.ndarray] = None,
                             depth_map: Optional[np.ndarray] = None,
                             session_dir: Optional[str] = None) -> Dict[str, str]:
        """
        분석 결과 일괄 저장
        
        Args:
            results_data: 분석 결과 데이터
            snapshot_image: 원본 스냅샷 이미지 (기본값: None)
            result_image: 결과 시각화 이미지 (기본값: None)
            depth_map: 깊이 맵 이미지 (기본값: None)
            session_dir: 세션 디렉토리 (기본값: None, 새로 생성)
            
        Returns:
            저장된 파일 경로 맵 {"json": path, "snapshot": path, "result": path, "depth": path}
        """
        # 세션 디렉토리 확인 (항상 새로 생성)
        if session_dir is None:
            session_dir = self.create_session()
        else:
            os.makedirs(session_dir, exist_ok=True)
            self.current_session_dir = session_dir
        
        saved_paths = {}
        
        # JSON 데이터 저장
        try:
            json_path = self.save_json(results_data, "analysis_results.json", session_dir)
            saved_paths["json"] = json_path
        except Exception as e:
            self.logger.error(f"JSON 데이터 저장 중 오류: {e}")
        
        # 스냅샷 이미지 저장
        if snapshot_image is not None:
            try:
                snapshot_path = self.save_image(snapshot_image, "snapshot.jpg", session_dir)
                saved_paths["snapshot"] = snapshot_path
            except Exception as e:
                self.logger.error(f"스냅샷 이미지 저장 중 오류: {e}")
        
        # 결과 이미지 저장
        if result_image is not None:
            try:
                result_path = self.save_image(result_image, "result.jpg", session_dir)
                saved_paths["result"] = result_path
            except Exception as e:
                self.logger.error(f"결과 이미지 저장 중 오류: {e}")
        
        # 깊이 맵 이미지 저장 (새로 추가)
        if depth_map is not None:
            try:
                # 깊이 맵이 단채널인 경우 처리
                if len(depth_map.shape) == 2:
                    # 단채널 깊이 맵을 컬러맵으로 변환
                    colored_depth = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                    depth_path = self.save_image(colored_depth, "depth_map.jpg", session_dir)
                else:
                    # 이미 컬러맵으로 변환된 경우
                    depth_path = self.save_image(depth_map, "depth_map.jpg", session_dir)
                saved_paths["depth"] = depth_path
                self.logger.info(f"깊이 맵 저장 완료: {depth_path}")
            except Exception as e:
                self.logger.error(f"깊이 맵 이미지 저장 중 오류: {e}")
        
        # 로그 메시지 생성
        log_message = f"타임스탬프: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if "detections" in results_data:
            log_message += f"감지된 객체 수: {len(results_data['detections'])}\n"
        if "goal_point" in results_data and results_data["goal_point"]:
            log_message += f"목표 지점: {results_data['goal_point']}\n"
        if "user_prompt" in results_data and results_data["user_prompt"]:
            log_message += f"사용자 프롬프트: {results_data['user_prompt']}\n"
        if "timing" in results_data:
            log_message += f"처리 시간: {results_data['timing'].get('total', 0):.3f}초\n"
        
        # 로그 저장
        try:
            log_path = self.save_text(log_message, "summary.txt", session_dir)
            saved_paths["log"] = log_path
        except Exception as e:
            self.logger.error(f"로그 파일 저장 중 오류: {e}")
        
        self.logger.info(f"분석 결과 일괄 저장 완료: {session_dir}")
        return saved_paths 