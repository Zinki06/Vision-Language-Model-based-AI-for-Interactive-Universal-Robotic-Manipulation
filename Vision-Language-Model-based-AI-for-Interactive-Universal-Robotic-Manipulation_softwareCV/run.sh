#!/bin/bash

# LLM2PF6 프로젝트 실행 스크립트

# 스크립트 위치 기반으로 프로젝트 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 필요한 디렉토리 생성
mkdir -p logs
mkdir -p sessions
mkdir -p debug

# 가상 환경 활성화 (가상 환경이 있는 경우)
if [ -d ".venv" ]; then
    echo "가상 환경 활성화 중..."
    source .venv/bin/activate || {
        echo "가상 환경 활성화 실패"
        exit 1
    }
fi

# 필요한 의존성 확인
if ! python -c "import cv2" &> /dev/null; then
    echo "OpenCV가 설치되어 있지 않습니다. 의존성을 설치하시겠습니까? (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        pip install -r requirements.txt
    else
        echo "의존성 설치를 건너뜁니다. 애플리케이션이 제대로 작동하지 않을 수 있습니다."
    fi
fi

# 애플리케이션 실행
echo "LLM2PF6 애플리케이션을 시작합니다..."
echo "동일한 오류 메시지는 중복 필터링되어 한 번만 표시됩니다."
python main.py

# 종료 메시지
echo "애플리케이션이 종료되었습니다." 