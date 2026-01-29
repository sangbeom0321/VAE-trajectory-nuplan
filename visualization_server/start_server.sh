#!/bin/bash

# VAE-Planner Visualization Server 시작 스크립트
# 사용법: ./start_server.sh [체크포인트_경로] [포트] [호스트]

# 기본 설정
CHECKPOINT_PATH="${1:-}"  # 기본값 없음: 자동 탐색 사용
CONFIG_PATH="${2:-train/config.yaml}"
PORT="${3:-5001}"
HOST="${4:-0.0.0.0}"  # 0.0.0.0으로 설정하면 외부 접근 가능

# 프로젝트 루트로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 체크포인트 자동 탐색 (와일드카드 사용 시)
if [[ "$CHECKPOINT_PATH" == *"*"* ]]; then
    CHECKPOINT_PATH=$(ls -t $CHECKPOINT_PATH 2>/dev/null | head -1)
fi

# 서버 시작
echo "=========================================="
echo "VAE-Planner Visualization Server"
echo "=========================================="
if [[ -n "$CHECKPOINT_PATH" && -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint: $CHECKPOINT_PATH"
else
    echo "Checkpoint: Auto-detect from train/train_output"
    CHECKPOINT_PATH=""
fi
echo "Config: $CONFIG_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "=========================================="
echo ""

python visualization_server/app.py \
    ${CHECKPOINT_PATH:+--checkpoint "$CHECKPOINT_PATH"} \
    --config "$CONFIG_PATH" \
    --port "$PORT" \
    --host "$HOST"
