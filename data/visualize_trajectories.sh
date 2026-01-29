#!/bin/bash
###################################
# Trajectory 시각화 스크립트
###################################

# 시각화할 데이터 경로 (환경 변수로 오버라이드 가능)
DATA_PATH="${DATA_PATH:-$HOME/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz}"

# 시각화할 샘플 수 (최대)
NUM_SAMPLES="${NUM_SAMPLES:-100000}"

# 실제로 그릴 샘플 수 (오버레이용)
MAX_DISPLAY="${MAX_DISPLAY:-100000}"

# 결과 저장 디렉토리
SAVE_DIR="${SAVE_DIR:-./trajectory_visualizations}"

###################################

python visualize_trajectories.py \
    --data_path "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --max_display "$MAX_DISPLAY" \
    --save_dir "$SAVE_DIR"
