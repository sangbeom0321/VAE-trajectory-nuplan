#!/bin/bash
###################################
# Planning Vocabulary 생성 스크립트
###################################

# 데이터 경로
DATA_PATH="/home/daniel/99_dataset/01_nuplan/dataset/vae/trajectories_8s.npz"

# 어휘집 크기 (클러스터 수)
K=100

# 샘플링할 궤적 수
NUM_SAMPLES=1000

# 랜덤 시드
RANDOM_SEED=42

# 결과 저장 디렉토리
SAVE_DIR="./planning_vocabulary"

###################################

python create_planning_vocabulary.py \
    --data_path $DATA_PATH \
    --k $K \
    --num_samples $NUM_SAMPLES \
    --random_seed $RANDOM_SEED \
    --save_dir $SAVE_DIR
