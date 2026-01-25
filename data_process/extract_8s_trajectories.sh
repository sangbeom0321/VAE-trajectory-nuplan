###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="$HOME/99_dataset/01_nuplan/dataset/nuplan-v1.1/splits/trainval" # nuplan training data path
NUPLAN_MAP_PATH="$HOME/99_dataset/01_nuplan/dataset/maps" # nuplan map path

# 8초 경로 추출 결과 저장 경로
TRAJECTORY_SAVE_PATH="$HOME/99_dataset/01_nuplan/dataset/exp2/trajectories_8s.npz"

# Log names JSON file path (상대 경로 또는 절대 경로 가능)
# 기본값: 현재 스크립트 디렉토리의 nuplan_train.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_NAMES_FILE="${LOG_NAMES_FILE:-$SCRIPT_DIR/nuplan_train.json}" # 기본값 사용 또는 환경변수로 오버라이드 가능

# 추출할 샘플 수
NUM_SAMPLES=100000

# 배치 저장 간격 (메모리 절약)
BATCH_SAVE=10000

# 사용할 db 파일 개수 (랜덤 선택)
NUM_DB_FILES=50

# 랜덤 시드 (재현 가능성을 위해)
RANDOM_SEED=42
###################################

python extract_8s_trajectories.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAJECTORY_SAVE_PATH \
--log_names_file $LOG_NAMES_FILE \
--num_samples $NUM_SAMPLES \
--num_db_files $NUM_DB_FILES \
--random_seed $RANDOM_SEED
