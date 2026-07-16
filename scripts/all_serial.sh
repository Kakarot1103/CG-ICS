#!/bin/bash
set -euo pipefail

# ================= Basic configuration =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SESSION_NAME="all_serial_ddp_$$"
CONDA_ENV_NAME="sam3"
PYTHON_ENTRY="main.py"

# ================= Parameter configuration =================
DATAPATH="data"
BSZ=1
NWORKER=0
NSHOT=1
IMG_SIZE=512
SAVE_IMAGE=0
SEED=0
NUM_TEST_SAMPLES=0
ALPHA=1
BETA=1
# Tree Search parameters
NUM_LOOPS=3
NUM_EXPAND_PER_NODE=5
EXPANSION_THRESHOLD=0.5
MAX_RESTARTS=3
EARLY_STOP_THRESHOLD=0.8

BENCHMARKS=(
  pascal
  coco
  fss
  lvis
)
BENCHMARKS_STR="${BENCHMARKS[*]}"

# ================= Multi-GPU settings =================
GPU_DEVICES=(0 1 2 3 4 5)
NPROC=${#GPU_DEVICES[@]}
MASTER_PORT=29501
CUDA_VISIBLE_DEVICES_STR=$(IFS=, ; echo "${GPU_DEVICES[*]}")

# ================= Launch tmux + serial DDP inference =================
tmux new-session -d -s "${SESSION_NAME}" bash -c "
  cd \"${SCRIPT_DIR}/..\" || exit 1
  source activate \"${CONDA_ENV_NAME}\" 2>/dev/null || conda activate \"${CONDA_ENV_NAME}\" 2>/dev/null || true
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_STR}
  echo \"[SERIAL] Using GPUs: \${CUDA_VISIBLE_DEVICES}\"
  echo \"[SERIAL] Will run datasets sequentially: ${BENCHMARKS_STR}\"

  BENCHMARKS=(${BENCHMARKS_STR})

  for BENCHMARK in \"\${BENCHMARKS[@]}\"; do
    case \"\${BENCHMARK}\" in
      pascal) FOLDS=(0 1 2 3) ;;
      coco) FOLDS=(0 1 2 3) ;;
      lvis) FOLDS=(0 1 2 3 4 5 6 7 8 9) ;;
      fss) FOLDS=(0) ;;
      pascal_part) FOLDS=(0 1 2 3) ;;
      *) echo \"[SERIAL] Unknown dataset \${BENCHMARK}, skipping\"; continue ;;
    esac

    SAVE_DIR=\"./results/release/\${BENCHMARK}\"
    FOLDS_STR=\"\${FOLDS[*]}\"

    echo \"==============================\"
    echo \"[SERIAL] Starting dataset \${BENCHMARK}, folds: \${FOLDS_STR}\"

    for FOLD in \"\${FOLDS[@]}\"; do
      echo \"[SERIAL] Launching dataset \${BENCHMARK} fold \${FOLD}\"
      torchrun \\
        --nproc_per_node=${NPROC} \\
        --master_port=${MASTER_PORT} \\
        ${PYTHON_ENTRY} \\
        --datapath \"${DATAPATH}\" \\
        --benchmark \"\${BENCHMARK}\" \\
        --bsz \"${BSZ}\" \\
        --nworker \"${NWORKER}\" \\
        --fold \"\${FOLD}\" \\
        --nshot \"${NSHOT}\" \\
        --img_size \"${IMG_SIZE}\" \\
        --num_test_samples \"${NUM_TEST_SAMPLES}\" \\
        --save_dir \"\${SAVE_DIR}\" \\
        --save_image \"${SAVE_IMAGE}\" \\
        --seed \"${SEED}\" \\
        --alpha \"${ALPHA}\" \\
        --beta \"${BETA}\" \\
        --num_loops \"${NUM_LOOPS}\" \\
        --num_expand_per_node \"${NUM_EXPAND_PER_NODE}\" \\
        --expansion_threshold \"${EXPANSION_THRESHOLD}\" \\
        --max_restarts \"${MAX_RESTARTS}\" \\
        --early_stop_threshold \"${EARLY_STOP_THRESHOLD}\"
      echo \"[SERIAL] Dataset \${BENCHMARK} fold \${FOLD} done\"
    done

    echo \"[SERIAL] Dataset \${BENCHMARK} finished all folds\"
    echo \"==============================\"
  done

  echo \"\"
  echo \"[SERIAL] All datasets finished, press Enter to close the tmux window...\"
  read -r
"

echo "=============================================="
echo "Tmux session '${SESSION_NAME}' created for sequential DDP inference"
echo "Run:  tmux attach -t ${SESSION_NAME}  to view progress"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES_STR}"
echo "nproc_per_node: ${NPROC}"
echo "Dataset order: ${BENCHMARKS_STR}"
echo "=============================================="
