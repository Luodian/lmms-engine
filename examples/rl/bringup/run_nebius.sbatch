#!/usr/bin/env bash
# AMILABS-594 r2 -- Qwen3.5-4B + VizDoom RL single-node 8-GPU bring-up on NEBIUS.
# Adapted from examples/rl/bringup/run_bringup_1node.sbatch (A1) for Nebius:
#   - partition=scaling; image=vllm-openai-v0.23.0.sqsh; mounts=/home:/home
#   - all paths under /home/brian (Nebius has no /mnt/vast)
#   - A2 activation contract (PYTHONNOUSERSITE + venv activate + HF_HOME)
#   - source pinned to fork commit e24499d: 4d03f4c's WeightReloadWorkerExtension
#     conflicts with vLLM 0.23's NATIVE Worker.reload_weights; e24499d guards the
#     extension injection (skip when the worker has a native reload_weights).
#
# Runs FROZEN source: clones lmms-engine @ PIN_COMMIT and lmms-eval @
# LMMS_EVAL_COMMIT into a per-job work dir. lmms-eval is cloned INTO
# <engine>/src/lmms-eval -- the path the engine's Ray runtime_env prepends to
# worker PYTHONPATH -- so driver and every Ray worker import the same pinned code.
#
#SBATCH --job-name=nb-linear-594-2nd-vizdoom-bringup
#SBATCH --account=scaling
#SBATCH --partition=scaling
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/brian/rlbringup/logs/%x_%j.log
#SBATCH --error=/home/brian/rlbringup/logs/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Site config (Nebius; overridable) --------------------------------------
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/enroot/vllm-openai-v0.23.0.sqsh}"
ENV_PREFIX="${ENV_PREFIX:-/home/brian/envs/lmms-engine-rl}"
MODEL_DIR="${MODEL_DIR:-/home/brian/models/Qwen3.5-4B}"
RUN_ROOT="${RUN_ROOT:-/home/brian/rlbringup/runs/${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/output}"
SYNC_DIR="${SYNC_DIR:-${RUN_ROOT}/sync}"

# --- Pin-commit source (fork; reload guard fix included) ---------------------
PIN_REMOTE_URL="${PIN_REMOTE_URL:-https://github.com/Luodian/lmms-engine.git}"
PIN_BRANCH="${PIN_BRANCH:-brian/rl-bringup}"
PIN_COMMIT="${PIN_COMMIT:-e24499da37230d623f3d7bfc2d64f6eb0fa29d5a}"
LMMS_EVAL_URL="${LMMS_EVAL_URL:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
LMMS_EVAL_COMMIT="${LMMS_EVAL_COMMIT:-d45c1947dd0d15ead37627fab68defe94ce908eb}"

# --- Work + mounts ----------------------------------------------------------
WORK_ROOT="${WORK_ROOT:-/home/brian/work}"
JOB_TAG="linear-594-2nd-bringup-${SLURM_JOB_ID:-manual}"
ENGINE_DIR="${WORK_ROOT}/${JOB_TAG}/lmms-engine"
LMMS_EVAL_DIR="${ENGINE_DIR}/src/lmms-eval"
CONFIG_YAML="${ENGINE_DIR}/examples/rl/bringup/vizdoom_qwen3_5_4b_1node.yaml"
export TMPDIR="${TMPDIR:-${WORK_ROOT}/${JOB_TAG}/tmp}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home:/home}"

for _p in OUTPUT_DIR SYNC_DIR; do
  case "${!_p}" in
    /*) ;;
    *) echo "[sbatch] ${_p} must be absolute, got '${!_p}'" >&2; exit 1 ;;
  esac
done

echo "[sbatch] job=${SLURM_JOB_ID:-manual} node=$(hostname) engine_dir=${ENGINE_DIR}"
echo "[sbatch] pin: ${PIN_REMOTE_URL}@${PIN_COMMIT}  lmms-eval: ${LMMS_EVAL_COMMIT}"
mkdir -p "${WORK_ROOT}/${JOB_TAG}" "${OUTPUT_DIR}" "${SYNC_DIR}" "${TMPDIR}"

# --- Freeze source (host side) ----------------------------------------------
if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
  git clone "${PIN_REMOTE_URL}" "${ENGINE_DIR}"
fi
git -C "${ENGINE_DIR}" fetch origin "${PIN_BRANCH}" || true
git -C "${ENGINE_DIR}" checkout --force "${PIN_COMMIT}"
echo "[sbatch] lmms-engine @ $(git -C "${ENGINE_DIR}" rev-parse --short HEAD)"

rm -rf "${LMMS_EVAL_DIR}"
git clone "${LMMS_EVAL_URL}" "${LMMS_EVAL_DIR}"
git -C "${LMMS_EVAL_DIR}" checkout --force "${LMMS_EVAL_COMMIT}"
echo "[sbatch] lmms-eval  @ $(git -C "${LMMS_EVAL_DIR}" rev-parse --short HEAD)"

DATA_PATH="${DATA_PATH:-${ENGINE_DIR}/examples/rl/data/vizdoom_train.jsonl}"
[[ -f "${DATA_PATH}" ]] || { echo "[sbatch] DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }

export ENV_PREFIX ENGINE_DIR LMMS_EVAL_DIR CONFIG_YAML MODEL_DIR OUTPUT_DIR SYNC_DIR DATA_PATH

# Inner command runs INSIDE the container. Single-quoted heredoc: expansion
# happens there against the propagated env.
read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
# A2 activation contract (verified): user-site leaks without PYTHONNOUSERSITE.
export PYTHONNOUSERSITE=1
source "${ENV_PREFIX}/bin/activate"
export HF_HOME=/home/brian/.cache/huggingface
# Frozen clones must win over the editable installs. lmms_engine's editable .pth
# is a plain path (PYTHONPATH prepend wins); lmms_eval's PEP660 finder is APPENDED
# to sys.meta_path (fallback -> the default PathFinder resolves PYTHONPATH first).
export PYTHONPATH="${ENGINE_DIR}/src:${LMMS_EVAL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
cd "${ENGINE_DIR}"
echo "[container] python=$(command -v python)"
python - <<'PREFLIGHT'
import importlib.util
import lmms_engine, lmms_eval, torch, vllm
# Gate-0 evidence: the per-job frozen clone must win over the editable install.
print(f"[preflight] lmms_engine -> {lmms_engine.__file__}")
print(f"[preflight] lmms_eval   -> {lmms_eval.__file__}")
mods = ["torch", "vllm", "ray", "transformers", "tensorboard", "vizdoom", "lmms_engine", "lmms_eval", "openai", "safetensors"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] ENV missing modules: {missing}")
print(f"[preflight] ok torch={torch.__version__} vllm={vllm.__version__} cuda={torch.cuda.device_count()} gpus")
PREFLIGHT
exec python -m lmms_engine.launch.rl \
  hydra.output_subdir=null \
  hydra/job_logging=stdout \
  config_yaml="${CONFIG_YAML}" \
  --nnodes 1 --gpus-per-node 8
INNER_EOF

srun \
  --unbuffered \
  --gpus-per-node=8 \
  --cpu-bind=none \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${ENGINE_DIR}" \
  bash -lc "${INNER}" &
SRUN_PID=$!

trap 'echo "[sbatch] USR1 -> srun ${SRUN_PID}"; kill -USR1 "${SRUN_PID}" 2>/dev/null || true' USR1
trap 'echo "[sbatch] TERM -> srun ${SRUN_PID}"; kill -TERM "${SRUN_PID}" 2>/dev/null || true' TERM

set +e
wait "${SRUN_PID}"; EXIT_CODE=$?
while kill -0 "${SRUN_PID}" 2>/dev/null; do
  wait "${SRUN_PID}"; EXIT_CODE=$?
done
set -e
echo "[sbatch] srun exited ${EXIT_CODE}"
exit "${EXIT_CODE}"

