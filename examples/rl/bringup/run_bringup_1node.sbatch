#!/usr/bin/env bash
# Single-node 8-GPU bring-up for PR #186's RL stack (Linear AMILABS-594 r2).
# Trains Qwen3.5-4B on the lmms-eval VizDoom agentic task long enough to see
# reward/loss curves and >=2 weight-sync cycles. Bring-up only: make it RUN.
#
# Runs FROZEN source: clones lmms-engine @ PIN_COMMIT and lmms-eval @
# LMMS_EVAL_COMMIT into a per-job work dir, so a later local edit cannot change a
# running job. lmms-eval is cloned INTO <engine>/src/lmms-eval -- exactly the path
# the engine's Ray runtime_env prepends to PYTHONPATH (train/rl/runner.py:1015,
# rl/model_server/ray.py:57), so both the driver and every Ray worker import the
# same pinned lmms_eval.
#
#SBATCH --job-name=fs-linear-594-2nd-vizdoom-bringup
#SBATCH --account=scaling
#SBATCH --partition=main
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#
# CONFIRM before submit (values this script cannot know):
#   --partition  FS GPU partition name. Override at submit if wrong:
#                babysit submit --cluster fs -- --partition=<name> examples/rl/bringup/run_bringup_1node.sbatch
#   Logs land as <job-name>_<jobid>.log in the submit dir (%j present, no clobber).
#   Redirect with: sbatch --output=<logs>/slurm/%x_%j.log ...
#   --cpus-per-task intentionally unset: --exclusive grants the whole node; the
#                srun step uses --cpu-bind=none so the 24 CPU rollout actors spread
#                across all cores (node core count is unknown at authoring time).

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- A2-supplied environment (REQUIRED; fail loud if unset) -----------------
# CONTAINER_IMAGE : enroot/pyxis image (path.sqsh or registry ref) with CUDA +
#                   system libs. ENV_PREFIX : python env prefix (its bin/ goes on
#                   PATH; its site-packages provide the deps). See ENV_PREFIX
#                   contract at the bottom of this file.
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the enroot image (A2 supplies)}"
ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX to the staged python env prefix (A2 supplies)}"

# --- Site paths (REQUIRED; consumed by the YAML via ${oc.env:...}) ----------
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the Qwen3.5-4B HF checkpoint dir}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an absolute run output dir}"
SYNC_DIR="${SYNC_DIR:?set SYNC_DIR to an absolute shared policy-sync dir}"
# Ray trial working_dirs make relative paths unsafe; the runner rejects a relative
# sync dir at runtime -- catch it (and a relative output dir) loudly at submit.
for _p in OUTPUT_DIR SYNC_DIR; do
  case "${!_p}" in
    /*) ;;
    *) echo "[sbatch] ${_p} must be an absolute path, got '${!_p}'" >&2; exit 1 ;;
  esac
done

# --- Pin-commit source (overridable) ----------------------------------------
PIN_REMOTE_URL="${PIN_REMOTE_URL:-https://github.com/Luodian/lmms-engine.git}"
PIN_BRANCH="${PIN_BRANCH:-brian/rl-bringup}"
PIN_COMMIT="${PIN_COMMIT:-3aea6f7819479bb64aa9f13b18e3445605cd0d57}"
LMMS_EVAL_URL="${LMMS_EVAL_URL:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
LMMS_EVAL_COMMIT="${LMMS_EVAL_COMMIT:-d45c1947dd0d15ead37627fab68defe94ce908eb}"

# --- Work + mounts (overridable) --------------------------------------------
WORK_ROOT="${WORK_ROOT:-$HOME/work}"
VAST_MOUNT="${VAST_MOUNT:-/mnt/vast}"
JOB_TAG="linear-594-bringup-${SLURM_JOB_ID:-manual}"
ENGINE_DIR="${WORK_ROOT}/${JOB_TAG}/lmms-engine"
LMMS_EVAL_DIR="${ENGINE_DIR}/src/lmms-eval"
CONFIG_YAML="${ENGINE_DIR}/examples/rl/bringup/vizdoom_qwen3_5_4b_1node.yaml"
export TMPDIR="${TMPDIR:-${WORK_ROOT}/${JOB_TAG}/tmp}"
# All site paths must be visible inside the container. Extend if any live
# outside these roots (e.g. ENV_PREFIX/MODEL_DIR on a different mount).
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${VAST_MOUNT}:${VAST_MOUNT},${WORK_ROOT}:${WORK_ROOT}}"

echo "[sbatch] job=${SLURM_JOB_ID:-manual} node=$(hostname) engine_dir=${ENGINE_DIR}"
mkdir -p "${WORK_ROOT}/${JOB_TAG}" "${OUTPUT_DIR}" "${SYNC_DIR}" "${TMPDIR}"

# --- Freeze source (host side; needs git + network on the batch node) -------
if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
  git clone "${PIN_REMOTE_URL}" "${ENGINE_DIR}"
fi
git -C "${ENGINE_DIR}" fetch origin "${PIN_BRANCH}" || true
git -C "${ENGINE_DIR}" checkout --force "${PIN_COMMIT}"
echo "[sbatch] lmms-engine @ $(git -C "${ENGINE_DIR}" rev-parse --short HEAD)"

# Clone the pinned lmms-eval into <engine>/src/lmms-eval (replaces the empty
# submodule dir). Full clone then checkout the SHA (SHA lives on a non-default
# branch; a shallow fetch-by-SHA is not guaranteed by the server).
rm -rf "${LMMS_EVAL_DIR}"
git clone "${LMMS_EVAL_URL}" "${LMMS_EVAL_DIR}"
git -C "${LMMS_EVAL_DIR}" checkout --force "${LMMS_EVAL_COMMIT}"
echo "[sbatch] lmms-eval  @ $(git -C "${LMMS_EVAL_DIR}" rev-parse --short HEAD)"

# vizdoom_train.jsonl ships in the frozen engine clone; override to point elsewhere.
DATA_PATH="${DATA_PATH:-${ENGINE_DIR}/examples/rl/data/vizdoom_train.jsonl}"
[[ -f "${DATA_PATH}" ]] || { echo "[sbatch] DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }

# Propagate to the container (srun inherits the environment; the YAML reads these).
export ENV_PREFIX ENGINE_DIR LMMS_EVAL_DIR CONFIG_YAML MODEL_DIR OUTPUT_DIR SYNC_DIR DATA_PATH

# Inner command runs INSIDE the container. Single-quoted so expansion happens
# there against the propagated env; activates ENV_PREFIX via PATH/PYTHONPATH,
# preflights the imports (fail loud), then execs the RL launcher.
read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ENGINE_DIR}/src:${LMMS_EVAL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
cd "${ENGINE_DIR}"
echo "[container] python=$(command -v python)"
python - <<'PREFLIGHT'
import importlib, importlib.util
mods = ["torch", "vllm", "ray", "transformers", "tensorboard", "vizdoom", "lmms_engine", "lmms_eval"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] ENV_PREFIX is missing required modules: {missing}")
import torch, vllm
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

# Forward preempt/term signals to the training step (checkpoint-on-preempt hook).
trap 'echo "[sbatch] USR1 -> srun ${SRUN_PID}"; kill -USR1 "${SRUN_PID}" 2>/dev/null || true' USR1
trap 'echo "[sbatch] TERM -> srun ${SRUN_PID}"; kill -TERM "${SRUN_PID}" 2>/dev/null || true' TERM

# Re-wait across signal interrupts until the step truly exits.
set +e
wait "${SRUN_PID}"; EXIT_CODE=$?
while kill -0 "${SRUN_PID}" 2>/dev/null; do
  wait "${SRUN_PID}"; EXIT_CODE=$?
done
set -e
echo "[sbatch] srun exited ${EXIT_CODE}"
exit "${EXIT_CODE}"

# ===========================================================================
# ENV_PREFIX contract (A2 supplies). A python env whose site-packages provide:
#   - lmms-engine runtime deps: torch, torchvision, transformers>=4.57.1,
#     accelerate, deepspeed, liger-kernel, qwen-vl-utils, opencv-python, pillow,
#     decord, hydra-core, omegaconf(>=2.1 for ${oc.env:}), loguru, pydantic,
#     wandb, safetensors, tensorboard.
#   - RL extra (pyproject [rl]): ray, openai, pytz, vizdoom.
#   - vllm (NOT in any pyproject extra): a build that (a) registers the Qwen3.5
#     / Qwen3-VL architecture, (b) exposes LLM.collective_rpc, sleep, wake_up,
#     reset_prefix_cache, and worker_extension_cls (all present since v0.11.0).
#   lmms_engine / lmms_eval themselves come from the PINNED CLONES via PYTHONPATH
#   above -- do NOT also install them in ENV_PREFIX (or pin them to the same SHAs).
#   The engine module must be importable inside the vLLM worker process too (it
#   is: Ray prepends ${ENGINE_DIR}/src to worker PYTHONPATH), so
#   worker_extension_cls "lmms_engine.rl.model_server.vllm_worker_ext.
#   WeightReloadWorkerExtension" resolves in-worker.
#
# ---------------------------------------------------------------------------
# POST-SUBMIT VERIFICATION (do all three before leaving the session):
#   1. State:      squeue -j "$JOB_ID" -h -o '%T'      # PENDING or RUNNING (empty = submit failed)
#   2. Heartbeat at +5 min (LOG = <job-name>_<jobid>.log, or `scontrol show job $JOB_ID | grep StdOut`):
#        grep -E "preflight] ok"                              "$LOG"  # env good, GPUs visible
#        grep -E "Started a local Ray instance|Ray vLLM actor placement|Started RL model servers" "$LOG"  # ray + vLLM actor pools up
#        grep -E "Built [0-9]+ rollout episode spec"          "$LOG"  # rollout specs built
#        grep -E "Resolved RL train batch|RL train shard balance step=" "$LOG"  # first optimizer step(s)
#   3. Weight sync (the bring-up success signal; expect >=2, initial v0 + step 5 ...):
#        grep -c "Reloaded policy weights for version"        "$LOG"
# ===========================================================================
