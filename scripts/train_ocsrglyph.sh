#!/usr/bin/env bash
# Exact OCSRGlyph training launcher.
#
# Reproduces OCSRGlyph behind the published USPTO-5704 headline numbers
# (canonical exact-match 5360/5704 = 93.97%, graph exact match 5496/5704 =
# 96.35%, valid-SMILES 5692/5704 = 99.79%).
#
# Recipe capture:  configs/ocsr/ocsrglyph-stereo-enriched.yaml
# Trainer:         python -m glyph.ocsr.train  (argparse; DDP via torchrun)
# Model/optim knobs come from glyph.ocsr.baseline_config.BaselineRecipe and
# match the OCSRGlyph training run: Swin-B/384 encoder, 6-layer transformer decoder (embed 256, 8 heads,
# ff 1024), vocab 101, ImageNet-pretrained backbone, effective batch 256
# (32/GPU x 8), max_steps 100000, cosine LR enc/dec 4e-4, warmup 0.02,
# label_smoothing 0.1, seed 42.
#
# Data mix (dataset_size = 2,209,724): PubChem-1M + USPTO-680K + stereo_200k_white x2
# + adjacent_ring x2. The x2 repeats are ConcatDataset weighting (not
# physical duplication); the weighted sampler is OFF. The stereo_200k_white and
# adjacent_ring enrichment CSVs are shipped FROZEN in the
# EdisonScientific/glyph-datasets HF dataset repo (no committed generator);
# PubChem-1M + USPTO-680K are public (HF CYF200127/MolNexTR).
#
# Usage (auto-detects the GPUs present — zero env vars needed):
#   export GLYPH_DATA_ROOT=/path/to/data
#   export WANDB_API_KEY=...            # required only for ONLINE logging;
#                                       # set WANDB_MODE=offline (no key needed)
#                                       # for a network-free run.
#   scripts/train_ocsrglyph.sh          # NUM_GPUS defaults to `nvidia-smi -L` count (floor 1)
#
# Pin the topology explicitly (e.g. the published 8-GPU / 256-effective-batch
# run, 32/GPU x 8):
#   NUM_GPUS=8 scripts/train_ocsrglyph.sh
#
# Effective batch = NUM_GPUS x BATCH_SIZE_PER_GPU. BATCH_SIZE_PER_GPU (recipe 32)
# targets ~80 GB cards; on a smaller-VRAM GPU (e.g. 32 GB) lower it to avoid OOM:
#   BATCH_SIZE_PER_GPU=8 scripts/train_ocsrglyph.sh
# A single-process alternative (no torchrun) is equivalent for 1 GPU:
#   python -m glyph.ocsr.train --encoder_name swin_base_patch4_window12_384 ...
# (the launcher below builds exactly that argv; run it with NUM_GPUS=1).
#
# All data paths default under ${GLYPH_DATA_ROOT} and are override-able via the
# environment variables shown below.

set -euo pipefail

# --- Data root ---------------------------------------------------------------
GLYPH_DATA_ROOT="${GLYPH_DATA_ROOT:-./data}"
if [[ "$GLYPH_DATA_ROOT" != /* ]]; then
  GLYPH_DATA_ROOT="$PWD/${GLYPH_DATA_ROOT#./}"
fi
export GLYPH_DATA_ROOT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Launch topology ---------------------------------------------------------
# Auto-detect the GPUs present (floor 1) so a bare launch uses every available
# GPU with zero env vars; override NUM_GPUS to pin the topology (e.g. NUM_GPUS=8
# to reproduce the published 8-GPU run). Effective batch = NUM_GPUS x
# BATCH_SIZE_PER_GPU.
_detected_gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_GPUS="${NUM_GPUS:-$_detected_gpus}"
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
# BATCH_SIZE_PER_GPU is the published recipe value (32), tuned for ~80 GB cards
# (A100-80GB). On smaller-VRAM GPUs (e.g. a 32 GB card) 32 can OOM — lower it
# (e.g. BATCH_SIZE_PER_GPU=8). This is a hardware knob, deliberately NOT
# auto-sized; effective batch = NUM_GPUS x BATCH_SIZE_PER_GPU.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"   # 32 x 8 = 256 (published effective batch)
MAX_STEPS="${MAX_STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 2000))}"
RUN_NAME="${RUN_NAME:-ocsrglyph-stereo-enriched}"
GIT_COMMIT="${GIT_COMMIT:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# --- Output ------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${GLYPH_DATA_ROOT}/checkpoints/ocsr/${RUN_NAME}}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-10000}"  # + final step100000.pth
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-1000}"               # 64-example in-training proxy

# --- Data-mix paths (relative to ${GLYPH_DATA_ROOT}) -----------------------
# The trainer resolves relative paths under GLYPH_DATA_ROOT automatically.
PUBCHEM_CSV="${PUBCHEM_CSV:-datasets/ocsr/pubchem_1m/pubchem_1m.csv}"
USPTO_CSV="${USPTO_CSV:-datasets/ocsr/uspto_680k/train.csv}"
USPTO_IMAGE_ROOT="${USPTO_IMAGE_ROOT:-datasets/ocsr/uspto_680k/images}"
USPTO_PATH_STRIP_PREFIX="${USPTO_PATH_STRIP_PREFIX:-uspto_mol/}"
STEREO_CSV="${STEREO_CSV:-datasets/ocsr/stereo_200k_white/train.csv}"
# The white-crop CSV's `file_path` already carries the `train/` split segment,
# so the image root is the dataset dir itself (resolves to
# stereo_200k_white/train/<name>.png). This is exactly what
# `glyph data prepare --ocsr` writes via _prepare_stereo_200k_white.
STEREO_IMAGE_ROOT="${STEREO_IMAGE_ROOT:-datasets/ocsr/stereo_200k_white}"
STEREO_REPEAT="${STEREO_REPEAT:-2}"
# The adjacent_ring enrichment (this work's mined subset) carries a bare SMILES
# column and renders on the fly, so it auto-downloads from
# EdisonScientific/glyph-datasets via the trainer's `hub:` resolver — no manual
# huggingface-cli step. Override with a local CSV path if you have one.
ADJACENT_RING_CSV="${ADJACENT_RING_CSV:-hub:adjacent_ring}"
ADJACENT_RING_REPEAT="${ADJACENT_RING_REPEAT:-2}"

mkdir -p "$OUTPUT_DIR"

# --- Build the trainer invocation --------------------------------------------
# Recipe knobs (encoder, decoder dims, vocab, weight decay, warmup ratio, grad
# clip, seed) are supplied by BaselineRecipe; the flags below select the recipe,
# the exact data mix, and the optimization overrides that define OCSRGlyph.
cmd=(
  torchrun
  --standalone
  --nproc_per_node="$NUM_GPUS"
  --master_port "$MASTER_PORT"
  -m glyph.ocsr.train
  # --- model recipe ---
  --encoder_name swin_base_patch4_window12_384
  --input_size 384
  --backbone_pretrained
  # --- optimization ---
  --batch_size "$BATCH_SIZE_PER_GPU"
  --max_steps "$MAX_STEPS"
  --num_workers "$NUM_WORKERS"
  --encoder_lr 4e-4
  --decoder_lr 4e-4
  --label_smoothing 0.1
  --cosine_lr
  --bf16_amp
  # --- data mix (dataset_size = 2,209,724) ---
  --pubchem_csv "$PUBCHEM_CSV"
  --uspto_csv "$USPTO_CSV"
  --uspto_image_root "$USPTO_IMAGE_ROOT"
  --uspto_path_strip_prefix "$USPTO_PATH_STRIP_PREFIX"
  --stereo_csv "$STEREO_CSV"
  --stereo_image_root "$STEREO_IMAGE_ROOT"
  --stereo_repeat "$STEREO_REPEAT"
  --extra_pubchem_csv "$ADJACENT_RING_CSV"
  --extra_pubchem_repeat "$ADJACENT_RING_REPEAT"
  # --- checkpointing / eval / bookkeeping ---
  --output_dir "$OUTPUT_DIR"
  --write_checkpoint
  --checkpoint_every_steps "$CHECKPOINT_EVERY_STEPS"
  --eval_every_steps "$EVAL_EVERY_STEPS"
  --run_name "$RUN_NAME"
  --git_commit "$GIT_COMMIT"
)

if [[ "${WANDB_MODE:-}" == "offline" || "${WANDB_MODE:-}" == "disabled" ]]; then
  # Network-free logging: the OCSR trainer needs no WANDB_API_KEY in this mode.
  # Clear any stub key so W&B's client-side key-format check is not tripped.
  unset WANDB_API_KEY
fi

printf 'train_ocsrglyph='
printf '%q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
