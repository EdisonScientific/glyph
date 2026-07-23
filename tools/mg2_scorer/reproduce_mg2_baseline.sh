#!/usr/bin/env bash
# Reproduce the released MarkushGrapher-2 Table 3 baseline locally.
#
# This launcher pins and prepares DS4SD's published two-stage pipeline, runs
# ChemicalOCR and `python -m markushgrapher.eval` unchanged on all three public
# benchmarks, dual-grades with score_benchmark.sh, and compares the fresh MG2
# JSONL files row-for-row with the release's cached cross-checks.
#
# Default (full run, RTX 5090 / CUDA 13):
#   tools/mg2_scorer/reproduce_mg2_baseline.sh
#
# Reuse completed local inference and only grade/report:
#   tools/mg2_scorer/reproduce_mg2_baseline.sh --skip-setup --skip-inference \
#       --work-dir /path/to/existing/run
#
# Model weights, environments, source checkouts, datasets, and run outputs stay
# under --work-dir (git-ignored); none are added to the repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELPER="${SCRIPT_DIR}/mg2_reproduction.py"
CACHED_DIR="${SCRIPT_DIR}/cached_predictions"

MG_COMMIT="35c71a0b4423983920b4703f52f46c5bca9ae631"
GENERATOR_COMMIT="17580c20d266886ef9ae22a50aab51444e0898b2"
TRANSFORMERS_COMMIT="867ac6e128e2e71ecc12a9c1b65e48e451245a67"
MOLSCRIBE_COMMIT="b15c96e10da5e2ec4444a877d1031d8d10190c4d"
DATASET_REVISION="e90868cdd3b62f1fe175d9ede9e0c65485c4a96c"
MG_MODEL_REVISION="2e1091d5acad0046910971439cedced0df659367"
OCR_MODEL_REVISION="650c1c8e6ea7699ca1f1049d30490b89fab538f2"
MOLSCRIBE_MODEL_REVISION="a0189776b7415b82795c7ee81eed311bf5c8724b"
MOLSCRIBE_MODEL_FILE="swin_base_char_aux_1m680k.pth"

WORK_DIR="${MG2_WORK_DIR:-${SCRIPT_DIR}/local-mg2-reproduction}"
SKIP_SETUP=0
SKIP_INFERENCE=0
BENCHMARKS=(ip5m m2s uspto_markush)

usage() {
  cat <<EOF
reproduce_mg2_baseline.sh - local MarkushGrapher-2 Table 3 reproduction

Usage:
  tools/mg2_scorer/reproduce_mg2_baseline.sh [options]

Options:
  --work-dir PATH       local checkout/env/model/output root (default: ${WORK_DIR})
  --benchmark NAME      run one benchmark; repeatable (ip5m, m2s, uspto_markush)
  --skip-setup          reuse an already prepared MG2 checkout and environments
  --skip-inference      reuse predictions_<N>.jsonl under --work-dir/runs/<name>/mg2/evaluation
  -h, --help            show this help

The default runs all 1,055 images. It requires Linux, uv, git, a CUDA 13-capable
NVIDIA setup, sufficient disk space, and a 32 GB GPU. No remote scheduler is used.
EOF
}

selected=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --benchmark) selected+=("$2"); shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-inference) SKIP_INFERENCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ${#selected[@]} -gt 0 ]]; then
  BENCHMARKS=("${selected[@]}")
fi

for command in uv git nvidia-smi; do
  command -v "$command" >/dev/null || { echo "ERROR: required command not found: $command" >&2; exit 3; }
done

WORK_DIR="$(mkdir -p "$WORK_DIR" && cd "$WORK_DIR" && pwd)"
MG_DIR="${WORK_DIR}/MarkushGrapher"
RUN_ROOT="${WORK_DIR}/runs"
CACHE_DIR="${WORK_DIR}/cache"
mkdir -p "$RUN_ROOT" "$CACHE_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_DIR}/uv}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export UV_LINK_MODE=copy

verify_commit() {
  local repo="$1" expected="$2" name="$3" actual
  actual="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  [[ "$actual" == "$expected" ]] || {
    echo "ERROR: ${name} is at ${actual:-<missing>}, expected ${expected}." >&2
    exit 4
  }
}

clone_at() {
  local url="$1" dest="$2" commit="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --quiet "$url" "$dest"
  fi
  git -C "$dest" fetch --quiet origin "$commit" || git -C "$dest" fetch --quiet origin
  git -C "$dest" checkout --quiet "$commit"
}

setup_pipeline() {
  echo "[setup] fetching pinned public source checkouts"
  clone_at https://github.com/DS4SD/MarkushGrapher.git "$MG_DIR" "$MG_COMMIT"
  clone_at https://github.com/lucas-morin/transformers.git \
    "$MG_DIR/external/transformers" "$TRANSFORMERS_COMMIT"
  clone_at https://github.com/lucas-morin/MolScribe.git \
    "$MG_DIR/external/MolScribe" "$MOLSCRIBE_COMMIT"
  verify_commit "$MG_DIR" "$MG_COMMIT" MarkushGrapher
  verify_commit "$MG_DIR/external/transformers" "$TRANSFORMERS_COMMIT" transformers
  verify_commit "$MG_DIR/external/MolScribe" "$MOLSCRIBE_COMMIT" MolScribe

  echo "[setup] creating isolated Python 3.10 environments"
  [[ -x "$MG_DIR/chemicalocr-env/bin/python" ]] || uv venv --python 3.10 "$MG_DIR/chemicalocr-env"
  [[ -x "$MG_DIR/markushgrapher-env/bin/python" ]] || uv venv --python 3.10 "$MG_DIR/markushgrapher-env"
  local ocr_py="$MG_DIR/chemicalocr-env/bin/python"
  local mg_py="$MG_DIR/markushgrapher-env/bin/python"

  # Blackwell requires CUDA 13 wheels. These override MG2's pre-Blackwell torch
  # pins while leaving its published inference/evaluation implementation intact.
  uv pip install --python "$ocr_py" --editable "$MG_DIR" --no-deps
  uv pip install --python "$ocr_py" --torch-backend cu130 \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' \
    'vllm==0.25.1' 'transformers==5.14.1' 'tokenizers==0.22.2' \
    'datasets==5.0.0' 'pyarrow==21.0.0' pillow tqdm \
    huggingface-hub accelerate aiohttp s3fs 'fsspec>=2024.6.1' 'rdkit==2026.3.4' \
    'opencv-python-headless==5.0.0.93'
  # vLLM's current OpenCV metadata selects NumPy 2, but ChemicalOCR's image
  # stack is validated with NumPy 1.26. Force that runtime after resolution.
  uv pip install --python "$ocr_py" --no-deps 'numpy==1.26.4'
  uv pip install --python "$ocr_py" --no-deps \
    "git+https://github.com/DS4SD/MarkushGenerator.git@${GENERATOR_COMMIT}"

  uv pip install --python "$mg_py" --editable "$MG_DIR" --no-deps
  uv pip install --python "$mg_py" --prerelease allow \
    --index https://download.pytorch.org/whl/nightly/cu130 \
    --index-strategy unsafe-best-match \
    'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0' 'numpy==1.24.4' \
    'datasets==2.14.6' 'pyarrow==12.0.1' 'accelerate==0.21.0' \
    'tokenizers==0.13.3' 'huggingface-hub==0.19.4' 'fsspec==2023.10.0' \
    's3fs==2023.10.0' 'OpenNMT-py==2.2.0' 'pyonmttok==1.37.1' \
    'rdkit==2024.3.5' 'opencv-python-headless==4.11.0.86' \
    'albumentations==1.4.2' 'timm==0.4.12' SmilesPE evaluate matplotlib pillow \
    pytorch-lightning sentencepiece scikit-learn wandb pytesseract levenshtein \
    pyyaml aiohttp nltk rouge-score svgpathtools lxml cairosvg ipykernel \
    ipywidgets protobuf
  uv pip install --python "$mg_py" --no-deps \
    "git+https://github.com/DS4SD/MarkushGenerator.git@${GENERATOR_COMMIT}" \
    --editable "$MG_DIR/external/MolScribe" \
    --editable "$MG_DIR/external/transformers"

  # Apply plumbing-only compatibility patches to the local pinned checkout:
  # torch's removed private iterator import and an HF revision option for the
  # otherwise unchanged published dataset-preparation routine.
  "$mg_py" "$HELPER" patch-upstream --checkout "$MG_DIR"

  echo "[setup] downloading pinned model snapshots (weights stay local)"
  "$mg_py" - "$MG_DIR/models/markushgrapher-2" <<PY
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    "docling-project/MarkushGrapher-2",
    revision="$MG_MODEL_REVISION",
    local_dir=sys.argv[1],
    local_dir_use_symlinks=False,
)
PY
  "$ocr_py" - "$MG_DIR/models/chemicalocr" <<PY
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    "docling-project/ChemicalOCR",
    revision="$OCR_MODEL_REVISION",
    local_dir=sys.argv[1],
    ignore_patterns=["checkpoint-*/*", "optimizer.pt", "scheduler.pt"],
)
PY
  mkdir -p "$MG_DIR/external/MolScribe/ckpts"
  "$mg_py" - "$MG_DIR/external/MolScribe/ckpts/$MOLSCRIBE_MODEL_FILE" <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil, sys
source = hf_hub_download(
    "yujieq/MolScribe",
    "$MOLSCRIBE_MODEL_FILE",
    revision="$MOLSCRIBE_MODEL_REVISION",
)
target = Path(sys.argv[1])
if not target.exists():
    shutil.copyfile(source, target)
PY
}

if [[ "$SKIP_SETUP" -eq 0 ]]; then
  setup_pipeline
fi

OCR_PY="$MG_DIR/chemicalocr-env/bin/python"
MG_PY="$MG_DIR/markushgrapher-env/bin/python"
[[ -x "$OCR_PY" && -x "$MG_PY" ]] || {
  echo "ERROR: environments missing under $MG_DIR; run without --skip-setup." >&2
  exit 5
}
verify_commit "$MG_DIR" "$MG_COMMIT" MarkushGrapher
verify_commit "$MG_DIR/external/transformers" "$TRANSFORMERS_COMMIT" transformers
verify_commit "$MG_DIR/external/MolScribe" "$MOLSCRIBE_COMMIT" MolScribe

read -r gpu_name compute_capability <<<"$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | sed 's/, /|/' | tr ' ' '_' | tr '|' ' ')"
echo "[hardware] ${gpu_name//_/ } (compute capability ${compute_capability})"
"$OCR_PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "ChemicalOCR environment cannot see CUDA"
print(f"[hardware] ChemicalOCR torch={torch.__version__}, CUDA={torch.version.cuda}, capability={torch.cuda.get_device_capability()}")
PY
"$MG_PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "MarkushGrapher environment cannot see CUDA"
print(f"[hardware] MG2 torch={torch.__version__}, CUDA={torch.version.cuda}, capability={torch.cuda.get_device_capability()}")
PY

bench_config() {
  case "$1" in
    ip5m) echo 'ip5-markush 878 mg2_ip5m_predictions_878.jsonl' ;;
    m2s) echo 'm2s 103 mg2_m2s_predictions_103.jsonl' ;;
    uspto_markush) echo 'uspto-markush 74 mg2_uspto-markush_predictions_74.jsonl' ;;
    *) echo "ERROR: unknown benchmark: $1" >&2; exit 2 ;;
  esac
}

run_inference() {
  local benchmark="$1" config="$2" count="$3" run="$RUN_ROOT/$benchmark"
  local raw="$run/raw" modern="$run/ocr-modern" legacy="$run/ocr-legacy" mg="$run/mg2"
  mkdir -p "$run" "$mg"

  echo "[${benchmark}] preparing ${count}-row pinned dataset"
  PYTHONPATH="$MG_DIR" "$OCR_PY" "$MG_DIR/scripts/dataset/prepare_hf_dataset.py" \
    --dataset docling-project/MarkushGrapher-2-Datasets --config "$config" \
    --revision "$DATASET_REVISION" --split test --output_dir "$raw" --target_split test
  echo "[${benchmark}] ChemicalOCR"
  PYTHONPATH="$MG_DIR" "$OCR_PY" "$MG_DIR/scripts/ocr/apply_ocr.py" \
    --dataset_dir "$raw" --model_path "$MG_DIR/models/chemicalocr" \
    --output_dir "$modern" --split test
  "$OCR_PY" "$HELPER" convert-dataset --input "$modern" --output "$legacy"

  cat > "$mg/datasets_predict.yaml" <<EOF
mdu_dataset:
  apply_ocr: false
  augment_test: false
  class_name: MDU_Dataset
  condense_labels: true
  dataset_path: $legacy
  encode_definition_group: false
  encode_index: true
  encode_position: false
  grounded_smiles: false
  load_from_cache: false
  mask_ratio: 1
  module_name: mdu_dataset
  name: mdu
  normalize_bbox: true
  splits: [test]
  stream: false
  task: Question Answering
  training_dataset_name: mdu
  type: supervised
  udop_tokenizer_only: false
EOF
  cat > "$mg/predict.yaml" <<EOF
{
 model_name_or_path: $MG_DIR/models/markushgrapher-2,
 tokenizer_path: auto,
 output_dir: $mg/evaluation,
 datasets_config: $mg/datasets_predict.yaml,
 max_seq_length: 512,
 image_size: 512,
 max_seq_length_decoder: 512,
 model_type: UdopUnimodel,
 architecture_variant: me-lf-stack-1,
 beam_search: True,
 normalize_bbox: True,
 use_pretrained_molscribe: True,
 freeze_ocsr_encoder: True,
 freeze_vtl_decoder: False,
 freeze_mlp_projector: False,
 do_train: False,
 do_eval: False,
 do_predict: True,
 dataloader_num_workers: 1,
 log_level: "INFO",
 viz_out_dir: $mg/visualization,
 prediction_loss_only: True,
 label_names: ["labels"],
 unit: word,
 apply_ocr: False,
 max_eval_samples: $count,
}
EOF
  echo "[${benchmark}] MG2 beam-search evaluation"
  # The two pinned checkpoints are trusted public pickle artifacts. New torch
  # defaults to restricted weight loading, while the pinned MG2/MolScribe code
  # predates that change and calls torch.load without an explicit policy.
  (cd "$MG_DIR" && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$MG_DIR" \
    "$MG_PY" -m markushgrapher.eval "$mg/predict.yaml")
}

for benchmark in "${BENCHMARKS[@]}"; do
  read -r config count cached_name <<<"$(bench_config "$benchmark")"
  run="$RUN_ROOT/$benchmark"
  predictions="$run/mg2/evaluation/predictions_${count}.jsonl"
  if [[ "$SKIP_INFERENCE" -eq 0 ]]; then
    run_inference "$benchmark" "$config" "$count"
  fi
  [[ -f "$predictions" ]] || { echo "ERROR: missing fresh predictions: $predictions" >&2; exit 6; }

  "$MG_PY" "$HELPER" adapt-for-scoring --input "$predictions" --output "$run/scorer_input.jsonl"
  echo "[${benchmark}] dual grading"
  GLYPH_MARKUSH_EXTRA="${GLYPH_MARKUSH_EXTRA:-markush}" \
    "$SCRIPT_DIR/score_benchmark.sh" --benchmark "$benchmark" \
    --input-json "$run/scorer_input.jsonl" --output-json "$run/scores.json"
  "$MG_PY" "$HELPER" compare --fresh "$predictions" \
    --cached "$CACHED_DIR/$cached_name" --output "$run/comparison.json"
done

if [[ ${#BENCHMARKS[@]} -eq 3 ]]; then
  "$MG_PY" "$HELPER" summarize --run-root "$RUN_ROOT" --output "$RUN_ROOT/summary.json"
else
  echo "[done] selected benchmark outputs are under $RUN_ROOT"
fi
