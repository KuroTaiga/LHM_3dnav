#!/usr/bin/env bash
# render_gs_from_img_smplx_muti.sh
# Batch rendering that keeps one LHM load for multiple images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -------- Defaults (override via CLI env or flags below) --------
MODEL_NAME="${MODEL_NAME:-LHM-1B-HF}"
IMAGE_DIR="${IMAGE_DIR:-../../inputs/images/SHHQ-1.0_samples}"
MOTION_SEQS_DIR="${MOTION_SEQS_DIR:-../../inputs/motion_seq_cleaned/walk_fbx}"
OUT_ROOT="${OUT_ROOT:-exps/SHHQ_exp_new}"
NAS_ROOT="${NAS_ROOT:-/mnt/nas/jiankundong/SHHQ_walk_fbx_new}"   # <- verify path
EXPORT_GS="${EXPORT_GS:-true}"
RENDER_FPS="${RENDER_FPS:-30}"
MOTION_READ_FPS="${MOTION_READ_FPS:-30}"
VIS_MOTION="${VIS_MOTION:-true}"
MOTION_IMG_NEED_MASK="${MOTION_IMG_NEED_MASK:-true}"
MOTION_IMG_DIR="${MOTION_IMG_DIR:-None}"  # e.g., set to a folder or keep 'None'
ZERO_HANDS="${ZERO_HANDS:-false}"
ASYNC_MESH_SYNC="${ASYNC_MESH_SYNC:-true}"  # run rsync in background to avoid pausing renders
# ---------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $(basename "$0") [--model NAME] [--images DIR] [--motions DIR] [--out DIR] [--nas DIR]
                        [--export-gs true|false] [--zero-hands]
                        [--render-fps N] [--motion-read-fps N]
                        [--vis-motion true|false] [--mask true|false] [--motion-img-dir PATH|None]

Runs a single LHM process over all images under --images to reuse model load.
EOF
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_NAME="$2"; shift 2;;
    --images) IMAGE_DIR="$2"; shift 2;;
    --motions) MOTION_SEQS_DIR="$2"; shift 2;;
    --out) OUT_ROOT="$2"; shift 2;;
    --nas) NAS_ROOT="$2"; shift 2;;
    --export-gs) EXPORT_GS="$2"; shift 2;;
    --render-fps) RENDER_FPS="$2"; shift 2;;
    --motion-read-fps) MOTION_READ_FPS="$2"; shift 2;;
    --vis-motion) VIS_MOTION="$2"; shift 2;;
    --mask) MOTION_IMG_NEED_MASK="$2"; shift 2;;
    --motion-img-dir) MOTION_IMG_DIR="$2"; shift 2;;
    --zero-hands) ZERO_HANDS="true"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Checks
[[ -d "$IMAGE_DIR" ]] || { echo "[ERR] IMAGE_DIR not found: $IMAGE_DIR"; exit 1; }
[[ -d "$MOTION_SEQS_DIR" ]] || { echo "[ERR] MOTION_SEQS_DIR not found: $MOTION_SEQS_DIR"; exit 1; }

echo "MODEL_NAME        : $MODEL_NAME"
echo "IMAGE_DIR         : $IMAGE_DIR"
echo "MOTION_SEQS_DIR   : $MOTION_SEQS_DIR"
echo "OUT_ROOT          : $OUT_ROOT"
echo "NAS_ROOT          : $NAS_ROOT"
echo "EXPORT_GS         : $EXPORT_GS"
echo "RENDER_FPS        : $RENDER_FPS"
echo "MOTION_READ_FPS   : $MOTION_READ_FPS"
echo "VIS_MOTION        : $VIS_MOTION"
echo "MOTION_IMG_NEED_MASK : $MOTION_IMG_NEED_MASK"
echo "MOTION_IMG_DIR    : $MOTION_IMG_DIR"
echo "ZERO_HANDS        : $ZERO_HANDS"
echo "ASYNC_MESH_SYNC   : $ASYNC_MESH_SYNC"
echo

if [[ "${ZERO_HANDS,,}" == "true" ]]; then
  echo "[INFO] Overwriting lhand_pose/rhand_pose to zeros in $MOTION_SEQS_DIR"
  python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$SCRIPT_DIR")
from render_gs_from_img_smplx import normalize_motion_jsons
normalize_motion_jsons(Path("$MOTION_SEQS_DIR"), zero_hands=True)
PY
fi

mkdir -p "$OUT_ROOT" "$NAS_ROOT"
mkdir -p "$OUT_ROOT"/{images,meshes,videos,tmp}
mkdir -p "$NAS_ROOT"/meshes

# Collect images (just to sanity check that we have input)
mapfile -d '' IMAGES < <(find "$IMAGE_DIR" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) -print0)

if [[ "${#IMAGES[@]}" -eq 0 ]]; then
  echo "[WARN] No images found in $IMAGE_DIR"
  exit 0
fi

echo "[INFO] Running single-process batch over ${#IMAGES[@]} images to reuse model load."
python -m LHM.launch infer.human_lrm \
  model_name="$MODEL_NAME" \
  motion_seqs_dir="$MOTION_SEQS_DIR" \
  image_input="$IMAGE_DIR" \
  video_dump="$OUT_ROOT/videos" \
  image_dump="$OUT_ROOT/images" \
  mesh_dump="$OUT_ROOT/meshes" \
  save_tmp_dump="$OUT_ROOT/tmp" \
  render_fps="$RENDER_FPS" \
  motion_video_read_fps="$MOTION_READ_FPS" \
  vis_motion="$VIS_MOTION" \
  motion_img_need_mask="$MOTION_IMG_NEED_MASK" \
  export_gs="$EXPORT_GS" \
  motion_img_dir="$MOTION_IMG_DIR" \
  export_video=false

RSYNC_PIDS=()
if [[ "${ASYNC_MESH_SYNC,,}" == "true" ]]; then
  rsync -a --info=progress2 "$OUT_ROOT/meshes"/ "$NAS_ROOT/meshes"/ &
  pid=$!
  RSYNC_PIDS+=("$pid")
  echo "[INFO] Mesh sync running in background (pid $pid)"
else
  rsync -a --info=progress2 "$OUT_ROOT/meshes"/ "$NAS_ROOT/meshes"/
fi

if [[ "${#RSYNC_PIDS[@]}" -gt 0 ]]; then
  echo "[INFO] Waiting for mesh sync to finish..."
  for pid in "${RSYNC_PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "[ERROR] Mesh sync process $pid failed" >&2
      exit 1
    fi
  done
fi

# Keep only meshes locally; other dumps are temporary
rm -rf "$OUT_ROOT"/images "$OUT_ROOT"/videos "$OUT_ROOT"/tmp

echo
echo "[ALL DONE] Outputs in $OUT_ROOT (meshes only) and mirrored to $NAS_ROOT."
