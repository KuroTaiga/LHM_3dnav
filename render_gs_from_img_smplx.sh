#!/usr/bin/env bash
# render_gs_from_img_smplx.sh
# Run LHM per image and mirror outputs to NAS without any "sub python run" wrappers.

set -euo pipefail

# -------- Defaults (override via CLI env or flags below) --------
MODEL_NAME="${MODEL_NAME:-LHM-1B-HF}"
IMAGE_DIR="${IMAGE_DIR:-../../inputs/images/SHHQ-1.0_samples}"
MOTION_SEQS_DIR="${MOTION_SEQS_DIR:-../../inputs/motion_seq_cleaned/walk_fbx}"
OUT_ROOT="${OUT_ROOT:-exps/SHHQ_exp}"
NAS_ROOT="${NAS_ROOT:-/media/lenovo/VariedHumanPlys/SHHQ_walk_fbx}"   # <- verify path
EXPORT_GS="${EXPORT_GS:-true}"
RENDER_FPS="${RENDER_FPS:-30}"
MOTION_READ_FPS="${MOTION_READ_FPS:-30}"
VIS_MOTION="${VIS_MOTION:-true}"
MOTION_IMG_NEED_MASK="${MOTION_IMG_NEED_MASK:-true}"
MOTION_IMG_DIR="${MOTION_IMG_DIR:-None}"  # e.g., set to a folder or keep 'None'
JOBS="${JOBS:-1}"   # >1 requires GNU parallel
# ---------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $(basename "$0") [--model NAME] [--images DIR] [--motions DIR] [--out DIR] [--nas DIR]
                        [--export-gs true|false] [--jobs N]
                        [--render-fps N] [--motion-read-fas N]
                        [--vis-motion true|false] [--mask true|false] [--motion-img-dir PATH|None]

Examples:
  $(basename "$0") --model LHM-1B-HF --images ../../inputs/images/SHHQ-1.0_samples \\
                   --motions ../../inputs/motion_seq_cleaned/walk_fbx --jobs 4
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
    --jobs) JOBS="$2"; shift 2;;
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
echo "JOBS              : $JOBS"
echo

mkdir -p "$OUT_ROOT" "$NAS_ROOT"

do_one() {
  local img="$1"
  local uid
  uid="$(basename "$img")"
  uid="${uid%.*}"

  local img_dump="$OUT_ROOT/images/$uid"
  local mesh_dump="$OUT_ROOT/meshes/$uid"
  local vid_dump="$OUT_ROOT/videos/$uid"
  local tmp_dump="$OUT_ROOT/tmp/$uid"

  mkdir -p "$img_dump" "$mesh_dump" "$vid_dump" "$tmp_dump"

  echo
  echo "[INFO] Processing UID: $uid | image: $(basename "$img")"

  # Run the LHM runner
  python -m LHM.runners.infer.human_lrm \
    model_name="$MODEL_NAME" \
    motion_seqs_dir="$MOTION_SEQS_DIR" \
    image_input="$img" \
    video_dump="$vid_dump" \
    image_dump="$img_dump" \
    mesh_dump="$mesh_dump" \
    save_tmp_dump="$tmp_dump" \
    render_fps="$RENDER_FPS" \
    motion_video_read_fps="$MOTION_READ_FPS" \
    vis_motion="$VIS_MOTION" \
    motion_img_need_mask="$MOTION_IMG_NEED_MASK" \
    export_gs="$EXPORT_GS" \
    motion_img_dir="$MOTION_IMG_DIR" \
    export_video=false

  # Mirror to NAS (preserve substructure)
  rsync -a --info=progress2 "$img_dump"/  "$NAS_ROOT/images/$uid"/
  rsync -a --info=progress2 "$mesh_dump"/ "$NAS_ROOT/meshes/$uid"/
  rsync -a --info=progress2 "$vid_dump"/  "$NAS_ROOT/videos/$uid"/
  rsync -a --info=progress2 "$tmp_dump"/  "$NAS_ROOT/tmp/$uid"/

  echo "[DONE] $uid copied to NAS."
}

export -f do_one
export MODEL_NAME MOTION_SEQS_DIR OUT_ROOT NAS_ROOT EXPORT_GS \
       RENDER_FPS MOTION_READ_FPS VIS_MOTION MOTION_IMG_NEED_MASK MOTION_IMG_DIR

# Collect images
mapfile -d '' IMAGES < <(find "$IMAGE_DIR" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) -print0)

if [[ "${#IMAGES[@]}" -eq 0 ]]; then
  echo "[WARN] No images found in $IMAGE_DIR"
  exit 0
fi

# Run (parallel if available & requested)
if command -v parallel >/dev/null 2>&1 && [[ "$JOBS" -gt 1 ]]; then
  printf '%s\0' "${IMAGES[@]}" | parallel -0 -j "$JOBS" do_one {}
else
  for img in "${IMAGES[@]}"; do
    do_one "$img"
  done
fi

echo
echo "[ALL DONE] Outputs in $OUT_ROOT and mirrored to $NAS_ROOT."
