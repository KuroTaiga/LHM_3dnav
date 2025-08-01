#!/bin/bash
# Given pose sequence, generating animation video with Gaussian Splatting export

MODEL_NAME=LHM-1B
IMAGE_INPUT="./inputs/images/example/"
MOTION_SEQS_DIR="./outputs/custom_motion/T_frontview/walk_45/inplace_smplx_params"
EXPORT_GS=true

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${3:-$MOTION_SEQS_DIR}
EXPORT_GS=${4:-$EXPORT_GS}

echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
echo "EXPORT_GS: $EXPORT_GS"

echo "INFERENCE VIDEO WITH GAUSSIAN SPLATTING EXPORT"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=True
EXPORT_GS=True  # ✅ Fixed syntax

# Set export_gs to enable per-frame GS export
if [ "$EXPORT_GS" = "true" ] || [ "$EXPORT_GS" = "True" ] || [ "$EXPORT_GS" = "1" ]; then
    EXPORT_GS_FLAG=true
    echo "Gaussian Splatting export enabled"
else
    EXPORT_GS_FLAG=false
    echo "Gaussian Splatting export disabled"
fi

python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO \
        export_gs=$EXPORT_GS_FLAG \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS