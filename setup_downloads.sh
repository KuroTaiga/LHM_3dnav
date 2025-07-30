# 设置目标根目录
DATA_ROOT="/root/autodl-tmp/LHM_Data"
mkdir -p "$DATA_ROOT"

echo ">>> Downloading LHM prior model..."
wget -O "$DATA_ROOT/LHM_prior_model.tar" https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar
mkdir -p "$DATA_ROOT/LHM_prior_model"
tar -xvf "$DATA_ROOT/LHM_prior_model.tar" -C "$DATA_ROOT/LHM_prior_model"
rm "$DATA_ROOT/LHM_prior_model.tar"

echo ">>> Downloading motion video..."
wget -O "$DATA_ROOT/motion_video.tar" https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/motion_video.tar
mkdir -p "$DATA_ROOT/motion_video"
tar -xvf "$DATA_ROOT/motion_video.tar" -C "$DATA_ROOT/motion_video"
rm "$DATA_ROOT/motion_video.tar"

echo ">>> Downloading custom pose estimate models..."
POSE_DIR="$DATA_ROOT/pretrained_models/human_model_files/pose_estimate"
mkdir -p "$POSE_DIR"
wget -P "$POSE_DIR" https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/yolov8x.pt
wget -P "$POSE_DIR" https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/vitpose-h-wholebody.pth

echo ">>> Installing pose estimation dependencies..."
cd ./engine/pose_estimation
pip install mmcv==1.3.9
pip install -v -e third-party/ViTPose
pip install ultralytics
cd ../../

echo ">>> Creating input/output directories under data root..."
mkdir -p "$DATA_ROOT/inputs/images" "$DATA_ROOT/inputs/videos" "$DATA_ROOT/outputs"

echo "✅ All setup complete. Data stored under: $DATA_ROOT"
