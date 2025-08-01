import os
import builtins
import torch
import json
from tqdm import tqdm
from plyfile import PlyData
from LHM.models.rendering.gs_renderer import GS3DRenderer, GaussianModel
from LHM.models.rendering.smpl_x import SMPLXModel
from LHM.models.rendering.utils.sh_utils import RGB2SH
from LHM.utils.model_query_utils import AutoModelSwitcher
from LHM.utils.rot6d import matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
builtins.matrix_to_axis_angle = matrix_to_axis_angle
builtins.matrix_to_quaternion = matrix_to_quaternion
builtins.quaternion_to_axis_angle = quaternion_to_axis_angle
builtins.axis_angle_to_matrix = axis_angle_to_matrix
# === CONFIGURATION ===
MODEL_NAME = "LHM-1B"
canonical_ply_path = "./exps/meshs/video_human_benchmark/human-lrm-1B/T_frontview.ply"
json_folder = "./outputs/custom_motion/T_frontview/walk_45/smplx_params"
human_model_path = "./pretrained_models/human_model_files"
gender = "neutral"
# === OUTPUT DIRECTORY ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "./outputs/animated_3dgs"
os.makedirs(output_dir, exist_ok=True)

def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


# === LOAD CANONICAL .PLY AS GAUSSIAN MODEL ===
cano_gs = GaussianModel(
    xyz=None, opacity=None, rotation=None,
    scaling=None, shs=None, use_rgb=False
)
cano_gs.load_ply(canonical_ply_path)


# --- Load pretrained model ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === LOAD SMPLX MODEL FOR TRANSFORMATIONS ===
smplx_model = SMPLXModel(
    human_model_path=human_model_path,
    gender="neutral",
    subdivide_num=2,
    shape_param_dim=10,
    expr_param_dim=10,
    cano_pose_type=0,
    apply_pose_blendshape=False,
).to(device)

# === LOAD CANONICAL BASE PARAM FOR TRANSFORM MATRIX ===
with open(os.path.join(json_folder, "00001.json")) as f:
    base_params = json.load(f)

# NOTE: Convert to tensors
def to_tensor_dict(params):
    return {
        "root_pose": torch.tensor(params["root_pose"]).view(1, 1, 3).to(device),
        "body_pose": torch.tensor(params["body_pose"]).view(1, 1, 21, 3).to(device),
        "jaw_pose": torch.tensor(params["jaw_pose"]).view(1, 1, 3).to(device),
        "leye_pose": torch.tensor(params["leye_pose"]).view(1, 1, 3).to(device),
        "reye_pose": torch.tensor(params["reye_pose"]).view(1, 1, 3).to(device),
        "lhand_pose": torch.tensor(params["lhand_pose"]).view(1, 1, 15, 3).to(device),
        "rhand_pose": torch.tensor(params["rhand_pose"]).view(1, 1, 15, 3).to(device),
        "expr": torch.zeros(1, 1, 10).to(device),  # use zero if not present
        "trans": torch.zeros(1, 1, 3).to(device),  # or torch.tensor(params["trans"]).view(1, 1, 3)
        "betas": torch.tensor(params["betas"]).view(1, 10).to(device),
    }

base_tensor_params = to_tensor_dict(base_params)

# === GENERATE TRANSFORM MATRIX ===
_, _, T = smplx_model.get_query_points(base_tensor_params, device=device)
base_tensor_params["transform_mat_neutral_pose"] = T  # (1, 55, 4, 4)

# === ANIMATE ALL FRAMES ===
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])

for frame_idx, fname in tqdm(enumerate(json_files), total=len(json_files)):
    with open(os.path.join(json_folder, fname)) as f:
        params = json.load(f)

    tensor_params = to_tensor_dict(params)
    tensor_params["transform_mat_neutral_pose"] = base_tensor_params["transform_mat_neutral_pose"]

    # === ANIMATE ===
    renderer = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=768,
        query_dim=None,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
        clip_scaling=0.2,
        shape_param_dim=100,
        expr_param_dim=50,
        cano_pose_type=0,#default=0,
        fix_opacity=False,
        fix_rotation=False,
        decoder_mlp=False,
        skip_decoder=False,
        decode_with_extra_info=None,
        gradient_checkpointing=False,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,
    )


    anim_gs_list, _ = renderer.animate_gs_model(
        gs_attr=cano_gs,
        query_points=cano_gs.xyz,
        smplx_data=tensor_params
    )

    # Save frame
    anim_gs_list[0].save_ply(os.path.join(output_dir, f"frame_{frame_idx:05d}.ply"))

print(" Animation completed and saved to:", output_dir)
