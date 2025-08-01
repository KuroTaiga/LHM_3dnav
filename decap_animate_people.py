import os
import json
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from LHM.models.rendering.smplx import smplx
from tqdm import tqdm

# === CONFIG ===
json_folder = "./outputs/custom_motion/T_frontview/walk_45/smplx_params"
base_ply_path ="./exps/meshs/video_human_benchmark/human-lrm-1B/T_frontview.ply"
output_dir = "./outputs/animated_3dgs"
human_model_path = "./pretrained_models/human_model_files"
gender = "neutral"
os.makedirs(output_dir, exist_ok=True)

# === LOAD BASE PLY ===
plydata = PlyData.read(base_ply_path)
data = plydata['vertex'].data
base_positions = np.vstack([data['x'], data['y'], data['z']]).T

scales = np.vstack([data['scale_0'], data['scale_1'], data['scale_2']]).T
rotations = np.vstack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']]).T
colors = np.vstack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']]).T
opacity = data['opacity']
sh_dim = 16
sh_features = None
if f'f_rest_0' in data.dtype.names:
    sh_features = np.vstack([data[f'f_rest_{i}'] for i in range(sh_dim)]).T

# === LOAD SMPLX MODEL ===
model = smplx.create(
    human_model_path,
    model_type='smplx',
    gender=gender,
    use_pca=False,
    num_betas=10,
    num_expression_coeffs=10,
    flat_hand_mean=True,
    use_face_contour=False,
    create_transl=True,
    create_body_pose=True,
    create_global_orient=True,
    create_expression=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_betas=True
)

# === ASSIGN GAUSSIANS TO JOINTS IN BASE T-POSE ===
with open(os.path.join(json_folder, "00001.json")) as f:
    base_params = json.load(f)

output_tpose = model(
    betas=torch.tensor(base_params['betas']).unsqueeze(0),
    expression=torch.tensor(base_params['expression']).unsqueeze(0),
    body_pose=torch.zeros((1, 63)),
    global_orient=torch.zeros((1, 3)),
    transl=torch.zeros((1, 3)),
    return_full_pose=True
)
joint_locations = output_tpose.joints[0].detach().cpu().numpy()  # (J, 3)

# Assign each Gaussian to its nearest joint
gaussian_joint_ids = np.argmin(np.linalg.norm(
    base_positions[:, None, :] - joint_locations[None, :, :], axis=-1), axis=1)

# === ANIMATE ===
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])

for frame_idx, fname in tqdm(enumerate(json_files), total=len(json_files)):
    with open(os.path.join(json_folder, fname)) as f:
        params = json.load(f)

    output = model(
        betas=torch.tensor(params['betas']).unsqueeze(0),
        expression=torch.tensor(params['expression']).unsqueeze(0),
        body_pose=torch.tensor(params['body_pose']).unsqueeze(0),
        global_orient=torch.tensor(params['global_orient']).unsqueeze(0),
        # transl=torch.tensor(params['trans']).unsqueeze(0),
        transl = torch.zeros((1, 3)),  # Use zero translation for now, so that it doesn't move
        left_hand_pose=torch.tensor(params['left_hand_pose']).unsqueeze(0),
        right_hand_pose=torch.tensor(params['right_hand_pose']).unsqueeze(0),
        jaw_pose=torch.tensor(params['jaw_pose']).unsqueeze(0),
        leye_pose=torch.tensor(params['leye_pose']).unsqueeze(0),
        reye_pose=torch.tensor(params['reye_pose']).unsqueeze(0),
        return_full_pose=True
    )

    joints = output.joints[0].detach().cpu().numpy()  # (J, 3)

    # Compute offset per Gaussian
    offsets = joints[gaussian_joint_ids] - joint_locations[gaussian_joint_ids]
    new_positions = base_positions + offsets

    # Repack to .ply
    records = []
    for i in range(len(new_positions)):
        row = (
            new_positions[i, 0], new_positions[i, 1], new_positions[i, 2],
            scales[i, 0], scales[i, 1], scales[i, 2],
            rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],
            colors[i, 0], colors[i, 1], colors[i, 2],
            opacity[i]
        )
        if sh_features is not None:
            row += tuple(sh_features[i])
        records.append(row)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4')
    ]
    if sh_features is not None:
        dtype += [(f'f_rest_{i}', 'f4') for i in range(sh_dim)]

    vertex_array = np.array(records, dtype=dtype)
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el], text=True).write(os.path.join(output_dir, f"frame_{frame_idx:05d}.ply"))

print("Done generating animated 3DGS .ply files.")
