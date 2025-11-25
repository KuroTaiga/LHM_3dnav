#!/usr/bin/env python3
"""
Batch driver for inference_gc.sh that routes every run into:
    SHHQ_exp/{images,meshs,videos,save_tmp}/{uid}/...
and mirrors the same structure to NAS at /media/lenvono/VariedHumanPlys/SHHQ_walk_fbx.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable
import json

DEFAULT_MODEL = "LHM-1B-HF"
DEFAULT_IMAGES = Path("../../inputs/images/SHHQ-1.0_samples")
DEFAULT_MOTION = Path("../../inputs/motion_seq_cleaned/walk_fbx")
DEFAULT_OUTPUT_ROOT = Path("SHHQ_exp")
DEFAULT_NAS_ROOT = Path("/mnt/nas/jiankundong/SHHQ_walk_fbx")

CMD_BOOL = {True: "True", False: "False"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"}

def normalize_motion_jsons(sequence_folder: Path) -> None:
    """Ensure every SMPL-X JSON uses the keys/shapes LHM expects."""
    rename_map = {
        "global_orient": "root_pose",
        "transl": "trans",
        "left_hand_pose": "lhand_pose",
        "right_hand_pose": "rhand_pose",
    }
    default_vectors = {
        "root_pose": [0.0, 0.0, 0.0],
        "trans": [0.0, 0.0, 0.0],
        "jaw_pose": [0.0, 0.0, 0.0],
        "leye_pose": [0.0, 0.0, 0.0],
        "reye_pose": [0.0, 0.0, 0.0],
    }
    default_hands = {
        "lhand_pose": [[0.0, 0.0, 0.0] for _ in range(15)],
        "rhand_pose": [[0.0, 0.0, 0.0] for _ in range(15)],
    }

    json_files = sorted(sequence_folder.glob("*.json"))
    if not json_files:
        print(f"[WARN] No motion JSONs under {sequence_folder}")
        return

    for json_path in json_files:
        data = json.loads(json_path.read_text())
        mutated = False

        # Rename SMPL-X keys to what LHM expects
        for old_key, new_key in rename_map.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)
                mutated = True

        # Drop metadata blocks that break torch.FloatTensor()
        if "meta" in data:
            data.pop("meta")
            mutated = True

        # Ensure required keys exist
        for key, default in default_vectors.items():
            if key not in data:
                data[key] = list(default)
                mutated = True

        for key, default in default_hands.items():
            if key not in data:
                # deep copy so later edits don’t mutate the template
                data[key] = [row[:] for row in default]
                mutated = True

        # Force tuple → list so JSON dumps cleanly
        for key in (*default_vectors.keys(), "betas", "focal", "princpt", "img_size_wh"):
            if key in data and isinstance(data[key], tuple):
                data[key] = list(data[key])
                mutated = True

        if mutated:
            json_path.write_text(json.dumps(data, indent=2))
            print(f"[SANITIZE] {json_path.name} normalized")


def run_command(cmd: Iterable[str]) -> None:
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"[CMD] {printable}")
    subprocess.run(list(map(str, cmd)), check=True)


def build_output_dirs(output_root: Path, uid: str) -> Dict[str, Path]:
    layout = {
        "image_dump": output_root / "images" / uid,
        "mesh_dump": output_root / "meshs" / uid,
        "video_dump": output_root / "videos" / uid,
        "save_tmp_dump": output_root / "save_tmp" / uid,
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def sync_uid_to_nas(uid: str, output_root: Path, nas_root: Path) -> None:
    for sub in ("images", "meshs", "videos", "save_tmp"):
        src = output_root / sub / uid
        if not src.exists():
            continue
        dst = nas_root / sub / uid
        dst.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "rsync",
                "-a",
                "--delete",
                f"{src}/",
                f"{dst}/",
            ]
        )
        print(f"[NAS] {sub}/{uid} synced to {dst}")


def render_gs(
    img_folder: Path,
    sequence_folder: Path,
    model_name: str = DEFAULT_MODEL,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    nas_root: Path = DEFAULT_NAS_ROOT,
    motion_img_dir: str = "None",
    vis_motion: bool = True,
    motion_img_need_mask: bool = True,
    render_fps: int = 30,
    motion_video_read_fps: int = 30,
    export_video: bool = False,
    export_gs: bool = True,
    export_mesh: bool = True,
):
    if export_mesh:
        print("[INFO] export mesh set, cano ply will be exported and not the plys")
        output_root = output_root / "cano"
        nas_root = nas_root / "cano"
    else:
        print("[INFO] export mesh not set, only gs will be exported")
        output_root = output_root / "gs"
        nas_root = nas_root / "gs"
    img_folder = img_folder.expanduser().resolve()
    sequence_folder = sequence_folder.expanduser().resolve()
    normalize_motion_jsons(sequence_folder)
    output_root = output_root.expanduser().resolve()
    nas_root = nas_root.expanduser().resolve()

    print(f"[INFO] processing seq from {sequence_folder}")
    images = sorted(
        p for p in img_folder.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES
    )
    if not images:
        print(f"[WARN] No images found under {img_folder}")
        return

    for img_path in images:
        uid = img_path.stem
        print(f"[INFO] Rendering {uid} ({img_path.name})")

        layout = build_output_dirs(output_root, uid)
        if not export_mesh:
            cmd = [
                "python",
                "-m",
                "LHM.launch",
                "infer.human_lrm",
                f"model_name={model_name}",
                f"image_input={img_path}",
                f"motion_seqs_dir={sequence_folder}",
                f"motion_img_dir={motion_img_dir}",
                f"vis_motion={CMD_BOOL[vis_motion]}",
                f"motion_img_need_mask={CMD_BOOL[motion_img_need_mask]}",
                f"render_fps={render_fps}",
                f"motion_video_read_fps={motion_video_read_fps}",
                f"export_video={CMD_BOOL[export_video]}",
                f"export_gs={CMD_BOOL[export_gs]}",
                f"image_dump={layout['image_dump']}",
                f"mesh_dump={layout['mesh_dump']}",
                f"video_dump={layout['video_dump']}",
                f"save_tmp_dump={layout['save_tmp_dump']}",
            ]
        else:
            cmd = [
                "python",
                "-m",
                "LHM.launch",
                "infer.human_lrm",
                f"model_name={model_name}",
                f"image_input={img_path}",
                f"motion_seqs_dir={sequence_folder}",
                f"motion_img_dir={motion_img_dir}",
                f"vis_motion={CMD_BOOL[vis_motion]}",
                f"motion_img_need_mask={CMD_BOOL[motion_img_need_mask]}",
                f"render_fps={render_fps}",
                f"motion_video_read_fps={motion_video_read_fps}",
                f"export_video={CMD_BOOL[export_video]}",
                f"export_gs={CMD_BOOL[export_gs]}",
                f"export_mesh={CMD_BOOL[export_mesh]}",
                f"image_dump={layout['image_dump']}",
                f"mesh_dump={layout['mesh_dump']}",
                f"video_dump={layout['video_dump']}",
                f"save_tmp_dump={layout['save_tmp_dump']}",
            ]
        try:
            run_command(cmd)
            print("[INFO] Syncing results to NAS...")
            sync_uid_to_nas(uid, output_root, nas_root)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Failed on {uid}: {exc}")
            continue
        print(f"[DONE] {uid} processed.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch GS inference with SHHQ_exp+NAS layout."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nas-root", type=Path, default=DEFAULT_NAS_ROOT)
    parser.add_argument("--export-video", action="store_true", default=False)
    parser.add_argument("--export-mesh", type=bool ,default=True)
    parser.add_argument("--no-export-gs", action="store_false", dest="export_gs")
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--motion-video-fps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("[INFO] Rendering the Cano PLY first")
    # render_gs(
    #     img_folder=args.images,
    #     sequence_folder=args.motion,
    #     model_name=args.model,
    #     output_root=args.output_root,
    #     nas_root=args.nas_root,
    #     render_fps=args.render_fps,
    #     motion_video_read_fps=args.motion_video_fps,
    #     export_video=args.export_video,
    #     export_gs=args.export_gs,
    #     export_mesh=args.export_mesh,
    # )
    print("[INFO] Done with Cano PLY")
    render_gs(
        img_folder=args.images,
        sequence_folder=args.motion,
        model_name=args.model,
        output_root=args.output_root,
        nas_root=args.nas_root,
        render_fps=args.render_fps,
        motion_video_read_fps=args.motion_video_fps,
        export_video=args.export_video,
        export_gs=args.export_gs,
        export_mesh=False, # second pass without the cano ply export
    )
    print("[INFO] All done.")