# render_gs_from_img_smplx usage

Helpers for turning a single reference image plus a SMPL-X motion JSON sequence into Gaussian-splat outputs. The Python entrypoint normalizes every motion JSON before rendering (renaming SMPL-X keys, stripping `meta`, forcing list types) and can also zero out hand poses.

## Canonical pose PLY (cano)
- Command: `python external/LHM_3dnav/render_gs_from_img_smplx.py --images <img_dir> --motion <motion_dir> --export-mesh [--no-export-gs] [--zero-hands] [--flat-hand-mean]`
- `--export-mesh` triggers the canonical pass; outputs land under `<output_root>/cano/{images,meshs,videos,save_tmp}/<uid>/` and mirror to `<nas_root>/cano/...` if set.
- Add `--no-export-gs` if you only need the canonical mesh; omit it to run the GS sequence afterward in the same invocation.
- `--zero-hands` overwrites `lhand_pose`/`rhand_pose` with zeros even if the JSON already contains hand data.

## PLY sequence (GS) and video
- Default run (no `--export-mesh`) renders the GS sequence: `python external/LHM_3dnav/render_gs_from_img_smplx.py --images <img_dir> --motion <motion_dir> [--export-video] [--zero-hands] [--flat-hand-mean]`
- Outputs land under `<output_root>/gs/{images,meshs,videos,save_tmp}/<uid>/` and mirror to `<nas_root>/gs/...` if set.
- Use `--export-video` to write the rendered video alongside the ply sequence. Control frame rates via `--render-fps` (output) and `--motion-video-fps` (input motion).

## Shell wrapper
- `external/LHM_3dnav/render_gs_from_img_smplx.sh` wraps the same flow with rsync to NAS and optional GNU parallel. Example:  
  `bash external/LHM_3dnav/render_gs_from_img_smplx.sh --images <img_dir> --motions <motion_dir> --nas <nas_root> --out <out_root> --jobs 4 --zero-hands`
- The `--zero-hands` flag pre-normalizes all motion JSONs by importing the Python helper above, guaranteeing closed/flat hands before inference.

## Utility switches (Python)
- `--output-root` / `--nas-root`: where to save and mirror results; canonical and GS runs automatically append `cano` or `gs`.
- `--zero-hands`: force zero hand poses on every motion frame (overrides existing hand poses).
- `--flat-hand-mean`: interpret zero hand poses with `flat_hand_mean=True`. Omit it to keep the default MANO-style curved hand mean.
- `--render-fps` / `--motion-video-fps`: control output vs. motion sequence FPS.
- `--export-video`: also render a video for the current pass.
