# comfyui-plate-relight

Plate Relight: relight and clean up real footage with AI renders and passes, without replacing the plate's pixels.

Three VFX post nodes for ComfyUI. They all work on plain IMAGE / MASK batches (one image per frame), keep the original plate's pixels, and run on CPU with NumPy + OpenCV.

| Node (display name) | Class | What it does |
|---|---|---|
| **Light Transfer** (plate keeps its pixels) | `LightTransfer` | Relights a plate without regenerating it: copies the light and colour from a relit render onto the original pixels. |
| **Temporal Deflicker** (flow-guided, for AI passes) | `TemporalDeflicker` | Removes frame-to-frame shimmer from per-frame AI passes (albedo, normals, specular and so on). It follows the plate's motion to blend each frame with its neighbours. |
| **Add Light from Passes** (plate + light) | `AddLightFromPasses` | Adds a new light on top of the original footage, using the albedo, normals and specular passes, plus cast shadows from the depth pass. |

All three are in the `image/postprocessing` category.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/dughogan/comfyui-plate-relight
```

Or copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

Dependencies are `numpy` and `opencv-python` (see `requirements.txt`). Both normally ship with ComfyUI already. `torch` comes from ComfyUI.
- There are **no** model downloads, no network calls, no subprocesses, and no writes to disk.
- `opencv-contrib-python` is **not** needed. The guided filter is implemented in-house (`_guided_filter`), and SIFT and DIS optical flow are in core OpenCV ≥ 4.5.

## Nodes in detail

### Light Transfer
**Inputs**
- `plate`: the original footage.
- `relit`: the same shot re-rendered in new light, for example by MiniMax H3 or Qwen Image Edit.
- `detail_sigma` (2.0): the scale of light detail taken from the relit render.
- `guide_radius` (8): the edge-aware smoothing radius.
- `shadow_damping` (0.02): stops glow in deep shadow.
- `exposure` (1.2).
- `align_relit` (on): generative relights zoom and drift. When on, each relit frame is aligned to the plate first, using SIFT homography on lighting-normalised images along one smooth quadratic path through time, so there's no jitter.
- Optional `protect_mask`: e.g. a SAM 3.1 "eye / eyebrow / mouth" matte. Inside it, the light is rebuilt from its surroundings, so the relit render's own eyes and expression never print onto the plate. It comes with `protect_hold_frames` (8, which bridges blinks) and `protect_grow` (12).

**Outputs**
- `IMAGE`: the plate with the new light.
- `aligned_relit`: the relit render warped onto the plate's framing.

### Temporal Deflicker
**Inputs**
- `pass_frames`: the flickering pass.
- `beauty`: the plate. Motion is measured on this.
- `radius` (6): frames either side that are blended.
- `iterations` (2).
- `occlusion_sigma` (0.05): lower is stricter.
- `flow_scale` (0.5).

It uses DIS optical flow on the plate. A neighbour only counts where the warped plate agrees, so occlusions fall back to the frame's own value instead of smearing.

### Add Light from Passes
**Inputs**
- `plate`, `albedo`, `normals`, `specular` and `matte`.
- Optional `depth`, which enables cast shadows.
- `yaw_start` / `yaw_end` / `pitch` / `sweep`: a directional light that can sweep during the clip.
- `colour`, `intensity`, `falloff`, `specular_amount` and `shininess`.
- `shadow_strength` and `shadow_depth_scale`.

**Outputs**
- `lit_plate`.
- `light_only`.

**How it works**
- The light is `albedo × N·L^falloff + Blinn-Phong specular`, added in linear light. The plate is never replaced.
- Normals use Marigold's camera-space convention (x right, y up, z toward camera, stored as (n+1)/2). Depth is bright = near.
- Cast shadows are screen-space: a ray march across the depth height field toward the light.

## Performance and limits
- Processing is per frame on CPU. Expect seconds per frame at 1080p for Light Transfer with `align_relit` on, and for Temporal Deflicker.
- The whole clip is held in RAM as float32, so very long or 4K+ clips need a lot of memory. Batch them through in chunks if needed.
- Add Light from Passes is a directional light only: there's no point or spot light, and no viewport gizmo yet.

## Test
From `ComfyUI/custom_nodes/`:
```
python comfyui-plate-relight/tests/smoke_test.py
```
This runs all three nodes on synthetic clips and prints `OK: ...`. It stubs `comfy.utils`, so it doesn't need a running ComfyUI.

## Example workflow
`example_workflows/custom_nodes_showcase.json` wires all three nodes to real passes. It references local test clips and uses VideoHelperSuite (`VHS_LoadVideo`) as loaders, so you'll need to swap in your own inputs.

## License

MIT, see `LICENSE`.
