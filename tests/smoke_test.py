"""Smoke test: runs all three nodes on small synthetic clips, outside ComfyUI.

Run from the folder that CONTAINS this pack (e.g. ComfyUI/custom_nodes):
    python comfyui-plate-relight/tests/smoke_test.py
Needs torch, numpy and opencv-python (plain, not contrib).
"""
import importlib
import os
import sys
import types

import numpy as np
import torch

# ComfyUI provides comfy.utils.ProgressBar at runtime; stub it so the test runs standalone.
comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.ProgressBar = lambda total: types.SimpleNamespace(update=lambda n: None)
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.utils", comfy.utils)

pack_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(pack_dir))
pack = importlib.import_module(os.path.basename(pack_dir))
M = pack.NODE_CLASS_MAPPINGS
assert set(M) == {"LightTransfer", "TemporalDeflicker", "AddLightFromPasses"}, M

N, H, W = 6, 96, 160
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def clip(shift=0.0, noise=0.0):
    f = []
    for i in range(N):
        base = 0.5 + 0.3 * np.sin((xx + i * 2 + shift) / 9) * np.cos(yy / 11)
        f.append(np.dstack([base, base * 0.9, base * 0.8]) + noise * rng.standard_normal((H, W, 3)))
    return torch.from_numpy(np.clip(np.stack(f), 0, 1).astype(np.float32))


plate = clip()
relit = clip() * torch.linspace(0.4, 1.2, W).view(1, 1, W, 1).clamp(0, 1)
normals = torch.zeros(N, H, W, 3); normals[..., 2] = 1.0; normals[..., :2] = 0.5   # flat, facing camera
matte = torch.ones(N, H, W)
depth = clip()

lt, aligned = M["LightTransfer"]().run(plate, relit, 2.0, 8, 0.02, 1.2, align_relit=False)
assert lt.shape == plate.shape and aligned.shape == plate.shape and torch.isfinite(lt).all()

lt2, _ = M["LightTransfer"]().run(plate, relit, 2.0, 8, 0.02, 1.2, align_relit=True, protect_mask=torch.zeros(N, H, W))
assert lt2.shape == plate.shape and torch.isfinite(lt2).all()

(df,) = M["TemporalDeflicker"]().run(clip(noise=0.05), plate, 2, 1, 0.05, 0.5)
assert df.shape == plate.shape and torch.isfinite(df).all()

lit, only = M["AddLightFromPasses"]().run(plate, plate, normals, plate, matte, -60.0, 60.0, 15.0, "ease in-out", "#FFC785",
                                          3.2, 3.0, 1.0, 24.0, depth=depth, shadow_strength=1.0, shadow_depth_scale=0.002)
assert lit.shape == plate.shape and only.shape == plate.shape and torch.isfinite(lit).all()
assert (lit >= plate - 1e-4).all(), "Add Light should only ever add light"

print("OK: LightTransfer, TemporalDeflicker, AddLightFromPasses")
