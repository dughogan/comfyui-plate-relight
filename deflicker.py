"""Flow-guided temporal deflicker for per-frame AI passes (albedo, shading, specular, normals...).

Per-frame estimators (Marigold, IID) have no memory, so each frame's answer shimmers.
For every frame t we warp the pass from its neighbours t-R..t+R onto frame t using optical
flow measured on the *beauty* plate (the real motion), then blend them.  A neighbour only
counts where the beauty plate agrees after warping, so occlusions and fast motion fall back
to the frame's own value instead of smearing.
"""
import cv2
import numpy as np


def _flow(dis, a, b, scale):
    if scale != 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    f = dis.calc(a, b, None)
    if scale != 1.0:
        f = cv2.resize(f, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR) / scale
    return f


def deflicker(passes, guide, radius=5, sigma_t=3.0, occ_sigma=0.04, flow_scale=0.5, iterations=1, progress=None):
    """passes: list of N,H,W,C float32 [0..1] arrays (all share the guide's motion)
    guide:  N,H,W,3 float32 [0..1] beauty plate.  Returns list of smoothed arrays."""
    n, h, w = guide.shape[:3]
    gray = [(cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)) for g in guide]
    gf = [g.astype(np.float32) / 255 for g in gray]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    outs = [p.copy() for p in passes]
    for _ in range(iterations):
        src = [o.copy() for o in outs]
        for t in range(n):
            acc = [np.zeros_like(s[t]) for s in src]
            wsum = np.zeros((h, w, 1), np.float32)
            for k in range(-radius, radius + 1):
                j = t + k
                if j < 0 or j >= n:
                    continue
                wt = np.exp(-(k * k) / (2 * sigma_t * sigma_t))
                if k == 0:
                    wk = np.full((h, w, 1), wt, np.float32)
                    for a, s in zip(acc, src):
                        a += s[t] * wk
                    wsum += wk
                    continue
                f = _flow(dis, gray[t], gray[j], flow_scale)
                mx, my = xx + f[..., 0], yy + f[..., 1]
                gw = cv2.remap(gf[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                err = cv2.GaussianBlur(np.abs(gw - gf[t]), (0, 0), 1.5)
                wk = (wt * np.exp(-(err * err) / (2 * occ_sigma * occ_sigma)))[..., None].astype(np.float32)
                for a, s in zip(acc, src):
                    a += cv2.remap(s[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(s[t].shape) * wk
                wsum += wk
            for o, a in zip(outs, acc):
                o[t] = a / np.maximum(wsum, 1e-6)
            if progress:
                progress(t + 1, n)
    return outs
