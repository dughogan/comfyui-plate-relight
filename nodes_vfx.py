"""VFX post nodes: temporal deflicker for AI passes, add a light from passes, light transfer."""
import math

import cv2
import numpy as np
import torch


from .add_light import add_light, light_dir
from .deflicker import deflicker



def _np(img):
    return img.detach().cpu().float().numpy()


class TemporalDeflicker:
    CATEGORY = "image/postprocessing"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    DESCRIPTION = ("Removes frame-to-frame shimmer from per-frame AI passes (albedo, shading, specular, normals). "
                   "Neighbouring frames are warped onto each frame with optical flow measured on the beauty plate, "
                   "then blended where the plate agrees, so real motion stays sharp.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pass_frames": ("IMAGE", {"tooltip": "The flickering pass, one image per frame."}),
            "beauty": ("IMAGE", {"tooltip": "The original plate, same frame count. Motion is measured on this."}),
            "radius": ("INT", {"default": 6, "min": 1, "max": 16, "tooltip": "Frames either side that are blended."}),
            "iterations": ("INT", {"default": 2, "min": 1, "max": 4}),
            "occlusion_sigma": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005, "tooltip": "Lower = stricter: neighbours count only where the warped plate matches closely."}),
            "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05}),
        }}

    def run(self, pass_frames, beauty, radius, iterations, occlusion_sigma, flow_scale):
        n = min(len(pass_frames), len(beauty))
        p, g = _np(pass_frames[:n]), _np(beauty[:n])
        if g.shape[1:3] != p.shape[1:3]:
            g = np.stack([cv2.resize(f, (p.shape[2], p.shape[1]), interpolation=cv2.INTER_AREA) for f in g])
        import comfy.utils
        bar = comfy.utils.ProgressBar(n * iterations)
        out = deflicker([p], g, radius=radius, sigma_t=radius * 0.6, occ_sigma=occlusion_sigma, flow_scale=flow_scale,
                        iterations=iterations, progress=lambda i, t: bar.update(1))[0]
        return (torch.from_numpy(np.clip(out, 0, 1)),)


class AddLightFromPasses:
    CATEGORY = "image/postprocessing"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("lit_plate", "light_only")
    DESCRIPTION = ("Adds a virtual light ON TOP of the original footage. The passes only shape the light "
                   "(albedo x N.L + a specular highlight); every original pixel stays. The light can sweep across the clip.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",), "albedo": ("IMAGE",), "normals": ("IMAGE",), "specular": ("IMAGE",), "matte": ("MASK",),
            "yaw_start": ("FLOAT", {"default": -85.0, "min": -180, "max": 180}),
            "yaw_end": ("FLOAT", {"default": 85.0, "min": -180, "max": 180}),
            "pitch": ("FLOAT", {"default": 15.0, "min": -89, "max": 89}),
            "sweep": (["ease in-out", "linear", "fixed (start)"],),
            "colour": ("STRING", {"default": "#FFC785"}),
            "intensity": ("FLOAT", {"default": 3.2, "min": 0, "max": 20, "step": 0.1}),
            "falloff": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 8, "step": 0.1, "tooltip": "Exponent on N.L: higher = tighter, more directional light."}),
            "specular_amount": ("FLOAT", {"default": 1.0, "min": 0, "max": 5, "step": 0.05}),
            "shininess": ("FLOAT", {"default": 24.0, "min": 1, "max": 256}),
        }, "optional": {
            "depth": ("IMAGE", {"tooltip": "Depth pass (bright = near). Connect it to get cast shadows: nose, brow and brim shadows that move with the light."}),
            "shadow_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.05}),
            "shadow_depth_scale": ("FLOAT", {"default": 0.002, "min": 0.0001, "max": 0.02, "step": 0.0001, "tooltip": "Depth units per pixel: how far the surface sticks out toward camera. Higher = longer shadows."}),
        }}

    def run(self, plate, albedo, normals, specular, matte, yaw_start, yaw_end, pitch, sweep, colour, intensity, falloff, specular_amount, shininess, depth=None, shadow_strength=1.0, shadow_depth_scale=0.002):
        spec_amt = specular_amount
        c = colour.lstrip("#"); col = tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
        n = min(len(plate), len(albedo), len(normals), len(specular), len(matte))
        P, A, N, S, M = (_np(x[:n]) for x in (plate, albedo, normals, specular, matte))
        Z = _np(depth) if depth is not None else None
        H, W = P.shape[1:3]
        fit = lambda f: f if f.shape[:2] == (H, W) else cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
        lit, only = [], []
        for i in range(n):
            u = i / max(1, n - 1)
            u = 0.5 - 0.5 * math.cos(math.pi * u) if sweep == "ease in-out" else (0 if sweep.startswith("fixed") else u)
            L = light_dir(yaw_start + (yaw_end - yaw_start) * u, pitch)
            z = fit(Z[min(i, len(Z) - 1)]).mean(2) if Z is not None else None
            r = add_light(P[i], fit(A[i]), fit(N[i]), fit(S[i]), fit(M[i]), L, colour=col, intensity=intensity,
                          softness=falloff, spec_amount=spec_amt, shininess=shininess,
                          depth=z, shadow_strength=shadow_strength, shadow_depth_scale=shadow_depth_scale)
            lit.append(r); only.append(np.clip(r - P[i], 0, 1))
        return (torch.from_numpy(np.stack(lit)), torch.from_numpy(np.stack(only)))


def _lnorm(img):
    y = np.log(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) + 8)
    m = cv2.GaussianBlur(y, (0, 0), 12); sd = np.sqrt(cv2.GaussianBlur((y - m) ** 2, (0, 0), 12)) + 0.05
    return ((np.clip((y - m) / sd, -3, 3) + 3) / 6 * 255).astype(np.uint8)


def _estimate_h(plate8, relit8):
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(_lnorm(plate8), None); k2, d2 = sift.detectAndCompute(_lnorm(relit8), None)
    if d1 is None or d2 is None:
        return None
    good = [a for a, b in cv2.BFMatcher().knnMatch(d2, d1, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None
    Hm, inl = cv2.findHomography(np.float32([k2[g.queryIdx].pt for g in good]), np.float32([k1[g.trainIdx].pt for g in good]), cv2.RANSAC, 4.0)
    return Hm if Hm is not None and inl.sum() >= 10 else None


def _protect_features(g, t, mask, grow=12, feather=6):
    """Inside the performer's eyes / brows / mouth, rebuild the light map from its surroundings instead of copying it
    from the relit render: a generative relight's own eyes and expression never print onto the plate.
    Coarse-to-fine normalized convolution: each finer scale is trusted only where it has enough surrounding samples."""
    m = (mask > 0.3).astype(np.uint8)
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), feather)[..., None]
    keepw = (1 - m).astype(np.float32)

    def fill(x):
        out = None
        for sig in (60, 30, 14, 6):
            w = cv2.GaussianBlur(keepw, (0, 0), sig)
            v = cv2.GaussianBlur(x * keepw, (0, 0), sig) / (w + 1e-4)
            if out is None:
                out = v
            else:
                c = np.clip(w * 3, 0, 1)
                out = v * c + out * (1 - c)
        return out

    lg = np.log(np.clip(g[..., 0], 1e-3, None)).astype(np.float32)
    gi = np.exp(fill(lg))[..., None]
    lt = np.log(np.clip(t, 1e-3, None)).astype(np.float32)
    ti = np.exp(np.dstack([fill(np.ascontiguousarray(lt[..., c])) for c in range(3)]))
    return g * (1 - mf) + gi * mf, t * (1 - mf) + ti * mf


def _guided_filter(guide, src, r, eps):
    """Grey-guide guided filter (He et al.), the same maths as cv2.ximgproc.guidedFilter without needing opencv-contrib."""
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mi, mp = box(guide), box(src)
    a = (box(guide * src) - mi * mp) / (box(guide * guide) - mi * mi + eps)
    b = mp - a * mi
    return box(a) * guide + box(b)


class LightTransfer:
    CATEGORY = "image/postprocessing"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "aligned_relit")
    OUTPUT_TOOLTIPS = ("The plate with only the relit render's light transferred onto it (every plate pixel kept).",
                       "The relit render itself, warped onto the plate's framing (generative relights re-frame the shot: zoom, shift, slow drift).")
    DESCRIPTION = ("Relight a plate without regenerating it: the plate keeps every pixel, only the light comes from the relit render. "
                   "Small-scale light ratio guided by the relit render's own light edges (crisp gobos), local tint, damped in deep shadow (no glow).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",), "relit": ("IMAGE",),
            "detail_sigma": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 16, "step": 0.5}),
            "guide_radius": ("INT", {"default": 8, "min": 1, "max": 64}),
            "shadow_damping": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
            "exposure": ("FLOAT", {"default": 1.2, "min": 0.25, "max": 4, "step": 0.05}),
            "align_relit": ("BOOLEAN", {"default": True, "tooltip": "Generative relights drift (scale, shift) and change expression. Align each relit frame to the plate first (SIFT homography on lighting-normalised images, one smooth quadratic path through time), or the relit eyes and features ghost onto the plate."}),
        }, "optional": {
            "protect_mask": ("MASK", {"tooltip": "Performer features to protect (e.g. SAM 3.1 'eye', 'eyebrow', 'mouth'). Inside it the light is filled in from the surrounding light, so the relit render's own eyes and expression never print onto the plate."}),
            "protect_hold_frames": ("INT", {"default": 8, "min": 0, "max": 30, "tooltip": "Hold the protect matte over +-N frames (running max): segmentation drops the eyes while they blink."}),
            "protect_grow": ("INT", {"default": 12, "min": 0, "max": 64}),
        }}

    def run(self, plate, relit, detail_sigma, guide_radius, shadow_damping, exposure, align_relit=True, protect_mask=None, protect_hold_frames=8, protect_grow=12):
        n = min(len(plate), len(relit))
        P, R = _np(plate[:n]), _np(relit[:n])
        H, W = P.shape[1:3]
        R = np.stack([r if r.shape[:2] == (H, W) else cv2.resize(r, (W, H)) for r in R])
        if align_relit:
            u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
            raw = [_estimate_h(u8(P[i]), u8(R[i])) for i in range(n)]
            ok = [h for h in raw if h is not None]
            fb = np.median(np.stack(ok), 0) if ok else np.eye(3)
            raw = [h / h[2, 2] if h is not None else fb for h in raw]
            # generative drift is slow: fit ONE smooth quadratic path through time (no frame-to-frame jitter)
            ref = np.float32([[W * .15, H * .15], [W * .85, H * .15], [W * .85, H * .85], [W * .15, H * .85]])
            tt = np.arange(n, dtype=np.float64)
            good = [i for i in range(n) if raw[i] is not fb]
            mp = np.stack([cv2.perspectiveTransform(ref.reshape(-1, 1, 2), raw[i]).reshape(-1, 2) for i in good]) if good else None
            Hs = []
            for i in range(n):
                if mp is None or len(good) < 3:
                    Hs.append(fb)
                    continue
                pts = np.zeros((4, 2), np.float32)
                for k in range(4):
                    for c in range(2):
                        pts[k, c] = np.polyval(np.polyfit(tt[good], mp[:, k, c], 2), i)
                Hs.append(cv2.getPerspectiveTransform(ref, pts))
            R = np.stack([cv2.warpPerspective(R[i], Hs[i], (W, H), borderMode=cv2.BORDER_REPLICATE) for i in range(n)])
        PM = None
        if protect_mask is not None:
            pm = _np(protect_mask)
            if pm.ndim == 2:
                pm = pm[None]
            pm = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) if x.shape != (H, W) else x for x in pm])
            hf = int(protect_hold_frames)
            PM = np.stack([pm[max(0, i - hf):i + hf + 1].max(0) for i in range(len(pm))])
        wts = np.array([0.2126, 0.7152, 0.0722], np.float32)
        out = []
        for i in range(n):
            pl = np.clip(P[i], 0, 1) ** 2.2
            rl = np.clip(R[i], 0, 1) ** 2.2
            Yp, Yr = pl @ wts, rl @ wts
            guide = np.log(cv2.GaussianBlur(Yr, (0, 0), 1.0) + 1e-3).astype(np.float32)
            gf = lambda x: _guided_filter(guide, x.astype(np.float32), guide_radius, 0.016)
            g = (cv2.GaussianBlur(Yr, (0, 0), detail_sigma) + shadow_damping) / (cv2.GaussianBlur(Yp, (0, 0), detail_sigma) + shadow_damping)
            g = np.exp(gf(np.log(np.clip(g, 1e-3, 4.0))))[..., None]
            ts = detail_sigma * 2.5
            cr = cv2.GaussianBlur(rl, (0, 0), ts) / (cv2.GaussianBlur(Yr, (0, 0), ts)[..., None] + 1e-2)
            cp = cv2.GaussianBlur(pl, (0, 0), ts * 4) / (cv2.GaussianBlur(Yp, (0, 0), ts * 4)[..., None] + 1e-2)
            t = np.log(np.clip(cr / (cp + 1e-3), 0.2, 4))
            t = np.exp(np.dstack([gf(t[..., c]) for c in range(3)]))
            if PM is not None and PM[min(i, len(PM) - 1)].max() > 0.05:
                g, t = _protect_features(g, t, PM[min(i, len(PM) - 1)], grow=protect_grow)
            out.append(np.clip(pl * g * t * exposure, 0, 1) ** (1 / 2.2))
        return (torch.from_numpy(np.stack(out).astype(np.float32)), torch.from_numpy(np.clip(R, 0, 1).astype(np.float32)))
