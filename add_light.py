"""Add a virtual light to real footage, using AI passes only to shape the light.

    light = albedo * max(0, N.L)^softness * colour * intensity      (diffuse)
          + spec_mask * max(0, N.H)^shininess * colour * spec_amount (highlight)
    out   = plate + light * matte                                    (linear light)

The plate is never replaced: every original pixel is still there, the light is a plus on top.
Normals follow Marigold's camera-space convention: x right, y up, z towards the camera, stored as (n+1)/2.
"""
import math
import cv2
import numpy as np


def light_dir(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)], np.float32)


def cast_shadow(depth, L, depth_scale=0.002, max_px=160, step=2.0, softness=0.02, blur=2.5, depth_blur=2.0):
    """Screen-space cast shadows from a depth pass (bright = near, i.e. a height field facing the camera).
    From each pixel, march across the image toward the light; if the surface rises above the light ray anywhere,
    the pixel is in shadow. depth_scale = depth units per pixel (how far the face 'sticks out')."""
    h, w = depth.shape
    if depth_blur > 0:
        depth = cv2.GaussianBlur(depth, (0, 0), depth_blur)   # skin texture in the depth pass would speckle the shadows
    d2 = np.array([L[0], -L[1]], np.float32)          # image-space direction toward the light (y down)
    n2 = float(np.linalg.norm(d2))
    if n2 < 1e-3:
        return np.ones_like(depth)                     # light straight at camera: no visible cast shadows
    d2 /= n2
    rise = depth_scale * float(L[2]) / n2              # how much the light ray climbs per pixel travelled
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    occ = np.full_like(depth, -1e9)
    for s in np.arange(step, max_px + step, step, dtype=np.float32):
        smp = cv2.remap(depth, xx + d2[0] * s, yy + d2[1] * s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        occ = np.maximum(occ, smp - s * rise)
    lit = 1.0 - np.clip((occ - depth) / softness, 0, 1)
    return cv2.GaussianBlur(lit, (0, 0), blur) if blur > 0 else lit


def add_light(plate, albedo, normals, spec, matte, L, colour=(1.0, 0.78, 0.52), intensity=1.6,
              softness=1.0, spec_amount=0.6, shininess=24.0, matte_soft=2.0, normal_blur=1.5,
              depth=None, shadow_strength=1.0, shadow_depth_scale=0.002):
    """All images H,W,3 float32 sRGB [0..1]; matte H,W float [0..1]; depth H,W (bright = near) enables cast shadows. Returns sRGB."""
    lin = lambda x: np.clip(x, 0, 1) ** 2.2
    n = normals * 2 - 1
    if normal_blur > 0:
        n = cv2.GaussianBlur(n, (0, 0), normal_blur)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    ndl = np.clip(n @ L, 0, 1) ** softness
    h = L + np.array([0, 0, 1], np.float32); h /= np.linalg.norm(h)
    ndh = np.clip(n @ h, 0, 1) ** shininess
    smask = np.clip(lin(spec).mean(2) * 4, 0, 1)
    c = np.asarray(colour, np.float32)
    light = lin(albedo) * ndl[..., None] * c * intensity + (smask * ndh)[..., None] * c * spec_amount
    if depth is not None and shadow_strength > 0:
        sh = cast_shadow(depth.astype(np.float32), L, depth_scale=shadow_depth_scale)
        light = light * (1 - shadow_strength * (1 - sh))[..., None]
    m = cv2.GaussianBlur(matte.astype(np.float32), (0, 0), matte_soft) if matte_soft > 0 else matte
    out = lin(plate) + light * m[..., None]
    return np.clip(out, 0, 1) ** (1 / 2.2)
