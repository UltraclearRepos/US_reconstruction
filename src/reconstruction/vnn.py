from pathlib import Path

import numpy as np
import torch

from config import (
    VOXEL_SIZE,
    RECONSTRUCTION_MAX_PLANE_DISTANCE_MM,
    PIXEL_SPACING,
    BACKGROUND_THRESHOLD,
    CENTER,
    CHUNK_PIXELS,
    IMAGE_KEY,
    DEVICE,
)

from src.reconstruction.common import (
    load_image_and_poses,
    validate_max_plane_dist,
    is_binary_input_from_key,
    compute_volume_geometry,
    auto_chunk_pixels,
    make_world_points_block_flat,
    nearest_sample_numpy,
    apply_validity,
    finalize_and_save,
)


def voxel_nearest_reconstruct_torch(
    img,
    Ts,
    sx,
    sy,
    origin,
    voxel,
    background_threshold=None,
    center=True,
    chunk_pixels=None,
    binary_input=False,
    device_str="cuda",
    max_plane_dist=None,
):
    max_plane_dist = validate_max_plane_dist(max_plane_dist)
    device = torch.device(device_str)

    Tn, H, W = img.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    origin, voxel, Dx, Dy, Dz = compute_volume_geometry(
        Ts=Ts,
        W=W,
        H=H,
        sx=sx,
        sy=sy,
        voxel=voxel,
        center=center,
    )

    N = Dx * Dy * Dz

    if chunk_pixels is None or chunk_pixels <= 0:
        chunk_pixels = auto_chunk_pixels(Tn, device)

    print(
        f"[INFO] voxel_nearest reconstruction: "
        f"volume=(Dz={Dz}, Dy={Dy}, Dx={Dx}), voxels={N}, "
        f"chunk={chunk_pixels}, max_plane_dist={max_plane_dist} mm, device={device}"
    )

    Ts_t = torch.from_numpy(Ts).to(device=device, dtype=torch.float32)
    invTs_t = torch.linalg.inv(Ts_t).contiguous()

    out_flat = np.zeros((N,), dtype=np.float32)
    wts_flat = np.zeros((N,), dtype=np.float32)

    for s in range(0, N, chunk_pixels):
        e = min(s + chunk_pixels, N)

        pts = make_world_points_block_flat(origin, voxel, Dx, Dy, Dz, s, e)
        M = pts.shape[0]

        pts_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
        ones_t = torch.ones((M, 1), dtype=torch.float32, device=device)
        pts_h_t = torch.cat([pts_t, ones_t], dim=1)

        local = torch.einsum("tij,mj->mti", invTs_t, pts_h_t)

        x_local_all = local[..., 0]
        y_local_all = local[..., 1]
        abs_z_all = torch.abs(local[..., 2])

        nearest_dist_t, nearest_idx_t = torch.min(abs_z_all, dim=1)

        rows_t = torch.arange(M, device=device)
        x_local_t = x_local_all[rows_t, nearest_idx_t]
        y_local_t = y_local_all[rows_t, nearest_idx_t]

        if center:
            u_t = x_local_t / float(sx) + float(cx)
            v_t = y_local_t / float(sy) + float(cy)
        else:
            u_t = x_local_t / float(sx)
            v_t = y_local_t / float(sy)

        frame_idx = nearest_idx_t.detach().cpu().numpy().astype(np.int32)
        u = u_t.detach().cpu().numpy().astype(np.float32)
        v = v_t.detach().cpu().numpy().astype(np.float32)

        vals, inside = nearest_sample_numpy(img, frame_idx, u, v)
        if max_plane_dist is not None:
            within_distance = (nearest_dist_t <= max_plane_dist).detach().cpu().numpy()
            inside = inside & within_distance

        vals_use, valid = apply_validity(
            vals=vals,
            inside=inside,
            background_threshold=background_threshold,
            binary_input=binary_input,
        )

        out_flat[s:e] = vals_use
        wts_flat[s:e] = valid.astype(np.float32)

        print(f"[INFO] processed voxels: {e}/{N}")

    out = out_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()
    wts = wts_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()

    return out, wts, origin, voxel


def reconstruct_voxel_nearest_for_file(
    path,
    out_root,
    voxel=VOXEL_SIZE,
    pixel_spacing=PIXEL_SPACING,
    background_threshold=BACKGROUND_THRESHOLD,
    center=CENTER,
    chunk_pixels=CHUNK_PIXELS,
    image_key=IMAGE_KEY,
    device_str=DEVICE,
    max_plane_dist=RECONSTRUCTION_MAX_PLANE_DISTANCE_MM,
):
    path = Path(path)
    base = path.stem

    img, Ts, (sx_attr, sy_attr), resolved_image_key = load_image_and_poses(
        path,
        image_key=image_key,
    )

    sx = pixel_spacing[0] if pixel_spacing else sx_attr
    sy = pixel_spacing[1] if pixel_spacing else sy_attr

    is_mask = is_binary_input_from_key(resolved_image_key)

    if background_threshold is None and is_mask:
        background_threshold = 0.5

    out, wts, origin, voxel = voxel_nearest_reconstruct_torch(
        img=img,
        Ts=Ts,
        sx=sx,
        sy=sy,
        origin=None,
        voxel=voxel,
        background_threshold=background_threshold,
        center=center,
        chunk_pixels=chunk_pixels,
        binary_input=is_mask,
        device_str=device_str,
        max_plane_dist=max_plane_dist,
    )

    finalize_and_save(
        out_root=out_root,
        base=base,
        volume=out,
        weights=wts,
        voxel=voxel,
        origin=origin,
        method="voxel_nearest",
        extra={
            "max_plane_dist": np.float32(-1.0 if max_plane_dist is None else max_plane_dist),
        },
    )
