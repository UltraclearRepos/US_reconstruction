from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates, spline_filter

from config import (
    VOXEL_SIZE,
    PIXEL_SPACING,
    BACKGROUND_THRESHOLD,
    CENTER,
    CHUNK_PIXELS,
    IMAGE_KEY,
    N_NEAREST,
    SPLINE_ORDER,
    DISTANCE_EPS,
    DISTANCE_POWER,
    DEVICE,
)

from src.reconstruction.common import (
    load_image_and_poses,
    is_binary_input_from_key,
    compute_volume_geometry,
    auto_chunk_pixels,
    make_world_points_block_flat,
    finalize_and_save,
)


def prepare_spline_frames(img, order):
    """
    Przygotowuje klatki 2D do B-spline interpolation.

    img: (T, H, W)
    order:
      0, 1 -> bez spline_filter
      2-5  -> scipy spline_filter
    """
    if order <= 1:
        return img.astype(np.float32)

    out = np.empty_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        out[i] = spline_filter(
            img[i].astype(np.float32),
            order=order,
        )

    return out


def sample_bspline_grouped(
    spline_frames,
    frame_idx_flat,
    u_flat,
    v_flat,
    inside_flat,
    spline_order,
):
    """
    Grupowane próbkowanie B-spline po numerze klatki.

    spline_frames: (T, H, W)
    frame_idx_flat: (M * K,)
    u_flat: (M * K,)
    v_flat: (M * K,)
    inside_flat: (M * K,)
    """
    vals_flat = np.zeros_like(u_flat, dtype=np.float32)

    valid_ids = np.flatnonzero(inside_flat)
    if valid_ids.size == 0:
        return vals_flat

    frame_valid = frame_idx_flat[valid_ids]
    unique_frames = np.unique(frame_valid)

    for fi in unique_frames:
        loc = valid_ids[frame_valid == fi]

        coords = np.vstack(
            [
                v_flat[loc],
                u_flat[loc],
            ]
        ).astype(np.float32)

        sampled = map_coordinates(
            spline_frames[int(fi)],
            coords,
            order=spline_order,
            mode="nearest",
            prefilter=False if spline_order > 1 else True,
        ).astype(np.float32)

        vals_flat[loc] = sampled

    return vals_flat


def bspline_reconstruct_hybrid_torch(
    img,
    Ts,
    sx,
    sy,
    voxel,
    background_threshold=None,
    center=True,
    chunk_pixels=None,
    binary_input=False,
    n_nearest=4,
    spline_order=3,
    distance_eps=1e-6,
    distance_power=1.0,
    device_str="cuda",
):
    """
    Hybrid GPU/CPU B-spline reconstruction.

    GPU:
      - transformacja punktów świata do lokalnych układów klatek,
      - wybór n_nearest najbliższych płaszczyzn po |z_local|.

    CPU:
      - scipy B-spline sampling,
      - agregacja wartości.
    """
    device = torch.device(device_str)

    Tn, H, W = img.shape

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    n_nearest = max(1, min(int(n_nearest), Tn))
    spline_order = int(np.clip(spline_order, 0, 5))

    origin, voxel, Dx, Dy, Dz = compute_volume_geometry(
        Ts=Ts,
        W=W,
        H=H,
        sx=sx,
        sy=sy,
        voxel=voxel,
        center=center,
        margin=5.0,
    )

    N = Dx * Dy * Dz

    if chunk_pixels is None or chunk_pixels <= 0:
        chunk_pixels = auto_chunk_pixels(Tn, device)

    print(
        f"[INFO] B-spline reconstruction: "
        f"volume=(Dz={Dz}, Dy={Dy}, Dx={Dx}), "
        f"voxels={N}, chunk={chunk_pixels}, "
        f"n_nearest={n_nearest}, spline_order={spline_order}, "
        f"device={device}"
    )

    Ts_t = torch.from_numpy(Ts).to(device=device, dtype=torch.float32)
    invTs_t = torch.linalg.inv(Ts_t).contiguous()

    spline_frames = prepare_spline_frames(
        img=img,
        order=spline_order,
    )

    out_flat = np.zeros((N,), dtype=np.float32)
    wts_flat = np.zeros((N,), dtype=np.float32)

    for s in range(0, N, chunk_pixels):
        e = min(s + chunk_pixels, N)

        pts = make_world_points_block_flat(
            origin=origin,
            voxel=voxel,
            Dx=Dx,
            Dy=Dy,
            Dz=Dz,
            start=s,
            end=e,
        )

        M = pts.shape[0]

        pts_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
        ones_t = torch.ones((M, 1), dtype=torch.float32, device=device)
        pts_h_t = torch.cat([pts_t, ones_t], dim=1)

        local = torch.einsum("tij,mj->mti", invTs_t, pts_h_t)

        x_local_all = local[..., 0]
        y_local_all = local[..., 1]
        abs_z_all = torch.abs(local[..., 2])

        nearest_dist_t, nearest_idx_t = torch.topk(
            abs_z_all,
            k=n_nearest,
            dim=1,
            largest=False,
            sorted=False,
        )

        x_local_t = torch.gather(
            x_local_all,
            dim=1,
            index=nearest_idx_t,
        )

        y_local_t = torch.gather(
            y_local_all,
            dim=1,
            index=nearest_idx_t,
        )

        if center:
            u_t = x_local_t / float(sx) + float(cx)
            v_t = y_local_t / float(sy) + float(cy)
        else:
            u_t = x_local_t / float(sx)
            v_t = y_local_t / float(sy)

        inside_t = (
            (u_t >= 0.0)
            & (u_t <= (W - 1))
            & (v_t >= 0.0)
            & (v_t <= (H - 1))
        )

        nearest_idx = nearest_idx_t.detach().cpu().numpy().astype(np.int32, copy=False)
        nearest_dist = nearest_dist_t.detach().cpu().numpy().astype(np.float32, copy=False)
        u = u_t.detach().cpu().numpy().astype(np.float32, copy=False)
        v = v_t.detach().cpu().numpy().astype(np.float32, copy=False)
        inside = inside_t.detach().cpu().numpy()

        frame_idx_flat = nearest_idx.reshape(-1)
        dist_flat = nearest_dist.reshape(-1)
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)
        inside_flat = inside.reshape(-1)

        vals_flat = sample_bspline_grouped(
            spline_frames=spline_frames,
            frame_idx_flat=frame_idx_flat,
            u_flat=u_flat,
            v_flat=v_flat,
            inside_flat=inside_flat,
            spline_order=spline_order,
        )

        vals = vals_flat.reshape(M, n_nearest)
        z_abs = dist_flat.reshape(M, n_nearest)

        if binary_input:
            thr = 0.5 if background_threshold is None else background_threshold
            valid = inside & (vals > thr)
            vals_use = valid.astype(np.float32)
        else:
            if background_threshold is None:
                valid = inside
            else:
                valid = inside & (vals > background_threshold)

            vals_use = np.where(valid, vals, 0.0).astype(np.float32)

        w = np.zeros((M, n_nearest), dtype=np.float32)

        w[valid] = 1.0 / np.power(
            z_abs[valid] + float(distance_eps),
            float(distance_power),
        )

        sum_vals = np.sum(vals_use * w, axis=1, dtype=np.float32)
        sum_wts = np.sum(w, axis=1, dtype=np.float32)

        nz = sum_wts > 0

        block_out = np.zeros((M,), dtype=np.float32)
        block_out[nz] = sum_vals[nz] / sum_wts[nz]

        out_flat[s:e] = block_out
        wts_flat[s:e] = sum_wts

        print(f"[INFO] processed voxels: {e}/{N}")

    out = out_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()
    wts = wts_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()

    return out, wts, origin, voxel


def reconstruct_bspline_for_file(
    path,
    out_root,
    voxel=VOXEL_SIZE,
    pixel_spacing=PIXEL_SPACING,
    background_threshold=BACKGROUND_THRESHOLD,
    center=CENTER,
    chunk_pixels=CHUNK_PIXELS,
    image_key=IMAGE_KEY,
    n_nearest=N_NEAREST,
    spline_order=SPLINE_ORDER,
    distance_eps=DISTANCE_EPS,
    distance_power=DISTANCE_POWER,
    device_str=DEVICE,
):
    """
    Rekonstrukcja B-spline dla jednego pliku H5.

    Input H5:
      img   -> (N,H,W)
      poses -> (N,4,4)

    Output:
      .npz
      .vti
    """
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
        print("[INFO] Binary mask detected -> background_threshold=0.5")

    out, wts, origin, voxel = bspline_reconstruct_hybrid_torch(
        img=img,
        Ts=Ts,
        sx=sx,
        sy=sy,
        voxel=voxel,
        background_threshold=background_threshold,
        center=center,
        chunk_pixels=chunk_pixels,
        binary_input=is_mask,
        n_nearest=n_nearest,
        spline_order=spline_order,
        distance_eps=distance_eps,
        distance_power=distance_power,
        device_str=device_str,
    )

    finalize_and_save(
        out_root=out_root,
        base=base,
        volume=out,
        weights=wts,
        voxel=voxel,
        origin=origin,
        method="bspline",
        is_mask=is_mask,
        extra={
            "n_nearest": np.int32(n_nearest),
            "spline_order": np.int32(spline_order),
            "distance_power": np.float32(distance_power),
            "distance_eps": np.float32(distance_eps),
        },
    )
