from pathlib import Path

import numpy as np
import torch

try:
    import vtk
    from vtk.util import numpy_support as vtk_np
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False

from src.data_loading.sweep import load_sweep_h5


def load_image_and_poses(path, image_key=None):
    result = load_sweep_h5(path, image_key=image_key)
    print(f"[INFO] file='{Path(path).name}', image_key='{result[3]}', shape={result[0].shape}")
    return result


def is_binary_input_from_key(image_key):
    key_l = image_key.lower() if image_key is not None else ""
    return ("mask" in key_l) or ("seg" in key_l) or ("label" in key_l)


def compute_bounds_for_sequence(Ts, W, H, sx, sy, margin=5.0, center=True):
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    corners_uv = np.array(
        [
            [0, 0],
            [W - 1, 0],
            [0, H - 1],
            [W - 1, H - 1],
            [cx, cy],
        ],
        dtype=np.float32,
    )

    if center:
        xy = np.stack(
            [
                (corners_uv[:, 0] - cx) * sx,
                (corners_uv[:, 1] - cy) * sy,
            ],
            axis=1,
        )
    else:
        xy = np.stack(
            [
                corners_uv[:, 0] * sx,
                corners_uv[:, 1] * sy,
            ],
            axis=1,
        )

    pts_local = np.c_[
        xy,
        np.zeros((len(corners_uv), 1), dtype=np.float32),
        np.ones((len(corners_uv), 1), dtype=np.float32),
    ]

    pts_world = np.einsum("tij,pj->tpi", Ts, pts_local)[:, :, :3]

    pmin = pts_world.reshape(-1, 3).min(axis=0) - margin
    pmax = pts_world.reshape(-1, 3).max(axis=0) + margin

    return pmin.astype(np.float32), pmax.astype(np.float32)


def compute_volume_geometry(Ts, W, H, sx, sy, voxel, center=True, margin=5.0):
    pmin, pmax = compute_bounds_for_sequence(
        Ts=Ts,
        W=W,
        H=H,
        sx=sx,
        sy=sy,
        margin=margin,
        center=center,
    )

    voxel = np.asarray(voxel, dtype=np.float32)
    origin = pmin.astype(np.float32)

    size = np.maximum(np.ceil((pmax - origin) / voxel).astype(np.int32), 1)
    Dx, Dy, Dz = int(size[0]), int(size[1]), int(size[2])

    return origin, voxel, Dx, Dy, Dz


def auto_chunk_pixels(Tn, device):
    if isinstance(device, str):
        device = torch.device(device)

    target_bytes = 700 * 1024 * 1024 if device.type == "cuda" else 256 * 1024 * 1024
    bytes_per_voxel = max(24 * int(Tn), 256)

    chunk = target_bytes // bytes_per_voxel
    chunk = max(2000, int(chunk))
    chunk = min(chunk, 100_000)

    return chunk


def make_world_points_block_flat(origin, voxel, Dx, Dy, Dz, start, end):
    ox, oy, oz = map(float, origin)
    vx, vy, vz = map(float, voxel)

    idx = np.arange(start, end, dtype=np.int64)

    yz_stride = Dx * Dz

    y = idx // yz_stride
    rem = idx % yz_stride
    x = rem // Dz
    z = rem % Dz

    xs = ox + (x.astype(np.float32) + 0.5) * vx
    ys = oy + (y.astype(np.float32) + 0.5) * vy
    zs = oz + (z.astype(np.float32) + 0.5) * vz

    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def save_as_vti(path_vti, vol_zyx, spacing_xyz, origin_xyz, array_name="intensity"):
    if not VTK_AVAILABLE:
        print("[WARN] vtk niedostępny — pomijam zapis .vti")
        return

    path_vti = Path(path_vti)
    path_vti.parent.mkdir(parents=True, exist_ok=True)

    Dz, Dy, Dx = vol_zyx.shape

    img = vtk.vtkImageData()
    img.SetDimensions(Dx, Dy, Dz)
    img.SetSpacing(*map(float, spacing_xyz))
    img.SetOrigin(*map(float, origin_xyz))

    flat = np.ascontiguousarray(vol_zyx.transpose(2, 1, 0)).ravel(order="F")

    vtk_arr = vtk_np.numpy_to_vtk(
        num_array=flat,
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_arr.SetName(array_name)

    img.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path_vti))
    writer.SetInputData(img)
    writer.Write()


def save_reconstruction_npz(
    path_npz,
    volume,
    weights,
    voxel_size,
    origin,
    method,
    extra=None,
):
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "volume": volume.astype(np.float32),
        "weights": weights.astype(np.float32),
        # NumPy volume axes are (z, y, x); keep spacing in the same order.
        "voxel_size": np.asarray(voxel_size, dtype=np.float32)[::-1],
        "origin_xyz": np.asarray(origin, dtype=np.float32),
        "method": np.array(method),
    }

    if extra:
        for k, v in extra.items():
            payload[k] = v

    np.savez_compressed(path_npz, **payload)


def finalize_and_save(
    out_root,
    base,
    volume,
    weights,
    voxel,
    origin,
    method,
    is_mask=False,
    extra=None,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    array_name = "mask" if is_mask else "intensity"

    vti_path = out_root / f"{base}.vti"
    npz_path = out_root / f"{base}.npz"

    save_as_vti(
        path_vti=vti_path,
        vol_zyx=volume,
        spacing_xyz=voxel,
        origin_xyz=origin,
        array_name=array_name,
    )

    save_reconstruction_npz(
        path_npz=npz_path,
        volume=volume,
        weights=weights,
        voxel_size=voxel,
        origin=origin,
        method=method,
        extra=extra,
    )

    print(f"[OK] {base} [{method}] -> {vti_path.name} / {npz_path.name}")


def nearest_sample_numpy(img, frame_idx, u, v):
    """
    img: (T,H,W)
    frame_idx, u, v: 1D arrays o tej samej długości.
    """
    Tn, H, W = img.shape

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    inside = (
        (frame_idx >= 0)
        & (frame_idx < Tn)
        & (ui >= 0)
        & (ui < W)
        & (vi >= 0)
        & (vi < H)
    )

    vals = np.zeros_like(u, dtype=np.float32)

    idx = np.flatnonzero(inside)
    if idx.size > 0:
        vals[idx] = img[frame_idx[idx], vi[idx], ui[idx]]

    return vals, inside


def bilinear_sample_grouped_numpy(img, frame_idx, u, v):
    """
    Grupowane bilinear sampling po klatkach.
    """
    Tn, H, W = img.shape

    inside = (
        (frame_idx >= 0)
        & (frame_idx < Tn)
        & (u >= 0.0)
        & (u <= W - 1)
        & (v >= 0.0)
        & (v <= H - 1)
    )

    vals = np.zeros_like(u, dtype=np.float32)

    valid_ids = np.flatnonzero(inside)
    if valid_ids.size == 0:
        return vals, inside

    frame_valid = frame_idx[valid_ids]
    unique_frames = np.unique(frame_valid)

    for fi in unique_frames:
        loc = valid_ids[frame_valid == fi]

        uu = u[loc]
        vv = v[loc]

        u0 = np.floor(uu).astype(np.int32)
        v0 = np.floor(vv).astype(np.int32)

        u1 = np.clip(u0 + 1, 0, W - 1)
        v1 = np.clip(v0 + 1, 0, H - 1)

        du = (uu - u0).astype(np.float32)
        dv = (vv - v0).astype(np.float32)

        frame = img[int(fi)]

        Ia = frame[v0, u0].astype(np.float32)
        Ib = frame[v0, u1].astype(np.float32)
        Ic = frame[v1, u0].astype(np.float32)
        Id = frame[v1, u1].astype(np.float32)

        wa = (1.0 - du) * (1.0 - dv)
        wb = du * (1.0 - dv)
        wc = (1.0 - du) * dv
        wd = du * dv

        vals[loc] = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return vals, inside


def apply_validity(vals, inside, background_threshold, binary_input):
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

    return vals_use, valid
