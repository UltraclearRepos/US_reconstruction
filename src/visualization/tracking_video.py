"""Animate an opaque exterior surface from recorded probe poses and export MP4.

Edit the settings below, then run: .venv/bin/python -m src.visualization.tracking_video
The last recorded pose is the common reference; no GT alignment is applied.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from config import ROOT, SEGMENTATION_MODEL_NAME, SEGMENTATION_RECONSTRUCTION_METHOD
from src.isosurface_extraction.extract import load_mesh_to_mapping
from src.utils.paths import (
    predicted_surface_dir, prepared_sweep_path, segmentation_reconstruction_path,
)


# Video settings. Optional paths can be set to Path("..."); None uses case paths.
CASE_NAME = "003_01"
MODEL_NAME = SEGMENTATION_MODEL_NAME
RECONSTRUCTION_METHOD = SEGMENTATION_RECONSTRUCTION_METHOD
CLASS_ID = 1

SURFACE_PATH = None
SWEEP_PATH = None
VOLUME_PATH = None
OUTPUT_PATH = None

DURATION_SECONDS = 12.0
FPS = 24
WIDTH = 1440
HEIGHT = 810


def validate_rigid(matrices, name):
    matrices = np.asarray(matrices, dtype=np.float64)
    if (matrices.shape[-2:] != (4, 4) or not np.isfinite(matrices).all()
            or not np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-5)):
        raise ValueError(f"{name}: expected finite homogeneous 4x4 matrices")
    rotation = matrices[..., :3, :3]
    if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise ValueError(f"{name}: expected rigid rotations without scale or reflection")
    return matrices


def load_scene(surface_path, sweep_path, volume_path):
    with np.load(surface_path, allow_pickle=False) as data:
        vertices = np.asarray(data["verts"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
        mesh_to_mapping = data["mesh_to_mapping"] if "mesh_to_mapping" in data else None
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Surface must contain finite XYZ vertices and valid triangle faces")
    if mesh_to_mapping is None:
        with np.load(volume_path, allow_pickle=False) as data:
            spacing_zyx = data["voxel_size"]
        mesh_to_mapping = load_mesh_to_mapping(volume_path, spacing_zyx)
        if mesh_to_mapping is None:
            raise ValueError("Surface has no mesh_to_mapping and source volume has no origin_xyz")
        print("[TRACKING] Legacy surface: recovered mesh_to_mapping from volume metadata", flush=True)
    mesh_to_mapping = validate_rigid(mesh_to_mapping, "mesh_to_mapping")
    with h5py.File(sweep_path, "r") as data:
        poses = validate_rigid(data["poses"][...], "poses")
    if len(poses) < 2:
        raise ValueError("At least two poses are required")
    # Rebase BOTH mesh and poses, so the last frame defines the identity pose.
    mapping_to_last = np.linalg.inv(poses[-1])
    mesh_to_last = mapping_to_last @ mesh_to_mapping
    vertices_last = vertices @ mesh_to_last[:3, :3].T + mesh_to_last[:3, 3]
    poses_last = mapping_to_last @ poses
    return vertices_last, faces, poses_last


def tracking_trajectory(poses, frame_count):
    """Smooth forward/backward replay; interpolate rotations on SO(3)."""
    source_times = np.arange(len(poses), dtype=float)
    phase = np.arange(frame_count) / frame_count
    positions = (0.5 - 0.5 * np.cos(2 * np.pi * phase)) * (len(poses) - 1)
    rotations = Slerp(source_times, Rotation.from_matrix(poses[:, :3, :3]))(positions)
    result = np.repeat(np.eye(4)[None], frame_count, axis=0)
    result[:, :3, :3] = rotations.as_matrix()
    for axis in range(3):
        result[:, axis, 3] = np.interp(positions, source_times, poses[:, axis, 3])
    return result, positions, rotations.as_euler("xyz", degrees=True)


def vtk_matrix(array):
    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy(np.asarray(array, dtype=np.float64).ravel())
    return matrix


def polydata(vertices, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(len(faces) + 1, dtype=np.int64) * 3, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.ravel()), deep=True),
    )
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def actor(source, color, opacity=1.0, line_width=1.0):
    mapper = vtk.vtkPolyDataMapper()
    if isinstance(source, vtk.vtkPolyData):
        mapper.SetInputData(source)
    else:
        mapper.SetInputConnection(source.GetOutputPort())
    mapper.ScalarVisibilityOff()
    result = vtk.vtkActor()
    result.SetMapper(mapper)
    result.GetProperty().SetColor(*color)
    result.GetProperty().SetOpacity(opacity)
    result.GetProperty().SetLineWidth(line_width)
    return result


def text(renderer, label, x, y, size=20, color=(0.88, 0.92, 0.98)):
    result = vtk.vtkTextActor()
    result.SetInput(label)
    result.SetPosition(x, y)
    result.GetTextProperty().SetFontFamilyToArial()
    result.GetTextProperty().SetFontSize(size)
    result.GetTextProperty().SetColor(*color)
    renderer.AddViewProp(result)
    return result


def set_camera(renderer, points):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)), 1)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + np.array([1.3, -1.7, 1.1]) * radius))
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    direction = center - np.asarray(camera.GetPosition())
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    aspect = renderer.GetRenderWindow().GetSize()[0] / renderer.GetRenderWindow().GetSize()[1]
    half_height = np.max(np.abs((points - center) @ up))
    half_width = np.max(np.abs((points - center) @ right))
    camera.SetParallelScale(max(half_height, half_width / aspect, 1) * 1.4)
    renderer.ResetCameraClippingRange()


def render_video(surface_path, sweep_path, volume_path, output, duration=12, fps=24,
                 width=1440, height=810):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4 export; install it or add it to PATH")
    if (not np.isfinite(duration) or duration <= 0 or fps < 1 or width < 600
            or height < 400 or width % 2 or height % 2):
        raise ValueError("Use positive duration/fps and even image dimensions >= 600x400")
    count = max(2, round(duration * fps))
    vertices, faces, poses = load_scene(surface_path, sweep_path, volume_path)
    trajectory, source_positions, euler_xyz = tracking_trajectory(poses, count)
    mesh = polydata(vertices, faces)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(width, height)
    probe_view = vtk.vtkRenderer()
    probe_view.SetBackground(0.025, 0.035, 0.055)
    window.AddRenderer(probe_view)
    moving_mesh = actor(mesh, (0.75, 0.58, 0.96), opacity=1.0)
    moving_mesh.GetProperty().SetInterpolationToPhong()
    moving_mesh.GetProperty().SetAmbient(0.2)
    moving_mesh.GetProperty().SetDiffuse(0.75)
    moving_mesh.GetProperty().SetSpecular(0.15)
    moving_mesh.GetProperty().SetSpecularPower(25)
    probe_view.AddActor(moving_mesh)
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(8, 8, 8)
    axes.AxisLabelsOff()
    probe_view.AddActor(axes)

    # Frame the complete trajectory once. The camera stays fixed so translations
    # and rotations visibly move the whole exterior mesh, as in a 3D viewer.
    mesh_bounds = np.array([[x, y, z] for x in (vertices[:, 0].min(), vertices[:, 0].max())
                            for y in (vertices[:, 1].min(), vertices[:, 1].max())
                            for z in (vertices[:, 2].min(), vertices[:, 2].max())])
    probe_bounds = np.concatenate([
        (mesh_bounds - pose[:3, 3]) @ pose[:3, :3] for pose in trajectory
    ])
    set_camera(probe_view, probe_bounds)

    text(probe_view, "POWIERZCHNIA 3D  /  TRACKING", 30, height - 50, 28)
    text(probe_view, "Widok z zewnatrz - automatyczny obrot i przesuniecie wzgledem sondy",
         30, height - 80, 18)
    status = text(probe_view, "", 30, 105, 19)
    pose_status = text(probe_view, "", 30, 45, 19)
    text(probe_view, "Odniesienie: ostatnia klatka mappingu", width - 445, 35, 17)

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_suffix(".png")
    temporary = tempfile.NamedTemporaryFile(suffix=".mp4", prefix="tracking_", dir=output.parent, delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(temporary_path)]
    try:
        with tempfile.TemporaryFile() as encoder_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=encoder_log)
            try:
                for index, pose in enumerate(trajectory):
                    inverse = np.linalg.inv(pose)
                    probe_transform = vtk_matrix(inverse)
                    moving_mesh.SetUserMatrix(probe_transform)
                    direction = "W PRZOD" if index < count / 2 else "POWROT"
                    status.SetInput(f"Odtwarzanie trackingu: {direction}\n"
                                    f"Klatka {source_positions[index] + 1:.1f} / {len(poses)}"
                                    f"     czas {index / fps:.1f} s")
                    tx, ty, tz = pose[:3, 3]
                    rx, ry, rz = euler_xyz[index]
                    pose_status.SetInput(f"Pozycja [mm]: {tx:+.1f}  {ty:+.1f}  {tz:+.1f}\n"
                                         f"Obrot XYZ [deg]: {rx:+.1f}  {ry:+.1f}  {rz:+.1f}")
                    probe_view.ResetCameraClippingRange()
                    window.Render()
                    capture.Modified()
                    capture.Update()
                    if index == count // 4:
                        writer = vtk.vtkPNGWriter()
                        writer.SetFileName(str(preview))
                        writer.SetInputConnection(capture.GetOutputPort())
                        writer.Write()
                    pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
                    pixels = pixels.reshape(height, width, 3)[::-1].copy()
                    process.stdin.write(pixels.tobytes())
                    if index % max(1, count // 8) == 0:
                        print(f"[TRACKING] Rendered {index + 1}/{count}", flush=True)
                process.stdin.close()
                if process.wait() != 0:
                    encoder_log.seek(0)
                    raise RuntimeError(encoder_log.read().decode(errors="replace"))
            finally:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        temporary_path.replace(output)
    finally:
        window.Finalize()
        temporary_path.unlink(missing_ok=True)
    print(f"[TRACKING] MP4: {output}\n[TRACKING] Preview: {preview}", flush=True)
    return output


def main():
    surface = SURFACE_PATH or (predicted_surface_dir(CASE_NAME, MODEL_NAME, RECONSTRUCTION_METHOD)
                               / f"class_{CLASS_ID}_surface.npz")
    sweep = SWEEP_PATH or prepared_sweep_path(CASE_NAME)
    volume = VOLUME_PATH or segmentation_reconstruction_path(
        CASE_NAME, MODEL_NAME, RECONSTRUCTION_METHOD,
    )
    output = OUTPUT_PATH or ROOT / "outputs" / "tracking" / f"{CASE_NAME}_tracking_surface.mp4"
    render_video(surface, sweep, volume, output,
                 duration=DURATION_SECONDS, fps=FPS, width=WIDTH, height=HEIGHT)


if __name__ == "__main__":
    main()
