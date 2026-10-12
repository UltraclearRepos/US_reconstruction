"""Prepare the Linux runtime directory and Qt/OpenGL backends for viewers."""

import os
import stat
import sys
from pathlib import Path


def prepare_qt_environment():
    """Fix an owned runtime directory before Qt initializes; preserve overrides."""
    if sys.platform != "linux":
        return

    if (os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE") == "wayland"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    try:
        metadata = runtime_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            return
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            runtime_dir.chmod(0o700)
        os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
    except OSError:
        # Keep Qt's own diagnostic if the runtime directory cannot be repaired.
        return
