from contextlib import contextmanager
from time import perf_counter


def log_io(stage, inputs=None, outputs=None):
    print(f"[{stage}]")
    for name, path in (inputs or {}).items():
        if path is not None:
            print(f"  INPUT  {name}: {path}")
    for name, path in (outputs or {}).items():
        if path is not None:
            print(f"  OUTPUT {name}: {path}")


@contextmanager
def timed_stage(stage):
    """Print elapsed wall-clock seconds, including time up to an error."""
    started_at = perf_counter()
    print(f"[TIME] {stage}: START", flush=True)
    status = "DONE"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        elapsed = perf_counter() - started_at
        print(f"[TIME] {stage}: {elapsed:.3f} s ({status})", flush=True)
