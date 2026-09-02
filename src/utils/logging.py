def log_io(stage, inputs=None, outputs=None):
    print(f"[{stage}]")
    for name, path in (inputs or {}).items():
        if path is not None:
            print(f"  INPUT  {name}: {path}")
    for name, path in (outputs or {}).items():
        if path is not None:
            print(f"  OUTPUT {name}: {path}")
