"""
Common code for tests.
"""
import os
from pathlib import Path

def sandbox_fname(base_name, ext):
    work_dir = "sandbox"
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(work_dir, f"{base_name}.{ext}")