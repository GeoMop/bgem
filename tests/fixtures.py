"""
Common code for tests.
"""
import os
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

def sandbox_fname(base_name, ext):
    work_dir = "sandbox"
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(work_dir, f"{base_name}.{ext}")

# Timing context manager


#@contextmanager
class catch_time(object):
    """
    Usage:
    with catch_time() as t:
        ...
    print(f"... time: {t}")
    """
    def __enter__(self):
        self.t = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.t = perf_counter() - self.t

    def __str__(self):
        return f"{self.t:.4f} s"

    def __repr__(self):
        return str(self)



