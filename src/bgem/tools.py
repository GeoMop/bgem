import logging
from time import perf_counter
from contextlib import contextmanager


@contextmanager
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

    def log_info(self, msg):
        logging.info(f"{msg} : T={str(self)}")

