import logging
from time import perf_counter


# @contextmanager
class catch_time:
    """
    Usage:
    with catch_time() as t:
        ...
    print(f"... time: {t}")
    """

    def __init__(self, msg):
        logging.info(f"{msg} ...")

    def __enter__(self):
        self.t = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.t = perf_counter() - self.t
        logging.info(f" : T={str(self)}")

    def __str__(self):
        return f"{self.t:.4f} s"

    def __repr__(self):
        return str(self)


def func_timer(func):
    """
    Measure time of the function call and report to logging.info
    """

    def _f(*args, **kwargs):
        with catch_time(f"{func.__name__}"):
            func(*args, **kwargs)

    return _f
