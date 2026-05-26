"""
Common code for tests.
"""
import os
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

from bgem import stochastic
import numpy as np


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





fracture_stats = dict(
    NS={'concentration': 17.8,
     'p_32': 0.094,
     'plunge': 1,
     'power': 2.5,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 292},
    NE={'concentration': 14.3,
     'p_32': 0.163,
     'plunge': 2,
     'power': 2.7,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 326},
    NW={'concentration': 12.9,
     'p_32': 0.098,
     'plunge': 6,
     'power': 3.1,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 60},
    EW={'concentration': 14.0,
     'p_32': 0.039,
     'plunge': 2,
     'power': 3.1,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 15},
    HZ={'concentration': 15.2,
     'p_32': 0.141,
     'power': 2.38,
     'r_max': 564,
     'r_min': 0.038,
     #'trend': 5
     #'plunge': 86,
     'strike': 95,
     'dip': 4
     })

def get_dfn_sample(box_size=100, seed=123):
    # generate fracture set
    np.random.seed(seed)
    fracture_box = 3 * [box_size]
    # volume = np.product()
    pop = stochastic.Population.from_cfg(fracture_stats, fracture_box)
    # pop.initialize()
    pop = pop.set_range_from_size(sample_size=30)
    mean_size = pop.mean_size()
    print("total mean size: ", mean_size)
    pos_gen = stochastic.UniformBoxPosition(fracture_box)
    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    return fractures