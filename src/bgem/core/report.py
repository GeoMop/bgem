from functools import wraps
import logging
import time

__report_indent_level = 0

def report(fn):
    @wraps(fn)
    def do_report(*args, **kwargs):
        global __report_indent_level
        __report_indent_level += 1
        init_time = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - init_time
        __report_indent_level -= 1
        indent = (__report_indent_level * 2) * " "
        logging.info(f"{indent}DONE {fn.__module__}.{fn.__name__} @ {duration}")
        return result
    return do_report

