from contextlib import contextmanager
from time import perf_counter


def format_duration(seconds, style="human"):
    if style == "days":
        return f"{seconds / 86400:.6f} days"

    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):02d}:{remaining_seconds:05.2f}m"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(remaining_minutes):02d}:{remaining_seconds:05.2f}h"


@contextmanager
def timed(label, results=None, style="human"):
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    if results is not None:
        results[label] = elapsed
    print(f"{label}: {format_duration(elapsed, style=style)}")
