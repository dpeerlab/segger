import numpy as np
from skimage.filters import threshold_li

def threshold_li_custom(arr: np.ndarray, max_iter: int = 100) -> float:
    """Fallback to StopIteration if can't converge. Not implemented in threshold_li."""
    n_iter = 0
    def _callback(threshold: float) -> None:
        nonlocal n_iter
        n_iter += 1
        if n_iter > max_iter:
            raise StopIteration

    return threshold_li(arr, iter_callback=_callback)