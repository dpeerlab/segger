from skimage.filters import threshold_li

def threshold_li_custom(arr, max_iter=100):
    """Fallback to StopIteration if can't converge. Not implemented in threshold_li."""
    n_iter = 0
    def _callback(threshold):
        nonlocal n_iter
        n_iter += 1
        if n_iter > max_iter:
            raise StopIteration

    return threshold_li(arr, iter_callback=_callback)