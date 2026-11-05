"""Bounded, ordered host prefetch for immutable feature caches."""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def enabled():
    return os.environ.get('FEDPASA_IO_OPTIMIZE') == '1'


def batches(arrays, indices, size):
    """Yield exactly the old batches, with at most one read ahead.

    The background thread performs NumPy reads only, never CUDA operations.
    Sparse mmap access disables speculative read-ahead of unselected images.
    """
    if enabled() and len(indices) > 1 and (indices[-1] - indices[0] + 1) > 1.5 * len(indices):
        import mmap
        for array in arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and hasattr(mapping, 'madvise'):
                mapping.madvise(mmap.MADV_RANDOM)
    def fetch(start):
        ids = indices[start:start + size]
        return start, ids, tuple(np.asarray(a[ids], dtype=np.float32) for a in arrays)
    if not enabled():
        for start in range(0, len(indices), size):yield fetch(start)
        return
    if not len(indices):return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='cache-prefetch') as pool:
        future = pool.submit(fetch, 0)
        for start in range(0, len(indices), size):
            batch = future.result()
            if start + size < len(indices):future = pool.submit(fetch, start + size)
            yield batch


def release_host_memory():
    """Release freed large CPU buffers before entering the patch-cache phase."""
    if not enabled():return
    import gc
    gc.collect()
    if os.name == 'posix':
        import ctypes
        libc = ctypes.CDLL(None)
        trim = getattr(libc, 'malloc_trim', None)
        if trim is not None:trim(0)
