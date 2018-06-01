# https://github.com/jni/llc-tools

import numba
import numpy as np
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable

import platform
if platform.system() == "Windows":
    raise RuntimeError("This doesn't work on Windows yet.")


def jit_filter_function(filter_function):
    """Decorator for use with scipy.ndimage.generic_filter."""
    jitted_function = numba.jit(filter_function, nopython=True)

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    return LowLevelCallable(wrapped.ctypes)


def jit_filter1d_function(filter_function):
    """Decorator for use with scipy.ndimage.generic_filter1d."""
    jitted_function = numba.jit(filter_function, nopython=True)

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), intp, voidptr))
    def wrapped(in_values_ptr, len_in, out_values_ptr, len_out, data):
        in_values = carray(in_values_ptr, (len_in,), dtype=float64)
        out_values = carray(out_values_ptr, (len_out,), dtype=float64)
        jitted_function(in_values, out_values)
        return 1
    return LowLevelCallable(wrapped.ctypes)


# @jit_filter_function
# def mode(values):
#     return np.bincount(values.astype(np.uint8).ravel()).argmax()
