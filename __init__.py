#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dphutils.py
"""
This is for small utility functions that don't have a proper home yet

Copyright (c) 2016, David Hoffman
"""

import numpy as np
import scipy as sp
import re
import io
import requests
from skimage.external import tifffile as tif
from scipy.optimize import curve_fit
from scipy.ndimage.fourier import fourier_gaussian
from scipy.ndimage._ni_support import _normalize_sequence
from scipy.signal import signaltools as sig
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift, fftn, ifftn,
                                             rfftn, irfftn)
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    FFTW = True
except ImportError:
    from numpy.fft import (fftshift, ifftshift, fftn, ifftn,
                           rfftn, irfftn)
    FFTW = False
eps = np.finfo(float).eps


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.
    """
    operation = operation.lower()
    if operation not in {'sum', 'mean'}:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def scale(data, dtype=None):
    """
    Scales data to [0.0, 1.0] range, unless an integer dtype is specified
    in which case the data is scaled to fill the bit depth of the dtype.

    Parameters
    ----------
    data : numeric type
        Data to be scaled, can contain nan
    dtype : integer dtype
        Specify the bit depth to fill

    Returns
    -------
    scaled_data : numeric type
        Scaled data

    Examples
    --------
    >>> from numpy.random import randn
    >>> a = randn(10)
    >>> b = scale(a)
    >>> b.max()
    1.0
    >>> b.min()
    0.0
    >>> b = scale(a, dtype = np.uint16)
    >>> b.max()
    65535
    >>> b.min()
    0
    """
    if np.issubdtype(data.dtype, np.complex):
        raise TypeError("`scale` is not defined for complex values")
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if np.issubdtype(dtype, np.integer):
        tmin = np.iinfo(dtype).min
        tmax = np.iinfo(dtype).max
    else:
        tmin = 0.0
        tmax = 1.0
    return ((data - dmin) / (dmax - dmin) * (tmax - tmin) + tmin).astype(dtype)


def scale_uint16(data):
    """Convenience function to scale data to the uint16 range."""
    return scale(data, np.uint16)


def radial_profile(data, center=None, binsize=1.0):
    """Take the radial average of a 2D data array

    Taken from http://stackoverflow.com/a/21242776/5030014

    Parameters
    ----------
    data : ndarray (2D)
        the 2D array for which you want to calculate the radial average
    center : sequence
        the center about which you want to calculate the radial average

    Returns
    -------
    radialprofile : ndarray
        a 1D radial average of data

    Examples
    --------
    >>> radial_profile(np.ones((11, 11)))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    """
    # test if the data is complex
    if np.iscomplexobj(data):
        # if it is complex, call this function on the real and
        # imaginary parts and return the complex sum.
        real_prof, real_std = radial_profile(np.real(data), center, binsize)
        imag_prof, imag_std = radial_profile(np.imag(data), center, binsize)
        return real_prof + imag_prof * 1j, np.sqrt(real_std**2 + imag_std**2)
        # or do mag and phase
        # mag_prof, mag_std = radial_profile(np.abs(data), center, binsize)
        # phase_prof, phase_std = radial_profile(np.angle(data), center, binsize)
        # return mag_prof * np.exp(phase_prof * 1j), mag_std * np.exp(phase_std * 1j)
    # pull the data shape
    idx = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) // 2
    else:
        # make sure center is an array.
        center = np.asarray(center)
    # calculate the radius from center
    idx2 = idx - center[[Ellipsis] + [np.newaxis] * (data.ndim)]
    r = np.sqrt(np.sum([i**2 for i in idx2], 0))
    # convert to int
    r = np.round(r / binsize).astype(np.int)
    # sum the values at equal r
    tbin = np.bincount(r.ravel(), data.ravel())
    # sum the squares at equal r
    tbin2 = np.bincount(r.ravel(), (data**2).ravel())
    # find how many equal r's there are
    nr = np.bincount(r.ravel())
    # calculate the radial mean
    # NOTE: because nr could be zero (for missing bins) the results will
    # have NaN for binsize != 1
    radial_mean = tbin / nr
    # calculate the radial std
    radial_std = np.sqrt(tbin2 / nr - radial_mean**2)
    # return them
    return radial_mean, radial_std


def mode(data):
    """Quickly find the mode of data

    up to 1000 times faster than scipy mode
    but not nearly as feature rich

    Note: we can vectorize this to work on different
    axes with numba"""
    # will not work with negative numbers (for now)
    return np.bincount(data.ravel()).argmax()


def slice_maker(xs, ws):
    """
    A utility function to generate slices for later use.

    Parameters
    ----------
    y0 : int
        center y position of the slice
    x0 : int
        center x position of the slice
    width : int
        Width of the slice

    Returns
    -------
    slices : list
        A list of slice objects, the first one is for the y dimension and
        and the second is for the x dimension.

    Notes
    -----
    The method will automatically coerce slices into acceptable bounds.

    Examples
    --------
    >>> slice_maker((30,20),10)
    [slice(25, 35, None), slice(15, 25, None)]
    >>> slice_maker((30,20),25)
    [slice(18, 43, None), slice(8, 33, None)]
    """
    # normalize inputs
    xs = np.asarray(xs)
    ws = np.asarray(_normalize_sequence(ws, len(xs)))
    if not np.isrealobj((xs, ws)):
        raise TypeError("`slice_maker` only accepts real input")
    if np.any(ws < 0):
        raise ValueError("width cannot be negative, width = {}".format(ws))
    # ensure integers
    xs = np.rint(xs).astype(int)
    ws = np.rint(ws).astype(int)
    # use _calc_pad
    toreturn = []
    for x, w in zip(xs, ws):
        half2, half1 = _calc_pad(0, w)
        xstart = x - half1
        xend = x + half2
        assert xstart <= xend, "xstart > xend"
        if xend <= 0:
            xstart, xend = 0, 0
        # the max calls are to make slice_maker play nice with edges.
        toreturn.append(slice(max(0, xstart), xend))
    # return a list of slices
    return toreturn



def fft_pad(array, newshape=None, mode='median', **kwargs):
    """Pad an array to prep it for fft"""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(sig.fftpack.helper.next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def padding_slices(oldshape, newshape):
    """This function takes the old shape and the new shape and calculates
    the required padding or cropping.newshape

    Can be used to generate the slices needed to undo fft_pad above"""
    # generate pad widths from new shape
    padding = tuple(_calc_pad(o, n) if n is not None else _calc_pad(o, o)
                    for o, n in zip(oldshape, newshape))
    # Make a crop list, if any of the padding is negative
    slices = [_calc_crop(s1, s2) for s1, s2 in padding]
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices

# def easy_rfft(data, axes=None):
#     """utility method that includes fft shifting"""
#     return fftshift(
#         rfftn(
#             ifftshift(
#                 data, axes=axes
#             ), axes=axes
#         ), axes=axes)


# def easy_irfft(data, axes=None):
#     """utility method that includes fft shifting"""
#     return ifftshift(
#         irfftn(
#             fftshift(
#                 data, axes=axes
#             ), axes=axes
#         ), axes=axes)

# add np.pad docstring
fft_pad.__doc__ += np.pad.__doc__


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (2, 3)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    # if oldnum is odd and newnum is even
    # we want to pull things backward
    if oldnum % 2:
        pad1, pad2 = pad_s, pad_b
    else:
        pad1, pad2 = pad_b, pad_s
    return pad1, pad2


# If we have fftw installed than make a better fftconvolve
if FFTW:
    def fftconvolve(in1, in2, mode="same", threads=1):
        """Same as above but with pyfftw added in"""
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return np.array([])

        s1 = np.array(in1.shape)
        s2 = np.array(in2.shape)
        complex_result = (np.issubdtype(in1.dtype, complex) or
                          np.issubdtype(in2.dtype, complex))
        shape = s1 + s2 - 1

        # Check that input sizes are compatible with 'valid' mode
        if sig._inputs_swap_needed(mode, s1, s2):
            # Convolution is commutative; order doesn't have any effect on output
            in1, s1, in2, s2 = in2, s2, in1, s1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [sig.fftpack.helper.next_fast_len(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.
        if not complex_result and (sig._rfft_mt_safe or sig._rfft_lock.acquire(False)):
            try:
                sp1 = rfftn(in1, fshape, threads=threads)
                sp2 = rfftn(in2, fshape, threads=threads)
                ret = (irfftn(sp1 * sp2, fshape, threads=threads)[fslice].copy())
            finally:
                if not sig._rfft_mt_safe:
                    sig._rfft_lock.release()
        else:
            # If we're here, it's either because we need a complex result, or we
            # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
            # is already in use by another thread).  In either case, use the
            # (threadsafe but slower) SciPy complex-FFT routines instead.
            sp1 = fftn(in1, fshape, threads=threads)
            sp2 = fftn(in2, fshape, threads=threads)
            ret = ifftn(sp1 * sp2, threads=threads)[fslice].copy()
            if not complex_result:
                ret = ret.real

        if mode == "full":
            return ret
        elif mode == "same":
            return sig._centered(ret, s1)
        elif mode == "valid":
            return sig._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")
    #fftconvolve.__doc__ = "DPH Utils: " + sig.fftconvolve.__doc__
else:
    fftconvolve = sig.fftconvolve


def fftconvolve_fast(data, kernel, **kwargs):
    """A faster version of fft convolution

    In this case the kernel ifftshifted before FFT but the data is not.
    This can be done because the effect of fourier convolution is to 
    "wrap" around the data edges so whether we ifftshift before FFT
    and then fftshift after it makes no difference so we can skip the
    step entirely.
    """
    # TODO: add error checking like in the above and add functionality
    # for complex inputs. Also could add options for different types of
    # padding.
    dshape = np.array(data.shape)
    kshape = np.array(kernel.shape)
    # find maximum dimensions
    maxshape = np.max((dshape, kshape), 0)
    # calculate a nice shape
    fshape = [sig.fftpack.helper.next_fast_len(int(d)) for d in maxshape]
    # pad out with reflection
    pad_data = fft_pad(data, fshape, "reflect")
    # calculate padding
    padding = tuple(_calc_pad(o, n)
                    for o, n in zip(data.shape, pad_data.shape))
    # so that we can calculate the cropping, maybe this should be integrated
    # into `fft_pad` ...
    fslice = tuple(slice(s, -e) if e != 0 else slice(s, None)
                   for s, e in padding)
    if kernel.shape != pad_data.shape:
        # its been assumed that the background of the kernel has already been
        # removed and that the kernel has already been centered
        kernel = fft_pad(kernel, pad_data.shape, mode='constant')
    k_kernel = rfftn(ifftshift(kernel), pad_data.shape, **kwargs)
    k_data = rfftn(pad_data, pad_data.shape, **kwargs)
    convolve_data = irfftn(k_kernel * k_data, pad_data.shape, **kwargs)
    # return data with same shape as original data
    return convolve_data[fslice]


def win_nd(size, win_func=sp.signal.hann, **kwargs):
    """
    A function to make a multidimensional version of a window function

    Parameters
    ----------
    size : tuple of ints
        size of the output window
    win_func : callable
        Default is the Hanning window
    **kwargs : key word arguments to be passed to win_func

    Returns
    -------
    w : ndarray
        window function
    """
    ndim = len(size)
    newshapes = tuple([
        tuple([1 if i != j else k for i in range(ndim)])
        for j, k in enumerate(size)])

    # Initialize to return
    toreturn = 1.0

    # cross product the 1D windows together
    for newshape in newshapes:
        toreturn = toreturn * win_func(max(newshape), **kwargs
                                       ).reshape(newshape)

    # return
    return toreturn


def anscombe(data):
    """Apply Anscombe transform to data

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    return 2 * np.sqrt(data + 3 / 8)


def anscombe_inv(data):
    """Apply inverse Anscombe transform to data

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    part0 = 1 / 4 * data**2
    part1 = 1 / 4 * np.sqrt(3 / 2) / data
    part2 = -11 / 8 / (data**2)
    part3 = 5 / 8 * np.sqrt(3 / 2) / (data**3)
    return part0 + part1 + part2 + part3 - 1 / 8


def fft_gaussian_filter(img, sigma):
    """FFT gaussian convolution

    Parameters
    ----------
    img : ndarray
        Image to convolve with a gaussian kernel
    sigma : int or sequence
        The sigma(s) of the gaussian kernel in _real space_

    Returns
    -------
    filt_img : ndarray
        The filtered image
    """
    # This doesn't help agreement but it will make things faster
    # pull the shape
    s1 = np.array(img.shape)
    # s2 = np.array([int(s * 4) for s in _normalize_sequence(sigma, img.ndim)])
    shape = s1  # + s2 - 1
    # calculate a nice shape
    fshape = [sig.fftpack.helper.next_fast_len(int(d)) for d in shape]
    # pad out with reflection
    pad_img = fft_pad(img, fshape, "reflect")
    # calculate the padding
    padding = tuple(_calc_pad(o, n) for o, n in zip(img.shape, pad_img.shape))
    # so that we can calculate the cropping, maybe this should be integrated
    # into `fft_pad` ...
    fslice = tuple(slice(s, -e) if e != 0 else slice(s, None)
                   for s, e in padding)
    # fourier transfrom and apply the filter
    kimg = rfftn(pad_img, fshape)
    filt_kimg = fourier_gaussian(kimg, sigma, pad_img.shape[-1])
    # inverse FFT and return.
    return irfftn(filt_kimg, fshape)[fslice]

# def read_system_monitors(path):
#     data = pd.read_csv(path)
#     

def exponent(xdata, amp, rate, offset):
    """Utility function to fit nonlinearly"""
    return amp * np.exp(-rate * xdata) + offset


def _estimate_exponent_params(data, xdata):
    """utility to estimate sine params"""
    if data[0] > data[-1]:
        # decay
        offset = np.nanmin(data)
        data_corr = data - offset
        log_data_corr = np.log(data_corr)
        valid_pnts = np.isfinite(log_data_corr)
        m, b = np.polyfit(xdata[valid_pnts], log_data_corr[valid_pnts], 1)
        return np.nan_to_num((np.exp(b), -m, offset))
    else:
        amp, rate, offset = _estimate_exponent_params(-data, xdata)
        return np.array((-amp, rate, -offset))


def exponent_fit(data, xdata=None):
    """Utility function that fits data to the sine function

    Assumes evenaly spaced data.

    Parameters
    ----------
    data : ndarray (1d)
        data that can be modeled as a single frequency sinusoid
    periods : numeric
        Estimated number of periods the sine wave covers

    Returns
    -------
    popt : ndarray
        optimized parameters for the sine wave
        - amplitude
        - frequency
        - phase
        - offset
    pcov : ndarray
        covariance of optimized paramters
    """
    # only deal with finite data
    # NOTE: could use masked wave here.
    if xdata is None:
        xdata = np.arange(len(data))

    finite_pnts = np.isfinite(data)
    data_fixed = data[finite_pnts]
    xdata_fixed = xdata[finite_pnts]
    # we need at least 4 data points to fit
    if len(data_fixed) > 3:
        # we can't fit data with less than 4 points
        # make guesses
        pguess = _estimate_exponent_params(data_fixed, xdata_fixed)
        # The jacobian actually slows down the fitting my guess is there
        # aren't generally enough points to make it worthwhile
        return curve_fit(exponent, xdata_fixed, data_fixed, p0=pguess, maxfev=2000)
        # fix signs, we want phase to be positive always

        # popt, pcov = curve_fit(sine, x, data_fixed, p0=pguess,
        #                        Dfun=sine_jac, col_deriv=True)
    else:
        raise RuntimeError("Not enough good points to fit.")


def get_tif_urls(baseurl):
    """Return all the links from a webpage with .tif"""
    r = requests.get(baseurl)
    links = re.findall('"([^"]*\.tif)"', r.content.decode())
    head = "/".join(r.url.split("/")[:3])
    return [head + l for l in links]


def url_tifread(url):
    """Read a tif into memory from a url"""
    r = requests.get(url)
    return tif.imread(io.BytesIO(r.content))
