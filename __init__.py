#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dphutils.py
"""
This is for small utility functions that don't have a proper home yet

Copyright (c) 2016, David Hoffman
"""

import numpy as np
import scipy as sp
import warnings
from scipy.ndimage.fourier import fourier_gaussian
from scipy.ndimage.filters import gaussian_filter
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


def scale(data, dtype=None):
    """
    Scales data to 0 to 1 range

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
    """
    Scales data to uint16 range

    Examples
    --------
    >>> from numpy.random import randn
    >>> a = randn(10)
    >>> a.dtype
    dtype('float64')
    >>> b = scale_uint16(a)
    >>> b.dtype
    dtype('uint16')
    >>> b.max()
    65535
    >>> b.min()
    0
    """

    return scale(data, np.uint16)


def radial_profile(data, center=None, binsize=1.0):
    """
    Take the radial average of a 2D data array

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
        return real_prof + imag_prof * 1j, real_std + imag_std * 1j
    # pull the data shape
    idx = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) // 2
    # calculate the radius from center
    idx2 = (idx - (np.array(data.shape) // 2)[[Ellipsis] + [np.newaxis] * (data.ndim)])
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


def slice_maker(y0, x0, width):
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
    >>> slice_maker(30,20,10)
    [slice(25, 35, None), slice(15, 25, None)]
    >>> slice_maker(30,20,25)
    [slice(18, 43, None), slice(8, 33, None)]
    """
    # ensure integers
    y0, x0 = int(y0), int(x0)
    # calculate the start and end
    half1 = width // 2
    # we need two halves for uneven widths
    half2 = width - half1
    ystart = y0 - half1
    xstart = x0 - half1
    yend = y0 + half2
    xend = x0 + half2
    # the max calls are to make slice_maker play nice with edges.
    toreturn = [slice(max(0, ystart), yend), slice(max(0, xstart), xend)]

    # return a list of slices
    return toreturn


def fft_pad(array, pad_width=None, mode='median', **kwargs):
    """
    Pad an array to prep it for fft
    """
    # pull the old shape
    oldshape = array.shape
    if pad_width is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(sig.fftpack.helper.next_fast_len(n) for n in oldshape)
    else:
        if isinstance(pad_width, int):
            newshape = tuple(pad_width for n in oldshape)
        else:
            newshape = tuple(pad_width)
    # generate pad widths from new shape
    padding = tuple(_calc_pad(o, n) if n is not None else _calc_pad(o, o)
                    for o, n in zip(oldshape, newshape))
    # Make a crop list, if any of the padding is negative
    slices = [_calc_crop(s1, s2) for s1, s2 in padding]
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return np.pad(array[slices], padding, mode=mode, **kwargs)


# add np.pad docstring
fft_pad.__doc__ += np.pad.__doc__


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """
    We have three cases:
    - old number even new number even
    - old number odd new number even
    - old number odd new number odd
    - old number even new number odd

    >>> _calc_pad(10, 16)
    (3, 3)
    >>> _calc_pad(11, 16)
    (3, 2)
    >>> _calc_pad(11, 17)
    (3, 3)
    >>> _calc_pad(10, 17)
    (4, 3)
    >>> _calc_pad(17, 17)
    (0, 0)
    >>> _calc_pad(17, 10)
    (-4, -3)
    """

    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side
    pad1 = width // 2
    # calculate the other
    pad2 = width - pad1
    return (pad2, pad1)


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
    # if any of the sizes is 32 or smaller, revert to proper filter
    if any(s1 < 33):
        warnings.warn(("Input is small along a dimension,"
                       " will revert to `gaussian_filter`"))
        return gaussian_filter(img, sigma)
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
