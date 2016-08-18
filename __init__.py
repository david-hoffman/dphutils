# -*- coding: utf-8 -*-

'''
This is for small utility functions that don't have a proper home yet
'''

import numpy as np
import numexpr as ne
import scipy.signal as sig
from scipy.ndimage.fourier import fourier_gaussian
from scipy.signal.signaltools import (_rfft_lock, _rfft_mt_safe,
                                      _inputs_swap_needed, _centered)
from scipy.fftpack.helper import next_fast_len
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import (ifftshift, fftshift, fftn, ifftn,
                                             rfftn, irfftn)
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import (ifftshift, fftshift, fftn, ifftn,
                           rfftn, irfftn)
# import unitary fourier transforms
from .uft import urfftn, uirfftn
eps = np.finfo(float).eps


def scale(data, dtype=None):
    '''
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
    '''

    dmin = data.min()
    dmax = data.max()

    if dtype is None:
        tmin = 0.0
        tmax = 1.0
    else:
        if np.issubdtype(dtype, np.integer):
            tmin = np.iinfo(dtype).min
            tmax = np.iinfo(dtype).max
        else:
            tmin = np.finfo(dtype).min
            tmax = np.finfo(dtype).max

    return ((data - dmin) / (dmax - dmin) * (tmax - tmin) + tmin).astype(dtype)


def scale_uint16(data):
    '''
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
    '''

    return (scale(data) * (2**16 - 1)).astype('uint16')


def radial_profile(data, center=None, binsize=1.0):
    '''
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
    >>> radial_profile(np.ones((11,11)),(5,5))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    '''
    # test if the data is complex
    if np.iscomplexobj(data):
        # if it is complex, call this function on the real and
        # imaginary parts and return the complex sum.
        real_prof, real_std = radial_profile(np.real(data), center, binsize)
        imag_prof, imag_std = radial_profile(np.imag(data), center, binsize)
        return real_prof + imag_prof * 1j, real_std + imag_std * 1j
    # pull the data shape
    y, x = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) / 2
    # split the cetner
    y0, x0 = center
    # calculate the radius from center
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
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
    '''
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
    '''

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


def nextpow2(n):
    '''
    Returns the next power of 2 for a given number

    Parameters
    ----------
    n : int
        The number for which you want to know the next power of two

    Returns
    -------
    m : int

    Examples
    --------
    >>> nextpow2(10)
    16
    '''

    if n < 0 or not isinstance(n, int):
        raise ValueError('n must be a positive integer, n = {}'.format(n))

    return 1 << (n - 1).bit_length()


def fft_pad(array, pad_width=None, mode='median', **kwargs):
    '''
    Parameters
    ----------
    array : array_like of rank N
        Input array
    pad_width : {sequence, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode : {str, function}
        One of the following string values or a user supplied function.

        'constant'
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        <function>
            Padding function, see Notes.
    stat_length : {sequence, int}, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ((before_1, after_1), ... (before_N, after_N)) unique statistic
        lengths for each axis.

        ((before, after),) yields same before and after statistic lengths
        for each axis.

        (stat_length,) or int is a shortcut for before = after = statistic
        length for all axes.

        Default is ``None``, to use the entire axis.
    constant_values : {sequence, int}, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ((before_1, after_1), ... (before_N, after_N)) unique pad constants
        for each axis.

        ((before, after),) yields same before and after constants for each
        axis.

        (constant,) or int is a shortcut for before = after = constant for
        all axes.

        Default is 0.
    end_values : {sequence, int}, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ((before_1, after_1), ... (before_N, after_N)) unique end values
        for each axis.

        ((before, after),) yields same before and after end values for each
        axis.

        (constant,) or int is a shortcut for before = after = end value for
        all axes.

        Default is 0.
    reflect_type : str {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extented part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    .. versionadded:: 1.7.0

    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes.  This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should return a rank 1 array equal in
    length to the vector argument with padded values replaced. It has the
    following signature::

        padding_func(vector, iaxis_pad_width, iaxis, **kwargs)

    where

        vector : ndarray
            A rank 1 array already padded with zeros.  Padded values are
            vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].
        iaxis_pad_width : tuple
            A 2-tuple of ints, iaxis_pad_width[0] represents the number of
            values padded at the beginning of vector where
            iaxis_pad_width[1] represents the number of values padded at
            the end of vector.
        iaxis : int
            The axis currently being calculated.
        kwargs : misc
            Any keyword arguments the function requires.


    '''
    # pull the old shape
    oldshape = array.shape

    if pad_width is None:
        # update each dimenstion to next power of two
        newshape = tuple([nextpow2(n) for n in oldshape])
    else:
        if isinstance(pad_width, int):
            newshape = tuple([pad_width for n in oldshape])
        else:
            newshape = tuple(pad_width)

    # generate pad widths from new shape

    padding = tuple([_calc_pad(o, n) for o, n in zip(oldshape, newshape)])

    return np.pad(array, padding, mode=mode, **kwargs)


def _calc_pad(oldnum, newnum):
    '''
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
    '''

    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side
    pad1 = width // 2
    # calculate the other
    pad2 = width - pad1

    return (pad2, pad1)

'''
Examples
--------
>>> a = [1, 2, 3, 4, 5]
>>> np.lib.pad(a, (2,3), 'constant', constant_values=(4,6))
array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])

>>> np.lib.pad(a, (2,3), 'edge')
array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])

>>> np.lib.pad(a, (2,3), 'linear_ramp', end_values=(5,-4))
array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

>>> np.lib.pad(a, (2,), 'maximum')
array([5, 5, 1, 2, 3, 4, 5, 5, 5])

>>> np.lib.pad(a, (2,), 'mean')
array([3, 3, 1, 2, 3, 4, 5, 3, 3])

>>> np.lib.pad(a, (2,), 'median')
array([3, 3, 1, 2, 3, 4, 5, 3, 3])

>>> a = [[1,2], [3,4]]
>>> np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')
array([[1, 1, 1, 2, 1, 1, 1],
       [1, 1, 1, 2, 1, 1, 1],
       [1, 1, 1, 2, 1, 1, 1],
       [1, 1, 1, 2, 1, 1, 1],
       [3, 3, 3, 4, 3, 3, 3],
       [1, 1, 1, 2, 1, 1, 1],
       [1, 1, 1, 2, 1, 1, 1]])

>>> a = [1, 2, 3, 4, 5]
>>> np.lib.pad(a, (2,3), 'reflect')
array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

>>> np.lib.pad(a, (2,3), 'reflect', reflect_type='odd')
array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

>>> np.lib.pad(a, (2,3), 'symmetric')
array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

>>> np.lib.pad(a, (2,3), 'symmetric', reflect_type='odd')
array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

>>> np.lib.pad(a, (2,3), 'wrap')
array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

>>> def padwithtens(vector, pad_width, iaxis, kwargs):
...     vector[:pad_width[0]] = 10
...     vector[-pad_width[1]:] = 10
...     return vector

>>> a = np.arange(6)
>>> a = a.reshape((2,3))

>>> np.lib.pad(a, 2, padwithtens)
array([[10, 10, 10, 10, 10, 10, 10],
       [10, 10, 10, 10, 10, 10, 10],
       [10, 10,  0,  1,  2, 10, 10],
       [10, 10,  3,  4,  5, 10, 10],
       [10, 10, 10, 10, 10, 10, 10],
       [10, 10, 10, 10, 10, 10, 10]])
'''


class Pupil(object):

    '''
    A class defining the pupil function and its closely related methods

    Eventually will be extended to use for pupil reconstruction

    [(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
    Phase-Retrieved Pupil Functions in Wide-Field Fluorescence Microscopy.
    Journal of Microscopy 2004, 216 (1), 32–48.](dx.doi.org/10.1111/j.0022-2720.2004.01393.x)
    '''

    def __init__(self, k_max=1 / 97.5, wl=600, NA=0.85, n=1.0, size=512):
        # We'll be doing a lot of work in k-space so we need to know kmax,
        # which is inverse pixel size. All dimensional units are in nanometer
        # Create attributes k_max
        self.k_max = k_max
        self.wl = wl
        self.NA = NA
        self.n = n
        self.size = size
        self.gen_kr()
        # initialize pupil

    def gen_kr(self):
        # we're generating complex data in k-space which means the total
        # bandwidth is k_max, but the positive max is half that
        k = np.linspace(-self.k_max / 2, self.k_max / 2, self.size)

        kyy, kxx = np.meshgrid(k, k)

        kr = np.sqrt(kxx**2 + kyy**2)

        self.kr = kr
        self.phi = np.arctan2(kyy, kxx)  # Azimuthal angle

    def gen_pupil(self):
        '''
        Generate ideal pupil function
        '''

        # we'll be using the distance from zero in a lot of cases, precompute
        self.gen_kr()
        kr = self.kr

        # define the diffraction limit
        # remember we're working with _coherent_ data _not_ intensity,
        # so drop the factor of 2
        diff_limit = self.NA / self.wl

        # return a circle of intensity 1 over the ideal passband of the
        # objective make sure data is complex
        self.pupil = (kr < diff_limit).astype(complex)

    def gen_psf(self, zrange, dipoles='total'):
        '''
        A function that generates a point spread function over the desired
        `zrange` from the given pupil

        It is assumed that the `pupil` has indices ordered as (y, x) to be
        consistent with image data conventions

        Parameters
        ---
        zrange

        Returns
        ---
        3D PSF
        '''

        self.gen_pupil()

        kr = self.kr
        # helper function

        wavelength = self.wl
        n = self.n

        kmag = n / wavelength

        my_kz = np.real(np.sqrt((kmag**2 - kr**2).astype(complex)))

        theta = np.arcsin((kr < kmag) * kr / kmag)  # Incident angle
        phi = self.phi
        Px1 = np.cos(theta) * np.cos(phi)**2 + np.sin(phi)**2
        Px2 = (np.cos(theta) - 1) * np.sin(phi) * np.cos(phi)
        Py1 = Px2
        Py2 = np.cos(theta) * np.sin(phi)**2 + np.cos(phi)**2
        Pz1 = np.sin(theta) * np.cos(phi)
        Pz2 = np.sin(theta) * np.sin(phi)

        pupil = np.array([
            self.pupil * np.exp(2 * np.pi * 1j * my_kz * z) for z in zrange
        ])

        plist = {'total': [Px1, Px2, Py1, Py2, Pz1, Pz2],
                 'x': [Px1, Px2],
                 'y': [Py1, Py2],
                 'z': [Pz1, Pz2],
                 'none': [np.ones_like(Px1)]
                 }

        pupils = [pupil * p for p in plist[dipoles]]

        PSFa_sub = [
            ifftshift(
                fftn(
                    fftshift(pupil, axes=[1, 2]),
                    axes=[1, 2]),
                axes=[1, 2])
            for pupil in pupils
        ]

        PSFi_sub = [abs(PSF)**2 for PSF in PSFa_sub]

        PSFi = np.array(PSFi_sub).sum(axis=0)

        PSFa = np.array(PSFa_sub).sum(axis=0)

        self.PSFi = PSFi
        self.PSFa = PSFa

    def gen_otf(self, **kwargs):
        raise NotImplementedError
        # need to implement this function which returns OTFs


def richardson_lucy(image, psf, iterations=10, clip=False, prediction_order=2,
                    win_func=None):
    """
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5, False)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    (2) Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image Restoration
    Algorithms. Applied Optics 1997, 36 (8), 1766.

    """
    # Stolen from the dev branch of skimage because stable branch is slow
    # checked against matlab on 20160805 and agrees to within machine precision
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    if win_func is None:
        window = 1.0
    else:
        winshape = np.array(image.shape)
        winshape[-1] = winshape[-1] // 2 + 1
        window = fftshift(win_nd(winshape, win_func=win_func))
    # Build the dictionary to pass around and update
    psf_norm = fft_pad(scale(psf), image.shape, mode='constant')
    psf_norm /= psf_norm.sum()
    u_tm2 = None
    u_tm1 = None
    g_tm2 = None
    g_tm1 = None
    u_t = None
    y_t = image
    # below needs to be normalized.
    otf = window * urfftn(fftshift(psf_norm))

    for i in range(iterations):
        # call the update function
        # make mirror psf
        # calculate RL iteration using the predicted step (y_t)
        reblur = np.real(uirfftn(otf * urfftn(y_t)))
        # assert (reblur > eps).all(), 'Reblur 0 or negative'
        im_ratio = image / reblur
        # assert (im_ratio > eps).all(), 'im_ratio 0 or negative'
        estimate = np.real(uirfftn(np.conj(otf) * urfftn(im_ratio)))
        # assert (estimate > eps).all(), 'im_ratio 0 or negative'
        u_tp1 = y_t * estimate

        # enforce non-negativity
        u_tp1[u_tp1 < 0] = 0

        # update
        u_tm2 = u_tm1
        u_tm1 = u_t
        u_t = u_tp1
        g_tm2 = g_tm1
        g_tm1 = ne.evaluate("u_tp1 - y_t")
        # initialize alpha to zero
        alpha = 0
        # run through the specified iterations
        if i > 1:
            # calculate alpha according to 2
            alpha = (g_tm1 * g_tm2).sum() / (g_tm2**2).sum()

            alpha = max(min(alpha, 1), 0)
            if not np.isfinite(alpha):
                print(alpha)
                alpha = 0
            assert alpha >= 0, alpha
            assert alpha <= 1, alpha

        # if alpha is positive calculate predicted step
        if alpha != 0:
            if prediction_order > 0:
                # first order correction
                h1_t = u_t - u_tm1
                if prediction_order > 1:
                    # second order correction
                    h2_t = (u_t - 2 * u_tm1 + u_tm2)
                else:
                    h2_t = 0
            else:
                h1_t = 0
        else:
            h2_t = 0
            h1_t = 0

        y_t = u_t + alpha * h1_t + alpha**2 / 2 * h2_t
        enusure_positive(y_t)
        assert (y_t >= 0).all()

    im_deconv = u_t

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def enusure_positive(a, eps=0):
    '''
    ensure the array is positive with the smallest value equal to eps
    '''
    assert np.isfinite(a).all(), 'The array has NaNs'
    a[a < 0] = eps


def fftconvolve(in1, in2, mode="full", threads=1, win_func=np.ones):

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
    # shape = s1 + s2 - 1
    # if you double pad the shape, which the above line does then you don't
    # need to take care of any shifting. But you can just pad to the max size
    # and fftshift one of the inputs.
    shape = np.maximum(s1, s2)
    if _inputs_swap_needed(mode, s1, s2):
        # Convolution is commutative; order doesn't have any effect on output
        in1, s1, in2, s2 = in2, s2, in1, s1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.
    if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
        try:
            winshape = np.array(fshape)
            winshape[-1] = winshape[-1] // 2 + 1
            ret = (irfftn(
                rfftn(fft_pad(in1, fshape), threads=threads) *
                rfftn(
                    fftshift(fft_pad(in2, fshape, mode='constant')),
                    threads=threads) *
                # need to fftshift the window so that HIGH
                # frequencies are damped, NOT low frequencies
                fftshift(win_nd(winshape, win_func)), fshape,
                threads=threads)[fslice].copy())
        finally:
            if not _rfft_mt_safe:
                _rfft_lock.release()
    else:
        # If we're here, it's either because we need a complex result, or we
        # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
        # is already in use by another thread).  In either case, use the
        # (threadsafe but slower) SciPy complex-FFT routines instead.
        ret = ifftn(fftn(in1, fshape) * fftn(in2, fshape))[fslice].copy()
        if not complex_result:
            ret = ret.real

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")


def win_nd(size, win_func=sig.hann, **kwargs):
    '''
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
    '''
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
    '''
    Apply Anscombe transform to data
    https://en.wikipedia.org/wiki/Anscombe_transform
    '''
    return 2 * np.sqrt(data + 3 / 8)


def anscombe_inv(data):
    '''
    Apply inverse Anscombe transform to data
    https://en.wikipedia.org/wiki/Anscombe_transform
    '''
    part0 = 1 / 4 * data**2
    part1 = 1 / 4 * np.sqrt(3 / 2) / data
    part2 = -11 / 8 / (data**2)
    part3 = 5 / 8 * np.sqrt(3 / 2) / (data**3)
    return part0 + part1 + part2 + part3 - 1 / 8


def fft_gaussian_filter(img, sigma):
    '''
    FFT gaussian convolution

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
    '''
    kimg = rfftn(img)
    filt_kimg = fourier_gaussian(kimg, sigma, img.shape[-1])
    return irfftn(filt_kimg)
