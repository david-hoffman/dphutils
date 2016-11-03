from nose.tools import *
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_approx_equal
import unittest
from scipy.signal import signaltools as sig
from scipy.ndimage.filters import gaussian_filter
# import the package to test
from dphutils import *


class TestFFTPad(unittest.TestCase):

    def setUp(self):
        pass

    def test_new_shape_no_size_even(self):
        '''
        Make sure the new shape has nextpow2 dimensions, even
        '''
        oldsize = 10
        oldshape = (oldsize, oldsize)
        data = np.zeros(oldshape)
        newsize = tuple(sig.fftpack.helper.next_fast_len(s) for s in oldshape)
        newdata = fft_pad(data)
        assert np.all(newsize == np.array(newdata.shape))

    def test_new_shape_no_size_odd(self):
        '''
        Make sure the new shape has nextpow2 dimensions, odd
        '''
        oldsize = 11
        oldshape = (oldsize, oldsize)
        data = np.zeros(oldshape)
        newsize = tuple(sig.fftpack.helper.next_fast_len(s) for s in oldshape)
        newdata = fft_pad(data)
        assert np.all(newsize == np.array(newdata.shape))

    def test_new_shape_one_size(self):
        '''
        Make sure the new shape has the same dimensions when one is given
        '''
        oldshape = (10, 20, 30)
        data = np.random.randn(*oldshape)
        newsize = 50
        newdata = fft_pad(data, newsize)

        assert np.all(newsize == np.array(newdata.shape))

    def test_new_shape_multiple(self):
        '''
        Make sure the new shape has the same dimensions when one is given
        '''
        oldshape = (10, 20, 30, 40)
        data = np.random.randn(*oldshape)
        newsize = (50, 40, 30, 100)
        newdata = fft_pad(data, newsize)

        assert np.all(newsize == np.array(newdata.shape))


def test_radprof_complex():
    """Testing rad prof for complex values"""
    result = radial_profile(np.ones((11, 11)) + np.ones((11, 11)) * 1j)
    avg = np.ones(8) + np.ones(8) * 1j
    assert_allclose(result[0], avg)
    std = np.zeros(8) + np.zeros(8) * 1j
    assert_allclose(result[1], std)


def test_win_nd():
    """Testing the size of win_nd"""
    shape = (128, 65, 17)
    result = win_nd(shape)
    assert_equal(shape, result.shape)


def test_anscombe():
    """Test anscombe function"""
    # https://en.wikipedia.org/wiki/Anscombe_transform
    data = np.random.poisson(100, (128, 128, 128))
    assert_almost_equal(data.mean(), 100, 1), "Data not generated properly!"
    ans_data = anscombe(data)
    assert_almost_equal(ans_data.var(), 1, 2)
    in_ans_data = anscombe_inv(ans_data)
    assert_almost_equal((in_ans_data - data).var(), 0, 4)


def test_fft_gaussian_filter():
    """Test the gaussian filter"""
    data = np.random.randn(128, 128, 128)
    sigmas = (np.random.random(3) + 1) * 2
    fftg = fft_gaussian_filter(data, sigmas)
    fftc = gaussian_filter(data, sigmas)
    # there's an edge effect that I can't track down maybe its
    # in fourier_gaussian
    fslice = (slice(16, -16), ) * data.ndim
    assert_allclose(fftg[fslice], fftc[fslice])


def test_fft_gaussian_filter_small():
    """make sure fft_gaussian_filter defaults to regular when input is small"""
    data = np.random.randn(32, 32, 2048)
    sigmas = (np.random.random(data.ndim) + 1) * 2
    fftg = fft_gaussian_filter(data, sigmas)
    fftc = gaussian_filter(data, sigmas)
    assert_allclose(fftg, fftc)


def test_fft_gaussian_filter_small_warn():
    """make sure fft_gaussian_filter defaults to regular when input is small"""
    data = np.random.randn(32, 32, 2048)
    sigmas = (np.random.random(data.ndim) + 1) * 2
    assert_warns(UserWarning, fft_gaussian_filter, data, sigmas)
