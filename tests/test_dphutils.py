from nose.tools import *
import numpy as np
import unittest

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
        newsize = nextpow2(oldsize)
        newdata = fft_pad(data)

        print(newsize)
        print(newdata.shape)
        assert np.all(newsize == np.array(newdata.shape))

    def test_new_shape_no_size_odd(self):
        '''
        Make sure the new shape has nextpow2 dimensions, odd
        '''
        oldsize = 11
        oldshape = (oldsize, oldsize)
        data = np.zeros(oldshape)
        newsize = nextpow2(oldsize)
        newdata = fft_pad(data)

        print(newsize)
        print(newdata.shape)
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


def test_nextpow2_arg():
    '''
    Tests whether `nextpow2` rejects negative integers
    '''
    assert_raises(ValueError, nextpow2, -2)
