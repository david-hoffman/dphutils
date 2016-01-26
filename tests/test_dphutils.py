from nose.tools import *
import os
import numpy as np
import unittest

#import the package to test
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
        data = np.random.randn(*oldshape)
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
        data = np.random.randn(*oldshape)
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
        newdata = fft_pad(data,newsize)

        assert np.all(newsize == np.array(newdata.shape))

    def test_new_shape_multiple(self):
        '''
        Make sure the new shape has the same dimensions when one is given
        '''
        oldshape = (10, 20, 30, 40)
        data = np.random.randn(*oldshape)
        newsize = (50, 40, 30, 100)
        newdata = fft_pad(data,newsize)

        assert np.all(newsize == np.array(newdata.shape))


def test_nextpow2_arg():
    '''
    Tests whether `nextpow2` rejects negative integers
    '''
    assert_raises(ValueError, nextpow2, -2)

class TestPupil(unittest.TestCase):

    def setUp(self):
        self.pupil = Pupil()

    def test_size(self):
        '''
        Make sure when size is changed the output changes accordingly
        '''

        pupil = self.pupil

        pupil.size = 128
        pupil.gen_psf([-1,0,1])
        assert_tuple_equal( pupil.PSFi.shape, (3, 128, 128) )

        pupil.size = 512
        pupil.gen_psf([0])
        assert_tuple_equal( pupil.PSFi.shape, (1, 512, 512) )
