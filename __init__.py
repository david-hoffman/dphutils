'''
This is for small utility functions that don't have a proper home yet
'''

import numpy as np

def radial_profile(data, center):
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
    '''

    y, x = np.indices((data.shape))

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())

    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr

    return radialprofile


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
    >>> slice_maker(30,20,25)
    [slice(18, 43, None), slice(8, 33, None)]
    '''

    #calculate the start and end
    half1 = width//2
    #we need two halves for uneven widths
    half2 = width-half1
    ystart = y0 - half1
    xstart = x0 - half1
    yend = y0 + half2
    xend = x0 + half2

    toreturn = [slice(ystart,yend), slice(xstart, xend)]

    #return a list of slices
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

    if n < 0:
        raise ValueError('n must be a positive integer, n = {}'.format(n))

    return 1 << (n-1).bit_length()

def fft_pad(array, pad_width = None, mode = 'median', **kwargs):
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
    #pull the old shape
    oldshape = array.shape

    if pad_width is None:
        #update each dimenstion to next power of two
        newshape = array([nextpow2(n) for n in oldshape])
    else:


    len(pad_width) == 1:
        newshape = array([pad_width for n in oldshape])
