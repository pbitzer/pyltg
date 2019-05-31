
"""
Functions for various operations on numpy arrays
"""

import numpy as np

def rebin(arr, shape, scheme=None):
    """
    Given a input array, change the array such that the new array dimensions
    are an integral factor of the original.
    
    Unlike some implentations out there, this function can upsample or 
    downsample the array. Often used with arrays that represent images.
    
    This function is meant to mimic IDL's implementation. It's not *quite*
    the same, especially for upsampling. Right now, scipy.ndimage.zoom is
    used, and expansion with bilinear order/scheme produces slightly
    different results. But, this function is more powerful since it can
    do other interpolation schemes.
    
    Parameters
    ----------
    arr : numpy ndarray
        The array you wish to rebin. Must be two dimensions (right now).
    shape : tuple, list
        The shape of the rebinned array. Must be an integral factor of
        the original array, and must be integers. You can either do upsampling 
        (the new shape is bigger) or downsampling (the new shape is smaller), 
        but you can't do upsampling on one dimension and downsampling on the other.
    scheme : str (mostly)
        If downsampling:
            'sample'
                Rebinned elements are sample via strides into the array.
            None
                The rebinned elements are averaged. Note: not a string!
        If upsampling:
            'sample'
                No interpolation is done and elements are repeated.
            'bilinear' or None
                Bilinear interpolation is done. This is the default.
            'cubic'
                Cubic interpolation is done.
                
    """
    
    from scipy.ndimage import zoom as sci_zoom # for upsample
    
    # First, make sure we have the same number of dimensions:
    if arr.ndim != len(shape):
        raise ValueError("The new array must have the same number of dimensions")
        
    # Now, check that the new array dimensions are an integral factor
    # of the original:
    divisor_down = np.divmod(arr.shape, shape)  # if all zero, we're downsampling
    divisor_up = np.divmod(shape, arr.shape)  # if all zero, we're upsampling

    # Now, let's figure out if we're up/down sampling...
    if np.all(divisor_up[1] == divisor_down[1]):
        # If these are the same, then the the new shape is same
        # (so why are you here? :-D)    
        new_arr = arr
    elif not np.any(divisor_down[1]):
        # If this is all zero, then we are downsampling:
        # We can either resample the array or average
        if scheme is None:
            # https://scipython.com/blog/binning-a-2d-array-in-numpy/
            
            # NOTE: For future ndim, see:
            # https://gist.github.com/derricw/95eab740e1b08b78c03f
            
            # Average the array:
            blowup_shape = (shape[0], arr.shape[0]//shape[0], 
                            shape[1], arr.shape[1]//shape[1])
            new_arr = np.all(arr, blowup_shape).mean(-1).mean(1)
            
        elif scheme == 'sample':
            strides = divisor_down[0]
            # NOTE: Here's a 2d-only oepration. 
            # Possible something here might help when we go to n-dim
            # https://scipy-cookbook.readthedocs.io/items/Rebinning.html
            # https://github.com/sbrisard/rebin/blob/master/rebin.py
            new_arr = arr[::strides[0], ::strides[1]]
        else:
            raise ValueError("Invalid scheme for downsampling")
    elif not np.any(divisor_up[1]):
        pass
        # For upsampling, we'll use scipy.ndimage.zoom
        if scheme == 'sample':
            order = 0
        elif (scheme == 'bilinear') | (scheme == None):
            order = 1
        elif scheme == 'cubic':
            order = 2
        # NOTE: could add higher order - scipy's zoom goes up to 5
        else:
            raise ValueError("Invalid scheme for upsampling")
        new_arr = sci_zoom(arr, divisor_up[0], order=order)
        
    else:
        # If none of these are true, then we must have non-integral factos
        raise ValueError("New shape must be integral factor of old shape")
        print('hi')
        pass  # here be dragons
        
    return new_arr
   

def xcorr(data1, data2, times1=None, times2=None):
    """
    Given two arrays, find the lag that maximizes the cross correlation.
    
    For two data arrays, usually time series, this function will find the 
    lag between the arrays that gives the maximum of the cross correlation.
    
    Before cross correlating, the mean of each data is subtracted from
    the data.
    
    If corresponding time arrays for the data is not given, the lag is
    reported as the number of elements.
    
    If the corresponding time arrays are given, then these times are 
    interpolated to a common time grid. The lag is then reported in units
    of time. 
    
    These time arrays will be shifted by the minimum of `times1`, `times2`.
    The smaller of these minimums will be subtracted from each of the times
    arrays. This has no effect on the lag, since we're only interested in the
    difference between the two data sets. It does help the case if we
    have large numbers (like in the case you pass in a `numpy.datetime64[ns]`
    set of times).

    Generally speaking, these arrays do not have to be "times", 
    just some measure of how the data is spaced out. Right now, it only has
    only really been tested on times, however.

    Note: interpolation uses SciPy's UnivariateSpline, not interp1d
    See: `<https://github.com/scipy/scipy/issues/4304>`_
    
    Code based on info found here:
    `<https://mail.scipy.org/pipermail/scipy-user/2011-December/031177.html>`_

    Note: Future implementation may use pycorrelate. See
    `<https://pycorrelate.readthedocs.io/en/latest/index.html>`_

    Parameters
    -----------
    data1 : array
        The reference data.
    data2 : array
        The data to be correlated against.
    times1 : array
        The times of each element of `data1`. If given as a `np.datetime64`, 
        this will be converted to a 64 bit integer. In this case, the lag
        is units of the `np.datetime64` you use. For example, if
        you provide `np.datetime64[ns]`, then the return value for the lags 
        is in units of nanoseconds. 
    times2 : array
        The times of each element of `data2`. If given as a `np.datetime64`, 
        this will be converted to a 64 bit integer. See `times1` for 
        additional notes. 
    
    Returns
    -------
    tuple : 
        - The lag of max correlation. A positive lag means the second array
        is after the first.
        - The lags; this will be the index of the lags. The size of the array
          is `data1.size + data2.size-1`
        - The cross correlation: the value of the cross correlation for the lags

    Examples
    ---------

    import numpy as np

    # make a triangle wave
    x1 = np.array([0, 0, 0, 0, 0, 0])
    x2 = np.array(np.arange(10))+1
    _x3 = np.array(np.arange(10))
    x3 = _x3[::-1] # reverse the array

    x = np.concatenate((x2, x3, x3))

    forced_lag = 4
    y = np.roll(x, forced_lag)  # make another array that lags the orig

    print(xcorr(x, y))  # should get value of forced_lag
    """

    import scipy.signal as sig
    from scipy.interpolate import UnivariateSpline
    
    if (times1 is None) or (times2 is None):
        # No times? Just find the lag
        _data1 = data1
        _data2 = data2

        dist_per_lag = 1
    else:
        # Since we have times for each array, we will interpolate the data
        # to a common set of times. 
        # TODO check to see if both times1, times2 are given        
        
        # The code for datetime is 'M'...
        if times1.dtype.kind == 'M': 
            times1 = times1.astype('int64')
        if times2.dtype.kind == 'M':
            times2 = times2.astype('int64')
        
        # We're only interested in delta between the two data arrays,
        # so we are going to shift the times:
        # TODO use a minmax function
        min_t = np.min(np.concatenate((times1, times2)))
        max_t = np.max(np.concatenate((times1, times2)))

        times1 = times1 - min_t
        times2 = times2 - min_t
        
        # Define the interpolations
        interp_args = {'ext':0, 'k':5, 's':0}  # keywords for UnivariateSpline
        
        intp1 = UnivariateSpline(times1, data1, **interp_args)
        intp2 = UnivariateSpline(times2, data2, **interp_args)
        
        # Now, we'll go about building the common time grid
        dt1 = np.min(np.diff(times1))
        dt2 = np.min(np.diff(times2))
        dist_per_lag = np.min((dt1, dt2))
        dist_per_lag/= 10  # We are going to create a times array on a finer grid
                
        times_common = np.arange(0, max_t-min_t, dist_per_lag)
        
        # Now, interpolate the data to this grid
        _data1 = intp1(times_common)
        _data2 = intp2(times_common)
    
    # Before correlation, we should subtract off means:
    # Note: avoiding compound ops to avoid any mutablity issues
    _data1 = _data1 - np.mean(_data1)
    _data2 = _data2 - np.mean(_data2)
    
    # Do the cross correlation, sweeping the data over each other.
    # This will return data1.size+data2.size-1 elements
    cross_corr = sig.correlate(_data1, _data2, "full")
    
    max_ind = np.argmax(cross_corr)
    
    # Now, generate an "axis" of lags
    lags = np.arange(cross_corr.size) - (_data1.size-1)
    
    # Get the lags for the max of the correlation. 
    loc = -lags[max_ind] * dist_per_lag
        
    return loc, lags, cross_corr
    
