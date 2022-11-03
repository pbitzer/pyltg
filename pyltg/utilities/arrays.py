
"""
Functions for various operations on numpy arrays
"""

import numpy as np


def rotation_z_matrix(angle):
    """
    Define a rotation matrix about the z axis (aka yaw matrix).

    Parameters
    ----------
    angle : numeric
        The angle about which to rotate (in radians).

    Returns
    -------
    rot_z : NumPy Array
        The rotation matrix

    """

    #aka yaw matrix
    cos = np.cos(angle)
    sin = np.sin(angle)

    rot_z = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
        ])

    return rot_z

def rotation_y_matrix(angle):
    """
    Define a rotation matrix about the y axis (aka pitch matrix).

    Parameters
    ----------
    angle : numeric
        The angle about which to rotate (in radians).

    Returns
    -------
    rot_y : NumPy Array
        The rotation matrix

    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    rot_y = np.array([
        [cos, 0, sin],
        [0, 1, 0],
        [-sin, 0, cos]
        ])

    return rot_y

def rotation_x_matrix(angle):
    """
    Define a rotation matrix about the x axis (aka roll matrix).

    Parameters
    ----------
    angle : numeric
        The angle about which to rotate (in radians).

    Returns
    -------
    rot_x : NumPy Array
        The rotation matrix
    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    rot_x = np.array([
        [1, 0, 0],
        [0, cos, -sin],
        [0, sin, cos]
        ])

    return rot_x

def rotation_matrix(yaw, pitch, roll):
    """
    Define a 3 dimension rotation matrix.

    This provides a quick way to get a rotation matrix. It is equivelant
    to

    .. math::
        R_z \cdot R_y \cdot R_x

    where :math:`R_x` is the rotation matrix about the x axis (and similar
    :math:`R_y` is the rotation matrix about the y axis, and
    :math:`R_z` is the rotation matrix about the z axis. This function is
    a bit faster than finding the three matrices individually and then (matrix)
    multiplying.

    Parameters
    ----------
    yaw : numeric
        The rotation angle about the x axis.
    pitch : numeric
        The rotation angle about the y axis.
    roll : numeric
        The rotation angle about the z axis.

    Returns
    -------
    rot_matrix : 3x3 NumPy array
        The matrix defining the rotation.

    """

    # Get the sines/cosines for the angles.
    cy = np.cos(yaw); sy = np.sin(yaw)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cr = np.cos(roll); sr = np.sin(roll)

    rot_matrix = np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr - cy*sr],
        [-sp  , cp*sr,          cp*cr]
        ])

    return rot_matrix

def perturb_points(x, y, delta=1, npts=16):
    """
    Draw a circle of points around a set of points.

    Given a set of points, effectively perturb them by drawing a circle
    of points around each. Useful for "expanding" a convex hull.

    Parameters
    ----------
    x : array-like
        The x-locations of the points
    y : array-like
        The y-locations of the points
    delta : numeric, optional
        The radius of points to draw around each input point. The default is 1.
    npts : int, optional
        The number of points to draw around each input point. The default is 16.

    Returns
    -------
    px : np.array
        The x-location of pertubed points, including the original points.
    py : np.array
        The y-location of pertubed points, including the original points.

    """

    npts = 16  # Number of points in the circle around each vertex
    theta = np.linspace(0, 2*np.pi, npts)

    dx = delta*np.cos(theta)
    dy = delta*np.sin(theta)

    px = np.broadcast_to(x, (npts, len(x))).copy()
    py = np.broadcast_to(y, (npts, len(y))).copy()

    px += dx[:, np.newaxis]
    py += dy[:, np.newaxis]

    px = px.flatten()
    py = py.flatten()

    return px, py

def hull_get_path(hull):
    """
    Given a ConvexHull using `scipy.spatial`, get the path defined by the vertices.

    Often we want to do something with a `ConvexHull` (like plot). This is a
    convenience function that gets the vertices and returns them as arrays.

    The path is closed (i.e., the last point returned is also the first point.)

    Parameters
    ----------
    hull : scipy.spatial.qhull.ConvexHull
        The convex hull

    Returns
    -------
    path_x : np.array
        The x locations of the hull.
    path_y : np.array
        The y locations of the hull.

    """

    path_x = np.append(hull.points[hull.vertices, 0],
                       hull.points[hull.vertices[0], 0])

    path_y = np.append(hull.points[hull.vertices, 1],
                       hull.points[hull.vertices[0], 1])

    return path_x, path_y

def histo_bins(data, bins=None, min_val=None, max_val=None):
    """
    Get suitable bins for a histogram.

    NumPy's histogram doesn't allow you to specify a particualar bin size,
    only either the number of bins or the actual bins to use. One of the
    purposes of this function is allow to generate the bins using a
    particular bin size, which can then be passed into NumPy's histogram.

    Parameters
    ----------
    data : NumPy array
        The array of data to base the bins off of. Should be 1D. Can include
        NaN's.
    bins : varies, optional
       If `None`, generate bins using Scott's choice. The default is None.
       If a scalar, this is used for the bin size.
       If it is an array, then these are used as the bins. Then, the only
       thing this function really does is find a min/max value (unless those
       are provided too, then this function does nothing!).
    min_val : float, optional
        The minimum value of data to consider when making the mins.
        If `None`, the mininum of the data is used. The default is None.
    max_val : TYPE, optional
        The maximum value of data to consider when making the mins.
        If `None`, the 95th percentile of the data is used. The default is None.

    Returns
    -------
    min_val : float
        The minimum value used to find the bins.
    max_val : float
        The maximum value used to find the bins.
    bin_start : NumPy array
        The starting edges of the bins.

    """


    # First, check for min/max values:
    if min_val is None:
        min_val = np.nanmin(data)
    if max_val is None:
        max_val = np.nanpercentile(data, 95)

    if bins is None:
        # No bins provided, so go get some using Scott's choice:
        binsize = 3.5 * np.nanstd(data, ddof=1)/data.size**(1/3)
        bin_start = np.arange(min_val, max_val, binsize)
    elif np.ndim(bins) == 0:
        # If it's not None, but a scalar, then it should be a binsize.
        # So, we'll make the bins
        bin_start = np.arange(min_val, max_val, bins)
    else:
        # Should be the actual bins, so what are you doing here?
        bin_start = bins

    return min_val, max_val, bin_start

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

    from scipy.ndimage import zoom as sci_zoom  # for upsample

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
        # For upsampling, we'll use scipy.ndimage.zoom
        if scheme == 'sample':
            order = 0
        elif (scheme == 'bilinear') | (scheme is None):
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

    return new_arr


def xcorr(data1, data2, times1=None, times2=None, delta=1):
    """
    Given two arrays, find the lag that maximizes the cross correlation.

    For two data arrays, usually time series, this function will find the
    lag between the arrays that gives the maximum of the cross correlation.

    Before cross correlating, the median of each data is subtracted from
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
    times1 : array, optional
        The times of each element of `data1`. If given as a `np.datetime64`,
        this will be converted to a 64 bit integer. In this case, the lag
        is units of the `np.datetime64` you use. For example, if
        you provide `np.datetime64[ns]`, then the return value for the lags
        is in units of nanoseconds.
    times2 : array, optional
        The times of each element of `data2`. If given as a `np.datetime64`,
        this will be converted to a 64 bit integer. See `times1` for
        additional notes.
    delta : numeric, optional
        If `times1` and `times2` are not given, this is the physical distance
        between adjacent elements of `data1` and `data2`. If not given, 
        the returned lag is simply the number of elements in which
        `data1` lags `data2`.

    Returns
    -------
    tuple :
        - The lag of max correlation. A positive lag means `data1` lags
          `data2`, i.e., data2 is earlier than data1.  
        - The lags; this will be the index of the lags. The size of the array
          is `data1.size + data2.size-1`
        - The cross correlation: the value of the cross correlation for the lags

    Examples
    ---------
    A basic usage example. We'll create a triangle wave, copy it, lag
    the copy, and compute the lag::

        import numpy as np

        # make a triangle wave
        x1 = np.array([0, 0, 0, 0, 0, 0])
        x2 = np.array(np.arange(10))+1
        _x3 = np.array(np.arange(10))
        x3 = _x3[::-1]  # reverse the array

        x = np.concatenate((x2, x3, x3))

        forced_lag = 4
        y = np.roll(x, forced_lag)  # make another array that lags the orig

        max_lag, lags, corr =  xcorr(x, y)
        print(max_lag)  # should get value of forced_lag

    """

    import scipy.signal as sig
    from scipy.interpolate import UnivariateSpline

    if (times1 is None) or (times2 is None):
        # No times? Just find the lag
        _data1 = data1
        _data2 = data2
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
        interp_args = {'ext': 0, 'k': 5, 's': 0}  # keywords for UnivariateSpline

        intp1 = UnivariateSpline(times1, data1, **interp_args)
        intp2 = UnivariateSpline(times2, data2, **interp_args)

        # Now, we'll go about building the common time grid
        dt1 = np.min(np.diff(times1))
        dt2 = np.min(np.diff(times2))
        delta = np.min((dt1, dt2))
        delta /= 10  # We are going to create a times array on a finer grid

        times_common = np.arange(0, max_t-min_t, delta)

        # Now, interpolate the data to this grid
        _data1 = intp1(times_common)
        _data2 = intp2(times_common)

    # Before correlation, we should subtract off means:
    # Note: avoiding compound ops to avoid any mutablity issues
    _data1 = _data1 - np.median(_data1)
    _data2 = _data2 - np.median(_data2)

    # Do the cross correlation, sweeping the data over each other.
    # This will return data1.size+data2.size-1 elements
    cross_corr = sig.correlate(_data1, _data2, "full")

    max_ind = np.argmax(cross_corr)

    # Now, generate an "axis" of lags
    lags = np.arange(cross_corr.size) - (_data1.size-1)

    # Get the lags for the max of the correlation.
    loc = lags[max_ind] * delta

    return loc, lags, cross_corr
    
def array2record(loc):
    """
    Convert a 3D array to a NumPy record array with fields `x, y, z`

    This provides a (relatively) easy way to get a record array suitable
    for locations used in, say, `source_retrieval`.

    Parameters
    ----------
    loc : NumPy array
        A n-by-3 NumPy array.

    Returns
    -------
    loc : NumPy record array
        A record array with fields `x, y, z` corresponding to the
        columns of the input.

    """

    loc = np.core.records.fromarrays(loc.T,
                                     dtype=[('x', 'float64'),
                                            ('y', 'float64'),
                                            ('z', 'float64')])

    return loc
