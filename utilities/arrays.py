
"""
Functions for various operations on numpy arrays
"""

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
    scheme : string, default: None
        If downsampling, this can be 'sample' or None. If None, then the 
        rebinned elements are averaged. If 'sample', the rebinned elements
        are sampled via strides.
        
        If upsampling, this can be 'sample', 'bilinear', 'cubic', or None.
        If 'bilinear' or None (default), the bilinear interpolation is done.
        If 'cubic', then cubic interpolation is done.
        If 'sample', then no interpolation is done and elements are repeated.
    
    arr - numpy 2 dim ndarray
    shape - tuple, list
    
    scheme = 'sample' (downsample, upsample), 'cubic' (upsample), 'bilinear' (upsample)
    
    """
    
    from scipy.ndimage import zoom as sci_zoom # for upsample
    from numpy import divmod as np_divmod

    from numpy import any as np_any
    from numpy import all as np_all
    from numpy import reshape as np_reshape
    
    
    # First, make sure we have the same number of dimensions:
    if arr.ndim != len(shape):
        raise ValueError("The new array must have the same number of dimensions")
        
    # Now, check that the new array dimensions are an integral factor
    # of the original:
    divisor_down = np_divmod(arr.shape, shape)  # if all zero, we're downsampling
    divisor_up = np_divmod(shape, arr.shape)  # if all zero, we're upsampling

    # Now, let's figure out if we're up/down sampling...
    if np_all(divisor_up[1] == divisor_down[1]):
        # If these are the same, then the the new shape is same
        # (so why are you here? :-D)    
        new_arr = arr
    elif not np_any(divisor_down[1]):
        # If this is all zero, then we are downsampling:
        # We can either resample the array or average
        if scheme is None:
            # https://scipython.com/blog/binning-a-2d-array-in-numpy/
            
            # NOTE: For future ndim, see:
            # https://gist.github.com/derricw/95eab740e1b08b78c03f
            
            # Average the array:
            blowup_shape = (shape[0], arr.shape[0]//shape[0], 
                            shape[1], arr.shape[1]//shape[1])
            new_arr = np_reshape(arr, blowup_shape).mean(-1).mean(1)
            
        elif scheme == 'sample':
            strides = divisor_down[0]
            # NOTE: Here's a 2d-only oepration. 
            # Possible something here might help when we go to n-dim
            # https://scipy-cookbook.readthedocs.io/items/Rebinning.html
            # https://github.com/sbrisard/rebin/blob/master/rebin.py
            new_arr = arr[::strides[0], ::strides[1]]
        else:
            raise ValueError("Invalid scheme for downsampling")
    elif not np_any(divisor_up[1]):
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
   