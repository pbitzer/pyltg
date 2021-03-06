# -*- coding: utf-8 -*-
"""
Module with common functions for lightning satellite data (e.g., GLM, LIS).

Most users won't need to call these functions directly.

"""

import numpy as np


def energy_colors(energies, min_val=None, max_val=None, satellite=None):
    """
    Map the given satellite lightning energies to a set of 256 colors.

    This provides a suitable color mapping for the energetic measurements
    provide by satellite lightning detectors. For GLM, an estimate of the energy
    is provided directly in the data, but for LIS you'll usually have the
    radiance (which is really the spectral energy density, but I digress).
    Either way, this is what should be passed in.

    Parameters
    ----------
    energies : array-like
        The energies (or radiances) to scale.
    min_val : numeric, optional
        The minimum value to consider for scaling. The default is None.
        If None, and `satellite` is None, use the minimum of `energies`.
    max_val : numeric, optional
        The maximum value to consider for scaling. The default is None.
        If None, and `satellite` is None, use the maximum of `energies`.
    satellite : str, optional
        Provide a string for the satellite (e.g., 'GLM', 'LIS') and we'll
        select a `min_val` and `max_val` for you. Upper case, please.

        For GLM, the values are scaled between 1 fJ and 50 fJ.
        For LIS, the values are scaled between 3000 and 100000 [data native units].

    Returns
    -------
    colors : NumPy array
        nx3 array of bytes. The last dimension corresponds to RGB
        values.


    .. note::
        Right now, the only color scale available to use a yellow->red. Others
        will be added later.

    """

    # Get some endpoints if a satellite was provided:
    if satellite == 'GLM':
        min_val = 1e-15
        max_val = 5e-14
    elif satellite == 'LIS':
        min_val = 3e3
        max_val = 1e5
    else:
        print('Invalid satellite')

    if min_val is None:
        min_val = np.min(energies)
    if max_val is None:
        max_val = np.max(energies)

    _min_val = np.log10(min_val)
    _max_val = np.log10(max_val)
    _values = np.log10(energies)

    # Linear scaling... Really, we should break this out into a function...
    m = (255-0)/(_max_val-_min_val)
    b = 255.-m * _max_val

    scl_colors = m*_values+b

    # First, clip to bounds:
    scl_colors = np.clip(scl_colors, 0, 255)

    # Make it a byte for indexing
    scl_colors = np.uint8(scl_colors)

    colors = np.zeros((len(_values), 3))

    nsteps = 256 # We'll get 256 colors

    # Yellow -> Red color map
    redV = np.repeat(np.uint8(255), nsteps)
    blueV = np.repeat(np.uint8(0), nsteps)
    scale = np.arange(nsteps)/(nsteps-1)
    n0 = 255
    n1 = 0
    greenV = np.uint8(n0 + (n1-n0) * scale)

    colors[:, 0] = redV[scl_colors]
    colors[:, 1] = greenV[scl_colors]
    colors[:, 2] = blueV[scl_colors]

    return colors

def get_children(ids, children):
    """
    Get the children of a set of groups or flashes.

    Typically, `children` will be a Dataframe containing more than just the
    children you're looking for (otherwise, this function is redunant).
    For example, if you want the children of a set of groups, the events
    would be passed in `children` and the group IDs would be passed in `ids`.

    Parameters
    -----------
    ids : array-like or scalar
        The IDs of the groups you want the children of.
    events : Pandas Dataframe
        The possible children. These would usually be a
        set that encompasses more that just those belonging to `ids`.
        Should be similar to output of `GLM.get_events` or `GLM.get_groups`

    Returns
    -------
    list
        A list of Pandas DataFrame's, one for each `ids`. If
        you provide a scalar for `ids`, an one element list is returned.


    """

    ids = np.atleast_1d(ids)

    these_children = [children[children.parent_id == _id] for _id in ids]

    return these_children