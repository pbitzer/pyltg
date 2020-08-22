# -*- coding: utf-8 -*-
"""
Module with common functions for lightning satellite data (e.g., GLM, LIS).

"""

import numpy as np

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