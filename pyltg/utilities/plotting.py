
"""
Functions for various operations while plotting (usually) with Matplotlib
"""

import numpy as np 

def gap_nan(arr1, arr2, gap):
    """
    Look for "gaps" in :param arr1: bigger than the specified :param gap: and
    insert NaNs in the gaps.

    When plotting data, especially time series data, there can be gaps in
    the data. For example, this can occur when looking at a time series of
    lightning data that comes in packets. We don't want to "connect the dots"
    between packets. This function introduces NaN to suppress the "connect
    the dots" that Matplotlib will nominally do.

    Right now, you must pass two arrays. This function could be extended so
    that only one, or more than two arrays, could be passed.

    Inspiration provided by Craig Markwardt's IDL routine by a similar name.

    Parameters
    ----------
    arr1 : array-like
        The array you wish to add NaNs to.
    arr2 : array-like
        A related array (to :param arr1:) that will get NaNs in the same
        location. A common use is that `arr1` will have times,
        and `arr2` will have the corresponding data to the time series.
    gap : numeric
        Any gap in `arr1` bigger than the provided value will have a
        NaN inserted.

    Returns
    -------
    tuple
        The arrays with NaNs (same order as passed in)
    """

    import numpy as np

    delta = np.diff(arr1)
    idxGap = np.where(delta > gap)[0]

    if len(idxGap) > 0:
        gapArr1 = list()
        gapArr2 = list()

        for _gapNum, idx in enumerate(idxGap):
            # TODO Is there better way than looping over a enumerate?
            if _gapNum == 0:
                chunk1 = arr1[0:idx]
                chunk2 = arr2[0:idx]

            else:
                chunk1 = arr1[idxGap[_gapNum - 1] + 1:idx]
                chunk2 = arr2[idxGap[_gapNum - 1] + 1:idx]

            gapArr1.append(chunk1)
            gapArr1.append(np.array([np.NaN]))

            gapArr2.append(chunk2)
            gapArr2.append(np.array([np.NaN]))

        # Get the last chunk of data
        gapArr1.append(arr1[idxGap[-1] + 1:-1])
        gapArr2.append(arr2[idxGap[-1] + 1:-1])

        # Finally, make everything an array again
        new1 = np.concatenate(gapArr1)
        new2 = np.concatenate(gapArr2)
    else:
        new1 = arr1
        new2 = arr2

    return new1, new2


def get_actual_ticks(axis):
    """
    Get ticks on the given axis that are actually on the plot.

    Annoyingly, the method `axis.get_ticklabel()` in matplotlib will return
    ticks that aren't actually on the plot. (Similar for axis methods.)
    This method will return the ticks that are.

    The ticks must be drawn for this to work, for example with::

        fig.canvas.draw()
        
    More generally::
        
        fig, ax = plt.subplots()
        get_actual_ticks(ax.xaxis)

    Parameters
    ----------
    axis : Matplotlib Axis
        The axis you wish to get the ticks from.
        
    Returns
    --------
    numpy array
        The tick values currently in view.

    """
    import numpy as np
    
    view = axis.get_view_interval() 
    loc = axis.get_majorticklocs()  # get_ticklocs?
    
    idx = np.where((loc >= view[0]) & (loc <= view[1]))
    goodTicks = loc[idx]
    
    return goodTicks


def _time_axis(axis, relative=True, title=True):
    """

    .. deprecated::
              This will be removed in a future version.

    Given a Matplotlib axis, format the ticks in a "time" format.

    Sometimes, the default formatting of Matplotlib is quite terrible. This
    function attempts to provide better formatting, including the ability to
    create a "relative" axis on the fly.

    Parameters
    ----------
    axis : Matplotlib axis
        The axis you wish to transform.
    relative : bool
        If true, "relative" ticks are used. This means the first tick is zero,
        and the other ticks are incremented relative to this.
        NOTE: Right now, this keyword has no effect - a relative axis
        is always returned.
    title : bool
        If True, a title is added to the axis with the "relative" time
    """
    import matplotlib.ticker as mtick
    import numpy as np
    
    @mtick.FuncFormatter
    def form(tick_val, pos=None):

        # AUTHOR NOTE: t0Plot,timeUnitMod is available in this scope from outer
        label = str((tick_val-t0Plot)/timeUnitMod)
        return label
    
    # Get the ticks that are actually on the plot:
    ticks = get_actual_ticks(axis)
    # Next, find how much the data spans
    dt = (ticks[-1]-ticks[0]).astype('timedelta64[ns]').astype('int64')

    # todo Because of rounding (I guess), the ticks may not be exactly spaced.
    # This particularly happens as the time span falls to ~ 1msec. So, 
    # set the ticks explicitly:
    # tick2tick = dt/(len(ticks)-1)

    # We're going to use power-of-ten exponent to decide the best units to use
    pwr = np.floor(np.log10(dt))
    
    expRange = np.array([9, 6, 3, 0])  # Assumes units of ns
    title = ['s', 'ms', 'us', 'ns']

    expIdx = np.clip(np.digitize(pwr, expRange), 0, 3)

    timeUnitMod = 10**expRange[expIdx]
    # noinspection PyTypeChecker
    timeUnitLabel = title[expIdx]
        
    # We want the first tick on the plot, as we'll set the tick labels
    # relative to this
    t0Plot = ticks[0] 

    axis.set_major_formatter(mtick.FuncFormatter(form))
    
    if title:
        # For the axis label, cast the first tick time back as a datetime
        # and extract the hr, min, sec:
        xTitle = 'Time (' + timeUnitLabel + '); Origin time: ' \
            + str(t0Plot.astype('datetime64[ns]')).split('T')[1]
        axis.set_label_text(xTitle)


def time_axis_label(ax, t0=None, which_ax = 'x'):
    """
    Given an Axes with a xaxis corresponding to times, label the axes with
    more readable ticks.

    Parameters
    -----------
    ax: MPL Axes
        The Axes containing the "sub"axis to be modified.
    t0: numpy datetime64[ns]
        The origin of the plot. Only used for labeling. This will get cast
        to datetime64[ns] if it isn't already.
    which_ax: str
        Which axis to label, either 'x' or 'y'.

    Returns
    --------
    tuple:
        Two elements are returned: exponent and label. The label is the label on
        the axis. The exponent corresponds to the unit used, in tens of nanoseconds.
        So, an exponent of 9 means the units are 10^9 ns, i.e., seconds.
    """

    which_ax = 'x'  # eventually, we could make this a keyword

    if which_ax == 'x':
        this_ax = ax.xaxis
        t_range = ax.get_xlim()
    elif which_ax == 'y':
        this_ax = ax.yaxis
        t_range = ax.get_ylim()

    # Get the range of the axis
    time_delta = t_range[1] - t_range[0]

    pwr = np.floor(np.log10(time_delta))

    # Define possible increments for the units and their corresponding labels
    exp_range = np.array([9, 6, 3, 0])
    exp_title = ['sec', 'ms', '$\mu$s', 'ns']

    exp_idx = np.digitize(pwr, exp_range)
    exp_idx = np.clip(exp_idx, 0, len(exp_range)-1)

    exp = exp_range[exp_idx]
    exp_title = exp_title[exp_idx]

    # Use this exponent for the multiplier.
    ax.ticklabel_format(axis='x', scilimits=(exp, exp))

    this_ax.offsetText.set_visible(False)

    the_label = exp_title

    if t0 is not None:
        the_label += '; Origin: ' + t0.astype('datetime64[ns]').astype('str')

    this_ax.set_label_text(the_label)

    return exp, the_label