
"""
Functions for various operations on MatplotLib Axes
"""


def gap_nan(arr1, arr2, gap):
    """
    Look for "gaps" in :param arr1: bigger than the specified :param gap: and
    insert NaNs in the gaps.

    When plotting data, especially time series data, there can be gaps in
    the data. For example, this can occur when looking at a time series of
    lightning data that comes in packets. We don't want to "connect the dots"
    between packets. This function introduces NaN to suppress the "connect
    the dots" that matplotlib will nominally do.

    Right now, you must pass two arrays. This function could be extended so
    that only one, or more than two arrays, could be passed.
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

    Annoyingly, the method axes.get_ticklabel() in matplotlib will return
    ticks that aren't actually on the plot. (Similar for axis methods.)
    This method will return the ticks
    that are.

    The ticks must be drawn for this to work, for example with

    fig.canvas.draw()

    """
    import numpy as np
    
    view = axis.get_view_interval() 
    loc = axis.get_majorticklocs() #get_ticklocs?
    
    idx = np.where((loc >= view[0]) & (loc<=view[1]))
    goodTicks = loc[idx]
    
    return goodTicks


def time_axis(axis, relative=True, title=True):
    """
    """
    import matplotlib.ticker as mtick
    import numpy as np
    
    @mtick.FuncFormatter
    def form(tickVal, pos=None):

        # AUTHOR NOTE: t0Plot,timeUnitMod is available in this scope from outer
        label = str((tickVal-t0Plot)/timeUnitMod)
        return label
    
    # Get the ticks that are actually on the plot:
    ticks = get_actual_ticks(axis)
    # Next, find how much the data spans
    dt = (ticks[-1]-ticks[0]).astype('timedelta64[ns]').astype('int64')

    # Because of rounding (I guess), the ticks may not be exactly spaced.
    # This particularly happens as the time span falls to ~ 1msec. So, 
    # set the ticks explicitly:
    tick2tick = dt/(len(ticks)-1)

    # We're going to use power-of-ten exponent to decide the best units to use
    pwr = np.floor(np.log10(dt))
    
    expRange = np.array([9, 6, 3, 0]) # Assumes units of ns
    title = ['s', 'ms', 'us', 'ns']

    expIdx = np.clip(np.digitize(pwr, expRange), 0, 3)

    timeUnitMod = 10**expRange[expIdx]
    timeUnitLabel = title[expIdx]
        
    # We want the first tick on the plot, as we'll set the tick labels
    # realtive to this 
    t0Plot = ticks[0] 

    axis.set_major_formatter(mtick.FuncFormatter(form))
    
    if title:
        # For the axis label, cast the first tick time back as a datetime
        # and extract the hr, min, sec:
        xTitle = 'Time ('+ timeUnitLabel +'); Origin time: ' \
            + str((t0Plot).astype('datetime64[ns]')).split('T')[1]
        axis.set_label_text(xTitle)