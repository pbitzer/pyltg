#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for functions related to time operations.
"""

import numpy as np


def tai93_to_utc(times):
    """
    Convert TAI93 times to UTC.

    The TAI93 epoch is Jan 1, 1993 at midnight. This function will convert
    times in this epoch to the more common UTC.

    Parameters
    ----------
    times : NumPy array
        An array of times correpsonding to seconds past the TAI93 epoch.
        Can be (and usually are) fractional seconds, so this is usually a
        float array.

    Returns
    -------
    NumPy Datetime64[ns]:
        The UTC times corresponding to the input.

    """

    # First, define all the dates after 1993 in which the leap second
    # was added. Many references in Google-land, but here's one:
    # http://hpiers.obspm.fr/eop-pc/index.php?index=TAI-UTC_tab&lang=en
    leap_sec_dates = np.array([
        '1993-07-01',
        '1994-07-01',
        '1996-01-01',
        '1997-07-01',
        '1999-01-01',
        '2006-01-01',
        '2009-01-01',
        '2012-07-01',
        '2015-07-01',
        '2017-01-01',
        ], dtype='datetime64[ns]')

    TAI93_EPOCH = np.datetime64('1993-01-01', 'ns')

    # We need to convert the leap_sec_dates to a TAI93 epoch,
    # and then to seconds.
    leap_tai93 = (leap_sec_dates - TAI93_EPOCH).astype('int64')/10**9

    # Count the number of leap seconds that were added:
    offsets = np.searchsorted(leap_tai93, times)
    offsets = np.array(offsets, dtype='timedelta64[s]')

    # Translate our TAI93 times to UTC epoch
    t_times = time_to_utc(times, TAI93_EPOCH)

    # To get UTC, we need to take the number of leap seconds "off" TAI93
    utc_times = t_times - offsets

    return utc_times


def time_to_utc(times, epoch):
    """
    Convert a set of fractional times of a given epoch to UTC.


    Datetimes/timedeltas are a bit of pain, since they can't handle
    fractional seconds. This is particularly a problem when you have data
    that contains the number of possibly fractional seconds past some
    epoch (e.g., seconds past midnight). This function will handle this
    common operation by splitting the fractional seconds into a whole
    count and a fractional count, then adding it to the epoch.

    One common use would be to add a measure of seconds past midnight
    to day of the measurement, to get a UTC time.

    Examples
    ---------

    Say you have a time measurement for the occurence for an event
    since midnight May 11, 2011::

        >>> times = np.array([1.1, 3600.1, 86399.9999999])
        >>> t0 = '2011-05-11'

    Then, to get these times in UTC::

        >>> time_to_utc(times, t0)
        array(['2011-05-11T00:00:01.100000000', '2011-05-11T01:00:00.099999999',
        '2011-05-11T23:59:59.999999899'], dtype='datetime64[ns]')


    Parameters
    ----------
    times : NumPy array
        The array of times relative to some epoch. Usually an array of floats.
    epoch : varies
        The epoch. This will be made into a NumPy Datetime64[ns] variable, so
        anything that can be accepted there is valid here. The epich should
        be relative to UTC.

    Returns
    -------
    NumPy Datetime64[ns]:
        The translated times to UTC.

    """

    # First, make sure our epoch is in units of nanoseconds.
    epoch = np.datetime64(epoch, 'ns')

    # To add possibly fractional seconds, we are going to split things
    # up into whole seconds and fractional seconds. We can't just use
    # np.floor, since that gives us a float. Sigh.
    # So, get a integer, then a timedelta:

    times_frac = times % 1
    # Get a nanosecond count:
    times_frac = np.floor(times_frac*10**9).astype('int').astype('timedelta64[ns]')

    # Go ahead and get the whole seconds, as a TimeDelta:
    times_whole = np.floor(times).astype('int').astype('timedelta64[s]')

    new_time = epoch + times_whole + times_frac

    return new_time
