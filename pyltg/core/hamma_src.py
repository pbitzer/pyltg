# -*- coding: utf-8 -*-
"""
Module to read in Huntsville Alabama Marx Meter Array (HAMMA) source data,
i.e., HAMMA Level 2 (L2) data.

.. note::
    This module (and class) only handles the source data. To get more
    information about the array that recorded the data, see the `hamma
    repository
    <https://www.nsstc.uah.edu/users/phillip.bitzer/python_doc/hamma/>`_.
    For example, to get the locations of the sensors,
    see the appropriate function in that repo.

.. warning::

    HAMMA L2 data is currently ASCII based, and not self-documented
    very well. This is likely to be changed in the near future. However,
    any changes to the underlying files will be accounted for in this
    class. The basic API will remain the same, so you can be comfortable
    using the same code.

.. warning::
    Right now, the L2 data contains (natively) x,y position relative to
    a coordinate system. This will be changed. While you do not have to
    worry about changing code for this change, this means right now
    this class only works for RELAMPAGO data, i.e., data from the CAMMA
    (Cordoba Argentina Marx Meter Array).


Examples
---------
Basic use is to just initialize the class with a HAMMA L2 file::

    from pyltg import HAMMA
    import numpy as np

    f1 = '20181103/2018-11-03T00-27-48-991_761_6.txt'
    h1 = HAMMA(f)

You can load multiple files at once::

    f = ['/Volumes/relampago/level2/v1.1/source_files/20181103/2018-11-03T00-27-48-991_761_6.txt',
         '/Volumes/relampago/level2/v1.1/source_files/20181103/2018-11-03T00-28-31-110_811_6.txt']

    h_src = HAMMA(f)

Print the time of all the sources::

    h_src.time

Just get indices of the sources in a a certain period::

    t0 = np.datetime64('2018-11-03T00:28:31.0', 'ns')  # times must be datetime64[ns]
    t1 = np.datetime64('2018-11-03T00:28:31.6', 'ns')
    cnt = h_src.limit(time=[t0, t1])  # cnt should be 519

The HAMMA data for these times::

    hamma_subset = h_src.get_active()
    hamma_subset.time

Of course, you don't need to extract the active data::

    h_src.time

Note this is returned as a Pandas Dataframe, not a new HAMMA object.

To restrict in time and space::

    cnt = h_src.limit(time=[t0, t1], lat=[-31.66, -31.6], lon=[-63.85, -63.8])

Plot all the data::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(h_src.data.lon, h_src.data.lat)

Now, overplot the "active" data, i.e., the data that falls within the limits provided::

    ax.scatter(h_src.lon, h_src.lat)

"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg
from pyltg.utilities.latlon import enu2lla
from pyltg.utilities.time import time_to_utc


def _col_names(n_columns):

    """
    Get the column names for a HAMMA source file.
    Right now, 11 columns will always be there, plus arrival times
    for all the sensors in the array.

    Parameters
    -----------
    n_columns: int
        The number of columns in the file. This is used to determine how
        many columns of arrival times are in the file beyond the "core"
        11 columns.

    """
    cols = ['x', 'y', 'z',  # The local Cartesian coords
            '_time',  # seconds past midnight
            'chi2',
            'current',
            'type',
            'err_x',
            'err_y',
            'err_z',
            'err_t',
            ]

    n_sensors = n_columns - len(cols)
    arrival_cols = ['arrival_' + '{0:0d}'.format(s) for s in np.arange(1, n_sensors+1)]

    return cols + arrival_cols


def _epoch_from_file_name(files):

    """
    Currently, the L2 data references times relative to an epoch. This
    is embedded in the filename. This function gets that time.
    """

    from pathlib import Path

    files = np.atleast_1d(files)

    times = list()

    for _f in files:
        base = Path(_f).name
        ymd = base.split('T')[0]

        times.append(pd.Timestamp(ymd))

    return times


class HAMMA(Ltg):
    """
    Class to handle HAMMA source data.

    Many of the following are not attributes of the class, but
    are columns in the underlying Dataframe. But, you can access them
    as you would an attribute. There may also be others, depending on
    the file read in.

    Attributes
    -----------
        _data : Dataframe
            The underlying data. A "real" attribute of the class.

        time : numpy.datetime64[ns]
            The source time of the source. UTC.
        lat : numpy.float
            The latitude of the source.
        lon : numpy.float
            The longitude of the source.
        alt : numpy.float
            The altitude of the source in km.
        chi2 : numpy.float
            The reduced :math:`\chi^2` value of the solution.
        x,y,z : numpy.float
            The Cartesian x,y,z position in the local reference frame. Will be
            deprecated.
        current : numpy.float
            The peak current of the source. Not currently used.
        type : unknown
            The classification (IC, CG) of the source. Not currently used.
        err_x : numpy.float
            The error in the x position (km)
        err_y : numpy.float
            The error in the y position (km)
        err_z : numpy.float
            The error in the height (km)
        arrival_1 : numpy.float
            The arrival time of the radiation to sensor 1. This is seconds
            past midnight. There will be `n` more of these fields,
            corresponding the the arrival times at each sensor. To get
            the locations of the sensors, see the appropriate function
            in the hamma repository.

    """

    def __init__(self, files=None):
        """

        Parameters
        ----------
        files : str
            The file(s) to be read in.
        """

        super().__init__()  # initialize the inherited baseclass

        if files is not None:
            self.readFile(files)

    def readFile(self, files):
        """
        Read the given file.

        If all the passed files are empty (i.e., no data) then then the `len`
        of the class is 0.

        Parameters
        ----------
        files : str
            The file(s) to be read in.

        """

        files = np.atleast_1d(files)

        CENTER = [-31.6671004, -63.8828453, 0]

        sources = list()

        for _file in files:
            this_src = pd.read_csv(_file, header=None)

            this_src.columns = _col_names(len(this_src.columns))

            # Convert time to UTC
            t0 = _epoch_from_file_name(_file)[0]

            time = time_to_utc(this_src._time, t0)

            this_src['time'] = time

            # Drop old seconds past midnight
            this_src.drop(columns='_time', inplace=True)

            sources.append(this_src)
        try:
            self._data = pd.concat(sources, ignore_index=True)
        except ValueError:
            # This can happen when we nothing to concat (all files empty)
            print('No data in these files')

        # Now, convert x,y,z to lla:
        lla = enu2lla(self._data.x.values, self._data.y.values, self._data.z.values, center=CENTER)

        self._data['lat'] = lla.lat
        self._data['lon'] = lla.lon
        self._data['alt'] = lla.alt

        # todo: drop x,y,z

        # todo: drop arrival times

    def get_arrival_times(self, idx):
        """
        Get the arrival times for a particular source(s).

        Sensors that don't participate will be NaN'd

        Parameters
        ----------
        idx : scalar or list/1D array
            The row(s) corresponding to the sources you want the
            arrival times for.

        Returns
        -------
        times: NumPy array
            The arrival times corresponding to the sources requested.
            If a scalar is used for `idx`, get an n-element array where `n`
            is the number of arrival times (sensors).
            Otherwise, get a m-by-n array, where `m` is the number of sources.
            Sensors that don't participate will be NaN'd.

        """

        col_names = [col for col in self.data.columns if col.startswith('arrival')]


        if np.isscalar(idx):
            n_locations = 1

        else:
            n_locations = len(idx)

        times = np.empty((n_locations, len(col_names)), dtype='float')

        for i, col in enumerate(col_names):
            times[:, i] = self.data.iloc[idx][col]

        if np.isscalar(idx):
            # get rid of the unused dim:
            times = times.flatten()

        times = np.where(times == 0.0, np.nan, times)

        return times
