# -*- coding: utf-8 -*-
"""

Module to read in National Lightning Detection Network (NLDN)
or Global Lightning Detection 360 (GLD360) data, both from
Vaisala.

This works only for stroke/pulse level data, not flash level.
(It might, just not tested.)

.. warning::
    Improvements really need to made to try to speed the
    read of the files. It can take a rather long time
    right now.

"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


class NLDN(Ltg):
    """
    Class to handle Vaisala data, either NLDN or GLD.

    Under the hood, the data is simply a pandas DataFrame.

    Attributes
    -----------
        _data : Dataframe
            The underlying data. A "real" attribute of the class.
        flashID : str
            The flash ID
        time : numpy.datetime64[ns]
            The source time of the stroke/pulse.
        lat : numpy.float
            The latitude of the stroke/pulse.
        lon : numpy.float
            The longitude of the stroke/pulse.
        alt : numpy.float
            This is zero-filled, for consistency with the base Ltg class.
        type : str
            The pulse type, either C (intracloud) or G (cloud-to-ground)
        current : numpy.float
            The peak current of the pulse, in kiloamps.
        _multi : int
            The multiplicity. For stroke/pulse data, this is zero for
            all strokes.
        semimajor : numpy.float
            The length of the semimajor axis of the 50% confidence interval
            of the solution, in kilometers.
        semiminor : numpy.float
            The length of the semiminor axis of the 50% confidence interval
            of the solution, in kilometers.
        axis_ratio : numpy.float
            The ratio of the semimajor and semiminor axes. NOTE: Likely will be
            deprecated in the future.
        azimuth : numpy.float
            The angle of the 50% confidence interval ellipse, relative to
            north (I think).
        chisq : numpy.float
            The (reduced, I think) :math:`\chi^2` of the solution. NOTE: Likely
            to be renamed to `chi2` in the future.
        num_sensors : numpy.int
            The number of sensors participating in the solution.

    """

    def __init__(self, filename=None):
        """
        If you don't provide a file name, you'll have to call :meth:`readFile`
        yourself to actually do anything useful.

        Parameters
        ----------
        filename : str
            The file name to be read in.

        """

        super().__init__()  # initialize the inherited baseclass

        if filename is not None:
            self.readFile(filename)

    def readFile(self, filename):
        """
        Given a filename, read the data and load it into the object.

        Parameters
        ----------
        filename : string
            The file name to be read.

        """

        if isinstance(filename, list):
            if len(filename) > 1:
                print('Multiple files not allowed yet')
            filename = filename[0]

        colNames = ('date', 'time', 'lat', 'lon', 'current', '_kA', '_multi', 'semimajor',
                    'semiminor', 'axis_ratio', 'azimuth', 'chisq', 'num_sensors', 'type')

        rawData = list()

        chunkSize = 100000  # How much of the file do we read at once?

        reader = pd.read_csv(filename, names=colNames, header=None,
                             delim_whitespace=True, iterator=True)

        # Read the first line:
        line = reader.get_chunk(1)

        # Get the date from this line, and assume all dates are the same:
        date = line.date.str.split('/|-')

        year = date.str[2].values
        month = date.str[0].values
        day = date.str[1].values

        if int(year[0]) < 90:
            year = '20' + year
        else:
            year = '19' + year

        # Drop columns we don't need:
        line.drop(['date', '_kA'], axis=1, inplace=True)

        rawData.append(line)

        # Now, read in the rest, chunking:
        reader.chunksize = chunkSize
        for chunk in reader:
            chunk.drop(['date', '_kA'], axis=1, inplace=True)

            rawData.append(chunk)

        rawData = pd.concat(rawData)

        # There are certain NLDN data files that have duplicates, namely
        # the "enhanced" GLD-NLDN data files (These contain locations from
        # both GLD360 and NLDN). Try to drop these duplicates:
        rawData.drop_duplicates(inplace=True)
        # NOTE: dropping duplicates along the way doesn't help speed

        ymd = year + '-' + month + '-' + day + 'T'

        rawData.time = np.array(ymd + rawData.time.values, dtype='datetime64')

        self._add_record(rawData)
