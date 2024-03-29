# -*- coding: utf-8 -*-
"""

Module for Earth Networks Total Lightning Network (ENTLN) data.

Examples
---------
Basic use is to just initialize the class with a ENTLN pulse file::

    f = 'LtgFlashPortions20180403.csv.gz'
    eni = ENTLN(f)

"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


class ENTLN(Ltg):
    """
    Class to handle Earth Networks Total Lightning Network data.


    Many of the following are not attributes of the class, but
    are columns in the underlying Dataframe. But, you can access them
    as you would an attribute....

    Attributes
    -----------
        _data : Dataframe
            The underlying data. A "real" attribute of the class.

        flashID : str
            The flash ID
        time : numpy.datetime64[ns]
            The source time of the pulse.
        lat : numpy.float
            The latitude of the pulse
        lon : numpy.float
            The longitude of the pulse
        alt : numpy.float
            The altitude of the pulse, in kilometers. This field is more
            or less nonsense.
        type : str
            The pulse type, either C (intracloud) or G (cloud-to-ground)
        currrent : numpy.float
            The peak current of the pulse, in kiloamps.
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

    def readFile(self, filename, full=False):
        """
        Given a filename(s), load the data into the object.

        Parameters
        ----------
        filename: str
            The file name(s) to be read in. (Multiple filenames OK.)
            For multiple filenames, use list-like type.
        full : bool, default: False
            If true, try to read in some of the solution attributes.
            Experimental.

        """

        # Define the columns/types, as they are in the file
        types = {'flashPortionHistoryID': np.int64, 'flashPortionID': str, 'flashID': str,
                 'nullTime': str, 'time': str,
                 'lat': float, 'lon': float, 'alt': float,
                 'type': str,
                 'amp': float}

        # Make a dict for the keywords to read_csv, no matter what:
        pdArgs = {'skiprows': 1,
                  'chunksize': 1000000}

        if full:
            # We're going to try to read some solution attributes
            # (but not all!)
            nExtra = 8
            extraNames = ['extra' + str(i).zfill(3) for i in np.arange(nExtra)]
            extraTypes = [str for i in np.arange(nExtra)]
            extra = dict(zip(extraNames, extraTypes))
            types = {**types, **extra}
        else:
            # If we're not doing full, add a couple more keywords
            # to read in faster
            pdArgs.update({'comment': "{",
                           'na_filter': False})

        # Finally, add in the rest of the keywords:
        # (Needs to be after the check for full in case we get more fields)
        pdArgs.update({'names': list(types.keys()),
                       'usecols': list(types.keys()),
                       'dtype': types})

        # We'll read the chunks into a list:
        rawData = list()

        for this_file in np.atleast_1d(filename):

            reader = pd.read_csv(this_file, **pdArgs)

            for chunk in reader:
                # The first two columns are not relevant for us, and
                # the "nullTime" field is largely redundant:
                # AUTHOR NOTE: Never could get usecols to work to not
                # have to do this
                chunk.drop(['flashPortionHistoryID', 'flashPortionID',
                            'nullTime'], axis=1, inplace=True)

                # Reinterpret the time string field as datetime64:
                chunk.time = chunk.time.astype('datetime64')

                # change the ype field to a string of G or C (for CG/IC)
                chunk.type.replace(to_replace={'0': 'G', '1': 'C'},
                                   inplace=True)

                # todo: process the "extra: fields
                # todo: change flashID to some sort of integer (from string)
                rawData.append(chunk)

        # Finally, make the "whole" dataframe
        rawData = pd.concat(rawData, ignore_index=True)

        # Rename the columns
        rawData.rename(columns={'amp': 'current'}, inplace=True)

        rawData.alt /= 1e3  # km, please
        rawData.current /= 1e3  # kA, please

        self._add_record(rawData)


def split_file(file):
    """
    Split a ENI daily pulse file into hourly files.

    Earth Networks pulse files are absurdly large, since it is pulse level
    data for the entire world for an entire day. This makes reading/moving/etc
    these files more difficult than it should be. This routine will take
    one of these files and split it into hourly files.

    Each of the new files will be placed in the same location as the original,
    with an 2 character hour indicator appended and gzipped. They will
    have the same header line as the daily file, so all your current code
    should work on them.

    Parameters
    ------------
    file : str
        The ENI pulse file. Must be gzipped. Files are assumed to have
        "two" extensions (e.g., `.csv.gz`)

    """

    import gzip
    from os.path import splitext

    # Because pulses could be out of order, we going to load everythin
    # into a dictionary of lists, separated by hour. Then, we'll write
    # everything out.

    data = {x: list() for x in np.arange(24)}

    # Get basefilename with path, but w/o extension.
    # Assume multiple extensions (e.g., basename.csv.gz)
    basename, ext1 = splitext(file)
    basename, ext2 = splitext(basename)

    # Open the file
    with gzip.open(file, 'rt') as _f:
        # Read the first line. We're going to put this at the top of
        # every new file.
        hdr = _f.readline()

        # Put the header in each list:
        for key, val in data.items():
            data[key].append(hdr)

        # Now, we'll go about reading each line (sigh)
        for line in _f:
            this_date = line.split(',')[4]
            _, hms = this_date.split('T')
            this_hr = int(hms[0:2])

            data[this_hr].append(line)

    # At this point, we have all the data. Write the files, But first,
    # get a helper function (relic from old code)
    def _write_file(new_name, data_to_write):
        with gzip.open(new_name+'.gz', 'wt') as _newf:
            for item in data_to_write:
                _newf.write("%s\n" % item)

    for hr, val in data.items():
        new_filename = basename + '_{:02}'.format(hr) + ext2
        _write_file(new_filename, val)
