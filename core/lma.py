# -*- coding: utf-8 -*-
"""
Module to read in Lightning Mapping Array (LMA) data.

Examples
---------
Basic use is to just initialize the class with a LMA source file::

    from pyltg import LMA
    import numpy as np

    f = 'nalma_lylout_20170427_07_3600.dat.gz'
    lma = LMA(f)

Print the time of all the sources::

    lma.time

Just get indices of the sources in a 10 minute period::

    t0 = np.datetime64('2017-04-27T07:00:00', 'ns')  # times must be datetime64[ns]
    t1 = np.datetime64('2017-04-27T07:10:00', 'ns')
    ind, cnt = lma.limit(time=[t0, t1])

The LMA data for these times::

    lma_subset = lma[ind[0]]

Note this is returned (for now) as a Pandas Dataframe, not a new LMA object.

.. warning::

    This is likely to be changed in the future!

To restrict in time and space::

    ind, cnt = lma.limit(time=[t0, t1], lat=[33, 35], lon=[-87, -85])

"""

import re

import numpy as np
import pandas as pd

import h5py

from pyltg.core.baseclass import Ltg


def flash_sort(files,
               ctr_lat=None, ctr_lon=None,
               logpath=None, savepath=None,
               stations=[6, 21], chi2=[0, 2.0], alt=[0, 20000],
               distance=3000.0,
               thresh_duration=3.0, thresh_critical_time=0.15,
               merge_critical_time=0.5, mask_length=8,
               ascii_flashes_out='flashes_out.dat'):
    """
    Sort the given files into flashes.

    This is essentially a wrapper for the flash sorting done by `lmatools`,
    so it requires this package.

    Most parameters are passed straight to the requisite function in
    `lmatools` and are documented there. Selected ones are documented here.

    Parameters
    -----------
    files: str or sequence of string
        The files to read in. Should ASCII files with LMA sources.
    logpath: str
        Where the flash sorting logs are saved. Defaults to tmp/
        relative to the path of the first `files`. If the path doesn't exist,
        we'll make it.
    savepath: str
        Where the flash sorting files are saved. Defaults to flash_sort/
        relative to the path of the first `files`. If the path doesn't exist,
        we'll make it.
    ctr_lat: float
        The center latitude of the LMA for the files. If this _or_ `ctr_lon`
        is not given, we'll peek in the first file to find the coordinate
        center. This is then used for all the files.
    ctr_lon: float
        The center longitude of the LMA for the files. If this _or_ `ctr_lat`
        is not given, we'll peek in the first file to find the coordinate
        center. This is then used for all the files.

    """

    from pathlib import Path

    from lmatools.flashsort.autosort.autorun import run_files_with_params, logger_setup
    from lmatools.flashsort.autosort.autorun_sklearn import cluster

    files = np.atleast_1d(files)

    if logpath is None:
        logpath = Path(files[0]).parent.joinpath('tmp/')

    logpath.mkdir(parents=True, exist_ok=True)

    logger_setup(logpath)

    if savepath is None:
        savepath = Path(files[0]).parent.joinpath('flash_sort/')

    savepath.mkdir(parents=True, exist_ok=True)

    if (ctr_lat is None) or (ctr_lon is None):
        ctr_lat, ctr_lon = _get_center_from_file(files[0])

    # Make keyword
    params = {'stations': (6,21),
              'chi2': (0,2.0),
              'ascii_flashes_out': 'flashes_out.dat',
              'ctr_lat': ctr_lat, 'ctr_lon': ctr_lat,
              'alt': (0, 20000.0),
              'distance': 3000.0,
              'thresh_duration': 3.0, 'thresh_critical_time': 0.15,
              'merge_critical_time': 0.5, 'mask_length': 8
              }

    run_files_with_params(files, savepath, params, cluster, 
                          retain_ascii_output=False, cleanup_tmp=True)


def get_flashes(lma_data, min_sources=None):
    """
    Given a Pandas DataFrame of LMA flash-sorted data (like from the `LMA`
    class), group the sources into the flashes.

    This does not do flash sorting. Instead, this function provides a way
    to extract the flashes that have already been sorted.

    Parameters
    -----------
    lma_data : DataFrame
        The DataFrame of LMA data. Should match the DataFrame used internally
        by the `LMA` class
    min_sources : scalar
        If not None, then only return flashes that have at least the specified
        number of sources. This seems to take some time, perhaps more time
        than simply skipping these flashes in subsequent processing.


    Returns
    -------
    Pandas GroupByDataFrame

    Each key in the GroupByDataFrame will be the flash ID.

    Examples
    ---------

    l = LMA(file)
    flashes = group_flashes(l.data)

    """

    def _get_bins(flash_ids):
        # This helper gets sorted, unique bins that span the given flash IDs
        bins = flash_ids
        bins.sort()
        bins = np.unique(bins)

        # Append one more bin to make sure we don't hit the weird
        # end-of-bins effects when "cutting
        bins = np.append(bins, bins[-1]+1)

        return bins

    # First, create bins for the flash IDs
    bins = _get_bins(lma_data.flash_id.values)

    # Define the cut to be used for the groupby:
    _cut = pd.cut(lma_data.flash_id, bins, labels=False, right=False)

    fl = lma_data.groupby(_cut)

    # todo: here, we would check for other flash constraints
    if min_sources is not None:

        fl_data = fl.filter(lambda x: len(x) >= min_sources)

        # Now, we cut and group again:
        bins = _get_bins(fl_data.flash_id.values)
        _cut = pd.cut(fl_data.flash_id, bins, labels=False, right=False)
        fl = fl_data.groupby(_cut)

    return fl


def _get_center_from_file(file):
    # Go get the coordinate center from a LMA source file
    from itertools import islice
    import gzip

    with gzip.open(file, 'rt') as thisFile:
        poss_hdr = list(islice(thisFile, 100))

    # This should contain at least the whole header for LMA files.
    # Parse it to find the start of the data
    # todo: refactor this a function?

    match_text = r"^Coordinate center"
    center_line = idxMatch(poss_hdr, match_text)[0]

    # From this line, split this into the fields.
    # The last three are lat, lon, alt.
    center_split = poss_hdr[center_line].split()

    # Everything up to this line is the header
    lat, lon = float(center_split[-3]), float(center_split[-2])

    return lat, lon


def idxMatch(lst, text2match):
    """
    Simple helper to parse through a text list to find the index
    that matches an input string.
    """

    txtRE = re.compile(text2match, re.IGNORECASE)  # TODO: keyword to ignore case?

    idx = [i for i, _ in enumerate(lst) if txtRE.search(_)]

    return idx


class LMA(Ltg):
    """
    Class to handle LMA source data.

    Many of the following are not attributes of the class, but
    are columns in the underlying Dataframe. But, you can access them
    as you would an attribute. There may also be others, depending on
    the file read in.


    .. warning:
        If you read in multiple flash HDF files, the flash IDs are not
        handled correctly (yet).

    Attributes
    -----------
        _data : Dataframe
            The underlying data. A "real" attribute of the class.

        flashID : str
            The flash ID. Note this is only present if you have a "flash" file
        time : numpy.datetime64[ns]
            The source time of the source.
        lat : numpy.float
            The latitude of the source.
        lon : numpy.float
            The longitude of the source.
        alt : numpy.float
            The altitude of the source.
        chi2 : numpy.float
            The reduced :math:`\chi^2` value of the solution.
        power : numpy.float
            The power of the pulse, in dB
        mask: str
            The station mask. This is a hex code that corresponds to
            which sensors participated in the solution.

    """
    def __init__(self, files=None):
        """

        Parameters
        ----------
        files : str
            The file(s) to be read in.
        """

        super().__init__()  # initialize the inherited baseclass

        # todo: this doesn't need to be a instance method. Either make it class method, or make it a function.
        self.__colNames()  # mapping from columns in the file to object props

        if files is not None:
            self.readFile(files)

    def __colNames(self):
        """
        Map the column names. Make sure time, lat, lon, alt are included.

        LMA files come in different flavors and use different internal names
        for the same fields (sigh). This method provides some of the most used
        internal names and map them to a common name.

        """

        # Each key is the name that's in the file, and the value is the
        # name that's used by the object. This means we could have
        # several keys with the same value...
        # Defining by inverting keys/values would mean a more difficult lookup
        self.colNames = {
                         # These are standard fields for Ltg class
                         'time (UT sec of day)': 'time',
                         'lat': 'lat',
                         'lon': 'lon',
                         'alt(m)': 'alt',
                         # LMA specific fields
                         'reduced chi^2': 'chi2',
                         'P(dBW)': 'power',
                         'mask': 'mask'}

    def readFile(self, files):
        """
        Read the given file.

        Parameters
        ----------
        files : str
            The file to be read in.

        """

        files = np.atleast_1d(files)

        sources = list()

        for _file in files:
            if h5py.is_hdf5(_file):
                this_data = self._readHDF(_file)

                # We need to modify flash IDs when reading multiple files
                # to ensure they are unique
                if len(sources) != 0:
                    _ctr = sources[-1].flash_id.max()
                    this_data.flash_id += _ctr+1
            else:
                # Assume it's ASCII
                this_data = self._readASCII(_file)

            sources.append(this_data)

        self._data = pd.concat(sources, ignore_index=True)

        self._data.alt /= 1e3  # convert to km

    def _readASCII(self, file):
        """
        Read in an LMA source ASCII file.

        """

        from itertools import islice
        import gzip

        with gzip.open(file, 'rt') as thisFile:
            possHdr = list(islice(thisFile, 100))

            # This should contain at least the whole header for (most) LMA files.
            # Parse it to find the start of the data

            dataText = r"^.*\*+.*data.*\*+.*"
            dataLine = idxMatch(possHdr, dataText)[0]

            # Everything up to this line is the header
            hdr = possHdr[0:dataLine]

            # Now, find the line that defines the columns in the file
            colText = "Data:"
            colLine = idxMatch(hdr, r"^"+colText)

            # We have the line, strip it down so that we only have the cols
            fileCols = hdr[colLine[0]][:-1]  # strip trailing /n
            fileCols = fileCols.split(colText)[1]  # strip the column line marker
            fileCols = fileCols.split(',')  # Make it a list
            fileCols = [_.strip() for _ in fileCols]  # remove trailing/leading space

            colNames = [self.colNames[col] for col in fileCols]

        # Get the data and assign the column names
        this_src = pd.read_csv(file, compression='gzip', header=None,
                               delim_whitespace=True,
                               skiprows=dataLine+1, names=colNames)

        # Extract the day of the data:
        dayString = r"Data start time:"
        dayLine = idxMatch(hdr, dayString)

        date_time = hdr[dayLine[0]].split(dayString)[1]
        # The date could be sep by / or -....
        date = re.findall(r"..[/|-]..[/|-]..", date_time)

        date = re.split(r"/|-", date[0])

        # Offset to either 2000 or 1900:
        if int(date[2]) < 90:
            date[2] = '20' + date[2]
        else:
            date[2] = '19' + date[2]

        date = np.datetime64(date[2] + '-' + date[0] + '-' + date[1], 'ns')

        secs = this_src.time.astype('int64').astype('timedelta64[s]')
        secs = secs.astype('timedelta64[ns]')
        secsFrac = ((this_src.time % 1)*1e9).astype('timedelta64[ns]')

        # Cast the times as datetime64[ns]....
        this_src.time = date + secs + secsFrac

        return this_src

    def _readHDF(self, file):
        """
        Read in a HDF5 Flash sorted files, of the type lmatools produces.

        This will only work if there's only one "events" dataset in
        the file. Nominally, this is how one would interact with lmatools,
        however, so it shouldn't be a big problem.

        """
        with h5py.File(file, 'r') as h5_file:

            keys = list(h5_file.keys())

            if keys.count('events') != 1:
                print('Invalid file - wrong number of "events" datasets')
                # todo: raise exception
                return

            # We need to get the group, then extrct the dataset
            # AND, get the "value" to get it into an array
            ev_dataset = list(h5_file.get('events').values())[0]

            data = ev_dataset.value

            t0_char = ev_dataset.attrs['start_time'].decode()

        # Make this a DataFrame before we start manipulating:
        data = pd.DataFrame(data)

        # First, drop columns we don't need:
        if 'charge' in data.columns:
            data.drop(columns='charge', inplace=True)

        # todo: Do we need to map the column names to ones used by Ltg Class?

        # Next, the time is saved as an offset to a epoch. We don't want
        # this, we want an absolute time. Things are slightly
        # complicated because this epoch is saved as a string.

        # The field are separated by "L". Extract the year, month, day.
        # We're going to assume the times are relative to midnight to this day.
        _start = t0_char.split('L')

        # todo: refactor this ... largely the same as in readFile
        date = np.datetime64('{1:0>4}-{3:0>2}-{5:0>2}'.format(*_start), 'ns')
        secs = data.time.astype('int64').astype('timedelta64[s]')
        secs = secs.astype('timedelta64[ns]')
        secsFrac = ((data.time % 1)*1e9).astype('timedelta64[ns]')

        # Cast the times as datetime64[ns]....
        data.time = date + secs + secsFrac

        return data
