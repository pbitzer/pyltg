# -*- coding: utf-8 -*-
"""

Module for Earth Networks Total Lightning Network (ENTLN) data.

Examples
---------
ENTLN files that contain pulse data come in two different types:
legacy CSV gzipped files and new(er) JSON files, which have both flashes
and pulses.

To read in JSON files:
    f = 'LtgFlashPortions20180403.csv.gz'
    eni = ENTLN(f)

Older use is to just initialize the class with a ENTLN pulse file::

    f = 'FLASHES_2023-12-18T01-10.json'
    eni = ENTLN(f)

If you want flash data from a JSON file, you can get it. However, it
won't be loaded into a `Ltg` class (yet). But if you want it, you can do it:

    f = 'FLASHES_2023-12-18T01-10.json'
    pulses, flashes = read_json(f, get_flashes=True)
    eni_fl = ENTLN()
    eni_fl._add_record(flashes)

"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


def _fix_fields(data: pd.DataFrame):
    # Rename fields and convert units

    data.rename(columns=
                {'amplitude': 'current',
                 'altitude': 'alt',
                 'timeStamp': 'time',
                 'latitude': 'lat',
                 'longitude': 'lon',
                 'numSensors': 'num_sensors'
                 },
                inplace=True)

    data.alt /= 1e3  # km, please
    data.current /= 1e3  # kA, please
    data.time = data.time.astype(np.datetime64)

    return data


def _exclude_fields():
    # Define the fields to be excluded when reading in JSON files.
    pulse_fields = [
        'kafkaInsertionEpoch',
        'id',
        'region',
        'version',
        'stationsDetecting',
        'fifthNearestSensorDistance',
        'residual',
        'solutionCount'
        'stationAmplitudes',
        'stationOffsets',
        'stationsOn',
        'stationsDetecting',
        'altTimeCorrection',
        'altQuality',
        'altAlgorithm',
        'biggestAngle',
        'waveformSignature',
        'classificationConfidence',
        'height',  # seems to be duplicated by altitude field
        'riseTime',
        'fallTime',
        'halfWidth',
    ]

    flash_fields = [
        'flashGuid',
        'version',
        'boundingBox',
        'buddies',
        'buddiesTimeStamps',
        'duration',
    ]

    return pulse_fields, flash_fields

def read_json(file, get_flashes=False):
    """
    Read in JSON-formatted ENTLN pulse files.

    Parameters
    ----------
    file: str
        The file name(s) to be read in. (Multiple filenames OK.)
        For multiple filenames, use list-like type.
    get_flashes : bool, default: False
        If true, get the flash data in the file. Right now, this
        gets returned as a separate `Ltg` class.

        Likely to be changed in the future.

    Returns
    -------
    Pandas dataframe with pulses, unless `get_flashes` is set.
    If so, then two `Ltg` classes are returned as a list: `[pulses, flashes]'
    (see note in the `get_flashes` parameter about future).

    """
    import json
    flashes = list()
    pulses = list()

    not_needed_pulse_fields, not_needed_flash_fields = _exclude_fields()

    # TODO Need flash id for tracking, and assign to pulse
    with open(file, 'r') as of:
        # Read in the file, but do it line-by-line.
        # Necessary because of how the files are formatted.
        for line in of:
            this_val = json.loads(line)

            # Separate out flashes and pulses
            this_pulses = this_val.pop('pulses')

            # Clean the pulses a bit
            for _p in this_pulses:
                # Get rid of the fields we don't need
                for _nn in not_needed_pulse_fields:
                    try:
                        del _p[_nn]
                    except KeyError:
                        pass

                # Unpack error ellipse
                err = _p.pop('errorEllipse')

                if err is not None:
                    major, minor, ellp = err['majorAxis'], err['minorAxis'], err['majorAxisBearing']
                else:
                    major, minor, ellp = np.nan, np.nan, np.nan

                _p['err_major_axis'] = major
                _p['err_minor_axis'] = minor
                _p['err_axis_bearing'] = ellp

            pulses.extend(this_pulses)

            # Now, the flashes!
            if get_flashes:
                # Get rid of the fields we don't need
                for _nn in not_needed_flash_fields:
                    try:
                        del this_val[_nn]
                    except KeyError:
                        pass

                flashes.append(this_val)

    # Do a little data sanitization....
    pulses = pd.DataFrame(pulses)
    pulses = _fix_fields(pulses)

    if get_flashes:
        flashes = pd.DataFrame(flashes)

        # Clean the flashes a bit.
        # First, the fields not handled internally:
        flashes.rename(columns=
                       {'startTimeStamp': 'start_time',
                        'endTimeStamp': 'end_time',
                        'height': 'alt',
                        },
                       inplace=True
                       )
        flashes.start_time = flashes.start_time.astype(np.datetime64)
        flashes.end_time = flashes.end_time.astype(np.datetime64)

        # Now the rest
        flashes = _fix_fields(flashes)

    if get_flashes:
        retval = [pulses, flashes]
    else:
        retval = pulses

    return retval


def read_ascii(file, full=False):
    """
     Read the data from a filename(s) for an ASCII ENTLN pulse file.

     Parameters
     ----------
     file: str
         The file name(s) to be read in. (Multiple filenames OK.)
         For multiple filenames, use list-like type.
     full : bool, default: False
         If true, try to read in some of the solution attributes.
         Experimental.

    Returns
    -------
    Pandas dataframe

    """

    # Define the columns/types, as they are in the file
    types = {'flashPortionHistoryID': np.int64, 'flashPortionID': str, 'flashID': str,
             'nullTime': str, 'time': str,
             'lat': np.float, 'lon': np.float, 'alt': np.float,
             'type': str,
             'amp': np.float}

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

    # TODO use fix_fields
    # Rename the columns
    rawData.rename(columns={'amp': 'current'}, inplace=True)

    rawData.alt /= 1e3  # km, please
    rawData.current /= 1e3  # kA, please

    return rawData

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

    def readFile(self, filename, file_type=None):
        """
        Read the data from a filename(s) for an ENTLN file.

        You can read in either ENTLN pulse ASCII files or JSON files
        with both pulses and flashes. See the functions (not methods)
        `read_ascii` and `read_json`. This method will attempt to figure out
        which one based on the file
        extension.

        Parameters
        ----------
        filename: str
            The file name(s) to be read in. (Multiple filenames OK.)
            For multiple filenames, use list-like type.
        file_type: str
            The type of file(s) to be read in. If None, the type of file will
            be guessed from the extenstion. Current allowed values are either
            'JSON' or 'GZ'.
        """

        if file_type is None:
            # Try to guess at the type of file....
            file_ext = Path(np.atleast_1d(filename)[0]).suffix.upper()

            if file_ext == '.JSON':
                file_type = 'JSON'
            elif file_ext == '.GZ':
                file_type = 'GZ'
            else:
                msg = "Unable to guess file type from extension."
                raise FileNotFoundError(msg)

        if file_type.upper() == 'JSON':
            data = read_json(filename)
        elif file_type.upper() == 'GZ':
            data = read_ascii(filename)
        else:
            msg = f"Invalid file type {file_type}"
            raise FileNotFoundError(msg)

        self._add_record(data)


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
