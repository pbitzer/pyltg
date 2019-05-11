# -*- coding: utf-8 -*-
"""

Module for Earth Networks Total Lightning Network (ENTLN) data.

Examples
---------
Basic use is to just initialize the class with a ENTLN pulse file::
    
    f = 'LtgFlashPortions20180403.csv.gz'
    eni = ENLTN(f)

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
        amp : numpy.float
            The peak current of the pulse, in kiloamps. NOTE: THIS WILL BE 
            CHANGED TO `current`
            
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
        if filename is not None:
            self.readFile(filename)

    def readFile(self, filename, full=False):
        """
        Given a filename, load the data into the object.

        Parameters
        ----------
        filename: str
            The file name to be read in.
        full : boolean, default: False
            If true, try to read in some of the solution attributes.
            Experimental.

        """

        if isinstance(filename, list):
            if len(filename) > 1:
                print('Multiple files not allowed yet')
            filename = filename[0]

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

        reader = pd.read_csv(filename, **pdArgs)

        for chunk in reader:
            # The first two columns are not relevant for us, and
            # the "nullTime" field is largely redundant:
            # AUTHOR NOTE: Never could get usecols to work to not have to do this
            chunk.drop(['flashPortionHistoryID', 'flashPortionID', 'nullTime'], 
                       axis=1, inplace=True)
           
            # Reinterpret the time string field as datetime64:
            chunk.time = chunk.time.astype('datetime64')
            
            # change the ype field to a string of G or C (for CG/IC)
            chunk.type.replace(to_replace={'0': 'G', '1': 'C'}, inplace=True)

            # todo: process the "extra: fields
            # todo: change flashID to some sort of integer (from string)
            rawData.append(chunk)
        
        # Finally, make the "whole" dataframe
        rawData = pd.concat(rawData, ignore_index=True)
        
        rawData.alt /= 1e3  # km, please
        rawData.amp /= 1e3  # kA, please
        
        self._data = rawData
        
        
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
        The ENI pulse file.
    
    """
    # 
    
    import gzip
    from os.path import splitext
    
    curr_hr = -1
    new_data = list()
    
    # Get basefilename with path, but w/o extension. 
    # Assume multiple extensions (e.g., basename.csv.gz)
    basename, ext1 = splitext(file)
    basename, ext2 = splitext(basename)
    
    def _write_file(new_name, data_to_write):
        with gzip.open(new_name+'.gz', 'wt') as _newf:
            for item in new_data:
                _newf.write("%s\n" % item)
    
    # Open the file
    with gzip.open(file, 'rt') as _f:
        # Read the first line. We're going to put this at the top of 
        # every new file.
        hdr = _f.readline()
        
        # Now, we'll go about reading each line (sigh)
        for line in _f:
            this_date = line.split(',')[4]
            _, hms = this_date.split('T')
            this_hr = int(hms[0:2])
            
            if this_hr != curr_hr:
                
                # Write data to file, if there's something to write
                if len(new_data) !=0:
                    new_filename = basename + '_{:02}'.format(curr_hr) + ext2
                    _write_file(new_filename, new_data)

                # Start a new list to write to file.
                curr_hr = int(this_hr)
                new_data = list()
                new_data.append(hdr)
                
            new_data.append(line)

        # At the end, so check if we have data, and if so, write the data
        # TODO: this code is exactly the same as above, so refactor
        if len(new_data) !=0:
            new_filename = basename + '_{:02}'.format(curr_hr) + ext2
            _write_file(new_filename, new_data)
            