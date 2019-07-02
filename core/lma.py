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

from pyltg.core.baseclass import Ltg


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
    
    
    Attributes
    -----------
        _data : Dataframe
            The underlying data. A "real" attribute of the class.
                
        flashID : str
            The flash ID
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
    def __init__(self, file):
        """
        Filename is required on initialization. This might be changed in
        the future.

        Parameters
        ----------
        file : str
            The file to be read in.
        """
        
        super().__init__()  # initialize the inherited baseclass
        
        self.__colNames()  # mapping from columns in the file to object props
        
        self.readFile(file)
                
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

    def readFile(self, file):
        """
        Read the given file.

        Parameters
        ----------
        file : str
            The file to be read in.

        """
        from itertools import islice
        import gzip

        # First, we need to find where in the file the data starts
        # This means we need to open the file and read in the beginning
        
        # NOTE: Is it better (faster) to find the data start as we read in?
        if isinstance(file, list):
            if len(file) > 1:
                print('Multiple files not allowed yet. Reading first.')
            file = file[0]

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
        self._data = pd.read_csv(file, compression='gzip', header=None,
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
        
        secs = self.time.astype('int64').astype('timedelta64[s]')
        secs = secs.astype('timedelta64[ns]')
        secsFrac = ((self.time % 1)*1e9).astype('timedelta64[ns]')
        
        # Cast the times as datetime64[ns]....
        self._data.time = date + secs + secsFrac

        self._data.alt /= 1e3  # convert to km