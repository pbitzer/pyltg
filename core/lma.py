# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:58:49 2017

@author: bitzer
"""

import re

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


def idxMatch(lst, text2match):
    """
    Simple helper to parse through a text list to find the index
    that matches an input string
    """

    txtRE = re.compile(text2match, re.IGNORECASE)  # TODO: keyword to ignore case?

    idx = [i for i, _ in enumerate(lst) if txtRE.search(_)]
    
    return idx


class LMA(Ltg):
    """
    Class to handle LMA source data.
    """
    def __init__(self, file):
        
        super().__init__()  # initialize the base class
        
        self.__colNames()  # mapping from columns in the file to object props
        
        self.readFile(file)
                
    def __colNames(self):
        # Map the column names. Make sure time, lat, lon, alt are included.
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