# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:14:18 2016

@author: bitzer
"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


class ENTLN(Ltg):
    """
    Class to handle Earth Networks Total Lightning Network data.

    Attributes:
        _data : Dataframe
            The underlying data.
    """
    
    """Methods
    
    """    
    def __init__(self, filename=None):
        """
        Initialization.

        If you don't provide a file name, you'll have to call `readFile`
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
