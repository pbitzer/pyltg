# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:14:18 2016

@author: bitzer
"""

import numpy as np
import pandas as pd

from pyltg.core.baseclass import Ltg


class NLDN(Ltg):
    """
    Class to handle Vaisala data, either NLDN or GLD.

    Under the hood, the data is simply a pandas DataFrame
    Attributes:
        _data
    """
    
    """Methods
    
    """    
    def __init__(self, fileName=None):
        if fileName is not None:
            self.readFile(fileName)

    def readFile(self, fileName):
        """
        Given a filename, read the data and load it into the object.
        Parameters
        ----------
        fileName

        Returns
        -------

        """

        if isinstance(fileName, list):
            if len(fileName) > 1:
                print('Multiple files not allowed yet')
            fileName = fileName[0]

        colNames = ('date', 'time', 'lat', 'lon', 'current', '_kA', '_multi', 'semimajor', 
                    'semiminor', 'axis_ratio', 'azimuth', 'chisq', 'num_sensors', 'type')

        rawData = list()
        
        chunkSize = 100000  # How much of the file do we read at once?
        
        reader = pd.read_csv(fileName, names=colNames, header=None,
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
        
        self._data = rawData