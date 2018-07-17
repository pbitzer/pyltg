# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:19:34 2017

@author: Bitzer
"""

import pandas as pd


class Ltg(object):
    """
    Generic class for lightning data.

    Very few users will ever need to initialize this class directly, but all
    lightning classes will inherit this class and its methods.
    """
    
    @property
    def data(self):
        return self._data

    def __init__(self, *args, **kwargs):
        self._data = pd.DataFrame(*args, **kwargs)  # initialize to an empty DataFrame

    def __len__(self):
        return self._data.shape[0]
        
    def __getattr__(self, name):
        # Check to see if this is a valid column name in the data DataFrame:
        if name in self._data.columns:
            return self._data[name].values
            # todo: try/except?
        else:
            raise KeyError('Unknown column name. Valid ones: ' + ', '.join(self._data.columns))

    def __getitem__(self, key):        
        # Overload the indexing operator to get the specified rows of the underlying data
        # todo: check for bad key (TypeError)
        # todo: check for bad indices (IndexError)
        return self._data.iloc[key]

    def __iadd__(self, other):
        print('overload +=')  # todo: overload +=

    def addField(self, fieldName, data):
        # Add a field (i.e., column) to the Dataframe
        # TODO: make sure the number of data matches existing element
        self._data[fieldName] = data
        
    def addRecord(self, data):
        # Add a record (i.e., row) to the Dataframe
        nRec = len(self._data)
        
        # If there's no record, we need to "initialize" the property a little differently...
        if nRec == 0:
            self._data = self.data.append(data, ignore_index=True)
        else:
            # TODO:make sure the record columns match the existing ones
            self._data.loc[nRec] = data
        
    def limit(self, **kwargs):
        # Pass in the data attributes of the data to be limited and their range, 
        # e.g., 
        # Ltg.limit(lat = [30, 40])
        # will return the indices of the data with lats between 30,40
        # Any attribute in the data will work.
        import numpy as np

        boolVal = np.full(len(self._data), True)  # an array of all true

        for arg, val in kwargs.items():
            # If any keywords are passed as None, skip them
            if val is None:
                continue
            try:
                thisData = self._data[arg].values
            except KeyError:
                print(arg + ' is an invalid data attribute name. Skipping...')
                continue
                
            # For the time field, be careful with type. Sometimes, a datetime64
            # might be passed in as a keyword, others an int64
            if arg is 'time':
                thisData = thisData.astype('int64')
                
                if type(val[0]) is not 'int64':
                    val = np.array(val).astype('int64')
            boolVal = boolVal & (thisData >= val[0]) & (thisData <= val[1])
            
        idx = np.where(boolVal)
        count = np.count_nonzero(boolVal)
            
        return idx, count