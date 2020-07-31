# -*- coding: utf-8 -*-
"""
The base class that other classes in the package use.
"""

import pandas as pd


class Ltg(object):
    """
    Generic class for lightning data.

    Very few users will ever need to initialize this class directly, but all
    lightning classes will inherit this class and its methods.
    """

    def __init__(self, *args, **kwargs):
        self._data = pd.DataFrame(*args, **kwargs)  # initialize to an empty DataFrame

        # Check to see if 'alt' is included. If not, add it:
        # todo: check for other columns
        if len(self._data) != 0:
            self.verifyColumns()


    def __len__(self):
        return self._data.shape[0]

    def __getattr__(self, name):
        # Override the get attribute to get the column name
        # of the data DataFrame:

        # We need to catch calls to access the (class) attribute data,
        # else we'll end up with inifinite recursion when finding a column
        # (seems to affect pickling mostly)
        if name == 'data' or name == '_data':
            return object.__getattribute__(self, '_data')
        elif name in self._data.columns:
            return self._data[name].values
            # todo: try/except?
        else:
            raise AttributeError('Unknown column name: ' + name + '. Valid ones: ' + ', '.join(self._data.columns))

    def __getitem__(self, key):
        """
        Overload the indexing operator to get the specified columns of the underlying data

        Parameters
        ----------
        key : str
            The name of the data attribute you want to get. This varies from data
            set to data set, but 'time', 'lat', 'lon' should always be there.

        Returns
        -------
        The data of interest, (usually) as a Pandas Series

        """

        # todo: check for bad key (TypeError)
        # todo: check for bad indices (IndexError)
        return self._data.iloc[key]

    def __iadd__(self, other):
        print('overload +=')  # todo: overload +=

    def verifyColumns(self):
        """
        Look at the columns in the underlying data, and ensure that
        `time`, `lat`, `lon`, `alt` are included. If they are not, add a
        column of zeros.
        """

        atts = ['time', 'lat', 'lon', 'alt']

        for att in atts:
            if att not in self._data.columns:
                self._data[att] = 0.0

    def addField(self, field_name, data):
        """
        Add a field (i.e., column) to the underlying Dataframe

        .. note::
            This isn't typically used outside of core developers.

        Parameters
        ----------
        field_name : str
            The name of the field to add.
        data : array
            The data to be added.
        """

        # TODO: make sure the number of data matches existing element
        self._data[field_name] = data

    def addRecord(self, data):
        """
        Add a record (i.e., row) to the Dataframe.

        This method will add some some data to the underlying Dataframe.

        .. warning::
            Very little error catching is implemented.

        Parameters
        ----------
        data : array
            The data to be added. The number of elements should match the number
            of columns currently in the Dataframe.

        Returns
        -------

        """

        nRec = len(self._data)

        # If there's no record, we need to "initialize" the property a little differently...
        if nRec == 0:
            self._data = self.data.append(data, ignore_index=True, sort=False)
            self.verifyColumns()
        else:
            # TODO: make sure the record columns match the existing ones
            self._data.loc[nRec] = data

    def limit(self, **kwargs):
        """
        Limit the underlying data based on the inputs.

        To use this, pass in the data attributes of the data to be limited and
        their range, e.g.,::

            Ltg.limit(lat = [30, 40])

        Parameters
        ----------
        kwargs : varies
            This should be provided in key-range pairs. The given key will be
            limited according to the provided range.

        Returns
        -------
        Tuple
            The returned tuple contains the indices (rows) and the count of the
            limited values.

        """
        # Pass in the data attributes of the data to be limited and their range,
        # e.g.,
        # Ltg.limit(lat = [30, 40])
        # will return the indices of the data with lats between 30,40
        # Any attribute in the data will work.
        # If times, either pass in int64 or datetime64[ns]
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
