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

        # Check to see if the standard columns are included:
        if len(self._data) != 0:
            self._verify_columns()

    def __len__(self):
        # Only count active

        return self._data[self._data.active].shape[0]

    def __getattr__(self, name):
        # Override the get attribute to get the column name
        # of the data DataFrame:

        # We need to catch calls to access the (class) attribute data,
        # else we'll end up with inifinite recursion when finding a column
        # (seems to affect pickling mostly)
        if name == 'data' or name == '_data':
            return object.__getattribute__(self, '_data')
        elif name in self._data.columns:
            # If we're getting something from the Dataframe, then
            # we only want the active rows:
            return self._data[self._data.active][name].values
        elif name == 'count':
            return self.__len__()
        else:
            raise AttributeError('Unknown column name: ' + name + '. Valid ones: ' + ', '.join(self._data.columns))

    def __getitem__(self, key):
        """
        Overload the indexing operator to get specified rows of the underlying data.

        Note that this is an index locator and is relative to active data.

        Parameters
        ----------
        key : slice, scalar
            The index location(s) you want. Will be passed to iloc of the
            underlying Dataframe.

        Returns
        -------
        The data of interest, (usually) as a Pandas Series

        """

        # TODO: check for bad indices (IndexError)

        # Only look at active rows
        return self._data[self._data.active].iloc[key]

    def __iadd__(self, other):
        print('overload +=')  # todo: overload +=

    def _verify_columns(self):
        """
        Look at the columns in the underlying data, and ensure that
        `time`, `lat`, `lon`, `alt` are included. If they are not, add a
        column of zeros. Also, check for an `active` column and if not present,
        add boolean True values.
        """

        atts = ['time', 'lat', 'lon', 'alt']

        for att in atts:
            if att not in self._data.columns:
                self._data[att] = 0.0

        if 'active' not in self._data.columns:
            self._data['active'] = True

    def _add_field(self, field_name, data):
        """
        Add a field (i.e., column) to the underlying Dataframe

        .. note::
            This shouldn't be used outside of core developers.

        Parameters
        ----------
        field_name : str
            The name of the field to add.
        data : array
            The data to be added.
        """

        # TODO: make sure the number of data matches existing element
        self._data[field_name] = data

    def _add_record(self, data):
        """
        Add a record (i.e., row) to the Dataframe.

        This method will add some some data to the underlying Dataframe.

        .. warning::
            Very little error catching is implemented.

        .. note::
            This shouldn't be used outside of core developers.

        Parameters
        ----------
        data : array
            The data to be added. The number of elements should match the number
            of columns currently in the Dataframe.

        Returns
        -------

        """

        num_rec = len(self._data)

        # If there's no record, we need to "initialize" the property a little differently...
        if num_rec == 0:
            self._data = self.data.append(data, ignore_index=True, sort=False)
            self._verify_columns()
        else:
            # TODO: make sure the record columns match the existing ones
            self._data.loc[num_rec] = data  # TODO: should this be iloc?

    def reset_active(self):
        """
        Set all data to be active.

        Returns
        -------
        None.

        """

        self._data.active = True

    def get_active(self):
        """
        Get a copy of the "active" data from the class.

        Returns
        -------
        Pandas Dataframe
            A copy of the active data.

        """

        return self._data[self._data.active].copy()


    def head(self, n=5):
        """
        Return the first `n` rows of the active data.

        The active data is the data in which the active attribute is True.
        Mimics Pandas `head`.

        Parameters
        ----------
        n : int, optional
            The number of rows to return. The default is 5.

        Returns
        -------
        Pandas Dataframe
            The first `n` rows of the active data.

        """

        return self._data[self._data.active].head(n)

    def limit(self, reset=False, active=None, **kwargs):
        """
        Limit the underlying data based on the inputs.

        To use this, pass in the data attributes of the data to be limited and
        their range, e.g.,::

            Ltg.limit(lat = [30, 40])

        Ny default, limits are only applied to active data.

        Parameters
        ----------
        reset : bool
            If `True`, reset the active state (to all active) before
            limiting. Default is `False`

        active : array-like
            The indices of the data you wish to keep active. Alternatively,
            an array of booleans the same length as the active data. In this
            case, elements that are `True` will correspond to data that
            is kept active. Use this keyword in a "which data to keep"
            manner.

            In general, you wouldn't try to limit other keys when using
            `active`. If you do, crazy thigs might happen!

        kwargs : varies
            This should be provided in key-range pairs. The keys correspond
            to the columns in the underlying data.
            The key(s) will be limited according to the
            provided range (inclusive).

            If you're passing in time, it needs to be int64 or datetime64[ns]
            (The nanosecond measurement is important!)

        Returns
        -------
        int
            The count of the values within the provided limits.

        """

        import numpy as np

        if reset:
            self.reset_active()

        is_active = self._data.active

        boolVal = self.active  # This is an array of True

        # First, check to see if active was passed in. If so, this is not
        # a range, but an array-like sequence, so we handle it differently.
        if active is not None:

            # We don't need the values moving forward, so pop it:
            active_idx = np.atleast_1d(active)

            # Since we're provided the indicies of what we want to keep,
            # flip all the values to False. This is easier than figuring
            # out which indices were not passed.
            boolVal = ~boolVal

            # Since it's the first keyword we're finding, we can just update the array
            boolVal[active_idx] = True

        for arg, val in kwargs.items():
            # If any keywords are passed as None, skip them
            if val is None:
                continue
            try:
                thisData = self.__getattr__(arg)
            except KeyError:
                print(arg + ' is an invalid data attribute name. Skipping...')
                continue

            # For the time field, be careful with type. Sometimes, a datetime64
            # might be passed in as a keyword, others an int64
            if arg == 'time':
                # Cast the data in the object as int64
                thisData = thisData.astype('int64')

                if type(val[0]) != 'int64':
                    val = np.array(val).astype('int64')

            # Each iteration of the loop, update if it's in range or not.
            boolVal = boolVal & (thisData >= val[0]) & (thisData <= val[1])

        self._data.loc[is_active, 'active'] = boolVal

        count = np.count_nonzero(boolVal)

        return count
