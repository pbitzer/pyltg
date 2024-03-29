# -*- coding: utf-8 -*-
"""
Read in Geostationary Lightning Mapper (GLM) data.

This uses Eric Bruning's glmtools under the hood
(https://github.com/deeplycloudy/glmtools).
Things are organized a little differently here. The focus is here is more
on class-based interaction with data, including reading multiple files
at once.

The operational feed for GLM data is missing important metrics that are
important to atmospheric electricity researchers. One good example
is the child count for each "grouping" of GLM data (e.g., the number
of groups in a flash). That's handled here.

The GLM class is essentially comprised of three
:class:`Ltg <pyltg.core.baseclass.Ltg>` classes: one for events,
one for groups, and one for flashes. These are attributes of the GLM class.
There are methods to connect these, e.g., for a given flash (or flashes),
get the child groups. Each grouping of GLM data has slightly different fields. The following are
describes each of the these, which again, are attributes of the "main"
GLM class.

flashes (see :func:`_extract_flashes`)
    :time: The time the flash starts
    :time_last: The time the flash ends (i.e., the last time of the last group
    :lat: Latitude of the radiance weighted centroid of the child groups
    :lon: Longitude of the radiance weighted centroid of the child groups
    :energy: Received optical energy of the flash (Joules)
    :id: 32bit value used as flash identifier
    :_orig_id: The original flash ID in the file (16 bit)
    :area: Area of the flash (in km**2)
    :quality_flag: Flag for data quality #todo: add the values
    :alt: Meaningless and zero-d out.


groups (see :func:`_extract_groups`)
    :time: The time of the group
    :lat: Latitude of the radiance weighted centroid of the child events
    :lon: Longitude of the radiance weighted centroid of the child events
    :energy: Received optical energy of the group (Joules)
    :id: 32bit value used as group identifier
    :_orig_id: The original group ID in the file (16 bit)
    :area: Area of the group (in km**2)
    :quality_flag: Flag for data quality #todo: add the values
    :alt: Meaningless and zero-d out.

events (see :func:`_extract_events`)
    :time: The time of the event
    :lat: Latitude of the event
    :lon: Longitude of the event
    :energy: Received optical energy of the group (Joules)
    :id: 32bit value used as event identifier
    :_orig_id: The original flash ID in the file (16 bit)
    :alt: Meaningless and zero-d out.

Still to be done:
    - Get grandchild count for flashes (i.e, the events).
    - Get event pixel address (in CCD coords)

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from pyltg.core.satellite import get_children, energy_colors
from pyltg.core.baseclass import Ltg

try:
    import glmtools.io.glm as glmt
except ImportError as err:
    print("**** Unable to find glmtools. You'll have limited lunctionality for GLM data ****")

    # To enable functionality that doesn't depend on glmtools, we will
    # create a dummy class to "catch" anything that tries to use it.
    # The only thing glmtools is used for (currently) is to enable reading
    # in GLM operational data.

    # Note: we might have to move this to somewhere else if glmtools
    # is ever used in another module
    class DummyGlmtools():
        def __getattr__(self, attr):
            raise RuntimeError('glmtools not installed')

    glmt = DummyGlmtools()  # variable name should match the import...as above


def search_path(path, time_range, lon_range=None, lat_range=None):
    """
    Search the given path for GLM data that satisfy the given ranges.

    This provides a way to search an archive of GLM data files for lightning
    in the give time/space bounds.

    Data is assumed to be in a folder structure path/doy/<sub_dirs>/<glm_files>

    Note: Time range can't cross dates (yet)

    Parameters
    ----------
    path : str
        The path to be searched. See above doc for assumed directory structure
        under this path. Will be passed to `pathlib.Path`
    time_range : list-like
        2 element iterable specifying the start/stop time you're interested in.
        The values are passed to `pandas.Timestamp`.
    lon_range : list-like, optional
        2 element iterable providing the longitude range. The default is None.
        If None, all longitudes are considered.
    lat_range : list-like, optional
        2 element iterable providing the latitude range. The default is None.
        If None, all latitudes are considered.

    Returns
    -------
    GLM
        A pyltg.GLM class with the data that satisfies the given ranges.

    """

    time_range = [pd.Timestamp(t) for t in time_range]

    # First, go the day-of-year folder.
    doy = time_range[0].dayofyear

    search_path = Path(path).joinpath('{:03}'.format(doy))

    all_files = np.array(sorted(search_path.rglob('OR_GLM*')))

    times = np.array(filename2date(all_files))

    # To make sure we don't miss any files, buffer the times a bit.
    # Since GLM files are 20 seconds, this should be sufficient.
    # We'll make it a little bigger just to be sure...
    BUFFER = pd.Timedelta(30, 's')

    idx = (times >= (time_range[0]-BUFFER)) & (times <= (time_range[1]+BUFFER))

    if not np.count_nonzero(idx):
        print('No files found')
        return None

    # These are the files that should contain our range.
    files = all_files[idx]

    all_g = GLM(files)

    fl_t = all_g.flashes.time

    # Sigh, can't do datetime64[ns] and Timestamp.
    # And, we don't have an quick way to convert an array of datetime64 to Timestamp.
    good_idx = (fl_t >= np.datetime64(time_range[0])) & (fl_t <= np.datetime64(time_range[1]))

    if lat_range is not None:
        fl_lat = all_g.flashes.lat
        lat_idx = (fl_lat >= lat_range[0]) & (fl_lat <= lat_range[1])

        good_idx = good_idx & lat_idx

    if lon_range is not None:
        fl_lon = all_g.flashes.lon
        lon_idx = (fl_lon >= lon_range[0]) & (fl_lon <= lon_range[1])

        good_idx = good_idx & lon_idx
    # Make sure we have _something_
    if not np.count_nonzero(good_idx):
        print('No data found in provided range')
        return None
    else:
        fl = all_g.flashes[good_idx]

        ev, grp = all_g.get_groups(fl.id, combine=True, events=True)

        g = GLM()
        g.flashes = Ltg(fl)
        g.groups = Ltg(grp)
        g.events = Ltg(ev)

        return g


def _convert_lon_360_to_180(lons):
    # Small helper to convert a set of 0->360 lons to -180->180
    return ((lons + 180) % 360)-180


def _convert_lm_time(lm_time):
    # Simple helper to take the times in LM files are convert to a suitable time

    # Note: the times slightly differ (~< 100 us precision) from IDL implementation

    # The times in the LM files are days past the epoch, but it's a float.
    # We're going to need to do some work to get it into a format
    # suitable for NumPy's datetime. Pandas will do it, but it's slow.

    epoch = np.datetime64('2000-01-01T12', 'ns')

    # First, extract the whole day count
    days = lm_time.astype('int32').astype('timedelta64[D]')

    # Next, get the fractional seconds past the start of the day
    sec_past_day = (lm_time % 1) * 86400

    # NumPy's datetime64 is a bit of pain, since it requires integers.
    # So, we'll extract them:
    whole_sec = sec_past_day.astype('int32').astype('timedelta64[s]')
    whole_nsec = ((sec_past_day % 1) * 1e9).astype('int64').astype('timedelta64[ns]')

    # After all that, we can finally build the time:
    times = epoch + days + whole_sec + whole_nsec

    return times

def read_lm_ev_mat(files):
    """
    Read a Lockheed Martin mat file (Matlab save file) that contains
    only events.

    As part of PLT, a particular processing chain of L0 data produces
    Matlab mat files with just events. Usually, this is done with no
    filters turned on, so we can naviagate all available events (at least,
    the ones on the earth).

    The underlying assumption is that there is a structure
    named `evOut` in the files.

    No attempt to ensure unique IDs is done.

    Parameters
    -----------
    files: str or sequence of string
        The files to read in

    """

    from scipy.io import loadmat

    files = np.atleast_1d(files)

    ev = list()

    # Go through the files, and get the structure...

    for f in files:
        mat_data = loadmat(f)

        ev_out = mat_data['evOut']

        # We need to do a little work, since loadmat seems to pull some
        # weird nesting in the record array.(For the record, squeeze_me
        # doens't seem to work.)

        # We'll just extract into a dict, to eventually put it into a DataFrame
        this_ev = dict(
                      time=_convert_lm_time(ev_out['origination_time'][0][0][0]),
                      px=ev_out['x'][0][0][0],
                      py=ev_out['y'][0][0][0],
                      intensity=ev_out['intensity'][0][0][0],
                      bg_msb=ev_out['bg_msb'][0][0][0],
                      filter_id=ev_out['filterID'][0][0][0],
                      energy=ev_out['energy'][0][0][0],
                      lat=ev_out['lat'][0][0][0],
                      lon=_convert_lon_360_to_180(ev_out['lon'][0][0][0])
                      )
        # Note: this ignores the following fields present in evOut:
        # 'device_status', 'consec', 'frame_id', 'df', 'rtep', 'pixel',
        # 'chan', 'ufid', 'isActive', 'time'

        ev.append(pd.DataFrame(this_ev))

    ev = pd.concat(ev, ignore_index=True)

    # Now, we need to add alt so that we can make it an Ltg class
    ev['alt'] = 0.0

    return Ltg(ev)

def read_lm_file(files, keepall=False):
    """
    Read Lockheed Martin netCDF file with events and groups.

    As part of PLT, files were generated using LM's Matlab code that contains
    Level 1 events and groups. (Both L1b and L1 events could be present).

    IDs (event, group) are not guaranteed to be unique if reading multiple files

    Parameters
    ----------
    files: str or sequence of string
        The files to read in
    keepall: bool
        If True, keep all events in the file, even ones that were filtered.
        Note that only 1b events are geo-navigated.

    """

    from netCDF4 import Dataset

    files = np.atleast_1d(files)

    all_ev = list()
    all_grp = list()

    for f in files:

        nc = Dataset(f)

        # Although a little tedious, we are going to get the fields and
        # put them into a dict and then to DataFrame...
        ev = dict()

        ev['lat'] = nc.variables['event_lat'][:]
        ev['lon'] = _convert_lon_360_to_180(nc.variables['event_lon'][:])
        ev['energy'] = nc.variables['event_energy'][:]
        ev['parent_id'] = nc.variables['event_parent_group_id'][:]
        ev['px'] = nc.variables['event_x_LM'][:]
        ev['py'] = nc.variables['event_y_LM'][:]
        ev['intensity'] = nc.variables['event_intensity_LM'][:]
        ev['bg_msb'] = nc.variables['event_bg_msb_LM'][:]
        ev['filter_id'] = nc.variables['event_filter_id_LM'][:]
        ev['time'] = _convert_lm_time(nc.variables['event_time_LM'][:].data)
        # Not using: event_frame_id_LM

        ev = pd.DataFrame(ev, columns=ev.keys())

        if not keepall:
            good_rows = ~ev.lat.isna()
            ev = ev[good_rows]

        all_ev.append(ev)

        # Now, the groups:
        # todo: make sure there are groups...
        grp = dict()
        grp['lat'] = nc.variables['group_lat'][:]
        grp['lon'] = _convert_lon_360_to_180(nc.variables['group_lon'][:])
        grp['energy'] = nc.variables['group_energy'][:]
        # Make sure IDs are unsigned...
        # todo: double check this is OK
        grp['id'] = nc.variables['group_id'][:].astype(np.uint32)
        grp['area'] = nc.variables['group_footprint_LM'][:]
        grp['child_count'] = nc.variables['group_child_count_LM'][:]

        grp['time'] = _convert_lm_time(nc.variables['group_time_LM'][:].data)

        grp = pd.DataFrame(grp, columns=grp.keys())

        all_grp.append(grp)

    all_ev = pd.concat(all_ev, ignore_index=True)
    all_grp = pd.concat(all_grp, ignore_index=True)

    return Ltg(ev), Ltg(all_grp)

def read_events_nc(files, no_error=True):
    """
    Read in the nc file produced by the Level 0 reader.

    Clem Tillier of Lockheed Martin has made a L0 reader available that
    will produce events. These events are not geo-navigated.

    Unique IDs are not implemented.

    Parameters
    -----------
    files: str or sequence of string
        The files to read in
    no_error: bool
        If True, then throw out any events associated with an error. Right now,
        this keyword has no effect - error events are always dropped.
        (In the future, we'll add an option to keep them!)
    """

    import xarray as xr

    files = np.atleast_1d(files)

    data = list()

    for f in files:
        ev = xr.open_dataset(f, decode_times=False)

        if (ev.error_count != 0) & True: # TODO Add ability to keep error events
            ev = ev.drop_dims('number_of_errors')

        # There's a few scalar variables we don't keep, as they become
        # columns that all have the same value:
        ev = ev.drop(['event_count', 'error_count',
                         'spw_dropped_event_count'])

        # Now, we should be able to get to a DataFrame that's sensible...
        ev = ev.to_dataframe()

        # Now, we build the time. It takes some awkward code gymnastics...
        time = (np.datetime64('2000-01-01T12', 'ns')
                + ev.event_day.astype('int64').values.astype('timedelta64[D]')
                + ev.event_millisecond.astype('int64').values.astype('timedelta64[ms]')
                + ev.event_microsecond.astype('int64').values.astype('timedelta64[us]'))

        ev['time'] = time

        data.append(ev)

    data = Ltg(pd.concat(data, ignore_index=True))

    return data


def event_poly(events, latlon=True,
               colors='yellow', fill=True):
    """
    Get a MPL PolyCollection fthat represents the polygons of the events.

    The color of each polygon represents the energy.

    Right now, a rough approximation for the event footprint is used, namely
    they are assumed to be approximately 4km x 4km.

    Parameters
    -----------
    events : Pandas Dataframe
        The events to be plotted. Intended to be similar to output of
        `GLM.get_events`
    latlon : bool
        If True, return the PolyCollection so it can be plotted on a
        Cartopy map plotted with lats and lons.
    colors : str
        The colors to be used to represent the energy of each event.
    fill : bool
        If True, fill the polygons according to `colors`


    Returns
    --------
    MPL PolyCollection
        A Matplotlib PolyCollection of the events. Of course, to add this
        to an existing axes (either MPL native or Cartopy GeoAxes) just use
        `ax.add_collection(poly)`

    """
    import cartopy.crs as ccrs

    # There doesn't seem to be "none" for transform, and the plotting
    # calls are similar whether or not we do a map. So, make a
    # dict with transform if we have it, otherwise leave it empty.
    trans_kw = {}
    if latlon:
        trans_kw['transform'] = ccrs.PlateCarree()

    centers = np.vstack((events.lon, events.lat)).T

    # assume 8 km square pixels for simplicity
    offsets = np.ones((4, len(events), 2))
    EVENT_EDGE = 0.04
    offsets[0, :, 0] = -EVENT_EDGE  # move ul, x
    offsets[1, :, 0] = -EVENT_EDGE  # move ll, x
    offsets[2, :, 0] =  EVENT_EDGE  # move lr, x
    offsets[3, :, 0] =  EVENT_EDGE  # move ur, x

    offsets[0, :, 1] =  EVENT_EDGE  # move ul, y
    offsets[1, :, 1] = -EVENT_EDGE  # move ll, y
    offsets[2, :, 1] = -EVENT_EDGE  # move lr, y
    offsets[3, :, 1] =  EVENT_EDGE  # move ur, y

    verts = centers + offsets
    verts = np.swapaxes(verts, 0, 1)

    if fill:
        # todo: here, we would pick a different color scheme
        colors = energy_colors(events.energy.values, satellite='GLM')/255
    else:
        colors = 'none'

    poly = PolyCollection(verts, edgecolors='black', facecolors=colors, **trans_kw)

    return poly


def filename2date(files, idx=3):
    """
    Get the date/time from a GLM filename.
    
    Slightly modified from the version in `glmtools`.

    Parameters
    ----------
    files : list-like strings
        The filenames to look at for the start times. Should be the base
        file name (with extension).
    idx : int, optional
        The field to look for the start time. For L2 files, the default
        of 3 is correct. For some L0 files, the index should be 2. 

    Returns
    -------
    list : 
        A n-element list of NumPy datetime64's with the start times of the
        data in the files. 

    """
    
    import datetime
    import os

    t0 = list()
    for _f in np.atleast_1d(files):

        this_file = os.path.basename(_f)  # no path, thanks
        this_file = os.path.splitext(this_file)[0]  # no extension, thanks

        parts = this_file.split('_')

        # Start time is in one of these fields:
        start = datetime.datetime.strptime(parts[idx][1:-1], '%Y%j%H%M%S')
        start = np.datetime64(start)

        # do we need fractional seconds?
        t0.append(start)

    return t0


def _extract_groups(glmdata):
    """
    Given a GLMDataset, extract groups and relevant attributes.

    Parameters
    ----------
    glmdata : GLMDataset
        The GLMDataset to extract groups from.

    Returns
    -------
    DataFrame :
        Pandas DataFrame with GLM data fields.

    """

    data = pd.DataFrame({
        'time': glmdata.dataset.group_time_offset.values,
        'time_frame': glmdata.dataset.group_frame_time_offset.values,
        'lat': glmdata.dataset.group_lat.values,
        'lon': glmdata.dataset.group_lon.values,
        'energy': glmdata.dataset.group_energy.values,
        'id': glmdata.dataset.group_id.values,
        '_orig_id': glmdata.dataset.group_id.values,  # this the id in the file
        'parent_id': glmdata.dataset.group_parent_flash_id.values,
        'area': glmdata.dataset.group_area.values,
        'quality_flag': glmdata.dataset.group_quality_flag.values
        })

    # For consistency with the Ltg parent class,
    # we need altitude, but zero it out since it's meaningless for GLM.
    data['alt'] = 0.0

    return data


def _extract_events(glmdata):
    """
    Given a GLMDataset, extract events and relevant attributes

    Parameters
    ----------
    glmdata

    Returns
    -------

    """

    data = pd.DataFrame({
        'time': glmdata.dataset.event_time_offset.values,
        'lat': glmdata.dataset.event_lat.values,
        'lon': glmdata.dataset.event_lon.values,
        'energy': glmdata.dataset.event_energy.values,
        'id': glmdata.dataset.event_id.values,
        '_orig_id': glmdata.dataset.event_id.values,  # this the id in the file
        'parent_id': glmdata.dataset.event_parent_group_id.values
        })

    # For consistency with the Ltg parent class,
    # we need altitude, but zero it out since it's meaningless for GLM.
    data['alt'] = 0.0

    return data


def _extract_flashes(glmdata):
    # Given a GLMDataset, extract flashes and relevant attributes

    # Because flash ids are 16 bit, we can get rollover of the id
    # in the "middle" of a file. This can cause problems we reading in
    # multiple files. So, de-16bit integer-ize them. We'll keep track
    # of the original ID for traceability however.

    data = pd.DataFrame({
        'time': glmdata.dataset.flash_time_offset_of_first_event.values,
        'time_last': glmdata.dataset.flash_time_offset_of_last_event.values,
        'lat': glmdata.dataset.flash_lat.values,
        'lon': glmdata.dataset.flash_lon.values,
        'energy': glmdata.dataset.flash_energy.values,
        'id': np.uint32(glmdata.dataset.flash_id.values),
        '_orig_id': glmdata.dataset.flash_id.values,  # this the id in the file
        'area': glmdata.dataset.flash_area.values,
        'quality_flag': glmdata.dataset.flash_quality_flag.values
        })

    # For consistency with the Ltg parent class,
    # we need altitude, but zero it out since it's meaningless for GLM.
    data['alt'] = 0.0

    return data


def _get_child_count(parent, child):
    # Given a parent and child dataset, find the number of children for
    # each parent.

    # First, define the bins to count in. Because of numpy's weirdness, we'll
    # need to add a bin that will be always empty:
    _bins = np.append(parent.id, parent.id.iloc[-1]+1)

    histo, bins = np.histogram(child.parent_id, bins=_bins)

    return histo


class GLM():
    """
    Class to handle GLM data.
    """

    def __init__(self, files=None):
        """
        Initialization

        Parameters
        ----------
        files : str
            The file(s) to be read in.
        """

        self.events = Ltg()
        self.groups = Ltg()
        self.flashes = Ltg()
        if files is not None:
            self.readFile(files)

    def readFile(self, files):
        """
        Read the given file(s).

        Use GLMDataset to (mininally) read in the files, but we're going to extract
        things and put them in certain places.

        Parameters
        ----------
        files : str
            The file to be read in.

        """
        files = np.atleast_1d(files)  # allow scalar input

        events = list()
        groups = list()
        flashes = list()

        ev_id_ctr = 0
        gr_id_ctr = 0
        fl_id_ctr = 0

        for _file in files:
            # Extract the GLM data. Since we'll handle the parent-child
            # relationship, don't do it here.
            this_glm = glmt.GLMDataset(_file, calculate_parent_child=False, change_energy_units=False)

            # Some GLM files have no data. Check for these cases:
            # todo: do we need to check groups and flashes too?
            if this_glm.dataset.dims['number_of_events'] == 0:
                continue

            this_event = _extract_events(this_glm)
            this_group = _extract_groups(this_glm)
            this_flash = _extract_flashes(this_glm)

            # We're going to modify the IDs a bit, since they can rollover.
            # The flash IDs seem to rollover at 2**16-1, but the group IDs
            # are MUCH weirder. The rollover seems to happen somewhere
            # between 2**29 and 2**30. To get reasonable IDs, we're going to
            # modify these IDs too.

            # First, get a "mapping" from the current IDs to unique values:
            new_flash_id = np.arange(len(this_flash))

            # Now, update the IDs to this new mapping:
            this_flash.id = new_flash_id

            # Update the parent IDs for the groups:
            # Get a dictionary to map the values:
            flash_id_map = dict(zip(this_flash._orig_id.values, new_flash_id))

            new_id = this_group.parent_id.map(flash_id_map.get)
            # Note: mapping is MUCH faster than using the DataFrame.replace method

            this_group.parent_id = new_id

            # Now, do the same thing with group/events:
            new_group_id = np.arange(len(this_group))
            this_group.id = new_group_id

            group_id_map = dict(zip(this_group._orig_id.values, new_group_id))

            this_event.parent_id = this_event.parent_id.map(group_id_map.get)

            # We'll sort these by id. Makes counting children easier.
            this_event.sort_values('id', inplace=True)
            this_group.sort_values('id', inplace=True)
            this_flash.sort_values('id', inplace=True)

            # Add in an offset to get unique values across files
            this_event['id'] += ev_id_ctr
            this_group['id'] += gr_id_ctr
            this_flash['id'] += fl_id_ctr

            # Offset the parent IDs for the children too:
            this_event['parent_id'] += gr_id_ctr
            this_group['parent_id'] += fl_id_ctr

            # Next, update the counters
            ev_id_ctr = this_event['id'].iloc[-1]+1
            gr_id_ctr = this_group['id'].iloc[-1]+1
            fl_id_ctr = this_flash['id'].iloc[-1]+1

            # Count children
            child_ev = _get_child_count(this_group, this_event)
            this_group['child_count'] = child_ev

            child_gr = _get_child_count(this_flash, this_group)
            this_flash['child_count'] = child_gr

            # todo: add option to not sort by time
            this_event.sort_values('time', inplace=True)
            this_group.sort_values('time', inplace=True)
            this_flash.sort_values('time', inplace=True)

            # Finally, add "this" data
            events.append(this_event)
            groups.append(this_group)
            flashes.append(this_flash)

        if not events:
            # todo: related to above todo, do we need to check groups/flashes?
            print('No GLM data found in files. Class will have no data.')
        else:
            # Put these as attributes of the class
            self.events = Ltg(pd.concat(events))
            self.groups = Ltg(pd.concat(groups))
            self.flashes = Ltg(pd.concat(flashes))

    def get_events(self, group_ids, combine=False):
        """
        Get child events for a set of groups.

        Parameters
        ----------
        group_ids : array-like
            The IDs for the groups for which you want the events
        combine: bool
            If True, return a Pandas DataFrame with all events. If False,
            return a list a DataFrames in which each element of the list
            corresponds to the events for each group ID.

        Returns
        -------
        Pandas DataFrame
            By default, a list of DataFrames is returned. To get one DataFrame,
            see `combine`.

        """

        evs = get_children(group_ids, self.events)

        if combine:
            evs = pd.concat(evs, ignore_index=True)

        return evs

    def get_groups(self, flash_ids, combine=False, events=False):
        """
        Get the child groups for a set of flashes.

        Parameters
        ----------
        flash_ids : array-like
            The IDs for the flashes for which you want the groups
        combine : bool
            If True, return a Pandas DataFrame with all groups. If False,
            return a list a DataFrames in which each element of the list
            corresponds to the groups for each flash ID.
        events : bool
            If True, also get the child events. If `combine` is True, then
            the events will be returned in one DataFrame. If not, get a
            list of DataFrames, one for each group.

        Returns
        -------
        Pandas DataFrame
            By default, a list of DataFrames is returned. To get one DataFrame,
            see `combine`. If `events` is True, then a tuple is returned, with
            (groups, events).
        """

        flash_ids = np.atleast_1d(flash_ids)

        grps = list()
        evs = list()
        for _id in flash_ids:
            this_grps = self.groups[self.groups.parent_id == _id]
            grps.append(this_grps)

            if events:
                evs.append(self.get_events(this_grps.id, combine=True))

        # TODO: We can get all the events at once, if we have the ids as a list

        if combine:
            grps = pd.concat(grps, ignore_index=True)
            if events:
                evs = pd.concat(evs, ignore_index=True)
        if events:
            return evs, grps
        else:
            return grps

    def plot_groups(self, groups=None, do_events=False, ax=None, latlon=True,
                    gridlines=True,
                    group_marker='.', group_color='black',
                    event_color='yellow', fill_events=True,
                    event_centers=False):
        """
        Make a spatial plot of groups.

        The plotting is done using the lons along the horizontal axis,
        lats along the vertical axis.

        .. warning:
            Right now, event polygons are only approximate. They are plotted
            as polygons with vertices 0.04 degrees from event the center.
            This is fine at GLM nadir, but becomes progressively worse as you
            look toward the edges of the FOV. Future work will try to
            geolocate the events edges.

        Parameters
        ----------
        groups : `Ltg` class or Pandas Dataframe
            The groups to be plotted. If `None`, plot the active
            groups. Default is None.
        do_events : bool
            If True, then plot the individual child events too. Right now,
            this is done in an approximate manner. The event footprint is
            approximated by drawing a 0.04 degree (roughly 4 km) box around
            the event lat/lon. This roughly matches the GLM pixel size at
            nadir, so event footprints off-nadir will not be not accurately
            represented.
        ax : MPL Axes
            If given, the plot will be made in the provided Axes.
        latlon: bool
            If True, make a map using Cartopy. If True and `ax` is given,
            then it is assumed that `ax` is a Cartopy GeoAxes or GeoAxesSubplot
        gridlines: bool
            If True, then gridlines will be added to the plot. Only valid
            if `latlon` is also True.
        group_marker: str
            The MPL marker used when plotting only groups
            i.e, when `do_events=False`.
        group_color: str
            The MPL color used when plotting only groups
            i.e, when `do_events=False`.
        event_color: str
            The color scheme used to scale the event colors by the energy.
            Hard coded for now to be the yellow scheme!
        fill_events: bool
            If True, fill the events with a color related to `event_color`.
            If False, just draw an empty polygon.
        event_centers: bool
            If True, plot a marker at the center of each event.

        Returns
        -------
        tuple

        Two element tuple. The first element is the Axes, and the second
        element is a `dict`. Depending on the arguments, you could have these:

            :groups: MPL Line2D
            :events_poly: List of MPL PolyCollection of event polygons (one element for each group)
            :events_pt: MPL Line 2D of event centroids
            :gridlines: Cartopy Gridliner

        """
        import cartopy.crs as ccrs

        if groups is None:
            groups = self.groups[self.groups.active]

        if ax is None:
            if latlon:
                fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mercator()))
            else:
                fig, ax = plt.subplots()

        # There doesn't seem to be "none" for transform, and the plotting
        # calls are similar whether or not we do a map. So, make a
        # dict with transform if we have it, otherwise leave it empty.
        trans_kw = {}
        if latlon:
            trans_kw['transform'] = ccrs.PlateCarree()

        retVal = dict()  # we'll return a dictionary of plot artists

        # Get the groups:
        if not do_events:
            # just make a scatter plot
            grp_plt = ax.plot(groups.lon, groups.lat, linestyle='None',
                              marker=group_marker, color=group_color, **trans_kw)
            retVal['groups'] = grp_plt[0]
        else:
            events = self.get_events(groups.id, combine=True)

            poly = event_poly(events, colors=event_color, latlon=latlon, fill=True)
            _ = ax.add_collection(poly)

            # If nothing else is plotted, then the x/y limits be MPL's default.
            # In this case, we'll want to set the x/y limits.
            # Otherwise, just add the events to the current view
            if (ax.get_xlim() == (0.0, 1.0)) & (ax.get_ylim() == (0.0, 1.0)):
                ax.autoscale()
            retVal['events_poly'] = poly

            if event_centers:
                ev_plt = ax.plot(events.lon, events.lat, linestyle='None',
                                 marker='.', color='black', markersize=0.5,
                                 **trans_kw)
                retVal['events_pt'] = ev_plt[0]

        if latlon & gridlines:
            gl = ax.gridlines(draw_labels=True, linestyle=':')
            retVal['gridlines'] = gl

        return ax, retVal
    
    def reset_active(self):
        """
        Reset the active state of the underlying Ltg classes. See
        `pyltg.baseclass.reset_active`.

        """
        
        self.events.reset_active()
        self.groups.reset_active()
        self.flashes.reset_active()
