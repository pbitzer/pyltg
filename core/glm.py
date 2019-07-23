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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from glmtools.io.glm import GLMDataset

from pyltg.core.baseclass import Ltg


def energy_colors(energies):
    """
    Map the given GLM energies to a set of 256 colors.

    Energies are scaled between 1 fJ and 50 fJ. #todo change this to keyword

    .. note::
        Right now, the only color scale available to use a yellow->red. Others
        will be added later.

    Parameters
    ----------
    energies n element array-like

    Returns
    -------
    array :
        nx3 NumPy array of bytes. The last dimension corresponds to RGB
        values.

    """
    # get RGB values that correspond to the energies.
    min_val = 1e-15
    max_val = 5e-14

    _min_val = np.log10(min_val)
    _max_val = np.log10(max_val)
    _values = np.log10(energies)

    # linear scale
    m = (255-0)/(_max_val-_min_val)
    b = 255.-m * _max_val

    scl_colors = m*_values+b

    # First, clip to bounds:
    scl_colors = np.clip(scl_colors, 0, 255)
    # Make it a byte for indexing
    scl_colors = np.uint8(scl_colors)

    colors = np.zeros((len(_values), 3))

    nsteps = 256
    redV = np.repeat(np.uint8(255), nsteps)
    blueV = np.repeat(np.uint8(0), nsteps)
    scale = np.arange(nsteps)/(nsteps-1)
    n0 = 255
    n1 = 0
    greenV = np.uint8(n0 + (n1-n0) * scale)

    colors[:, 0] = redV[scl_colors]
    colors[:, 1] = greenV[scl_colors]
    colors[:, 2] = blueV[scl_colors]

    return colors


def filename2date(files):
    # Take a filename and get the start time
    import datetime, os

    t0 = list()
    for _f in files:

        this_file = os.path.splitext(_f)[0]

        parts = this_file.split('_')

        # Start time is in the third to last element:
        # check start with s?

        start = datetime.datetime.strptime(parts[-3][1:-1], '%Y%j%H%M%S')
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

        self.events = None
        self.groups = None
        self.flashes = None
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
            this_glm = GLMDataset(_file, calculate_parent_child=False)

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

        group_ids = np.atleast_1d(group_ids)

        evs = [self.events[self.events.parent_id == _id] for _id in group_ids]

        if combine:
            evs = pd.concat(evs)

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
            grps = pd.concat(grps)
            if events:
                evs = pd.concat(evs)
        if events:
            return evs, grps
        else:
            return grps

    def plot_groups(self, groups, do_events=False, ax=None, latlon=True,
                    gridlines=True,
                    marker_group='.',
                    colors_events='yellow', fill_events=True,
                    event_centers=True):
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
        groups : array-like
            The groups to be plotted.
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
        marker_group: str
            The MPL marker used when plotting only groups
            i.e, when `do_events=False`.
        colors_events: str
            The color scheme used to scale the event colors by the energy.
            Hard coded for now to be the yellow scheme!
        fill_events: bool
            If True, fill the events with a color related to `colors_events`.
            If False, just draw an empty polygon.
        event_centers: bool
            If True, plot a marker at the center of each event.

        Returns
        -------
        dict
            A dictionary of the individual MPL plot artists. Depending on the
            arguments, you could have these:

            :groups: MPL Line2D
            :events_poly: MPL PolyCollection of event polygons
            :events_pt: MPL Line 2D of event centroids
            :gridlines: Cartopy Gridliner

        """
        import cartopy.crs as ccrs

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
                              marker=marker_group, **trans_kw)
            retVal['groups'] = grp_plt[0]
        else:
            events = self.get_events(groups.id, combine=True)

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

            if fill_events:
                # todo: here, we would pick a different color scheme
                colors = energy_colors(events.energy.values)/255
            else:
                colors = 'none'

            poly = PolyCollection(verts, edgecolors='black', facecolors=colors, **trans_kw)
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

        return retVal
