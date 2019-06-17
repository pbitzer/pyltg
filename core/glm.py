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
    b = 255.-m *_max_val

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
    import datetime
    
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
            'time': glmdata.dataset.group_time_offset, 
            'lat': glmdata.dataset.group_lat,
            'lon': glmdata.dataset.group_lon,
            'energy': glmdata.dataset.group_energy,
            'id': glmdata.dataset.group_id, 
            '_orig_id': glmdata.dataset.group_id,  # this the id in the file
            'parent_id': glmdata.dataset.group_parent_flash_id, 
            'area': glmdata.dataset.group_area, 
            'quality_flag': glmdata.dataset.group_quality_flag
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
            'time': glmdata.dataset.event_time_offset, 
            'lat' : glmdata.dataset.event_lat, 
            'lon' : glmdata.dataset.event_lon, 
            'energy' : glmdata.dataset.event_energy, 
            'id': glmdata.dataset.event_id, 
            '_orig_id': glmdata.dataset.event_id,  # this the id in the file
            'parent_id': glmdata.dataset.event_parent_group_id
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
            'time': glmdata.dataset.flash_time_offset_of_first_event, 
            'time_last': glmdata.dataset.flash_time_offset_of_last_event, 
            'lat': glmdata.dataset.flash_lat,
            'lon': glmdata.dataset.flash_lon,
            'energy': glmdata.dataset.flash_energy,
            'id': np.uint32(glmdata.dataset.flash_id), 
            '_orig_id': glmdata.dataset.flash_id,  # this the id in the file
            'area': glmdata.dataset.flash_area, 
            'quality_flag': glmdata.dataset.flash_quality_flag
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
    try:
        histo, bins = np.histogram(child.parent_id, bins=_bins)
    except:
        import pdb; pdb.set_trace() 
        
    return histo


class GLM():
    """
    Class to handle GLM data.
    """
        
    def __init__(self, files):
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
            
            this_event = _extract_events(this_glm)
            this_group = _extract_groups(this_glm)
            this_flash = _extract_flashes(this_glm)
            
            # Inflate the 16 bit id to 32 bit to prevent rollover issues:
            this_flash.id += np.uint32(2**16)
            
            # We also need to do the same to the group parent id:
            this_group.parent_id += np.uint32(2**16)
            
            # We'll sort these by id. Makes counting children easier.
            this_event.sort_values('id', inplace=True)
            this_group.sort_values('id', inplace=True)
            this_flash.sort_values('id', inplace=True)

            # Modify the ids to unique values
            # When reading in multiple files, the id's will be replicated 
            # (start over for each file). So, we'll modify the ids
            # to unique values.

            # First, go ahead an subtract off the smallest id value for each.
            # Since we've sorted by id, this is trivial:
            min_ev_id = this_event['id'].iloc[0]
            min_gr_id = this_group['id'].iloc[0]
            min_fl_id = this_flash['id'].iloc[0]
            
            this_event['id'] -= min_ev_id
            this_group['id'] -= min_gr_id
            this_flash['id'] -= min_fl_id
            
            # Don't forget our parents!
            this_event['parent_id'] -= min_gr_id
            this_group['parent_id'] -= min_fl_id
            
            # Next, add in an offset to get unique values
            this_event['id'] += ev_id_ctr
            this_group['id'] += gr_id_ctr
            this_flash['id'] += fl_id_ctr
            
            # Offset the parents too:
            this_event['parent_id'] += gr_id_ctr
            this_group['parent_id'] += fl_id_ctr
            
            # Next, update the counters
            ev_id_ctr += this_event['id'].iloc[-1]
            gr_id_ctr += this_group['id'].iloc[-1]
            fl_id_ctr += this_flash['id'].iloc[-1]
          
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

    def plot_groups(self, groups, do_events=False, ax=None):
        """
        Make a spatial plot of groups.

        The plotting is done using the lons along the horizontal axis,
        lats along the vertical axis.

        .. note::
            Future mod will allow plotting on a map.

        Parameters
        ----------
        groups : array-like
            The groups to be plotted.
        do_events : bool
            If True, then plot the individual child events too. Right now,
            this is done in an approximate manner. The event footprint is
            approximated by drawing a 0.04 dgree (roughly 4 km) box around
            the event lat/lon. This roughly matches the GLM pixel size at nadir,
            so event footprints off-nadir will not be not accurately
            represented.
        ax : MPL Axes
            If given, the plot will be made in the provided Axes.

        Returns
        -------
        dict
            A dictionary of the individual MPL plot artists. Depending on the
            arguments, you could have these:

            :groups: MPL Line2D
            :events_poly: MPL PolyCollection of event polygons
            :events_pt: MPL Line 2D of event centroids.

        """
        
        if ax is None:
            fig, ax = plt.subplots() 
            
        retVal = dict()  # we'll return a dictionary of plot artists
            
        # Get the groups:
        if not do_events:
            # just make a scatter plot
            grp_plt = ax.plot(groups.lon, groups.lat, linestyle='None', marker='.')
            retVal['groups'] = grp_plt[0]
        else:
            events = self.get_events(groups.id, combine=True)
            
            centers = np.vstack((events.lon, events.lat)).T
            
            # assume 8 km square pixels for simplicity
            offsets = np.ones((4, len(events), 2))
            EVENT_EDGE = 0.04
            offsets[0, :, 0] = -EVENT_EDGE # move ul, x
            offsets[1, :, 0] = -EVENT_EDGE # move ll, x
            offsets[2, :, 0] =  EVENT_EDGE # move lr, x
            offsets[3, :, 0] =  EVENT_EDGE # move ur, x
            
            offsets[0, :, 1] =  EVENT_EDGE # move ul, y
            offsets[1, :, 1] = -EVENT_EDGE # move ll, y
            offsets[2, :, 1] = -EVENT_EDGE # move lr, y
            offsets[3, :, 1] =  EVENT_EDGE # move ur, y
            
            verts = centers + offsets
            verts = np.swapaxes(verts, 0, 1)
            
            colors = energy_colors(events.energy.values)/255
            
            poly = PolyCollection(verts, edgecolors='black', facecolors=colors)
            _ = ax.add_collection(poly)
            retVal['events_poly'] = poly
            
            # overplot event centers
            ev_plt = ax.plot(events.lon, events.lat, linestyle='None', marker='.', color='black')
            retVal['events_pt'] = ev_plt[0]
            
        return retVal
    
def contiguous_groups(grps):
    # Get lengths and start positiions of time contiguous groups in the 
    # given dataframe
    # Based on https://stackoverflow.com/a/32681075, which in turn is based
    # on R's rle
    
    diff_t = np.diff(grps.time)  # this should be units of ns
    
    is_contig = diff_t.astype('int64') <= 2000000
    
    n = len(is_contig)
    
    change = np.array(is_contig[1:] != is_contig[:-1])
    vals = np.append(np.where(change), n-1)  # include the last element
    run_length = np.diff(np.append(-1, vals))
    start_pos = np.cumsum(np.append(0, run_length))[:-1]
    
    # pare down to "Trues":
    is_true = is_contig[vals]
    if np.count_nonzero(is_true) == 0:
        return None, None
    else:
        # OK, we have runs of time contiguous groups.
        # Add one to the length to account for the difference
        # (A run_length of 1 means two groups are contiguous)
        return run_length[is_true] + 1, start_pos[is_true]

    
def _contig_groups_stat(grps):
    # From a dataframe of grps, calculate several stats needed for finding
    # probability of CC.
    #
    # TODO: separate each calculation into seperate functions
    
    # Return a dictionary with keys consist with _probability_cc_coeff
    from pyltg.utilities.latlon import lla2enu
    from scipy.spatial.distance import pdist

    # Find the contiguous groups:
    run_len, start_pos = contiguous_groups(grps)
    
    # Find the max run:
    idx = np.argmax(run_len)

    # Now, get the stats of these:
    grp_subset = grps[start_pos[idx]:start_pos[idx]+run_len[idx]]
    
    # Max distance of groups
    # First, get Cartesian coords:
    xyz = lla2enu(grp_subset.lat.values, grp_subset.lon.values, np.zeros(len(grp_subset)))
    max_dist = np.max(pdist(np.array([xyz.x, xyz.y]).T))
    
    # Median energy
    med_e = np.median(grp_subset.energy)
    
    # Max energy
    max_e = np.max(grp_subset.energy)
    
    # Total energy
    tot_e = np.sum(grp_subset.energy)
    
    # Max footprint
    max_area = np.max(grp_subset.area)
    
    # Collect everything into a dictionary:
    stats = {'n_contig': run_len[idx], 
             'delta_dist': max_dist, 
             'med_energy': med_e, 
             'max_energy': max_e, 
             'tot_energy': tot_e, 
             'max_area': max_area
             }

    return stats
    
def _probability_cc_coeff():
    # Logit coefficients from SF, 20190129
    #TODO: add ability to read others from...somewhere
    
    #TODO: standardize the coefficient names b/t here and _contig_groups_stat
    coeff = {'n_contig': -0.1344, 
             'delta_dist': 0.1412, 
             'med_energy': -0.0545, 
             'max_energy': 0.0435, 
             'tot_energy': -0.0001, 
             'max_area': 0.0022,     
             'intercept': -1.9408}
    
    return coeff

def logistic_cc(data, coeff=None):
    # data is a dictionary of values, with keys match those in _probability_cc_coeff
    
    # Because GLM energies are quite small (relative to other parameters), 
    # provide a scale factor to divide by so that they're of similar order
    ENERGY_SCALE = 1e-15
    
    if coeff is None:
        coeff = _probability_cc_coeff()
    
    exponent = coeff['intercept'] ##############THIS NEEDS TO BE EXPANDED TO MATCH LENGTH OF EACH ARRAY IN DATA

    for key, val in data.items():
        
        # If part of the data is not included in the coefficients, skip it
        if key not in coeff.keys():
            # TODO: warn missing key
            continue
        elif 'energy' in key:
            val /= ENERGY_SCALE
        exponent += coeff[key]*val
        
    logisitic = 1/(1+np.exp(-exponent))
    
    return logisitic
    

def probability_cc(grps):
    # Pass a particular set of groups in a flash, 
    # find the probability it has continuing current.
    # Will look up the groups for that flash (so, need groups)

    # todo: make this able to accept a flash(id)?    

    # Find the contiguous groups (no need)
    
    # Calculate the statistics
    stats = _contig_groups_stat(grps)

    # Find the probabilities
    prob = logistic_cc(stats)
    
    return prob


if __name__ == '__main__':
    
    # Compare GLM/NLDN for a paticlaur time/space
    
    import numpy as np 
    import glob, os
    
    from pyltg import NLDN
    from pyltg.utilities.latlon import lla2enu
    
    # nldn sep by year
#    this_nldn_path = basepath_nldn + str(np.datetime64(time, 'Y'))
    
    
    df = pd.read_csv('/Users/bitzer/Downloads/bolide - Sheet1 (1).csv', parse_dates={'time': ['date_time']})
    
    
    basepath_nldn = '/Volumes/hammadev/NLDN/'
    basepath_glm = '/Volumes/hammadev/GLM/'

    
    stats = list()
    for idx, row in df.iterrows():
        
        # Check for no location data
        if np.isnan(row.lon_min): continue
            
        # for testing:
        if idx != 37: continue
#        if not row.has_flashes: continue
    
        time = np.datetime64(row.time, 's')

        if np.isnan(row.lat_max):
            lat_min = row.lat_min-0.5
            lat_max = row.lat_min+0.5
        else:
            lat_min = row.lat_min
            lat_max = row.lat_max
            
        if np.isnan(row.lon_max):
            lon_min = row.lon_min-0.5
            lon_max = row.lon_min+0.5
        else:
            lon_min = row.lon_min
            lon_max = row.lon_max
            
        lat = [lat_min, lat_max]
        lon = [lon_min, lon_max]
        
        # glm sep by date
        this_glm_path = basepath_glm + str(np.datetime64(time, 'D')).replace('-', '')
        
        # Get a listing of all GLM files in directory
        glm_files = sorted(glob.glob( this_glm_path+'/*.nc'))
        glm_start = filename2date(glm_files)
    
        # get GLM, but be specific with the time
        file_idx = np.searchsorted(glm_start, time)
        
        g = GLM(glm_files[file_idx-2:file_idx+3])
        
        flash_idx = g.flashes.limit(time=[np.datetime64(time-2, 'ns'), np.datetime64(time+2, 'ns')], 
                       lat=lat, lon=lon)
    
        # should only be one...
        if flash_idx[1] == 0:
            print('No flashes found for ' + str(time))

            # search for just time, useful for diagnostics
            _time_idx =  g.flashes.limit(time=[np.datetime64(time-2, 'ns'), np.datetime64(time+2, 'ns')])
            xyz = lla2enu(
                    g.flashes[_time_idx[0]].lat.values,
                    g.flashes[_time_idx[0]].lon.values, 
                    np.zeros(_time_idx[1]), 
                    center=[lat[0], lon[0], 0])
            min_dist = np.min(xyz.x**2 + xyz.y**2)
            
            print(min_dist)
            
            continue
            
        this_fl = g.flashes[flash_idx[0]]
        
        this_grp = g.get_groups(this_fl.id, combine=True)#[0] # returned as list
        
        this_stats = _contig_groups_stat(this_grp)
        prob = logistic_cc(this_stats)
        
        this_stats['prob'] = prob
        stats.append(this_stats)
        print(prob)
        
    out = pd.DataFrame(stats)
    
    # plot light curve
    import matplotlib.dates as md

    fig, ax = plt.subplots()
    line = ax.plot(this_grp.time, this_grp.energy)[0]
    xfmt = md.DateFormatter('%H:%M:%S.%f')
    ax.set_xlabel(this_grp.time[0].strftime('%Y-%m-%d'))
    ax.xaxis.set_major_formatter(xfmt)
#    ax.tick_params(axis='x', rotation=45)
    #MPL animation SUUUUCKS
    
    dt = np.timedelta64(100, 'ms')
    frame_times = np.arange(this_grp.time.min(), this_grp.time.max()+dt, dt) 

        
    
    

