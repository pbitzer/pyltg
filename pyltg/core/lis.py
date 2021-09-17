# -*- coding: utf-8 -*-
"""
Read in Lightning Imaging Sensor (LIS) data.

This module contains the LIS class and several functions related to LIS data.
The functions typically help with geolocation that you won't need to call
directly. The function :func:`geolocate_events_in_frame` may
be the most common one you'll want to use outside if you're doing work
outside of the LIS class.

The LIS class is essentially comprised of four
:class:`Ltg <pyltg.core.baseclass.Ltg>` classes: one for events,
one for groups, one for flashes, and one for one second data.
These are attributes of the LIS class.
There are methods to connect these, e.g., for a given flash (or flashes),
get the child groups. Each "sub class" of LIS data has slightly different fields.
We describe below each of the these, which again, are attributes of
the LIS class.

In general, the "address" in the netCDF variables maps to `id` in the fields here.
If you read in multiple files, the original id is kept in addition to
an unique one that the class uses (since ids are not unique across files).

Unlike for GLM, the grandchild count (i.e., the events in a flash) is contained
in the data files and, as such, are included here.

For an example of how to use some the features, see the relevant `example <../lis.html>`_.

"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from netCDF4 import Dataset

from pyltg.core.baseclass import Ltg
from pyltg.core.satellite import get_children, energy_colors
from pyltg.utilities.time import tai93_to_utc
from pyltg.utilities.arrays import rotation_matrix


def _ensure_dataframe(data):
    # Several functions can take several different input types, but
    # ultimately we want to work with a Dataframe. Make sure
    # we get one:

    # We have to do some checking of the input:
    if isinstance(data, pd.Series):
        # We were passed a Series; make it a DataFrame
        data = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        # We were passed several rows of a DataFrame, so nothing to do
        pass
    elif isinstance(data, Ltg):
        # We were passed a pyltg.Ltg object, get the Dataframe
        data = data.data

    return data


def event_poly(events, latlon=True,
               colors='yellow', fill=True,
               corners=None):
    """
    Get a MPL PolyCollection that represents the polygons of the events.

    The color of each polygon represents the energy.

    .. note::
        This is very similar to the same
        :func:`function in glm.py <pyltg.core.glm.event_poly>`. We'll refactor
        at some point and put this into satellite.py...

    Parameters
    -----------
    events : Pandas Dataframe
        The events to be plotted. Intended to be take the output of
        `LIS.get_events`
    latlon : bool
        If True, return the PolyCollection so it can be plotted on a
        Cartopy map plotted with lats and lons.
    colors : str
        The colors to be used to represent the energy of each event.
        This keyword has no effect right now (at some you'll be able
        to change the color scheme).
    fill : bool
        If True, fill the polygons according to `colors`
    corners: varies
        The corners to be used in drawing polygons. Usually used in
        conjuction with :func:`geolocate_events_in_frame`.
        If `None`, then an approximate footprint iof 4kmx4km is used.

    Returns
    --------
    MPL PolyCollection
        A Matplotlib PolyCollection of the events. Of course, to add this
        to an existing axes (either MPL native or Cartopy GeoAxes) just use
        `ax.add_collection(poly)`

    """

    # The plotting calls are similar whether or not we do a map, but there
    # doesn't seem to be "none" for transform (needed for map). So, make a
    # dict with transform if we have it, otherwise leave it empty.
    trans_kw = {}
    if latlon:
        import cartopy.crs as ccrs  # protect the import
        trans_kw['transform'] = ccrs.PlateCarree()

    # If we don't have corners provided, assume square pixels.
    if corners is None:

        centers = np.vstack((events.lon, events.lat)).T

        # assume 4 km square pixels for simplicity
        offsets = np.ones((4, len(events), 2))
        EVENT_EDGE = 0.02
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
    else:
        verts = np.stack((corners.lon, corners.lat))
        verts = np.moveaxis(verts, (0, 1, 2), (2, 0, 1))

    if fill:
        # todo: here, we would pick a different color scheme

        # Be smart about which satellite - if it's GLM, we have energies.
        if 'energy' in events.columns:
            satellite = 'GLM'
            energy = events.energy.values
        elif 'radiance' in events.columns:
            satellite = 'LIS'
            energy = events.radiance.values

        colors = energy_colors(energy, satellite=satellite)/255
    else:
        colors = 'none'

    poly = PolyCollection(verts, edgecolors='black', facecolors=colors, **trans_kw)

    return poly


def _extract_events(nc):
    # Given an open, valid netCDF file, get the events

    # We're not going to get lightning_event_location - it's just lats/lons

    data = pd.DataFrame({
        'time': nc.variables['lightning_event_TAI93_time'][:],
        'observe_time': nc.variables['lightning_event_observe_time'][:],
        'lat': nc.variables['lightning_event_lat'][:],
        'lon': nc.variables['lightning_event_lon'][:],
        'radiance': nc.variables['lightning_event_radiance'][:],
        'footprint': nc.variables['lightning_event_footprint'][:],
        'id': nc.variables['lightning_event_address'][:],
        '_orig_id': nc.variables['lightning_event_address'][:],

        'parent_id': nc.variables['lightning_event_parent_address'][:],


        'px': nc.variables['lightning_event_x_pixel'][:],
        'py': nc.variables['lightning_event_y_pixel'][:],
        'bg_value': nc.variables['lightning_event_bg_value'][:],
        'bg_radiance': nc.variables['lightning_event_bg_radiance'][:],
        'bg_value_flag': nc.variables['lightning_event_bg_value_flag'][:],
        'amplitude': nc.variables['lightning_event_amplitude'][:],

        'threshold': nc.variables['lightning_event_approx_threshold'][:],
        'alert_flag': nc.variables['lightning_event_alert_flag'][:],

        'cluster_index': nc.variables['lightning_event_cluster_index'][:],
        'density_index': nc.variables['lightning_event_density_index'][:],
        'noise_index': nc.variables['lightning_event_noise_index'][:],
        'glint_index': nc.variables['lightning_event_glint_index'][:],
        'sza_index': nc.variables['lightning_event_sza_index'][:],

        'grouping_sequence': nc.variables['lightning_event_grouping_sequence'][:],
        })

    return data


def _extract_groups(nc):
    # Given an open, valid netCDF file, get the groups

    # We're not going to get lightning_group_location - it's just lats/lons
    # lightning_flash_parent_address - it's the area ID

    data = pd.DataFrame({
        'time': nc.variables['lightning_group_TAI93_time'][:],
        'observe_time': nc.variables['lightning_group_observe_time'][:],
        'lat': nc.variables['lightning_group_lat'][:],
        'lon': nc.variables['lightning_group_lon'][:],
        'radiance': nc.variables['lightning_group_radiance'][:],
        'footprint': nc.variables['lightning_group_footprint'][:],
        'id': nc.variables['lightning_group_address'][:],
        '_orig_id': nc.variables['lightning_group_address'][:],
        'parent_id': nc.variables['lightning_group_parent_address'][:],
        'child_id': nc.variables['lightning_group_child_address'][:],
        'child_count': nc.variables['lightning_group_child_count'][:],

        'threshold': nc.variables['lightning_group_approx_threshold'][:],
        'alert_flag': nc.variables['lightning_group_alert_flag'][:],

        'cluster_index': nc.variables['lightning_group_cluster_index'][:],
        'density_index': nc.variables['lightning_group_density_index'][:],
        'noise_index': nc.variables['lightning_group_noise_index'][:],
        'oblong_index': nc.variables['lightning_group_oblong_index'][:],
        'glint_index': nc.variables['lightning_group_glint_index'][:],

        'grouping_sequence': nc.variables['lightning_group_grouping_sequence'][:],
        'grouping_status': nc.variables['lightning_group_grouping_status'][:],
        })

    return data


def _extract_flashes(nc):
    # Given an open, valid netCDF file, get the flashes

    # We're not going to get lightning_flash_locations - it's just lats/lons
    # lightning_flash_parent_address - it's the area ID

    data = pd.DataFrame({
        'time': nc.variables['lightning_flash_TAI93_time'][:],
        'delta_time': nc.variables['lightning_flash_delta_time'][:],
        'observe_time': nc.variables['lightning_flash_observe_time'][:],
        'lat': nc.variables['lightning_flash_lat'][:],
        'lon': nc.variables['lightning_flash_lon'][:],
        'radiance': nc.variables['lightning_flash_radiance'][:],
        'footprint': nc.variables['lightning_flash_footprint'][:],
        'id': nc.variables['lightning_flash_address'][:],
        '_orig_id': nc.variables['lightning_flash_address'][:],
        'child_id': nc.variables['lightning_flash_child_address'][:],
        'child_count': nc.variables['lightning_flash_child_count'][:],
        'grandchild_count': nc.variables['lightning_flash_grandchild_count'][:],

        'threshold': nc.variables['lightning_flash_approx_threshold'][:],
        'alert_flag': nc.variables['lightning_flash_alert_flag'][:],

        'cluster_index': nc.variables['lightning_flash_cluster_index'][:],
        'density_index': nc.variables['lightning_flash_density_index'][:],
        'noise_index': nc.variables['lightning_flash_noise_index'][:],
        'oblong_index': nc.variables['lightning_flash_oblong_index'][:],
        'glint_index': nc.variables['lightning_flash_glint_index'][:],

        'grouping_sequence': nc.variables['lightning_flash_grouping_sequence'][:],
        'grouping_status': nc.variables['lightning_flash_grouping_status'][:],
        })

    return data


def _extract_one_second(nc, background=False):
    # Given an open, valid netCDF file, get the one second data.

    # Depending on the file, the keys are named slightly differently...
    which_file = 'bg_info' if background else 'one_second'

    # Some fields are not in the background file, but are in the data files.
    # Get the common ones:
    data = {
        'time': nc.variables[which_file + '_TAI93_time'][:],
        'alert_summary': nc.variables[which_file + '_alert_summary'][:],
        'instrument_alert': nc.variables[which_file + '_instrument_alert'][:],
        'platform_alert': nc.variables[which_file + '_platform_alert'][:],
        'external_alert': nc.variables[which_file + '_external_alert'][:],
        'processing_alert': nc.variables[which_file + '_processing_alert'][:],
        'position_vector': nc.variables[which_file + '_position_vector'][:].tolist(),
        'velocity_vector': nc.variables[which_file + '_velocity_vector'][:].tolist(),
        'transform_matrix': nc.variables[which_file + '_transform_matrix'][:].tolist(),
        'solar_vector': nc.variables[which_file + '_solar_vector'][:].tolist(),
        'ephemeris_qulaity': nc.variables[which_file + '_ephemeris_quality_flag'][:],
        'altitude_quality': nc.variables[which_file + '_attitude_quality_flag'][:],
        'noise_index': nc.variables[which_file + '_noise_index'][:],
        'event_count': nc.variables[which_file + '_event_count'][:, :].tolist()
        }

    # Now, the other fields:
    if not background:
        more_data = {
            # The position/velocity x/y/z are already in position/velocity vector...
            'boresight_threshold': nc.variables[which_file + '_boresight_threshold'][:],
            'thresholds': nc.variables[which_file + '_thresholds'][:].tolist(),
            }
        data.update(more_data)

    # Make it a Dataframe...
    data = pd.DataFrame(data)

    # todo: Assign lat/lon to be nadir point?

    data.time = tai93_to_utc(data.time)

    return data


def is_alert_high(flag):
    """

    Find out if any alert flags are high for LIS.

    Every second, LIS data contains information about the spacecraft and
    instrument. Among other things, this includes an flag that describes any
    alerts and warnings for that one second time period.

    This function takes in a set of alert flags from LIS and decides whether
    any of the alerts were high. It only looks for the fatal flag and
    ignores the warning flags.


    For completeness, the alert summary is an integer. Each bit of each
    integer corresponds to a different type of alert:

        =======  ===================
        Bit      Meaning
        =======  ===================
        0        Instrument Fatal
        1        Instrument Warning
        2        Platform Fatal
        3        Platform Warning
        4        External Fatal
        5        External Warning
        6        Processing Fatal
        7        Processing Warning
        =======  ===================




    This function looks at bits 0, 2, 4, 6 to see if they are high.

    See the
    `LIS OTD User Guide <https://ghrc.nsstc.nasa.gov/pub/doc/lis/LISOTD_UserGuide.pdf>`_
    for more information about these flags. (Note that here we start counting
    bits at zero, while the User's Guide starts at one. )

    Examples
    ----------
    To find if there are any alerts high for some one seconds::

        has_problem = is_alert_high(lis.one_second.alert_summary)

    Here, `l` is read in as `lis = LIS(file)`

    Parameters
    ----------
    flag : array
        The alert summary (usually in the one second data)

    Returns
    -------
    NumPy array
        An array of booleans, one for each one second. For every element
        that's True, then one of the fatal flags was high.

    """

    instrument_fatal = (flag & 2**0).astype(bool)

    platform_fatal = (flag & 2**2).astype(bool)

    external_fatal = (flag & 2**4).astype(bool)

    processing_fatal = (flag & 2**6).astype(bool)

    # They're booleans, so just add them up so if any are false, they whole thins is
    is_fatal = instrument_fatal + platform_fatal + external_fatal + processing_fatal

    return is_fatal


def read_one_second(file, background=False):
    """
    Read in one second data from a LIS file.

    Information about the LIS position and operating state is updated every
    second and is contained in what as known as "one second" data. This
    function reads in said data.

    This is a convenience function to read in one second data from either a
    LIS data file or LIS background file. Use this if you don't want to
    have open/close file yourself and all you care about is one second data.


    Parameters
    ----------
    file: str
        The file to be read in. This is passed to `netcdf4.Dataset` so whatever
        is accepted there is accepted here.
    background: bool
        If True, you're passing in a background file. (This is necessary
        as the field names in the file are slightly different for data
        files and background files.)

    Returns
    -------
    Ltg class
        A Ltg class containing the one second data.

    """
    nc = Dataset(file)

    data = _extract_one_second(nc, background=background)

    nc.close()

    return Ltg(data)


def geolocate_events_in_frame(evs, one_secs, corners=True):
    """
    Geolocate events in a single frame.

    This can be used to quickly geolocate events in a single frame.
    For example, you might use this to geolocate events in a single group.
    This is essentially a wrapper for :func:`interp_one_sec`
    and :func:`geolocate_pixels`.

    Most of the time, this will be used for getting the footprint of a set
    of events.

    Parameters
    ----------
    evs : varies
        The events. This could be a Pandas Series, Pandas Dataframe, or a
        Ltg class.
    one_secs : Ltg
        The one second data that surrounding the frame. Must have at least
        two one seconds.
    corners : bool, optional
        If True, get the corners for each pixel. If not, get the center of
        each pixel. The default is True.

    Returns
    -------
    lla : SimpleNamespace
        Same as :func:`geolocate_pixels`. If `corners` is True, then the arrays
        are reshaped to be Nx4, where N is the number of pixels. The corners
        are counterclockwise around each pixel (not closed).

    """

    evs = _ensure_dataframe(evs)

    # First, use the one seconds to interpolate to get spacecraft attributes.
    # Remember, these are all in one frame, so the same time...

    pos, vel, xform_matrix = interp_one_sec(evs.iloc[0].time, one_seconds=one_secs)

    if corners:
        # Get the pixel locations as a flattened array in going CCW around each pixel
        px = (evs.px.values[:, np.newaxis] + np.array([-0.5, -0.5, 0.5, 0.5])).ravel()
        py = (evs.py.values[:, np.newaxis] + np.array([-0.5, 0.5, 0.5, -0.5])).ravel()
    else:
        px = evs.px.values
        py = evs.py.values

    lla = geolocate_pixels(pos, vel, xform_matrix, px, py)

    if corners:
        # Reshape each array to Nx4 arrays, where N is the number of pixels
        for key in lla.__dict__.keys():
            setattr(lla, key, getattr(lla, key).reshape((-1, 4)))

    return lla


def ccd_pixels(border=False, corners=False, full_frame=False):
    """
    Get x/y addresses corresponding to the CCD pixels.

    This is a convenience function to get the x/y pixel addresses. The
    upper left of the CCD is `x,y=0,0`. LIS's CCD is 128x128 pixels.

    At least one of the parameters must be set to `True`.

    Parameters
    ----------
    border : bool, optional
        If True, get the pixels along the border of the CCD. Especially
        useful when geolocating the field of view of LIS. The default is False.
    corners : bool, optional
        Not yet implemented. The default is False.
    full_frame : bool, optional
        If True, get x/y locations for the whole CCD. Especially useful when
        geolocating a background. The default is False.

    Returns
    -------
    SimpleNamespace
        Two fields: `x` and `y`, corresponding to pixel addresses of the CCD.

    """

    if border:
        edge = np.arange(127)  # one less than pixel array size
        n_border = len(edge)

        min_corner = 0
        max_corner = 127

        min_edge = np.full(n_border, min_corner)
        max_edge = np.full(n_border, max_corner)

        # We'll go CCW - bottom, right, top, left:
        xpix = np.concatenate([edge, max_edge, [max_corner], np.flip(edge), min_edge])
        ypix = np.concatenate([min_edge, edge, [max_corner], max_edge, np.flip(edge)])

        xy = np.vstack([xpix, ypix]).T

    elif corners:
        raise NotImplementedError('still working on this')
    elif full_frame:
        xpix = np.arange(128)
        ypix = np.arange(128)

        xy = np.array(np.meshgrid(xpix, ypix)).T.reshape(-1, 2)  # much faster than itertools
    else:
        raise Exception('You must set one of the arguments to True')

    return SimpleNamespace(x=xy[:, 0], y=xy[:, 1])


def get_pointing_vector(x_pixels, y_pixels):
    """
    Find the pointing vector for a set of pixels.

    This finds the vector pointing from the CCD to the earth for a given set
    of pixels. It's the first step needed to geolocating pixel.

    Note that the passed in pixels could be fractional. This is handy if you
    want to find a pixel's footprint.

    .. note::
        This function contains a magnification factor specific to ISS_LIS.

    Parameters
    ----------
    x_pixels : array-like or scalar
        The x address(es) of the pixel(s).
    y_pixels : array-like or scalar
        The y address(es) of the pixel(s).

    Returns
    -------
    NumPy array
        The `Nx3` array of vector components, where `N` is the number of pixels
        passed in.

    """

    x_pixels = np.atleast_1d(x_pixels)
    y_pixels = np.atleast_1d(y_pixels)

    # Coefficients related to LIS optics.
    coeff = np.array([1.4754537, -0.36224695, -0.088939824, -0.28203806])  # lis

    # otd coeff = [1.4037314D, .015948348D, -.86702375D, .24905820D]

    x = -1*(63.5 - x_pixels)/127
    y = -1*(y_pixels-63.5)/127

    xy = np.sqrt(x**2 + y**2)

    convert = coeff[0] + xy*(coeff[1]+xy*(coeff[2]+xy*(coeff[3])))

    look_vector = np.zeros((len(x_pixels), 3))
    MAGNIFICATION_FACTOR = 1.01
    look_vector[:, 0] = x*convert*MAGNIFICATION_FACTOR
    look_vector[:, 1] = y*convert*MAGNIFICATION_FACTOR
    look_vector[:, 2] = np.sqrt(1 - (look_vector[:, 0]**2 + look_vector[:, 1]**2))

    return look_vector


def lvlh_matrix(pos, vel):
    """
    Calculate the local vertical, local horizontal (LVLH) reference frame matrix.

    See `https://spaceflight.nasa.gov/feedback/expert/answer/mcc/sts-105/08_14_08_01_18.html`
    for more information.

    Parameters
    ----------
    pos: array-like
        The position of the spacecraft.
    vel: array-like
        The velocity of the spacecraft.

    Returns
    -------

    """

    # local vertical, local horizontal frame
    v_mag = np.sqrt(np.dot(vel, vel))
    p_mag = np.sqrt(np.dot(pos, pos))

    cross = np.cross(pos, vel)

    m_lvlh = np.array([
        vel/v_mag,
        cross/(-p_mag*v_mag),
        pos/(-p_mag)
        ])

    return m_lvlh


def yaw_pitch_roll(vel):
    """
    The yaw, pitch, roll angles for ISS LIS.

    These are empirically determined. See the work done by Tim Lang
    `<https://github.com/nasa/ISS_Camera_Geolocate/blob/master/iss_camera_geolocate/iss_lis_background_geolocate.py`>

    Parameters
    ----------
    vel : 3 element array
        The velocity of the instrument/platform. Typically, this comes from
        one second data.

    Returns
    -------
    tuple
        Three element tuple with thw yaw, pitch, and roll angles.
    """

    # These are ISS-LIS specific values, empirically determeined (link in docstring)
    ROTATE_FACTOR = 0
    LEFT_RIGHT_FACTOR = -0.0022
    UP_DOWN_FACTOR = 0.0205

    yaw = 3.182 - 0.0485 + \
        0.0485 * vel[2] / 6000.0 + ROTATE_FACTOR
    pitch = -0.020 + 0.015 - \
        0.0042 * vel[2] / 6000.0 + LEFT_RIGHT_FACTOR
    roll = 0.020 - 0.051 + UP_DOWN_FACTOR

    return yaw, pitch, roll


def find_slant_range(look_vec, pos, parallax=13.5e3):
    """
    Find the range from a spacecraft position to earth along a particular direction.

    Parameters
    ----------
    look_vec: NumPy array
        The look vector.  # TODO: how many elements/shape? nx3?
    pos: NumPy Array
        The position of the spacecraft # TODO: I think this has to be three elements for now, no more
    parallax: scalar
        The height, in meters, above earth's surface to find the range to. This is effectively
        cloud height. Default is 13,500 m (13.5 km)


    Returns
    -------
    Numpy Array
        The range to earth's surface for each look vector.

    """

    # Find the distance from where are (pos - ecef) to earth's "surface" (really,
    # the cloud height) along the look_vec
    axisA = 6378137.0 + parallax
    axisB = axisA
    axisC = 6356752.31 + parallax

    # square the axes for future use
    axis_sq = np.array([axisA**2, axisB**2, axisC**2])

    # We're going to rely a whole lot on NumPy's broadcasting to find the
    # solution....

    a_coef = np.sum(look_vec**2/axis_sq, axis=1)  # Broadcasting...

    b_coef = 2*np.sum(np.array(pos) * look_vec/axis_sq, axis=1)

    c_coef = -1 + np.sum(pos**2/axis_sq)

    radical = b_coef**2 - 4*a_coef*c_coef

    # todo: add check here - if radical < 0 or a_coef < 0

    # The solution is given by solving the quadratic equation:
    slant_range = (-1*b_coef - np.sqrt(radical)) / (2*a_coef)

    return slant_range


def earth_intersection(pos, vel, transform_matrix, ccd_vector):
    """

    Find the ECEF position corresponding to a satellite pixel.

    There are several transformations needed to locate a source
    correpsonding to a satellite pixel. Typically, you have a vector
    defining the where a pixel is looking (through the satellite optics).
    You also have the orientation of the instrument on the satellite and the
    orientation of the satellite.

    These transformations are applied to define a vector pointing from a
    pixel, through the optics, and accounting for the orientation/position
    of detector and satellite. Then, the Earth Centered, Earth Fixed (ECEF)
    position is found. This is done with the default values in
    :func:`find_slant_range` (so nominally above the earth's surface)

    Rarely would you need to call this function directly. Instead, just use
    :func:`geolocate_pixels`

    Parameters
    ----------
    pos: NumPy array
        The position of the satellite. Usually from one second data.
    vel: NumPy array
        The velocity of the satellite.  Usually from one second data.
    transform_matrix: NumPy array
        The transformation matrix from pixel coordinates to ECR coordinates.
         Usually from one second data.
    ccd_vector
        The vector pointing through the optics. See :func:`get_pointing_vector`

    Returns
    -------

    """

    # If pos is not an NumPy array, make it so:
    if type(pos) is not np.ndarray:
        pos = np.array(pos)

    # Get yaw, pitch, roll
    ypr = yaw_pitch_roll(vel)

    # This defines the orientation of LIS on the ISS:
    Mlisorientation = rotation_matrix(*ypr)
    Mypr = np.array(transform_matrix).reshape(3, 3)

    # This gives us the orientation of the ISS
    Mlvlh = lvlh_matrix(pos, vel)

    # Get the transformation from CCD coordinates to the satellite
    n_pix = ccd_vector.shape[0]
    look = np.empty((n_pix, 3))
    _transform = np.dot(Mypr, Mlisorientation)

    # Transform the CCD vector into coordinates of the satellite
    for idx in np.arange(n_pix):
        look[idx, :] = np.dot(np.dot(_transform, ccd_vector[idx, :]), Mlvlh)

    # TODO: should already be normalized, right?
    # This is the "look" vector for the pixels accounting for the transformations
    look_unit = look/np.sqrt(np.sum(look**2, axis=1)[:, np.newaxis])

    # Find the distance from where we are to earth's surface (well, cloud height)
    slant_range = find_slant_range(look_unit, pos)

    # Navigate to the source, in ECEF units.
    look_ecr = pos + slant_range[:, np.newaxis]*look_unit

    return look_ecr


def geolocate_pixels(pos, vel, transform_matrix, pixel_x, pixel_y):
    """
    Geolocate the given pixel address locations.

    Right now, this only works for pixels in a single frame, i.e., for one
    set of spacecraft attributes.

    Parameters
    ----------
    pos : array-like
        3 element array specifying the spacecraft position.
    vel : array-like
        3 element array specifying the spacecraft velocity.
    transform_matrix : array-like
        9 element array specifying the transformation matrix.
    pixel_x : scalar or array like
        X pixel address to be located. Can be fractional.
    pixel_y : scalar or array like
        Y pixel address to be located. Can be fractional.

    Returns
    -------
    SimpleNameSpace
        Returned values contains fields for lat, lon, and alt. Each field is
        an array with same number of elements as `pixel_x` (or `pixel_y`)

        .. note::
            This will likely be changed to a dataclass when the package moves to 3.7+

    """

    import pyproj

    # Define the transform from ECEF <-> lat/lon
    ecefProj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    llaProj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    ecef2lla = pyproj.Transformer.from_proj(ecefProj, llaProj)

    # Pointing vector
    pointing_vector = get_pointing_vector(pixel_x, pixel_y)

    # Earth intersection
    src_ecef = earth_intersection(pos, vel, transform_matrix, pointing_vector)
    # Transform
    lon, lat, alt = ecef2lla.transform(src_ecef[:, 0], src_ecef[:, 1], src_ecef[:, 2])

    return SimpleNamespace(lat=lat, lon=lon, alt=alt)


def interp_one_sec(time, one_seconds=None,
                   one_sec_times=None, one_sec_pos=None, one_sec_vel=None,
                   one_sec_transform_matrix=None):
    """
    Interpolate one second data to a given time(s).

    Sometimes, we want to be able to geolocate LIS data for a time not
    exactly given in one second data (e.g., a background, or most
    events, groups, etc.). This function will do so, although it's only
    really handy if you're reading in the data outside the
    :py:class:`~.LIS` class (like reading in background files with
    `xarray`)

    The interpolation is linear, which has been used for years with LIS data
    and seems to work well. You'll want to provide at least two one second
    data periods.

    See the `example <../lis.html>`_ for a instance how this might be used.

    Parameters
    ----------
    time : NumPy Datetime64[ns]
        The time (or times) to interpolate to. Usually something like a
        background time. Can be a scalar.
    one_seconds : Ltg class or Pandas Dataframe, optional
        The one second data used for interpolation. Either provide this,
        or all of the other `one_sec*` arguments. If you provide this
        the other `one_sec*` arguments are ignored.
        The default is None.
    one_sec_times : NumPy array, optional
        The time used for interpolation. Should be NumPy Datatime64 or
        the 64 bit integer corresponding to those times.
        The default is None.
    one_sec_pos : NumPy array, optional
        The positions used for interpolation. The default is None.
    one_sec_vel : NumPy array, optional
        The velocities used for interpolation. The default is None.
    one_sec_transform_matrix : NumPy array, optional
        The transformation matrix used for inteprolation. The default is None.

    Returns
    -------
    tuple
        Three element tuple given the interpolated position, velocity, and
        transformation matrix for desired times.

    """

    # return 3 element tuple of Nx? arrays
    from scipy.interpolate import interp1d

    # If one_seconds was given, use that. If not, make sure we have all the
    # other necessary parameters.

    # Interpolation doesn't play nicely with times. Work around it by using
    # integers.
    # todo: should this be floats for interp? (that would stink)
    if one_seconds is not None:
        frame_time = one_seconds.time.astype('int64')
        pos_stack = np.stack(one_seconds.position_vector, axis=0)
        vel_stack = np.stack(one_seconds.velocity_vector, axis=0)
        xform_matrix_data = np.stack(one_seconds.transform_matrix, axis=0)
    elif (one_sec_times is not None) & (one_sec_pos is not None) & (one_sec_vel is not None) & (one_sec_transform_matrix is not None):
        frame_time = one_sec_times.astype('int64')
        pos_stack = one_sec_pos
        vel_stack = one_sec_vel
        xform_matrix_data = one_sec_transform_matrix
    else:
        msg = 'Wrong number of parameters - either use one_seconds or pass all of time, position, velocity, and transform matrix'
        raise TypeError(msg)

    # Interpolation is much better behaved if we offset the times to an epoch
    t0 = np.min(frame_time)
    if isinstance(time, pd.Timestamp):
        # We hit this if a scalar is passed for time, e.g., LIS.events[0].time
        # (author note: but not LIS.events.time[0])
        time = time.to_datetime64()

    time_shift = time.astype('int64') - t0
    frame_time_shift = frame_time - t0

    pos = list()
    vel = list()
    for idx in np.arange(3):
        _interp_p = interp1d(frame_time_shift, pos_stack[:, idx])
        pos.append(_interp_p(time_shift))

        _interp_v = interp1d(frame_time_shift, vel_stack[:, idx])
        vel.append(_interp_v(time_shift))

    pos = np.array(pos)
    vel = np.array(vel)

    transform_matrix = list()
    for idx in np.arange(9):
        _interp = interp1d(frame_time_shift, xform_matrix_data[:, idx])
        transform_matrix.append(_interp(time_shift))

    transform_matrix = np.array(transform_matrix)

    return pos, vel, transform_matrix


def get_fov(one_sec, times=None):
    """
    Get the FOV corresponding to some one second data.

    This is mostly a convenience function and allows you to quickly
    get the geolocated field of view of LIS for the given one second data.

    Parameters
    ----------
    one_sec : varies
        The one second data. This could be a Pandas Series, Dataframe, or a
        Ltg class.

    times : NumPy datetime64 (scalar or array-like)
        The time(s) corresponding to a field of view you desire. In this case,
        `one_sec` must be at least two elements and is used to interpolate
        the spacecraft attributes to the time(s) you want.

    Returns
    -------
    fov : list
        The geolocated "border" of LIS's CCD - effectively the field of view.
        For n one seconds passed, you get a list of the same length. Each
        element of the list corresponds to the return value of
        :func:`geolocate_pixels`.

    Examples
    --------
    Given a LIS data file::

        file = 'ISS_LIS_SC_V1.0_20200108_184930_NRT_17139.nc'
        l = LIS(file)

    Find the field of view for every 5th second::
        fov = get_fov(l.one_second[0::5])

    Plot them::

        import matplotlib.pyplot as plt

    """

    # We have to do some checking of the input:
    if isinstance(one_sec, pd.Series):
        # We were passed a Series; make it a DataFrame
        one_sec = pd.DataFrame([one_sec])
    elif isinstance(one_sec, pd.DataFrame):
        # We were passed several rows of a DataFrame, so nothing to do
        pass
    elif isinstance(one_sec, Ltg):
        # We were passed a pyltg.Ltg object, get the Dataframe
        one_sec = one_sec.data
    else:
        raise TypeError('Unknown argument type for one_sec')

    # If we have times, interpolate to get the spacecraft position
    if times is not None:
        pos, vel, transform_matrix = interp_one_sec(times,
                                                    one_seconds=one_sec)

        if len(pos.shape) == 1:
            # If True, then we were passed a scalar for times.
            # Need to a little manipulation before moving on...
            pos = pos.reshape(-1, 1)
            vel = vel.reshape(-1, 1)
            transform_matrix = transform_matrix.reshape(-1, 1)

        one_sec = pd.DataFrame({
            'position_vector': list(pos.T),
            'velocity_vector': list(vel.T),
            'transform_matrix': list(transform_matrix.T)})

    # At this point, we've made sure we have a DataFrame. This makes
    # the following much more straightforward....

    pix = ccd_pixels(border=True)

    fov = list()

    for _, row in one_sec.iterrows():

        lla = geolocate_pixels(row.position_vector,
                               row.velocity_vector,
                               row.transform_matrix,
                               pix.x, pix.y)
        fov.append(lla)

    return fov


class LIS():
    """
    Class to handle LIS data.

    This class is comprised of four "sub"classes,`flashes`, `groups`, `events`,
    and `one_second`, all :py:class:`~pyltg.core.baseclass.Ltg` classes. The
    data attributes of these classes are:

    .. note::
        These descriptions are incomplete, i.e., there are fields not documented
        here. For a complete description, see the
        `LIS OTD User Guide <https://ghrc.nsstc.nasa.gov/pub/doc/lis/LISOTD_UserGuide.pdf>`_

    `LIS.flashes`:

    =================  ========================================================================
    Attribute          Description
    =================  ========================================================================
    time               The time the flash starts
    delta_time         How long the flash lasted (time last group - first group)
    lat                Latitude of the radiance weighted centroid of the child groups
    lon                Longitude of the radiance weighted centroid of the child groups
    radiance           Measured radiance (really spectral energy density) energy of the flash
    footprint          Area of the flash (in km**2) # TODO: change this to area to match GLM?
    id                 32bit value used as flash identifier
    child_id           The ids of the child groups
    child_count        The number of groups in the flash
    grandchild_count   The number of events in the flash
    alt                Meaningless and zero-d out.
    =================  ========================================================================

    `LIS.groups`:

    =================  ========================================================================
    Attribute          Description
    =================  ========================================================================
    time               The time of the group
    lat                Latitude of the radiance weighted centroid of the child events
    lon                Longitude of the radiance weighted centroid of the child events
    radiance           Measured radiance (really spectral energy density) energy of the group
    footprint          Area of the group (in km**2) # TODO: change this to area to match GLM?
    id                 32bit value used as group identifier
    parent_id          The id of the parent flash
    child_id           The ids of the child events
    child_count        The count of the events in the group
    alert_flag         The bit masked value for instrument, platform, and external factors and processing algorithm
    alt                Meaningless and zero-d out.
    =================  ========================================================================

    `LIS.events`:

    =================  ========================================================================
    Attribute          Description
    =================  ========================================================================
    time               The time of the event
    delta_time         Latitude of the event
    lat                Longitude of the event
    lon                Longitude of the radiance weighted centroid of the child groups
    radiance           Measured radiance (really spectral energy density) energy of the event
    footprint          Area of the event (in km**2) # TODO: change this to area to match GLM?
    id                 32bit value used as event identifier
    parent_id          The id of the parent group
    child_count        The number of groups in the flash
    px                 The CCD pixel address in the x direction
    py                 The CCD pixel address in the y direction
    bg_value           The 16 bit value of the background at the time of the event
    bg_radiance        The background radiance at the time of the event
    alt                Meaningless and zero-d out.
    =================  ========================================================================

    `LIS.one_second`:

    =================  ========================================================================
    Attribute          Description
    =================  ========================================================================
    time               The time of the one second
    lat                Right now, zero'd out (eventually, the latitude of nadir)
    lon                Right now, zero'd out (eventually, the longitude of nadir)
    intstrument_alert  Bit masked status of instrument
    platform_alert     Bit masked status of platform
    external_alert     Bit masked status of external factors
    processing_alert   Bit masked status of processing algorithms
    position_vector    The location of the satellite in ECR coordinates
    velocity_vector    The velocity of the satellite in ECR coordinates
    transform_matrix   Components of transform from pixel plane-boresight coordinates to ECR coordinates of boresight and pixel plane
    noise_index        A metric related to noise level (0-100)
    alt                Meaningless and zero-d out.
    =================  ========================================================================

    """

    def __init__(self, files=None):
        """
        Initialization.

        Parameters
        ----------
        files : str
            The file(s) to be read in.
        """

        self.events = Ltg()
        self.groups = Ltg()
        self.flashes = Ltg()
        self.one_second = Ltg()

        if files is not None:
            self.readFile(files)

    def readFile(self, files):
        """
        Read the given file(s).

        Normally, you'd provide a file (or files) on intialization, so you
        don't have to call this separately.

        Parameters
        ----------
        files : str
            The file(s) to be read in.

        """
        files = np.atleast_1d(files)  # allow scalar input

        events = list()
        groups = list()
        flashes = list()
        one_sec = list()

        ev_id_ctr = 0
        gr_id_ctr = 0
        fl_id_ctr = 0

        for _file in files:
            # todo: with...open
            nc = Dataset(_file)

            this_ev = _extract_events(nc)
            this_grp = _extract_groups(nc)
            this_fl = _extract_flashes(nc)
            this_one_sec = _extract_one_second(nc, background=False)

            nc.close()

            # TODO: do we need check for "empty" files like w/GLM?

            # IDs are not necessarily unique. We'll modify them so they are.
            # Similar to what is done with GLM data (glm.py in this package)
            # See there for details, but the gist is get unique values and map
            # TODO: refactor?

            this_ev.sort_values('id', inplace=True)
            this_grp.sort_values('id', inplace=True)
            this_fl.sort_values('id', inplace=True)

            new_flash_id = np.arange(len(this_fl))
            this_fl.id = new_flash_id
            flash_id_map = dict(zip(this_fl._orig_id.values, new_flash_id))

            # Update group parent
            new_id = this_grp.parent_id.map(flash_id_map.get)
            this_grp.parent_id = new_id

            # New id for the group:
            new_group_id = np.arange(len(this_grp))
            this_grp.id = new_group_id
            group_id_map = dict(zip(this_grp._orig_id.values, new_group_id))

            # Update event parent
            this_ev.parent_id = this_ev.parent_id.map(group_id_map.get)

            # New event ID (although I don't think is really necessary)
            new_event_id = np.arange(len(this_ev))
            this_ev.id = new_event_id

            # Add in an offset to get unique values across files
            this_ev['id'] += ev_id_ctr
            this_grp['id'] += gr_id_ctr
            this_fl['id'] += fl_id_ctr

            # Offset the parent IDs for the children too:
            this_ev['parent_id'] += gr_id_ctr
            this_grp['parent_id'] += fl_id_ctr

            # Next, update the counters
            ev_id_ctr = this_ev['id'].iloc[-1]+1
            gr_id_ctr = this_grp['id'].iloc[-1]+1
            fl_id_ctr = this_fl['id'].iloc[-1]+1

            # Modify the times to UTC:
            for val in [this_ev, this_grp, this_fl]:  # one seconds already converted
                val.time = tai93_to_utc(val.time)

            # todo: add option to not sort by time
            # this_event.sort_values('time', inplace=True)
            # this_group.sort_values('time', inplace=True)
            # this_flash.sort_values('time', inplace=True)

            # Finally, add "this" data
            events.append(this_ev)
            groups.append(this_grp)
            flashes.append(this_fl)
            one_sec.append(this_one_sec)

        # Put these as attributes of the class
        self.events = Ltg(pd.concat(events))
        self.groups = Ltg(pd.concat(groups))
        self.flashes = Ltg(pd.concat(flashes))
        self.one_second = Ltg(pd.concat(one_sec))

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
        # TODO: This (and more is very similar to GLM class. Possible refactor?

        evs = get_children(group_ids, self.events)

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

        # TODO: This (and more) is very similar to GLM class. Possible refactor?

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

    def plot_groups(self, groups=None, do_events=False, ax=None, latlon=True,
                    gridlines=True,
                    group_marker='.', group_color='black',
                    event_color='yellow', fill_events=True,
                    event_centers=False):
        """
        Make a spatial plot of groups.

        The plotting is done using the lons along the horizontal axis,
        lats along the vertical axis.

        Parameters
        ----------
        groups : `Ltg` class or Pandas Dataframe
            The groups to be plotted. If `None`, plot the active
            groups. Default is None.
        do_events : bool
            If True, then plot the individual child events too. The actual
            footprint of the events is found and plotted (so it might take a
            bit longer to plot).
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
                _, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mercator()))
            else:
                _, ax = plt.subplots()

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
            events = self.get_events(groups.id, combine=False)

            poly = list()
            for evs in events:  # These are the events, separated into groups
                # First, get the corners for these events
                corners = geolocate_events_in_frame(evs, self.one_second, corners=True)

                this_poly = event_poly(evs, colors=event_color, latlon=latlon,
                                       fill=fill_events, corners=corners)

                _ = ax.add_collection(this_poly)

                poly.append(this_poly)

            # If nothing else is plotted, then the x/y limits be MPL's default.
            # In this case, we'll want to set the x/y limits.
            # Otherwise, just add the events to the current view
            if (ax.get_xlim() == (0.0, 1.0)) & (ax.get_ylim() == (0.0, 1.0)):
                ax.autoscale()

            retVal['events_poly'] = poly

            # This should be probably combined with the above loop...
            if event_centers:
                ev_plt = list()
                for evs in events:
                    _this_plot = ax.plot(evs.lon, evs.lat, linestyle='None',
                                         marker='.', color='black', markersize=0.5,
                                         **trans_kw)
                    ev_plt.append(_this_plot)

                retVal['events_pt'] = ev_plt

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
        self.one_second.reset_active()

class LIS_Background():
    """
    Class to hold LIS Background data.

    This will be a "dirty" class, intended to show functionality. As needed,
    this can be cleaned to made more formal. For example, just basic data
    is read in from the file - just enough to make a plot.
    """

    def __init__(self, file):

        # Right now, this is "dirty" class and we won't separate readfile...

        nc = Dataset(file)  # todo: use xarray?

        self.bg_data = nc.variables['bg_data'][:]

        times = nc.variables['bg_data_summary_TAI93_time'][:]
        self.times = tai93_to_utc(times)

        self.one_second = Ltg(_extract_one_second(nc, background=True))

        nc.close()

    def plot(self, idx=None, time=None):
        """
        Generate a basic plot of the background.

        Parameters
        ----------
        idx : int, optional
            The index of the background (as stored in the file) you want.
            Provide either `idx` OR `time`. If both are given, `idx` is ignored.
            The default is None.
        time : np.datetime64, optional
            The time you want a background for. We'll find the closest.
            Provide either `idx` OR `time`. If both are given, `idx` is ignored.
            The default is None.

        Returns
        -------
        ax : Cartopy.GeoAxesSubplot
            The Cartopy axes of the plot.

        """

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if time is not None:
            # OK, we need to do some work to find which index is closest
            idx = np.abs(self.times-time).argmin()
            # todo: print the time used?

        bg_frame_attr = interp_one_sec(self.times[idx],
                                       one_seconds=self.one_second)

        ccd = ccd_pixels(full_frame=True)

        lla = geolocate_pixels(bg_frame_attr[0],
                               bg_frame_attr[1],
                               bg_frame_attr[2],
                               ccd.x, ccd.y
                               )

        map_proj = ccrs.Mercator()
        ll_proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(subplot_kw={'projection': map_proj})

        ax.pcolormesh(lla.lon.reshape(128, 128), lla.lat.reshape(128, 128),
                      self.bg_data[idx].T, transform=ll_proj,
                      alpha=0.8, cmap='bone')

        _ = ax.add_feature(cfeature.LAND, facecolor='burlywood')
        _ = ax.add_feature(cfeature.OCEAN, facecolor='steelblue')

        _ = ax.gridlines(draw_labels=True, linestyle=':')

        return ax
