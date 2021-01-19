
"""
Functions for various latitude/longitude operations.

While pyproj is a great package, it doesn't seem to be able to handle
Eaast-North-Up (ENU) coordinate systems. Among other things, this module
can.

"""

import numpy as np
import pyproj


def rot_matrix_ecef2enu(lam, phi):
    """
    Define the rotation matrix to go from ECEF coordinates to ENU.

    This doesn't seem to be in the package pyproj, so we'll define it here.
    Typically, you won't need to call this function.

    Parameters
    ----------
    lam : numeric
        The longitude of the center of the ENU system.
    phi : numeric
        The longitude of the center of the ENU system.

    Returns
    -------
    numpy array
        A 3x3 numpy array defining the rotation matrix.

    """
    # Make the matrix below a little easier to type by defining a few variables:
    sLam = np.sin(lam)
    cLam = np.cos(lam)
    sPhi = np.sin(phi)
    cPhi = np.cos(phi)

    rotMatrix = np.array([[-sLam,      cLam,       0],
                         [-sPhi*cLam,  -sPhi*sLam, cPhi],
                         [cPhi*cLam,   cPhi*sLam,  sPhi]])

    return rotMatrix


def lla2enu(lats, lons, alts, center=None):
    """
    Convert lat, lon, alt to ENU (East North Up) coordinates.

    If no center is given, use the average lat/lon/alt of input. This is the
    sister function to :func:`enu2lla`

    Parameters
    ----------
    lats : array-like
        The latitudes of the data
    lons : array-like
        The longitudes of the data
    alts : array-like
        The altitudes of the data (in km)
    center : three element array-like
        The lat, lon, alt of the center of the ENU coordinate system. If none
        is given, then then average of each input is used.

    Returns
    -------
    numpy record array
        An `n` element numpy record array, where `n` is the number of elements
        in lats/lons/alts. The record arrays has the fields `x`, `y`, and `z`.

    Examples
    --------

    Find the Cartesian location of a point located at a lat, lon of
    (34.681, -86.530) and a altitude of 1.1 km in a coordinate system
    centered at lat, lon = (34.726, -86.639) and 100 meters above the surface.
    But, remember, heights should be in kilometers and arguments are array-like
    (i.e., no scalars)::

        center_coord = (34.726, -86.639, 0.1)
        xyz = lla2enu([34.681], [-86.530], [1.1], center=center_coord)

    For a sanity check, convert back to x,y,z using the sister function
    func:`enu2lla`::
        lla = enu2lla(xyz.x, xyz.y, xyz.z, center=center_coord)

    """

    # Start by defining the projections for the conversion:
    ecefProj = pyproj.Proj(proj='geocent',  ellps='WGS84', datum='WGS84')
    llaProj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Define the transform:
    lla2ecef_xform = pyproj.Transformer.from_proj(llaProj, ecefProj)

    # First, convert to ECEF
    ecef = np.array(lla2ecef_xform.transform(lons, lats, np.array(alts)*1e3))

    # Next, convert the center:
    if center is None:
        center = (np.mean(lats), np.mean(lons), np.mean(alts)*1e3)

    centerEcef = np.array(lla2ecef_xform.transform(center[1], center[0], center[2]*1e3))

    # Now, we convert ECEF to ENU...

    # Start by finding the vector pointing from the origin to the point(s) of interest
    vec = (ecef.transpose() - centerEcef).transpose()

    # Now, we construct the rotation matrix at the origin:
    lam = np.radians(center[1])
    phi = np.radians(center[0])

    # Get the rotation matrix:
    rotMatrix = rot_matrix_ecef2enu(lam, phi)

    # Finally, transform ECEF to ENU
    enu = rotMatrix.dot(vec)/1e3  # return in km

    # Return as an numpy record array
    return np.core.records.fromarrays(enu, names='x,y,z')


def enu2lla(x, y, z, center):
    """
    Convert from ENU (East North Up) coordinates to latitude/longitude/altitude.

    This is the sister function to :func:`lla2enu`

    Parameters
    ----------
    x : array-like
        The east-west (x) coordinates of the data in kilometers.
    y : array-like
        The north-south (y) coordinates of the data in kilometers.
    z : array-like
        The up-down (z) coordinates of the data in kilometers.
    center : three element array-like
        The lat/lon/alt of the center of the ENU coordinate system.
        Altitude is in kilometers; lat/lon in degrees.

    Returns
    -------
    numpy record array
        An `n` element numpy record array, where `n` is the number of elements
        in x/y/z. The record arrays has the fields `lat`, `lon`, and `alt`.
        All values are in kilometers.

    Examples
    --------

    Find the lat/lon/altitude of a point that is 10 km east, 5 km south, and
    1 km above the Cartesian coordinate system centered at
    lat, lon = (34.726, -86.639) and 100 meters above the surface. But,
    remember, heights should be in kilometers and arguments are array-like
    (i.e., no scalars)::

        center_coord = (34.726, -86.639, 0.1)
        lla = enu2lla([10], [-5], [1], center=center_coord)

    For a sanity check, convert back to x,y,z using the sister function
    func:`lla2enu`::
        xyz = lla2enu(lla.lat, lla.lon, lla.alt, center=center_coord)

    """

    # Start by defining the projections for the conversion:
    ecefProj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    llaProj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Define the transform:
    lla2ecef_xform = pyproj.Transformer.from_proj(llaProj, ecefProj)

    # Take the center point and convert to ECEF, but convert km to m
    centerEcef = lla2ecef_xform.transform(center[1], center[0], center[2]*1e3)

    # Now, convert the ENU coordinates to a "delta" ECEF. This is the offset
    # (in ECEF coordinates) of the points from the center. To do so, get the
    # the rotation matrix, and apply the inverse
    # (since it's a rotation matrix, the inverse is the transpose).

    rotMatrix = rot_matrix_ecef2enu(np.radians(center[1]), np.radians(center[0])).T
    ecefDelta = rotMatrix.dot([x, y, z])*1e3  # again, convert to m

    # Now, translate the vector of points to the ECEF of the center:
    ecefVec = (ecefDelta.transpose() + centerEcef).transpose()

    # Convert these to LLA:
    lla = np.array(lla2ecef_xform.transform(ecefVec[0, :],
                                            ecefVec[1, :],
                                            ecefVec[2, :],
                                            direction='INVERSE'))

    # Now, we want to return this as a record array, but convert the alt to km:
    dtype = [('lon', np.float), ('lat', np.float), ('alt', np.float)]
    lla = np.core.records.fromarrays([lla[0], lla[1], lla[2]/1e3],
                                     dtype=dtype)

    return lla
