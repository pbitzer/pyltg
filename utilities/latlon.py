import numpy as np
import pyproj


def rot_matrix_ecef2enu(lam, phi):
    # Define the rotation matrix to go from ECEF coordinates to ENU.
    # lam = "lon"
    # phi = "lat"
    sLam = np.sin(lam)
    cLam = np.cos(lam)
    sPhi = np.sin(phi)
    cPhi = np.cos(phi)
    
    rotMatrix = np.array( [[-sLam,      cLam,       0], 
                          [-sPhi*cLam,  -sPhi*sLam, cPhi], 
                          [cPhi*cLam,   cPhi*sLam,  sPhi]] )
    
    return rotMatrix


def lla2enu(lats, lons, alts, center=None):
    """
    Convert lat, lon, alt to ENU coordinates.

    If no center is given, use the average lat/lon/alt of input

    Parameters
    ----------
    lats
    lons
    alts
    center

    Returns
    -------

    """

    # Start by defining the projections for the conversion:
    ecefProj = pyproj.Proj(proj='geocent',  ellps='WGS84', datum='WGS84')
    llaProj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    # First, convert to ECEF
    ecef = np.array(pyproj.transform(llaProj, ecefProj, lons, lats, alts))

    # Next, convert the center:
    if center is None:
        center = (np.mean(lats), np.mean(lons), np.mean(alts))
        
    centerEcef = np.array(pyproj.transform(llaProj, ecefProj, center[1], center[0], center[2]))

    # Now, we convert ECEF to ENU...
    
    # Start by finding the vector pointing from the origin to the point(s of interest)
    vec = (ecef.transpose() - centerEcef).transpose()
    
    # Now, we construct the rotation matrix at the origin:
    lam = np.radians(center[1])
    phi = np.radians(center[0])
    
    # Get the rotation matrix:    
    rotMatrix = rot_matrix_ecef2enu(lam, phi)
    
    # Finally, transform ECEF to ENU
    enu = rotMatrix.dot(vec)/1e3 #return in km
    
    # Return as an numpy record array
    return np.core.records.fromarrays(enu, names='x,y,z')
    

def enu2lla(x, y, z=None, center=None):
    """
    Convert from ENU coordinates to lat/lon/alt.

    Center is lat/lon/alt array like
    x, y, z in km

    Parameters
    ----------
    x
    y
    z
    center

    Returns
    -------

    """

    # Start by defining the projections for the conversion:
    ecefProj = pyproj.Proj(proj='geocent',  ellps='WGS84', datum='WGS84')
    llaProj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Take the center point and convert to ECEF (in km, please)
    centerEcef = np.array(pyproj.transform(llaProj, ecefProj, center[1], center[0], center[2]))

    # Now, convert the ENU coordinates to a "delta" ECEF. This is the offset
    # (in ECEF coordinates) of the points from the center. To do so, get the
    # the rotation matrix, and apply the inverse 
    # (since it's a rotation matrix, the inverse is the transpose).

    rotMatrix = rot_matrix_ecef2enu(np.radians(center[1]), np.radians(center[0])).T    
    ecefDelta = rotMatrix.dot([x, y, z])*1e3
    
    # Now, translate the vector of points to the ECEF of the center:
    ecefVec = (ecefDelta.transpose() + centerEcef).transpose()
    
    # Convert these to LLA:    
    lla = np.array(pyproj.transform(ecefProj, llaProj, ecefVec[0, :], ecefVec[1, :], ecefVec[2, :]))
    
    # Now, we want to return this as a record array:
    dtype = [('lon', np.float), ('lat', np.float), ('alt', np.float)]
    lla = np.core.records.fromarrays([lla[0], lla[1], lla[2]],
                                     dtype=dtype)
        
    return lla 