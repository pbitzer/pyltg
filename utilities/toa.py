#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module is for functions related to time-of-arrival (TOA) calculations.

Any spatial locations should be provided in kilometers (including height).

It is largely built around the package `lmfit`.

Examples
---------

Consider a simple array::

    loc_x = np.array([-10, -5, -5, 0, 5, 5, 10])
    loc_y = np.array([0, 10, -10, 0, 10, -10, 0])
    loc_z = np.array([0, 0, 0, 0, 0, 0, 0])

The arrival time for a source at (1, 1, 5) at time zero is then::

    t0 = 0.0
    source_pos = {'t':t0, 'x': 1, 'y':1, 'z':5}
    arrival_times = toa_model(source_pos, loc_x, loc_y, loc_z)

To show how to do the source retrieval, first peturb these times with a
100 nanosecond Gaussianly distributed error to simulate "real" data::

    import numpy as np

    err = 1e-7
    measured_arrival = arrival_times + np.random.normal(0, err, len(arrival_times))

Then, go get the source position for the "real" arrival times::

    import lmfit

    retrieved_source, fit = source_retrieval(measured_arrival, loc_x, loc_y, loc_z, err=err)
    print(lmfit.fit_report(fit))


"""

import numpy as np

import lmfit

SPEED_OF_LIGHT = 2.99792458e5/1.0002  # km/sec, matches that used in LMA algorithm

def toa_model(param, x, y, z, times=None, err=None):
    r"""
    Define a model consistent with lmfit for time of arrival.

    This is written in such a way in can be used in three primary ways::
        1) If times is None, return the arrival times
        2) If err is None (and times is not), return the residual ((arrival time - measured times))
        3) If err is not None and times is not None, divide the residual by the error(s)

    The general time-of-arrival equation is:

    .. math::
        t = t_i + \frac{1}{c} \sqrt{(x-x_i)^2 + (y-y_i)^2 + (z-z_i)^2}

    where :math:`c` is the speed of light, :math:`t_i, x_i, y_i, z_i` is the
    source spacetime location (so, those provided by param) and
    :math:`x, y, z` is the location of the sensor (so, `locs`).

    Parameters
    ----------
    param : dict
        The parameters of the source, with keys 't', 'x', 'y', 'z'.
    x : NumPy array
        A n-element array containing the `x` position of the sensors in km.
    y : NumPy array
        A n-element array containing the `y` position of the sensors in km.
    z : NumPy array
        A n-element array containing the `z` position of the sensors in km.
    times : NumPy array, optional
        A n-element array of arrival times, one for each element in `x`.
        When this is None, the arrival times from the source defined in
        `param` to each in position in `x,y,z` are returned.
        This is handy for simulations.
    err : scalar or n-element array, optional
        The error in arrival times. If None, then
        `times` should be defined. In this case, the residual between the
        measured `times` and the modelled times is returned.

    Returns
    -------
    NumPy array
        The value returned depends on the parameters supplied for `times` and
        `err`

    """

    model = param['t'] + 1/SPEED_OF_LIGHT * np.sqrt(
            (x-param['x'])**2 +
            (y-param['y'])**2 +
            (z-param['z'])**2)

    if times is None:
        return model
    if err is None:
        return (model-times)

    return (model-times)/err


def default_param(guess, minTime, tOffset):
    """
    Get a set of default parameters to be used in `toa_model`.

    The paramters will be constrained::
        t: No minimum, maximum set by `minTime`
        x: [-250, 250]
        y: [-250, 250]
        z: [0, 20]

    Parameters
    ----------
    guess : tuple, list, array
        The 4 element iterable for the guess (t, x, y, z).
    minTime : float
        The maximum value the parameter time can take.
    tOffset : float
        The offset for the times. Usually the nearest millisecond.

    Returns
    -------
    params : lmfit Parameter
        A Parameter class suitable for use with `lmfit`.

    """

    #get the default parameter dict for lmfit
    params = lmfit.Parameters()

    params.add('t', value=guess[0], max=minTime-tOffset)
    params.add('x', value=guess[1], min=-250., max=250.)
    params.add('y', value=guess[2], min=-250., max=250.)
    params.add('z', value=guess[3], min=-1., max=20.)

    return params

def guess_koshak(x, y, z, times):
    """
    Provide a initial guess for the spacetime retreival using the
    methodology in Koshak et al. (2004).

    In Koshak et al. (2004), an analytic solution for a source spacetime
    retrieval is provided, extending the results of
    Koshak and Solakiewicz (1996). It depends on inverting a matrix based
    on the geometry of the network and the arrival times.

    Parameters
    ----------
    x : NumPy array
        The Cartesian x location of the sensors (kilometers).
    y : NumPy array
        The Cartesian y location of the sensors (kilometers)..
    z : NumPy array
        The Cartesian z location of the sensors (kilometers)..
    times : NumPy array
        The arrival times to be used for the retrieval of the source
        spacetime location.

    Returns
    -------
    f : list
        The [time, x, y, z] of the intitial guess of the retrieval.

    """

    # Which sensor is the "reference" sensor? In Koshak 2004, it is assumed
    # to be sensor 1.
    ref_sensor = np.nanargmin(times)

    # All times are shifted to this arrival time:
    _times = times - times[ref_sensor]

    # Get the indices of the other sensors
    sensor_idx = np.arange(len(times))
    not_ref = sensor_idx != ref_sensor

    # Define L_i, as defined "near" eq4 in Koshak 2004
    Lsqr_ref = x[ref_sensor]**2 + y[ref_sensor]**2 + z[ref_sensor]**2

    Lsqr = x[not_ref]**2 + y[not_ref]**2 + z[not_ref]**2


    # These follow eqs 4-5 in Koshak 2004
    g = 0.5*(Lsqr - SPEED_OF_LIGHT**2*_times[not_ref]**2 - Lsqr_ref)

    # Note that we build the transpose of K (see eq 5), since it's easier
    # in code. Since Transpose(Ktrans) = K, this works just fine...
    Ktrans = np.array([x[not_ref]-x[ref_sensor],
                       y[not_ref]-y[ref_sensor],
                       z[not_ref]-z[ref_sensor],
                       -SPEED_OF_LIGHT*_times[not_ref]
                       ])

    # Eq 6 in Koshak 2004 (note "@" is matrix multiplication)
    f = np.linalg.inv(Ktrans@Ktrans.T)@Ktrans@g

    # For consistency with other functions here, reorder and modify
    # to return [t, x, y, z] with t the "absolute" time

    f = [f[3]/SPEED_OF_LIGHT+times[ref_sensor], f[0], f[1], f[2]]

    return f

def min_time_msec(times):
    """
    Take an array of times and find the minimum, to the nearest millisecond.

    Parameters
    ----------
    times : NumPy float array
        The times (in seconds) you wish to find the minimum of.

    Returns
    -------
    tOffset : float
        The minimum of `times`, rounded to the nearest millisecond.

    """
    #simple function to calculate the minimum time to the msec
    tOffset = np.min(np.floor(times*1e3))/1e3

    return tOffset


def initial_guess(times, x=0., y=0., z=5.):
    """
    Format an initial guess for the parameters of a source location.

    When doing TOA, we can have a number-size mismatch for times and spatial
    values: x,y,z are on the order of 100, while times (in seconds
    past midnight) are several orders of magnitude bigger. So, this will
    offset a set of these arrival times so that we have a better size match.

    The intital guess for the time parameter will be 100 us before the
    minimum arrival time.

    Parameters
    ----------
    times : NumPy array
        An n-element array of arrival times.
    x : float, optional
        The initial guess for x. The default is 0..
    y : float, optional
        The initial guess for y. The default is 0..
    z : float, optional
        The initial guess for z. The default is 5..

    Returns
    -------
    guess : tuple
        The (t, x, y, z) of the guess parameters.
    tOffset : float
        The time offset used (i.e., the time subtracted from) the arrival
        `times`.

    """

    # Get the minimum time, to the nearest msec.
    # This will be the amount that we offset the arrival times:
    tOffset = min_time_msec(times)

    #now get the earliest arrival time
    minTime = np.min(times)

    # Default guess is 100us before the min arrival time
    guess = (minTime-tOffset-100e-6, x, y, z)

    return guess, tOffset


def source_retrieval(times, x, y, z, params=None, guess=None,
                     fullResult=True, keys=None, err=1e-7):
    r"""
    Given a set of arrival times and locations, find the source spacetime
    position.

    The source spacetime location is done by minimizing the :math:`\chi^2`:

    .. math::
        \chi^2 = \sum_{i=0}^{n-1}\frac{(t_i-\tau_i)^2}{\sigma_i^2}

    where :math:`t_i` is the measured arrival time at the :math:`i`th sensor
    (i.e., the `times`), :math:`\sigma_i` is the error in the arrival time, and

    .. math::
        \tau_i = \frac{t_m}{c} + \sqrt{(x_i-x_m)^2 + (y_i-y_m)^2 + (z_i-z_m)^2}

    and :math:`t_m, x_m, y_m, z_m` are the model parameters for the source
    spacetime poistion (i.e., :math`\tau_i` is the modelled arrival time).

    Parameters
    ----------
    times : NumPy Aarray
        A n-element array of arrival times, for each sensor location in `x`.
    x : NumPy array
        A n-element array containing the `x` position of the sensors in km.
    y : NumPy array
        A n-element array containing the `y` position of the sensors in km.
    z : NumPy array
        A n-element array containing the `z` position of the sensors in km.
    params : lmfit Parameter, optional
        The parameters to use as the initial guess. If `None`, then
        :func:`default_param` will be used.
    guess : list/tuple/array, optional
        The (t, x, y, z) to be used for the initial guess. If `None`, then
        :func:`initial_guess` will be called with the provided `times`.
    fullResult : bool, optional
        If `True`, return the lmfit MinimizerResult along with the solution. See
        the output of `lmfit.minimize` for more.
    keys : tuple/list, optional
        The keys to extract from the result. If `None`, then the base
         ('t', 'x', 'y', 'z') are extracted.
    err : scalar or NumPy array, optional
        The error in the arrival times. If a scalar, this is used for
        all arrival times. The default is 1e-7.

    Returns
    -------
    NumPy array or tuple
        If `full_result` is True, then a tuple is returned:
        (solution, lmfit.MinimizerResult).

        If `full_result` is False, then just the solution is returned,
        such that `solution = [t, x, y, z]`

    """

    # First, look for bad data:
    idx = ~np.isnan(times)

    times = times[idx]
    x = x[idx]
    y = y[idx]
    z = z[idx]

    if guess is None:
        guess, tOffset = initial_guess(times)

    tOffset = min_time_msec(times)

    if params is None:
        params = default_param(guess, times.min(), tOffset)

    # Do the minimization, but shift the times
    result = lmfit.minimize(toa_model, params,
                            args=(x, y, z,),
                            kws={'times':times-tOffset,
                                 'err':err},
                            xtol=1e-10, gtol=1e-10, ftol=1e-10)

    if keys is None: # which keys do we get extract from result?
        keys = ('t', 'x', 'y', 'z')

    sol = np.array([result.params[_k].value for _k in keys])
    sol[0] += tOffset  # add back the time offset

    if fullResult:
        return sol, result
    else:
        return sol
