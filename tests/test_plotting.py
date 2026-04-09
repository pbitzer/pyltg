import matplotlib
matplotlib.use('Agg')

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, QuadMesh

from pyltg.core.baseclass import Ltg


def _make_ltg(n=100):
    """Create a minimal Ltg object with n rows of synthetic data."""
    t0 = np.datetime64('2020-01-01T00:00:00', 'ns')
    times = t0 + np.arange(n).astype('timedelta64[ms]')
    df = pd.DataFrame({
        'time': times,
        'lat': np.random.uniform(30, 35, n),
        'lon': np.random.uniform(-90, -85, n),
        'alt': np.random.uniform(0, 20, n),
    })
    return Ltg(df)


def test_plot_zt_scatter():
    obj = _make_ltg(50)
    val = obj.plot('zt')
    assert isinstance(val, Line2D)
    plt.close('all')


def test_plot_zt_scatter_color_array():
    obj = _make_ltg(50)
    val = obj.plot('zt', color=obj.time.astype('int64'))
    assert isinstance(val, PathCollection)
    plt.close('all')


def test_plot_zt_pcolormesh():
    obj = _make_ltg(2000)
    val = obj.plot('zt', max_pts=1000)
    assert isinstance(val, QuadMesh)
    plt.close('all')


def test_plot_zt_overplot():
    obj = _make_ltg(50)
    fig, ax = plt.subplots()
    val = obj.plot('zt', ax=ax)
    assert val.axes is ax
    plt.close('all')


def test_plot_zt_idx():
    obj = _make_ltg(50)
    idx = np.zeros(50, dtype=bool)
    idx[:10] = True
    val = obj.plot('zt', idx=idx)
    assert isinstance(val, Line2D)
    assert len(val.get_xdata()) == 10
    plt.close('all')


def test_plot_invalid_type():
    obj = _make_ltg(50)
    with pytest.raises(ValueError):
        obj.plot('bad')
    plt.close('all')
