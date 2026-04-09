import matplotlib
matplotlib.use('Agg')

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, QuadMesh
from pathlib import Path

from pyltg.core.baseclass import Ltg
from pyltg.core.lma import LMA
from pyltg.core.hamma_src import HAMMA
from pyltg.core.entln import ENTLN
from pyltg.core.nldn import NLDN

INTERNAL_DATA = Path(__file__).parent.parent / 'pyltg' / 'internal_testing_files'
PRIVATE_DATA = Path(__file__).parent.parent / 'private_data_for_tests'


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


def test_plot_ll_scatter():
    obj = _make_ltg(50)
    val = obj.plot('ll')
    assert isinstance(val, (Line2D, PathCollection))
    plt.close('all')


def test_plot_ll_pcolormesh():
    obj = _make_ltg(2000)
    val = obj.plot('ll', max_pts=1000)
    assert isinstance(val, QuadMesh)
    plt.close('all')


def test_plot_ll_nogrid():
    """Verify nogrid=True doesn't error."""
    obj = _make_ltg(50)
    val = obj.plot('ll', nogrid=True)
    assert isinstance(val, Line2D)
    plt.close('all')


def test_plot_ll_overplot():
    obj = _make_ltg(50)
    val1 = obj.plot('ll', nogrid=True)
    val2 = obj.plot('ll', ax=val1.axes, nogrid=True, color='red')
    assert val2.axes is val1.axes
    plt.close('all')


def test_plot_ll_no_cartopy(monkeypatch):
    """Simulate missing cartopy and verify fallback."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if 'cartopy' in name:
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)

    obj = _make_ltg(50)
    with pytest.warns(UserWarning, match="Cartopy not available"):
        val = obj.plot('ll')
    assert isinstance(val, Line2D)
    plt.close('all')


def test_lma_quick_plot_zt():
    f = INTERNAL_DATA / 'nalma_lylout_20170427_07_3600.dat.gz'
    lma = LMA(str(f))
    lma.limit(lat=[34, 35])
    val = lma.quick_plot('zt')
    assert val is not None
    plt.close('all')


def test_lma_quick_plot_ll():
    f = INTERNAL_DATA / 'nalma_lylout_20170427_07_3600.dat.gz'
    lma = LMA(str(f))
    lma.limit(lat=[34, 35])
    val = lma.quick_plot('ll')
    assert val is not None
    plt.close('all')


def test_hamma_quick_plot_zt():
    f = INTERNAL_DATA / '2018-11-10T19-47-49-776_1001_9.txt'
    h = HAMMA(str(f))
    val = h.quick_plot('zt')
    assert val is not None
    plt.close('all')


def test_entln_quick_plot_ll():
    f = PRIVATE_DATA / 'LtgFlashPortions20180403_mini.csv'
    if not f.exists():
        pytest.skip('Private test data not available')
    eni = ENTLN(str(f))
    eni.limit(lat=[0, 5], lon=[-75, -70])
    val = eni.quick_plot('ll')
    assert val is not None
    plt.close('all')


def test_nldn_quick_plot_ll():
    f = PRIVATE_DATA / 'vaistrokertns_20190824_daily_v1_lit_mini.raw'
    if not f.exists():
        pytest.skip('Private test data not available')
    n = NLDN(str(f))
    n.limit(lat=[34.5, 35], lon=[-95, -85])
    val = n.quick_plot('ll')
    assert val is not None
    plt.close('all')


def test_plot_overplot_scatter_color_array():
    """Overplotting with array color should not crash (scalex/scaley bug)."""
    obj = _make_ltg(50)
    fig, ax = plt.subplots()
    val = obj.plot('zt', ax=ax, color=obj.time.astype('int64'))
    assert isinstance(val, PathCollection)
    assert val.axes is ax
    plt.close('all')


def test_plot_overplot_pcolormesh():
    """Overplotting with pcolormesh should not crash (scalex/scaley bug)."""
    obj = _make_ltg(2000)
    fig, ax = plt.subplots()
    val = obj.plot('zt', ax=ax, max_pts=1000)
    assert isinstance(val, QuadMesh)
    plt.close('all')


def test_plot_idx_with_color_array():
    """idx + array color should subset both data and color."""
    obj = _make_ltg(50)
    idx = np.zeros(50, dtype=bool)
    idx[:10] = True
    colors = obj.time.astype('int64')
    val = obj.plot('zt', color=colors, idx=idx)
    assert isinstance(val, PathCollection)
    plt.close('all')


def test_plot_empty_data():
    """Plotting with no active data should raise ValueError."""
    obj = _make_ltg(50)
    obj.limit(lat=[99, 100])  # no data in this range
    with pytest.raises(ValueError, match="No active data"):
        obj.plot('zt')
    plt.close('all')
