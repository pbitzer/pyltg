# -*- coding: utf-8 -*-
"""
Tests for pyltg.core.nldn.NLDN

All tests use synthetic data written to tmp_path — no external data files needed.
"""

from pathlib import Path

import numpy as np
import pytest

from pyltg.core.nldn import NLDN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nldn_row(date='05/27/2024', time='12:00:00.000000',
              lat=34.7, lon=-86.6, current=-15.3,
              multi=0, semimajor=0.5, semiminor=0.3,
              axis_ratio=1.67, azimuth=45.0, chisq=1.2,
              num_sensors=8, pulse_type='G'):
    """Return one NLDN ASCII row string (no trailing newline)."""
    # Format matches the colNames tuple in NLDN.readFile:
    # date time lat lon current _kA _multi semimajor semiminor
    #     axis_ratio azimuth chisq num_sensors type
    # _kA is a duplicate of current (Vaisala quirk), included but dropped on read.
    return (f"{date} {time} {lat:.4f} {lon:.4f} "
            f"{current:.1f} {current:.1f} {multi} "
            f"{semimajor:.3f} {semiminor:.3f} {axis_ratio:.3f} "
            f"{azimuth:.1f} {chisq:.3f} {num_sensors} {pulse_type}")


def _write_nldn_file(path: Path, rows: list) -> Path:
    """Write a list of row strings to *path* and return it."""
    path.write_text("\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROW_A = _nldn_row(lat=34.7, lon=-86.6, current=-15.3, pulse_type='G')
ROW_B = _nldn_row(lat=35.1, lon=-87.0, current=10.2,
                  time='13:30:00.000000', pulse_type='C')
ROW_C = _nldn_row(lat=33.5, lon=-85.9, current=-22.0,
                  time='14:15:00.500000', pulse_type='G')


# ---------------------------------------------------------------------------
# Tests: single-file path
# ---------------------------------------------------------------------------

class TestSingleFile:

    def test_string_input(self, tmp_path):
        """Single file path as a plain string."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A, ROW_B])
        n = NLDN(str(f))
        assert n.count == 2

    def test_list_of_one_input(self, tmp_path):
        """Single file path wrapped in a list."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A, ROW_B])
        n = NLDN([str(f)])
        assert n.count == 2

    def test_pathlib_input(self, tmp_path):
        """Single file path as a Path object."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A, ROW_B])
        n = NLDN(f)
        assert n.count == 2

    def test_columns_present(self, tmp_path):
        """Required Ltg columns must be present after read."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A])
        n = NLDN(f)
        for col in ('time', 'lat', 'lon', 'alt'):
            assert col in n.columns, f"Missing required column: {col}"

    def test_time_is_datetime64(self, tmp_path):
        """time column should be datetime64 (not raw string)."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A])
        n = NLDN(f)
        assert np.issubdtype(n.time.dtype, np.datetime64)

    def test_lat_lon_values(self, tmp_path):
        """lat/lon values should round-trip through the ASCII format."""
        f = _write_nldn_file(tmp_path / 'nldn_a.txt', [ROW_A])
        n = NLDN(f)
        assert abs(n.lat[0] - 34.7) < 1e-3
        assert abs(n.lon[0] - -86.6) < 1e-3


# ---------------------------------------------------------------------------
# Tests: multiple-file path
# ---------------------------------------------------------------------------

class TestMultipleFiles:

    def test_two_files_concatenate(self, tmp_path):
        """Row count from two distinct files equals the sum."""
        f1 = _write_nldn_file(tmp_path / 'nldn_1.txt', [ROW_A, ROW_B])
        f2 = _write_nldn_file(tmp_path / 'nldn_2.txt', [ROW_C])
        n = NLDN([f1, f2])
        assert n.count == 3

    def test_two_files_list_of_strings(self, tmp_path):
        """List of plain strings (not Path objects) is accepted."""
        f1 = _write_nldn_file(tmp_path / 'nldn_1.txt', [ROW_A])
        f2 = _write_nldn_file(tmp_path / 'nldn_2.txt', [ROW_B])
        n = NLDN([str(f1), str(f2)])
        assert n.count == 2

    def test_cross_file_dedup(self, tmp_path):
        """Identical rows in two different files collapse to one row."""
        f1 = _write_nldn_file(tmp_path / 'nldn_1.txt', [ROW_A])
        f2 = _write_nldn_file(tmp_path / 'nldn_2.txt', [ROW_A])  # exact duplicate
        n = NLDN([f1, f2])
        assert n.count == 1

    def test_within_file_dedup_preserved(self, tmp_path):
        """Within-file dedup (existing behaviour) still works with multi-file path."""
        # Two identical rows in one file, one unique row in another
        f1 = _write_nldn_file(tmp_path / 'nldn_1.txt', [ROW_A, ROW_A])
        f2 = _write_nldn_file(tmp_path / 'nldn_2.txt', [ROW_B])
        n = NLDN([f1, f2])
        assert n.count == 2  # ROW_A deduped to 1, plus ROW_B

    def test_three_files(self, tmp_path):
        """More than two files work."""
        f1 = _write_nldn_file(tmp_path / 'nldn_1.txt', [ROW_A])
        f2 = _write_nldn_file(tmp_path / 'nldn_2.txt', [ROW_B])
        f3 = _write_nldn_file(tmp_path / 'nldn_3.txt', [ROW_C])
        n = NLDN([f1, f2, f3])
        assert n.count == 3
