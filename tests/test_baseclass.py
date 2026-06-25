# -*- coding: utf-8 -*-
"""
Tests for pyltg.core.baseclass.Ltg
"""

from pyltg.core.baseclass import Ltg


def test_len_empty():
    """An empty/uninitialized Ltg has no records, so len() should be 0.

    Regression test for #2: previously raised AttributeError because the
    empty internal DataFrame has no 'active' column.
    """
    assert len(Ltg()) == 0
