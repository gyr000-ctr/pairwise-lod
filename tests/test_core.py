import math

import pytest

import lod

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _l():
    return lod.LodSelector(4, 1.0)


def _check_lod_near():
    return _l().select(1.0, 1.0) == 0


def _check_lod_far():
    return _l().select(1000.0, 1.0) == 3


def _check_lod_same():
    return _l().select(5.0, 1.0) == _l().select(5.0, 1.0)


def _check_lod_bad_threshold():
    return _raises(lambda: lod.LodSelector(4, 0.0))


def _check_lod_bad_levels():
    return _raises(lambda: lod.LodSelector(0, 1.0))


def _check_lod_zero_radius():
    return _l().select(10.0, 0.0) == 3


def _check_lod_monotonic():
    s = _l()
    return s.select(1.0, 1.0) <= s.select(10.0, 1.0) <= s.select(100.0, 1.0)


def _check_lod_bounds():
    return 0 <= _l().select(0.0, 1.0) <= 3


def _check_lod_det():
    return _l().select(7.0, 2.0) == _l().select(7.0, 2.0)


def _check_lod_type():
    return isinstance(_l().select(1.0, 1.0), int)


def _check_lod_negative():
    return _raises(lambda: _l().select(-1.0, 1.0))


def _check_lod_max():
    return _l().select(1e9, 1.0) == 3


def _check_lod_min():
    return _l().select(0.0, 1.0) == 0


def _check_lod_ratio():
    s = _l()
    return s.select(4.0, 1.0) >= s.select(2.0, 1.0)


def _check_lod_no_mutation():
    s = _l()
    s.select(3.0, 1.0)
    return s.select(3.0, 1.0) == s.select(3.0, 1.0)

CASES = [("lod_near", "_check_lod_near"), ("lod_far", "_check_lod_far"), ("lod_same", "_check_lod_same"), ("lod_bad_threshold", "_check_lod_bad_threshold"), ("lod_bad_levels", "_check_lod_bad_levels"), ("lod_zero_radius", "_check_lod_zero_radius"), ("lod_monotonic", "_check_lod_monotonic"), ("lod_bounds", "_check_lod_bounds"), ("lod_det", "_check_lod_det"), ("lod_type", "_check_lod_type"), ("lod_negative", "_check_lod_negative"), ("lod_max", "_check_lod_max"), ("lod_min", "_check_lod_min"), ("lod_ratio", "_check_lod_ratio"), ("lod_no_mutation", "_check_lod_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
