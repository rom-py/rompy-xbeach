"""Checks that generated parameters use values XBeach accepts."""

import pytest

from rompy_xbeach.data.boundary import BoundaryOff
from rompy_xbeach.data.boundary.base import WBCTYPE_FROM_ID
from rompy_xbeach.grid import GeoPoint, RegularGrid

# Values accepted by XBeach for wbctype (setallowednames in params.F90)
XBEACH_WBCTYPES = {
    "params",
    "parametric",
    "swan",
    "vardens",
    "off",
    "jonstable",
    "reuse",
    "ts_1",
    "ts_2",
    "ts_nonh",
}


@pytest.mark.parametrize(
    "boundary_id,expected",
    [
        ("stat", "params"),
        ("bichrom", "params"),
        ("jons", "parametric"),
        ("parametric", "parametric"),
        ("jonstable", "jonstable"),
        ("swan", "swan"),
        ("off", "off"),
        ("reuse", "reuse"),
        ("ts_1", "ts_1"),
        ("ts_2", "ts_2"),
        ("ts_nonh", "ts_nonh"),
    ],
)
def test_wbctype_from_id(boundary_id, expected):
    """Boundary ids are written with the wbctype names XBeach accepts."""
    wbctype = WBCTYPE_FROM_ID.get(boundary_id, boundary_id)
    assert wbctype == expected
    assert wbctype in XBEACH_WBCTYPES


def test_boundary_off_writes_wave_params():
    """BoundaryOff writes the wave boundary params it accepts, e.g. the direction grid."""
    params = BoundaryOff(thetamin=-90, thetamax=90, dtheta=10).get(destdir=None)
    assert params == {
        "wbctype": "off",
        "thetamin": -90.0,
        "thetamax": 90.0,
        "dtheta": 10.0,
    }


def test_regular_grid_sets_vardx_zero():
    """The regular grid is defined by spacing, so XBeach must not expect xfile/yfile."""
    grid = RegularGrid(
        ori=GeoPoint(x=0, y=0, crs=28350), alfa=0, dx=10, dy=10, nx=5, ny=5, crs=28350
    )
    assert grid.params["vardx"] == 0
