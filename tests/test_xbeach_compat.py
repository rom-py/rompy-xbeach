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
        ("params", "params"),
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


def test_boundary_reuse_copies_referenced_series(tmp_path):
    """Reuse copies the list files and every series file they reference."""
    from rompy_xbeach.data.boundary import BoundaryReuse
    from rompy_xbeach.types import XBeachDirectoryBlob

    source = tmp_path / "previous"
    source.mkdir()
    row = "2160.000 2160.000 1.000 12.03 0.785 2.01 {}"
    (source / "ebcflist.bcf").write_text(row.format("E_series00001.bcf"))
    (source / "qbcflist.bcf").write_text(row.format("q_series00001.bcf"))
    (source / "esbcflist.bcf").write_text(row.format("Es_series00001.bcf"))
    for name in ["E_series00001.bcf", "q_series00001.bcf", "Es_series00001.bcf"]:
        (source / name).write_text("data")

    destdir = tmp_path / "run"
    BoundaryReuse(previous_run=XBeachDirectoryBlob(source=str(source))).get(destdir)
    assert sorted(p.name for p in destdir.iterdir()) == sorted(
        p.name for p in source.iterdir()
    )


def test_boundary_reuse_missing_series(tmp_path):
    """A clear error is raised when a referenced series file is missing."""
    from rompy_xbeach.data.boundary import BoundaryReuse
    from rompy_xbeach.types import XBeachDirectoryBlob

    source = tmp_path / "previous"
    source.mkdir()
    row = "2160.000 2160.000 1.000 12.03 0.785 2.01 {}"
    (source / "ebcflist.bcf").write_text(row.format("E_series00001.bcf"))
    (source / "qbcflist.bcf").write_text(row.format("q_series00001.bcf"))
    (source / "E_series00001.bcf").write_text("data")

    boundary = BoundaryReuse(previous_run=XBeachDirectoryBlob(source=str(source)))
    with pytest.raises(FileNotFoundError, match="q_series00001.bcf"):
        boundary.get(tmp_path / "run")


def test_boundary_params_constant_and_bichromatic():
    """Tlong is only written when set, which makes XBeach use bichromatic waves."""
    from rompy_xbeach.data.boundary import BoundaryParams

    constant = BoundaryParams(Hrms=1.0, Trep=10.0).get(destdir=None)
    assert constant["wbctype"] == "params"
    assert "Tlong" not in constant

    bichromatic = BoundaryParams(Hrms=1.0, Trep=10.0, Tlong=80.0).get(destdir=None)
    assert bichromatic["wbctype"] == "params"
    assert bichromatic["Tlong"] == 80.0
