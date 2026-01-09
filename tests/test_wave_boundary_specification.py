"""Tests for wave boundary classes in data/boundary/."""

import pytest
from rompy_xbeach.data.boundary import (
    BoundaryStat,
    BoundaryBichrom,
    BoundaryStatTable,
    BoundaryTs1,
    BoundaryTs2,
    BoundaryTsNonh,
    BoundaryOff,
    BoundaryReuse,
)


def test_boundary_stat():
    """Test BoundaryStat for stationary waves."""
    boundary = BoundaryStat(
        Hrms=2.0,
        Trep=12.0,
        dir0=285.0,
        m=10,
    )
    assert boundary.model_type == "stat"
    assert boundary.Hrms == 2.0
    assert boundary.Trep == 12.0
    assert boundary.dir0 == 285.0
    assert boundary.m == 10


def test_boundary_stat_get():
    """Test BoundaryStat.get() returns correct XBeach parameters."""
    boundary = BoundaryStat(
        Hrms=2.0,
        Trep=12.0,
        dir0=270.0,
        m=10,
    )
    params = boundary.get("/tmp")
    assert params["wbctype"] == "stat"
    assert params["Hrms"] == 2.0
    assert params["Trep"] == 12.0
    assert params["dir0"] == 270.0
    assert params["m"] == 10


def test_boundary_bichrom():
    """Test BoundaryBichrom for bichromatic waves."""
    boundary = BoundaryBichrom(
        Hrms=1.5,
        Trep=10.0,
        Tlong=80.0,
        dir0=270.0,
        m=10,
    )
    assert boundary.model_type == "bichrom"
    assert boundary.Hrms == 1.5
    assert boundary.Tlong == 80.0


def test_boundary_bichrom_get():
    """Test BoundaryBichrom.get() returns correct XBeach parameters."""
    boundary = BoundaryBichrom(
        Hrms=1.5,
        Trep=10.0,
        Tlong=80.0,
        dir0=270.0,
        m=10,
    )
    params = boundary.get("/tmp")
    assert params["wbctype"] == "bichrom"
    assert params["Hrms"] == 1.5
    assert params["Tlong"] == 80.0


def test_boundary_ts1():
    """Test BoundaryTs1 for time series at single location."""
    boundary = BoundaryTs1(
        bcfile="bc/gen.ezs",
        Hrms=2.0,
        Trep=12.0,
    )
    assert boundary.model_type == "ts_1"
    assert boundary.bcfile == "bc/gen.ezs"


def test_boundary_ts1_get():
    """Test BoundaryTs1.get() returns correct XBeach parameters."""
    boundary = BoundaryTs1(
        bcfile="bc/gen.ezs",
        Hrms=2.0,
        Trep=12.0,
    )
    params = boundary.get("/tmp")
    assert params["wbctype"] == "ts_1"
    assert params["bcfile"] == "bc/gen.ezs"
    assert params["Hrms"] == 2.0


def test_boundary_ts2():
    """Test BoundaryTs2 for time series at two locations."""
    boundary = BoundaryTs2(
        bcfile="bc/gen.ezs",
    )
    assert boundary.model_type == "ts_2"
    assert boundary.bcfile == "bc/gen.ezs"


def test_boundary_ts_nonh():
    """Test BoundaryTsNonh for non-hydrostatic time series."""
    boundary = BoundaryTsNonh(
        bcfile="Boun_u.bcf",
    )
    assert boundary.model_type == "ts_nonh"
    assert boundary.bcfile == "Boun_u.bcf"


def test_boundary_stat_table():
    """Test BoundaryStatTable for time-varying parametric waves."""
    boundary = BoundaryStatTable(
        bcfile="stat_table.txt",
    )
    assert boundary.model_type == "stat_table"
    assert boundary.bcfile == "stat_table.txt"


def test_boundary_off():
    """Test BoundaryOff for no wave forcing."""
    boundary = BoundaryOff()
    assert boundary.model_type == "off"


def test_boundary_off_get():
    """Test BoundaryOff.get() returns correct XBeach parameters."""
    boundary = BoundaryOff()
    params = boundary.get("/tmp")
    assert params["wbctype"] == "off"
    assert len(params) == 1


def test_boundary_reuse():
    """Test BoundaryReuse without file."""
    boundary = BoundaryReuse()
    assert boundary.model_type == "reuse"
    assert boundary.bcfile is None


def test_boundary_reuse_with_file():
    """Test BoundaryReuse with file."""
    boundary = BoundaryReuse(bcfile="path/to/ebcflist.bcf")
    assert boundary.model_type == "reuse"
    assert boundary.bcfile == "path/to/ebcflist.bcf"


def test_boundary_reuse_get():
    """Test BoundaryReuse.get() returns correct XBeach parameters."""
    boundary = BoundaryReuse()
    params = boundary.get("/tmp")
    assert params["wbctype"] == "reuse"

    boundary_with_file = BoundaryReuse(bcfile="ebcflist.bcf")
    params = boundary_with_file.get("/tmp")
    assert params["wbctype"] == "reuse"
    assert params["bcfile"] == "ebcflist.bcf"


def test_boundary_stat_with_wave_params():
    """Test BoundaryStat with additional wave boundary parameters."""
    boundary = BoundaryStat(
        Hrms=2.0,
        Trep=12.0,
        dir0=270.0,
        m=10,
        nmax=0.8,
        thetamin=-60,
        thetamax=60,
        dtheta=10,
    )
    params = boundary.get("/tmp")
    assert params["wbctype"] == "stat"
    assert params["Hrms"] == 2.0
    assert params["nmax"] == 0.8
    assert params["thetamin"] == -60
    assert params["thetamax"] == 60
    assert params["dtheta"] == 10
