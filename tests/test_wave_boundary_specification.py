"""Tests for wave boundary classes in data/boundary/."""

import pytest
from pathlib import Path
from rompy_xbeach.types import XBeachDataBlob
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


HERE = Path(__file__).parent


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


def test_boundary_ts1(tmp_path):
    """Test BoundaryTs1 for time series at single location."""
    # Create a test file
    test_file = tmp_path / "gen.ezs"
    test_file.write_text("test content")
    
    boundary = BoundaryTs1(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_1"
    assert boundary.id == "ts_1"


def test_boundary_ts1_get(tmp_path):
    """Test BoundaryTs1.get() returns correct XBeach parameters."""
    # Create a test file
    source_file = tmp_path / "source" / "gen.ezs"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("test content")
    
    boundary = BoundaryTs1(
        source=XBeachDataBlob(source=source_file),
    )
    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)
    assert params["wbctype"] == "ts_1"
    assert params["bcfile"] == "bc/gen.ezs"
    # Verify file was copied to bc/ subdirectory
    assert (destdir / "bc" / "gen.ezs").exists()


def test_boundary_ts2(tmp_path):
    """Test BoundaryTs2 for time series at two locations."""
    # Create a test file
    test_file = tmp_path / "gen.ezs"
    test_file.write_text("test content")
    
    boundary = BoundaryTs2(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_2"
    assert boundary.id == "ts_2"


def test_boundary_ts_nonh(tmp_path):
    """Test BoundaryTsNonh for non-hydrostatic time series."""
    # Create a test file
    test_file = tmp_path / "Boun_u.bcf"
    test_file.write_text("test content")
    
    boundary = BoundaryTsNonh(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_nonh"
    assert boundary.id == "ts_nonh"


def test_boundary_stat_table(tmp_path):
    """Test BoundaryStatTable for time-varying parametric waves."""
    # Create a test file
    test_file = tmp_path / "stat_table.txt"
    test_file.write_text("test content")
    
    boundary = BoundaryStatTable(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_stat_table"
    assert boundary.id == "stat_table"


def test_boundary_stat_table_get(tmp_path):
    """Test BoundaryStatTable.get() returns correct XBeach parameters."""
    # Create a test file
    source_file = tmp_path / "source" / "stat_table.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("test content")
    
    boundary = BoundaryStatTable(
        source=XBeachDataBlob(source=source_file),
    )
    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)
    assert params["wbctype"] == "stat_table"
    assert params["bcfile"] == "stat_table.txt"
    # Verify file was copied
    assert (destdir / "stat_table.txt").exists()


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
