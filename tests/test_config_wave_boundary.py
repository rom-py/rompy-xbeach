"""Tests for wave boundary classes in data/boundary/."""

import pytest
from pathlib import Path
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.grid import RegularGrid, GeoPoint
from rompy_xbeach.data.bathy import XBeachBathy
from rompy_xbeach.source import SourceGeotiff
from rompy_xbeach.data.boundary import (
    BoundaryStat,
    BoundaryBichrom,
    BoundaryOff,
    BoundaryReuse,
)
from rompy_xbeach.components.physics import Physics


HERE = Path(__file__).parent


@pytest.fixture
def grid():
    return RegularGrid(
        ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347.0,
        dx=10,
        dy=10,
        nx=100,
        ny=50,
    )


@pytest.fixture
def bathy():
    return XBeachBathy(
        source=SourceGeotiff(
            filename=str(HERE / "data" / "bathy.tif"),
        ),
    )


def test_boundary_stat_get():
    """Test BoundaryStat.get() returns correct parameters."""
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


def test_boundary_bichrom_get():
    """Test BoundaryBichrom.get() returns correct parameters."""
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


def test_boundary_off_get():
    """Test BoundaryOff.get() returns correct parameters."""
    boundary = BoundaryOff()
    params = boundary.get("/tmp")
    assert params["wbctype"] == "off"
    assert len(params) == 1


def test_boundary_reuse_get():
    """Test BoundaryReuse.get() returns correct parameters."""
    boundary = BoundaryReuse()
    params = boundary.get("/tmp")
    assert params["wbctype"] == "reuse"

    boundary_with_file = BoundaryReuse(bcfile="ebcflist.bcf")
    params = boundary_with_file.get("/tmp")
    assert params["wbctype"] == "reuse"
    assert params["bcfile"] == "ebcflist.bcf"


def test_config_input_optional():
    """Test that input field is optional in Config."""
    from pydantic import ValidationError

    # input should be optional
    try:
        Config()
    except ValidationError as e:
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        # Should fail on grid, bathy (required), but NOT on input (optional)
        assert "input" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields


def test_warn_wave_direction_params_without_swave(grid, bathy, caplog):
    """Test that a warning is logged when wave direction params are set but swave=False."""
    import logging

    caplog.set_level(logging.WARNING)

    Config(
        grid=grid,
        bathy=bathy,
        physics=Physics(swave=False),
        input=DataInterface(
            wave=BoundaryStat(
                Hrms=2.0,
                Trep=12.0,
                thetamin=-60,
                thetamax=60,
                dtheta=10,
            ),
        ),
    )

    # Check that warning was logged
    assert any(
        "Wave directional parameters" in record.message and "swave=0" in record.message
        for record in caplog.records
    )


def test_no_warn_wave_direction_params_with_swave(grid, bathy, caplog):
    """Test that no warning is logged when swave is enabled (default)."""
    import logging

    caplog.set_level(logging.WARNING)

    Config(
        grid=grid,
        bathy=bathy,
        input=DataInterface(
            wave=BoundaryStat(
                Hrms=2.0,
                Trep=12.0,
                thetamin=-60,
                thetamax=60,
                dtheta=10,
            ),
        ),
    )

    # Check that no wave direction warning was logged
    assert not any(
        "Wave directional parameters" in record.message for record in caplog.records
    )
