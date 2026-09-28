"""Checks that wave boundary types are only used with the wave models XBeach allows."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Nonh, Stationary, Surfbeat
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.bathy import XBeachBathy
from rompy_xbeach.data.boundary import (
    BoundaryFileSwan,
    BoundaryOff,
    BoundaryParams,
    BoundaryReuse,
    BoundaryTs1,
    BoundaryTsNonh,
)
from rompy_xbeach.grid import GeoPoint, RegularGrid
from rompy_xbeach.source import SourceGeotiff
from rompy_xbeach.types import XBeachDataBlob, XBeachDirectoryBlob

HERE = Path(__file__).parent

GRID = RegularGrid(
    ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
    alfa=347.0,
    dx=10,
    dy=10,
    nx=100,
    ny=50,
)
BATHY = XBeachBathy(source=SourceGeotiff(filename=str(HERE / "data" / "bathy.tif")))
WAVEMODELS = {"stationary": Stationary(), "surfbeat": Surfbeat(), "nonh": Nonh()}


@pytest.fixture
def boundaries(tmp_path):
    """One wave boundary of each kind, keyed by a short name."""
    bcfile = tmp_path / "bcfile.txt"
    bcfile.write_text("data")
    return {
        "params": BoundaryParams(Hrms=1.0, Trep=10.0),
        "bichromatic": BoundaryParams(Hrms=1.0, Trep=10.0, Tlong=80.0),
        "off": BoundaryOff(),
        "swan": BoundaryFileSwan(bcfile_source=XBeachDataBlob(source=bcfile)),
        "reuse": BoundaryReuse(previous_run=XBeachDirectoryBlob(source=str(tmp_path))),
        "ts_1": BoundaryTs1(source=XBeachDataBlob(source=bcfile)),
        "ts_nonh": BoundaryTsNonh(source=XBeachDataBlob(source=bcfile)),
    }


ALLOWED = {
    "params": {"stationary", "surfbeat", "nonh"},
    "bichromatic": {"surfbeat"},
    "off": {"stationary", "surfbeat", "nonh"},
    "swan": {"surfbeat", "nonh"},
    "reuse": {"surfbeat", "nonh"},
    "ts_1": {"surfbeat"},
    "ts_nonh": {"nonh"},
}


@pytest.mark.parametrize("wavemodel", list(WAVEMODELS))
@pytest.mark.parametrize("boundary", list(ALLOWED))
def test_wave_boundary_for_wavemodel(boundaries, boundary, wavemodel):
    """Allowed combinations validate, others raise a clear error."""
    kwargs = dict(
        grid=GRID,
        bathy=BATHY,
        physics=Physics(wavemodel=WAVEMODELS[wavemodel], swave=wavemodel != "nonh"),
        input=DataInterface(wave=boundaries[boundary]),
    )
    if wavemodel in ALLOWED[boundary]:
        Config(**kwargs)
    else:
        with pytest.raises(
            ValidationError, match=f"cannot be used with the {wavemodel}"
        ):
            Config(**kwargs)
