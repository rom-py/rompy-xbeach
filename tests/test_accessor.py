from pathlib import Path
import pytest
import xarray as xr

from rompy_xbeach.source import SourceGeotiff
from rompy_xbeach.data import XBeachBathy
from rompy_xbeach.grid import RegularGrid, GeoPoint

HERE = Path(__file__).parent


@pytest.fixture
def grid():
    grid = RegularGrid(
        ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347,
        dx=10,
        dy=15,
        nx=230,
        ny=220,
        crs="28350",
    )
    yield grid


@pytest.fixture
def xbeach_bathy_files(tmp_path, grid):
    source = SourceGeotiff(filename=HERE / "data/bathy.tif")
    data = XBeachBathy(source=source, posdwn=False)
    yield data.get(destdir=tmp_path, grid=grid)


def test_data(xbeach_bathy_files):
    xfile, yfile, depfile, grid = xbeach_bathy_files
    dset = xr.Dataset.xbeach.from_xbeach(depfile, grid)
    assert "xc" in dset.coords
    assert "yc" in dset.coords
    assert "dep" in dset.data_vars
    assert hasattr(dset, "rio")
    assert hasattr(dset, "xbeach")
