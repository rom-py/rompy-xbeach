from pathlib import Path
import pytest
import numpy as np

from rompy_xbeach.data import XBeachDataGrid, XBeachBathy, SeawardExtensionLinear
from rompy_xbeach.source import SourceGeotiff
from rompy_xbeach.grid import GeoPoint, RegularGrid


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def tif_path():
    yield HERE / "data/bathy.tif"


@pytest.fixture(scope="module")
def source():
    yield SourceGeotiff(filename=HERE / "data/bathy.tif")


def test_geotiff(tif_path):
    source = SourceGeotiff(filename=tif_path)
    dset = source._open()
    assert list(dset.data_vars.keys()) == ["data"]
    assert dset.rio.crs == 4326


def test_geotiff_band_index(tif_path):
    source = SourceGeotiff(filename=tif_path, band=1)
    assert "data" in source._open()
    with pytest.raises(KeyError):
        source = SourceGeotiff(filename=tif_path, band=2)
        source._open()


def test_geotiff_kwargs(tif_path):
    source = SourceGeotiff(filename=tif_path, kwargs={"chunks": {"x": 100, "y": 100}})
    dset = source._open()
    assert dset.data.chunks is not None


def test_geotiff_not_exposed_kwargs(tif_path):
    source = SourceGeotiff(
        filename=tif_path,
        kwargs={"band_as_variable": True, "default_name": "dummy"},
    )
    assert source.kwargs["band_as_variable"] is False
    assert "default_name" not in source.kwargs


def test_xbeach_data_grid(source):
    data = XBeachDataGrid(source=source)
    assert data.model_type == "xbeach_data_grid"
    assert hasattr(data.ds, "rio")


def test_xbeach_data_grid_rio_accessor(source):
    data = XBeachDataGrid(source=source)
    assert hasattr(data.ds, "rio")
    assert hasattr(data.ds.rio, "x_dim")
    assert hasattr(data.ds.rio, "y_dim")


def test_xbeach_bathy_get(source, tmp_path):
    data = XBeachBathy(
        source=source,
        posdwn=False,
        left=5,
        right=5,
    )
    grid = RegularGrid(
        ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347.0,
        dx=10,
        dy=15,
        nx=230,
        ny=220,
        crs="28350",
    )
    xfile, yfile, datafile, grid = data.get(destdir=tmp_path, grid=grid)


def test_xbeach_bathy_extend_seaward_linear(source, tmp_path):
    grid = RegularGrid(
        ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347.0,
        dx=10,
        dy=15,
        nx=230,
        ny=220,
        crs="28350",
    )
    data1 = XBeachBathy(
        source=source,
        posdwn=False,
        left=5,
        right=5,
    )
    XBeachBathy(
        source=source,
        posdwn=False,
        left=5,
        right=5,
        extension=SeawardExtensionLinear(
            depth=25,
            slope=0.05,
        ),
    )
    xfile1, yfile1, datafile1, grid1 = data1.get(destdir=tmp_path, grid=grid)
    xfile1, yfile2, datafile2, grid2 = data1.get(destdir=tmp_path, grid=grid)


def test_xbeach_bathy_fillna(source, tmp_path):
    grid = RegularGrid(
        ori=GeoPoint(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347.0,
        dx=10,
        dy=15,
        nx=230,
        ny=220,
        crs="28350",
    )
    data = XBeachBathy(
        source=source, posdwn=False, left=5, right=5, interpolate_na=False
    )
    xfile, yfile, datafile, grid = data.get(destdir=tmp_path, grid=grid)
    data = np.loadtxt(datafile)
    assert np.isnan(data).any()
    data = XBeachBathy(
        source=source,
        posdwn=False,
        left=5,
        right=5,
        interpolate_na=True,
        interpolate_na_kwargs={"method": "linear"},
    )
    xfile, yfile, datafile, grid = data.get(destdir=tmp_path, grid=grid)
    data = np.loadtxt(datafile)
    assert not np.isnan(data).any()
    # import matplotlib.pyplot as plt
    # import xarray as xr
    # dset = xr.Dataset.xbeach.from_xbeach(datafile, grid)
    # dset.xbeach.plot_model_bathy(grid, posdwn=False)
    # plt.show()
