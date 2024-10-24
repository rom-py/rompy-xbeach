from pathlib import Path
import pytest
import xarray as xr

from rompy_xbeach.data import XBeachDataGrid, XBeachBathy, SeawardExtensionLinear
from rompy_xbeach.source import SourceGeotiff, SourceXYZ, SourceCRSDataset, SourceCRSFile, SourceCRSIntake
from rompy_xbeach.grid import Ori, RegularGrid


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def tif_path():
    yield HERE / "data/bathy.tif"


@pytest.fixture(scope="module")
def source():
    yield SourceGeotiff(filename=HERE / "data/bathy.tif")


def test_source_xyz():
    source = SourceXYZ(
        filename=HERE / "data/bathy_xyz.zip",
        read_csv_kwargs=dict(sep="\t"),
        xcol="easting",
        ycol="northing",
        zcol="elevation",
        crs=4326,
        res=0.0005,
    )
    self = source
    ds = source._open()
    assert ds.rio.crs == 4326
    assert ds.rio.x_dim == "x"
    assert ds.rio.y_dim == "y"


def test_source_crs_dataset(tif_path):
    ds = xr.open_dataset(tif_path, engine="rasterio", band_as_variable=True)
    ds = ds.rio.reproject("epsg:28350").rename(x="easting", y="northing")
    source = SourceCRSDataset(obj=ds, crs=28350, x_dim="easting", y_dim="northing")
    ds2 = source._open()
    assert ds.rio.crs == 28350
    assert ds.rio.x_dim == "easting"
    assert ds.rio.y_dim == "northing"


def test_source_crs_intake(tif_path):
    source = SourceCRSIntake(
        catalog_uri=tif_path.parent/"catalog.yaml",
        dataset_id="bathy_netcdf",
        crs=4326,
    )
    ds = source._open()
    assert ds.rio.crs == 4326
    assert ds.rio.x_dim == "x"
    assert ds.rio.y_dim == "y"


def test_source_crs_file(tif_path):
    source = SourceCRSFile(uri=tif_path, crs=4326)
    ds = source._open()
    assert ds.rio.crs == 4326
    assert ds.rio.x_dim == "x"
    assert ds.rio.y_dim == "y"


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
        ori=Ori(x=115.594239, y=-32.641104, crs="epsg:4326"),
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
        ori=Ori(x=115.594239, y=-32.641104, crs="epsg:4326"),
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
    data2 = XBeachBathy(
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