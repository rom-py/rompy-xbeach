from pathlib import Path
import pytest
import xarray as xr

from rompy_xbeach.source import (
    SourceGeotiff,
    SourceXYZ,
    SourceCRSDataset,
    SourceCRSFile,
    SourceCRSIntake,
    SourceOceantide,
    SourceCRSOceantide
)


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
        catalog_uri=tif_path.parent / "catalog.yaml",
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


def test_source_oceantide():
    source = SourceOceantide(
        reader="read_otis_binary",
        kwargs=dict(
            gfile=HERE / "data/swaus_tide_cons/grid_m2s2n2k2k1o1p1q1mmmf",
            hfile=HERE / "data/swaus_tide_cons/h_m2s2n2k2k1o1p1q1mmmf",
            ufile=HERE / "data/swaus_tide_cons/u_m2s2n2k2k1o1p1q1mmmf",
        )
    )
    assert hasattr(source._open(), "tide")


def test_source_crs_oceantide():
    source = SourceCRSOceantide(
        reader="read_otis_binary",
        kwargs=dict(
            gfile=HERE / "data/swaus_tide_cons/grid_m2s2n2k2k1o1p1q1mmmf",
            hfile=HERE / "data/swaus_tide_cons/h_m2s2n2k2k1o1p1q1mmmf",
            ufile=HERE / "data/swaus_tide_cons/u_m2s2n2k2k1o1p1q1mmmf",
        ),
        crs=4326,
    )
    assert hasattr(source._open(), "tide")
    assert source._open().rio.crs == 4326
