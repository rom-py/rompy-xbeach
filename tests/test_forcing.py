import pytest
from pathlib import Path
import numpy as np

from rompy.core.time import TimeRange
from rompy.core.source import SourceTimeseriesCSV, SourceTimeseriesDataFrame
from rompy_xbeach.source import (
    SourceCRSFile,
    SourceCRSOceantide,
    SourceTideConsPointCSV,
)
from rompy_xbeach.grid import RegularGrid

from rompy_xbeach.components.forcing import Wind, WindFile
from rompy_xbeach.forcing import (
    WindGrid,
    WindStation,
    WindPoint,
    WindVector,
    WindScalar,
    TideConsGrid,
    TideConsPoint,
    WaterLevelGrid,
    WaterLevelStation,
    WaterLevelPoint,
)


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def time():
    yield TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h")


@pytest.fixture(scope="module")
def grid():
    yield RegularGrid(
        ori=dict(x=115.594239, y=-32.641104, crs="epsg:4326"),
        alfa=347.0,
        dx=10,
        dy=15,
        nx=230,
        ny=220,
        crs="28350",
    )


@pytest.fixture(scope="module")
def source_wind_grid():
    yield SourceCRSFile(
        uri=HERE / "data/era5-20230101.nc",
        kwargs=dict(engine="netcdf4"),
        crs=4326,
        x_dim="longitude",
        y_dim="latitude",
    )


@pytest.fixture(scope="module")
def source_wind_station():
    yield SourceCRSFile(
        uri=HERE / "data/smc-params-20230101.nc",
        crs=4326,
        x_dim="longitude",
        y_dim="latitude",
    )


@pytest.fixture(scope="module")
def source_wind_timeseries():
    yield SourceTimeseriesCSV(filename=HERE / "data/wind.csv", tcol="time")


@pytest.fixture(scope="module")
def source_tide_grid():
    yield SourceCRSOceantide(
        reader="read_otis_binary",
        kwargs=dict(
            gfile=HERE / "data/swaus_tide_cons/grid_m2s2n2k2k1o1p1q1mmmf",
            hfile=HERE / "data/swaus_tide_cons/h_m2s2n2k2k1o1p1q1mmmf",
            ufile=HERE / "data/swaus_tide_cons/u_m2s2n2k2k1o1p1q1mmmf",
        ),
        crs=4326,
        x_dim="lon",
        y_dim="lat",
    )


@pytest.fixture(scope="module")
def source_tide_point():
    yield SourceTideConsPointCSV(filename=HERE / "data/tide_cons_station.csv")


@pytest.fixture(scope="module")
def source_water_level_grid():
    yield SourceCRSFile(
        uri=HERE / "data/ssh_gridded.nc",
        kwargs=dict(engine="netcdf4"),
        crs=4326,
        x_dim="lon",
        y_dim="lat",
    )


@pytest.fixture(scope="module")
def source_water_level_station():
    yield SourceCRSFile(
        uri=HERE / "data/ssh_stations.nc",
        crs=4326,
        x_dim="lon",
        y_dim="lat",
    )


@pytest.fixture(scope="module")
def source_water_level_timeseries():
    yield SourceTimeseriesCSV(filename=HERE / "data/ssh.csv", tcol="time")


# =====================================================================================
# Components
# =====================================================================================
def test_wind_constant_component():
    wind = Wind(windv=10, windth=180)
    assert wind.write() == {"windv": 10, "windth": 180}


def test_wind_file_component(tmp_path):
    wind = WindFile(
        filename="wind.txt",
        tsec=[0, 1, 2, 3],
        windv=[10, 10, 10, 10],
        windth=[180, 180, 180, 180],
    )
    windfile = wind.write(tmp_path)
    assert windfile.is_file()
    data = np.loadtxt(windfile)
    assert data[:, 0].tolist() == wind.tsec
    assert data[:, 1].tolist() == wind.windv
    assert data[:, 2].tolist() == wind.windth


def test_wind_file_component_same_sizes():
    with pytest.raises(ValueError):
        WindFile(
            filename="wind.txt",
            tsec=[0, 1, 2, 3],
            windv=[10, 10, 10, 10],
            windth=[180, 180, 180],
        )


# =====================================================================================
# Data objects
# =====================================================================================
def test_wind_vector():
    wind_vars = WindVector(
        u="u10",
        v="v10",
    )
    assert wind_vars.model_type == "wind_vector"
    assert wind_vars.u == "u10"
    assert wind_vars.v == "v10"


def test_wind_scalar():
    wind_vars = WindScalar(
        spd="ws10",
        dir="wd10",
    )
    assert wind_vars.model_type == "wind_scalar"
    assert wind_vars.spd == "ws10"
    assert wind_vars.dir == "wd10"


def test_wind_grid(tmp_path, source_wind_grid, grid, time):
    wind = WindGrid(
        source=source_wind_grid,
        coords=dict(x="longitude", y="latitude"),
        wind_vars=WindVector(u="u10", v="v10"),
    )
    windfile = wind.get(destdir=tmp_path, grid=grid, time=time)
    assert (tmp_path / windfile["windfile"]).is_file()


def test_wind_station(tmp_path, source_wind_station, grid, time):
    wind = WindStation(
        source=source_wind_station,
        coords=dict(s="seapoint"),
        wind_vars=WindVector(u="uwnd", v="vwnd"),
    )
    windfile = wind.get(destdir=tmp_path, grid=grid, time=time)
    assert (tmp_path / windfile["windfile"]).is_file()


def test_wind_timeseries(tmp_path, source_wind_timeseries, grid, time):
    wind = WindPoint(
        source=source_wind_timeseries,
        wind_vars=WindVector(u="u10", v="v10"),
    )
    windfile = wind.get(destdir=tmp_path, grid=grid, time=time)
    assert (tmp_path / windfile["windfile"]).is_file()


def test_wind_timeseries_spd_dir(tmp_path, source_wind_timeseries, grid, time):
    # Extract the DataFrame object and test it with SourceTimeseriesDataFrame
    df = source_wind_timeseries._open_dataframe()
    wind = WindPoint(
        source=SourceTimeseriesDataFrame(obj=df),
        wind_vars=WindScalar(spd="wspd", dir="wdir"),
    )
    windfile = wind.get(destdir=tmp_path, grid=grid, time=time)
    assert (tmp_path / windfile["windfile"]).is_file()


def test_wind_timeseries_time_in_range(tmp_path, source_wind_timeseries, grid):
    wind = WindPoint(
        source=source_wind_timeseries, wind_vars=WindScalar(spd="wspd", dir="wdir")
    )
    time = TimeRange(start="1900-01-01T00", end="1900-01-01T12", interval="1h")
    with pytest.raises(ValueError):
        wind.get(destdir=tmp_path, grid=grid, time=time)


@pytest.mark.parametrize(
    "source_fixture,forcing_class,coords,variables",
    [
        (
            "source_tide_grid",
            TideConsGrid,
            {"x": "lon", "y": "lat"},
            None,
        ),
        (
            "source_tide_point",
            TideConsPoint,
            {},
            None,
        ),
        (
            "source_water_level_grid",
            WaterLevelGrid,
            {"x": "lon", "y": "lat"},
            ["ssh"],
        ),
        (
            "source_water_level_station",
            WaterLevelStation,
            {"s": "site", "x": "lon", "y": "lat"},
            ["ssh"],
        ),
        (
            "source_water_level_timeseries",
            WaterLevelPoint,
            {},
            ["ssh"],
        ),
    ],
)
def test_water_level_forcing(
    tmp_path, source_fixture, forcing_class, coords, variables, grid, time, request
):
    """Test water level forcing classes with different sources and configurations."""
    source = request.getfixturevalue(source_fixture)
    kwargs = {"source": source, "coords": coords}
    if variables is not None:
        kwargs["variables"] = variables

    forcing = forcing_class(**kwargs)
    namelist = forcing.get(destdir=tmp_path, grid=grid, time=time)

    filename = tmp_path / namelist["zs0file"]
    assert filename.is_file()
    data = np.loadtxt(filename)
    assert namelist["tidelen"] == data.shape[0]
    assert namelist["tideloc"] == 1
