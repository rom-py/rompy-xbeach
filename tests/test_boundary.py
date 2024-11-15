from pathlib import Path
import pytest
import xarray as xr

from rompy.core.time import TimeRange

from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.source import SourceCRSFile, SourceCRSWavespectra
from rompy_xbeach.boundary import BoundaryStation, BoundaryStationSpectraJons
from rompy_xbeach.components.boundary import (
    WaveBoundaryBase,
    WaveBoundaryJons,
    WaveBoundaryJonstable,
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
def source_crs_file():
    yield SourceCRSFile(
        uri=HERE / "data/smc-params-20230101.nc",
        kwargs=dict(engine="netcdf4"),
        crs=4326,
        x_dim="lon",
        y_dim="lat",
    )


@pytest.fixture(scope="module")
def source_crs_wavespectra():
    yield SourceCRSWavespectra(uri=HERE / "data/aus-20230101.nc", reader="read_ww3")


def test_wave_boundary_base():
    with pytest.raises(TypeError):
        WaveBoundaryBase()


def test_wave_boundary_spectral_defaults():
    wb = WaveBoundaryJons()
    assert wb.bcfile == "spectrum.txt"
    assert wb.rt is None
    assert wb.dbtc is None
    assert wb.tm01switch is None
    assert wb.correcthm0 is None
    assert wb.fcutoff is None
    assert wb.nonhspectrum is None
    assert wb.nspectrumloc is None
    assert wb.nspr is None
    assert wb.random is None
    assert wb.sprdthr is None
    assert wb.trepfac is None
    assert wb.wbcversion is None


def test_wave_boundary_spectral_valid_ranges():
    with pytest.raises(ValueError):
        WaveBoundaryJons(rt=1000)
        WaveBoundaryJons(dbtc=2.1)
        WaveBoundaryJons(dthetas_xb=-361)
        WaveBoundaryJons(fcutoff=41.0)
        WaveBoundaryJons(nspectrumloc=0)
        WaveBoundaryJons(sprdthr=1.1)
        WaveBoundaryJons(trepfac=-0.1)
        WaveBoundaryJons(wbcversion=4)
        WaveBoundaryJons(fnyq=1.0, dfj=0.01)


def test_wave_boundary_spectral_jons_valid_ranges():
    with pytest.raises(ValueError):
        WaveBoundaryJons(fnyq=1.0, dfj=0.00099)
        WaveBoundaryJons(fnyq=1.0, dfj=0.051)


def test_wave_boundary_spectral_jons_write(tmp_path):
    wb = WaveBoundaryJons(hm0=1.0, tp=12.0, bcfile="jons.txt")
    bcfile = wb.write(tmp_path)
    assert bcfile.is_file()


def test_wave_boundary_spectral_jonstable_same_sizes():
    with pytest.raises(ValueError):
        WaveBoundaryJonstable(
            hm0=[1.0, 2.0],
            tp=[10.0, 10.0],
            mainang=[180, 180],
            gammajsp=[3.3, 3.3],
            s=[10.0],
            duration=[1800, 1800],
            dtbc=[1.0, 1.0],
        )


def test_wave_boundary_spectral_jonstable_valid_ranges():
    with pytest.raises(ValueError):
        WaveBoundaryJonstable(
            hm0=[1.0, 5000.0],
            tp=[10.0, 10.0],
            mainang=[180, 180],
            gammajsp=[3.3, 3.3],
            s=[10.0, 10.0],
            duration=[1800, 1800],
            dtbc=[1.0, 1.0],
        )


def test_wave_boundary_spectral_jonstable_write(tmp_path):
    wb = WaveBoundaryJonstable(
        hm0=[1.0, 2.0],
        tp=[10.0, 10.0],
        mainang=[180, 180],
        gammajsp=[3.3, 3.3],
        s=[10.0, 10.0],
        duration=[1800, 1800],
        dtbc=[1.0, 1.0],
    )
    bcfile = wb.write(tmp_path)
    assert bcfile.is_file()


def test_boundary_station(tmp_path, source_crs_file, grid, time):
    wb = BoundaryStation(
        id="test",
        source=source_crs_file,
        coords=dict(x="longitude", y="latitude", s="seapoint")
    )
    ds = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert ds[wb.coords.s].size == 1
    tstart, tend = ds.time.to_index().to_pydatetime()[[0, -1]]
    assert time.start >= tstart and time.end <= tend


def test_boundary_station_spectra_jons(tmp_path, source_crs_wavespectra, grid, time):
    wb = BoundaryStationSpectraJons(
        id="test",
        source=source_crs_wavespectra,
    )
    bcfile = wb.get(destdir=tmp_path, grid=grid, time=time)
    # import ipdb; ipdb.set_trace()


# def test_xbeach_wave_station(tmp_path, source, grid, time):
#     # kind = WaveBoundaryJons(hm0=1.5, tp=12.0)
#     wb = XBeachSpectraStationSingle(id="test", source=source, kind="jons")
#     bcfile = wb.get(destdir=tmp_path, grid=grid, time=time)
#     assert bcfile.is_file()
#     # import ipdb; ipdb.set_trace()
