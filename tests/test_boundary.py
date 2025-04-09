from pathlib import Path
import pytest
from wavespectra import read_swan

from rompy.core.time import TimeRange
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.source import SourceCRSFile, SourceCRSWavespectra
from rompy_xbeach.boundary import (
    BoundaryBaseStation,
    BoundaryStationParamJons,
    BoundaryPointParamJons,
    BoundaryStationSpectraJons,
    BoundaryGridParamJons,
    BoundaryStationSpectraJonstable,
    BoundaryStationParamJonstable,
    BoundaryPointParamJonstable,
    BoundaryGridParamJonstable,
    BoundaryStationSpectraSwan,
)
from rompy_xbeach.components.boundary import (
    WaveBoundaryBase,
    WaveBoundaryJons,
    WaveBoundaryJonstable,
)


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def time():
    yield TimeRange(start="2023-01-01T00", end="2023-01-01T03", interval="1h")


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
def source_file():
    yield SourceCRSFile(
        uri=HERE / "data/smc-params-20230101.nc",
        kwargs=dict(engine="netcdf4"),
        crs=4326,
    )


@pytest.fixture(scope="module")
def source_gridded_file():
    yield SourceCRSFile(
        uri=HERE / "data/gridded_wave_parameters.nc",
        kwargs=dict(engine="netcdf4"),
        crs=4326,
    )


@pytest.fixture(scope="module")
def source_csv():
    yield SourceTimeseriesCSV(filename=HERE / "data/wave-params-20230101.csv")


@pytest.fixture(scope="module")
def source_wavespectra():
    yield SourceCRSWavespectra(uri=HERE / "data/aus-20230101.nc", reader="read_ww3")


# =====================================================================================
# Boundary Components
# =====================================================================================
def test_wave_boundary_base_abstract():
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


# =====================================================================================
# Base Boundary
# =====================================================================================
def test_boundary_station_is_abstract(source_file):
    with pytest.raises(TypeError):
        BoundaryBaseStation(
            id="base",
            source=source_file,
            coords=dict(x="longitude", y="latitude", s="seapoint"),
        )


# =====================================================================================
# JONS BCFILE
# =====================================================================================
def test_boundary_grid_jons_bctype(tmp_path, source_gridded_file, grid, time):
    """Test bctype can be defined as either jons or parametric."""
    kwargs = dict(
        source=source_gridded_file,
        coords=dict(x="longitude", y="latitude", t="time"),
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
        filelist=False,
    )
    wb = BoundaryGridParamJons(**kwargs)
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"


def test_boundary_jons_bctype(tmp_path, source_file, grid, time):
    """Test bctype can be defined as either jons or parametric."""
    kwargs = dict(
        source=source_file,
        coords=dict(s="seapoint", x="longitude", y="latitude", t="time"),
        filelist=False,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    # Jons
    wb = BoundaryStationParamJons(**kwargs)
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    # Parametric
    wb = BoundaryStationParamJons(id="parametric", **kwargs)
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "parametric"
    # Unsupported
    with pytest.raises(ValueError):
        wb = BoundaryStationParamJons(id="unsupported", **kwargs)


def test_boundary_station_param_jons_bcfile(tmp_path, source_file, grid, time):
    """Test single (bcfile) jons spectral boundary from stations param source."""
    wb = BoundaryStationParamJons(
        source=source_file,
        coords=dict(s="seapoint", x="longitude", y="latitude", t="time"),
        filelist=False,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filename = tmp_path / namelist["bcfile"]
    assert filename.is_file()
    # Assert parameters defined in bcfile
    bcdata = filename.read_text()
    for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
        assert keys in bcdata


def test_boundary_station_param_jons_filelist(tmp_path, source_file, grid, time):
    """Test multiple (filelist) jons spectral boundary from param source."""
    wb = BoundaryStationParamJons(
        source=source_file,
        coords=dict(s="seapoint"),
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filelist = tmp_path / namelist["bcfile"]
    lines = filelist.read_text().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        # Assert bcfile created
        filename = tmp_path / line.split()[-1]
        assert filename.is_file()
        # Assert parameters defined in bcfile
        bcdata = filename.read_text()
        for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
            assert keys in bcdata


def test_boundary_station_param_jons_filelist_float(tmp_path, source_file, grid, time):
    """Test multiple jons spectral boundary with one param defined as a float."""
    wb = BoundaryStationParamJons(
        source=source_file,
        coords=dict(s="seapoint"),
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp=3.3,
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filelist = tmp_path / namelist["bcfile"]
    lines = filelist.read_text().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        # Assert bcfile created
        filename = tmp_path / line.split()[-1]
        assert filename.is_file()
        # Assert parameters defined in bcfile
        bcdata = filename.read_text()
        for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
            assert keys in bcdata


def test_boundary_station_spectra_jons_bcfile(tmp_path, source_wavespectra, grid, time):
    """Test single (bcfile) jons spectral boundary from spectra source."""
    wb = BoundaryStationSpectraJons(
        source=source_wavespectra,
        filelist=False,
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filename = tmp_path / namelist["bcfile"]
    assert filename.is_file()
    # Assert parameters defined in bcfile
    bcdata = filename.read_text()
    for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
        assert keys in bcdata


def test_boundary_station_spectra_jons_filelist(
    tmp_path, source_wavespectra, grid, time
):
    """Test multiple (filelist) jons spectral boundary from spectra source."""
    wb = BoundaryStationSpectraJons(
        source=source_wavespectra,
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filelist = tmp_path / namelist["bcfile"]
    lines = filelist.read_text().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        # Assert bcfile created
        filename = tmp_path / line.split()[-1]
        assert filename.is_file()
        # Assert parameters defined in bcfile
        bcdata = filename.read_text()
        for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
            assert keys in bcdata


def test_boundary_point_param_jons_bcfile(tmp_path, source_csv, grid, time):
    """Test single (bcfile) jons spectral boundary from timeseries param source."""
    wb = BoundaryPointParamJons(
        source=source_csv,
        filelist=False,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filename = tmp_path / namelist["bcfile"]
    assert filename.is_file()
    # Assert parameters defined in bcfile
    bcdata = filename.read_text()
    for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
        assert keys in bcdata


def test_boundary_point_param_jons_filelist(tmp_path, source_csv, grid, time):
    """Test multiple (filelist) jons spectral boundary from timeseries param source."""
    wb = BoundaryPointParamJons(
        source=source_csv,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jons"
    filelist = tmp_path / namelist["bcfile"]
    lines = filelist.read_text().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        # Assert bcfile created
        filename = tmp_path / line.split()[-1]
        assert filename.is_file()
        # Assert parameters defined in bcfile
        bcdata = filename.read_text()
        for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
            assert keys in bcdata


# =====================================================================================
# JONSTABLE BCFILE
# =====================================================================================
def test_boundary_station_param_jonstable(tmp_path, source_file, grid, time):
    """Test multiple (filelist) jons spectral boundary from param source."""
    wb = BoundaryStationParamJonstable(
        source=source_file,
        coords=dict(s="seapoint"),
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jonstable"
    bcfile = tmp_path / namelist["bcfile"]
    bcdata = bcfile.read_text().split("\n")
    for line in bcdata[1:]:
        if not line:
            continue
        # Assert all parameters defined in bcfile
        params = line.split()
        assert len(params) == 7


def test_boundary_station_spectra_jonstable(tmp_path, source_wavespectra, grid, time):
    """Test single (bcfile) jons spectral boundary from spectra source."""
    wb = BoundaryStationSpectraJonstable(
        source=source_wavespectra,
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jonstable"
    filename = tmp_path / namelist["bcfile"]
    assert filename.is_file()
    # Assert all parameters defined in bcfile
    bcdata = filename.read_text().split("\n")
    for line in bcdata[1:]:
        if not line:
            continue
        params = line.split()
        assert len(params) == 7


def test_boundary_point_param_jonstable(tmp_path, source_csv, grid, time):
    """Test multiple (filelist) jons spectral boundary from timeseries param source."""
    wb = BoundaryPointParamJonstable(
        source=source_csv,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jonstable"
    bcfile = tmp_path / namelist["bcfile"]
    bcdata = bcfile.read_text().split("\n")
    for line in bcdata[1:]:
        if not line:
            continue
        # Assert all parameters defined in bcfile
        params = line.split()
        assert len(params) == 7


def test_boundary_grid_param_jonstable(tmp_path, source_gridded_file, grid, time):
    """Test multiple (filelist) jons spectral boundary from param source."""
    wb = BoundaryGridParamJonstable(
        source=source_gridded_file,
        hm0="phs1",
        tp="ptp1",
        mainang="pdp1",
        gammajsp="ppe1",
        dspr="pspr1",
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "jonstable"
    bcfile = tmp_path / namelist["bcfile"]
    bcdata = bcfile.read_text().split("\n")
    for line in bcdata[1:]:
        if not line:
            continue
        # Assert all parameters defined in bcfile
        params = line.split()
        assert len(params) == 7


# =====================================================================================
# SWAN BCFILE
# =====================================================================================
def test_boundary_station_spectra_swan_bcfile(tmp_path, source_wavespectra, grid, time):
    """Test single (bcfile) jons spectral boundary from param source."""
    wb = BoundaryStationSpectraSwan(
        source=source_wavespectra,
        filelist=False,
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "swan"
    filename = tmp_path / namelist["bcfile"]
    assert filename.is_file()
    # Assert swan file defined in bcfile
    ds = read_swan(filename)
    assert hasattr(ds, "spec")


def test_boundary_station_spectra_swan_filelist(
    tmp_path, source_wavespectra, grid, time
):
    """Test multiple (filelist) jons spectral boundary from param source."""
    wb = BoundaryStationSpectraSwan(
        source=source_wavespectra,
    )
    namelist = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert namelist["wbctype"] == "swan"
    filelist = tmp_path / namelist["bcfile"]
    lines = filelist.read_text().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        # Assert bcfile created
        filename = tmp_path / line.split()[-1]
        assert filename.is_file()
        # Assert swan file defined in bcfile
        ds = read_swan(filename)
        assert hasattr(ds, "spec")
