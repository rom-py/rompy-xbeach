from pathlib import Path
import pytest
from wavespectra import read_swan

from rompy.core.time import TimeRange
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.source import SourceCRSFile, SourceCRSWavespectra
from rompy_xbeach.data.boundary import (
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
from rompy_xbeach.data.boundary.writers import (
    BoundaryWriterBase,
    JonsWriter,
    JonstableWriter,
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
def test_boundary_writer_base_abstract():
    """Test that BoundaryWriterBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BoundaryWriterBase()


def test_jons_writer_defaults():
    """Test default values for JONSWAP file writer."""
    bf = JonsWriter()
    assert bf.bcfile == "spectrum.txt"
    assert bf.hm0 is None
    assert bf.tp is None
    assert bf.mainang is None
    assert bf.gammajsp is None
    assert bf.s is None
    assert bf.fnyq is None
    assert bf.dfj is None


def test_jons_writer_valid_ranges():
    """Test validation ranges for JONSWAP parameters."""
    with pytest.raises(ValueError):
        JonsWriter(fnyq=1.0, dfj=0.00099)
        JonsWriter(fnyq=1.0, dfj=0.051)


def test_jons_writer_write(tmp_path):
    """Test writing JONSWAP boundary file."""
    bf = JonsWriter(hm0=1.0, tp=12.0, bcfile="jons.txt")
    bcfile = bf.write(tmp_path)
    assert bcfile.is_file()


def test_jonstable_writer_same_sizes():
    """Test that JONSTABLE requires all parameter lists to be same size."""
    with pytest.raises(ValueError):
        JonstableWriter(
            hm0=[1.0, 2.0],
            tp=[10.0, 10.0],
            mainang=[180, 180],
            gammajsp=[3.3, 3.3],
            s=[10.0],
            duration=[1800, 1800],
            dtbc=[1.0, 1.0],
        )


def test_jonstable_writer_valid_ranges():
    """Test validation ranges for JONSTABLE parameters."""
    with pytest.raises(ValueError):
        JonstableWriter(
            hm0=[1.0, 5000.0],
            tp=[10.0, 10.0],
            mainang=[180, 180],
            gammajsp=[3.3, 3.3],
            s=[10.0, 10.0],
            duration=[1800, 1800],
            dtbc=[1.0, 1.0],
        )


def test_jonstable_writer_write(tmp_path):
    """Test writing JONSTABLE boundary file."""
    bf = JonstableWriter(
        hm0=[1.0, 2.0],
        tp=[10.0, 10.0],
        mainang=[180, 180],
        gammajsp=[3.3, 3.3],
        s=[10.0, 10.0],
        duration=[1800, 1800],
        dtbc=[1.0, 1.0],
        bcfile="jonstable.txt",
    )
    bcfile = bf.write(tmp_path)
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
        filelist=False,
    )
    wb = BoundaryGridParamJons(**kwargs)
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"


def test_boundary_jons_bctype(tmp_path, source_file, grid, time):
    """The JONSWAP boundary id is the XBeach wbctype, parametric."""
    kwargs = dict(
        source=source_file,
        coords=dict(s="seapoint", x="longitude", y="latitude", t="time"),
        filelist=False,
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    wb = BoundaryStationParamJons(**kwargs)
    assert wb.id == "parametric"
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    assert boundary_spec["bcfile"].startswith("parametric-")
    # The legacy instat name is no longer accepted
    with pytest.raises(ValueError):
        BoundaryStationParamJons(id="jons", **kwargs)


def test_boundary_station_param_jons_bcfile(tmp_path, source_file, grid, time):
    """Test single (bcfile) jons spectral boundary from stations param source."""
    wb = BoundaryStationParamJons(
        source=source_file,
        coords=dict(s="seapoint", x="longitude", y="latitude", t="time"),
        filelist=False,
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filename = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
        filelist=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filelist = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var=3.3,
        dspr_var="pspr1",
        filelist=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filelist = tmp_path / boundary_spec["bcfile"]
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
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filename = tmp_path / boundary_spec["bcfile"]
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
        filelist=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filelist = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filename = tmp_path / boundary_spec["bcfile"]
    assert filename.is_file()
    # Assert parameters defined in bcfile
    bcdata = filename.read_text()
    for keys in ["Hm0", "Tp", "mainang", "gammajsp", "s"]:
        assert keys in bcdata


def test_boundary_point_param_jons_filelist(tmp_path, source_csv, grid, time):
    """Test multiple (filelist) jons spectral boundary from timeseries param source."""
    wb = BoundaryPointParamJons(
        source=source_csv,
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
        filelist=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "parametric"
    filelist = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "jonstable"
    bcfile = tmp_path / boundary_spec["bcfile"]
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
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "jonstable"
    filename = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "jonstable"
    bcfile = tmp_path / boundary_spec["bcfile"]
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
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "jonstable"
    bcfile = tmp_path / boundary_spec["bcfile"]
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
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "swan"
    filename = tmp_path / boundary_spec["bcfile"]
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
        filelist=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    assert boundary_spec["wbctype"] == "swan"
    filelist = tmp_path / boundary_spec["bcfile"]
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


@pytest.fixture(scope="module")
def source_single_site(grid):
    """Station spectra source with a single site, the closest to the grid."""
    from rompy_xbeach.grid import GeoPoint
    from rompy_xbeach.source import SourceCRSDataset

    dset = SourceCRSWavespectra(
        uri=HERE / "data/aus-20230101.nc", reader="read_ww3"
    ).open()
    x, y = grid.offshore
    bnd = GeoPoint(x=x, y=y, crs=grid.crs).reproject(4326)
    isite = int(((dset.lon - bnd.x) ** 2 + (dset.lat - bnd.y) ** 2).argmin(dim="site"))
    yield SourceCRSDataset(
        obj=dset.isel(site=[isite]).load(), crs=4326, x_dim="lon", y_dim="lat"
    )


def test_boundary_station_single_site_idw_raises(
    tmp_path, source_single_site, grid, time
):
    """Default idw masks the data when a single site is available, raise clearly."""
    wb = BoundaryStationSpectraSwan(source=source_single_site, filelist=False)
    with pytest.raises(ValueError, match="sel_method='nearest'"):
        wb.get(destdir=tmp_path, grid=grid, time=time)


def test_boundary_station_single_site_nearest(tmp_path, source_single_site, grid, time):
    """A single-site station source works with sel_method='nearest'."""
    wb = BoundaryStationSpectraSwan(
        source=source_single_site, filelist=False, sel_method="nearest"
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    ds = read_swan(tmp_path / boundary_spec["bcfile"])
    assert not bool(ds.efth.isnull().any())
    assert float(ds.spec.hs().squeeze()) > 0


# =====================================================================================
# Data-interface field leakage guard
# =====================================================================================
def test_data_interface_fields_not_serialized(tmp_path, source_file, grid, time):
    """Data interface fields must not leak into the XBeach params dict.

    Concrete data-driven boundary classes inherit many fields from the rompy data
    interfaces (source, coords, etc.). These must be stripped from the serialized
    params (by BoundaryBase._serialize), otherwise they end up in params.txt.
    """
    wb = BoundaryStationParamJons(
        source=source_file,
        coords=dict(s="seapoint", x="longitude", y="latitude", t="time"),
        filelist=False,
        hm0_var="phs1",
        tp_var="ptp1",
        mainang_var="pdp1",
        gammajsp_var="ppe1",
        dspr_var="pspr1",
        wbcScaleEnergy=True,
    )
    boundary_spec = wb.get(destdir=tmp_path, grid=grid, time=time)
    leaked = {
        "source",
        "coords",
        "location",
        "variables",
        "time_buffer",
        "buffer",
        "crop_data",
        "filter",
        "sel_method",
        "sel_method_kwargs",
    } & set(boundary_spec)
    assert not leaked, f"Data interface fields leaked into params: {sorted(leaked)}"
    # Wave parameters must still be present
    assert "wbcScaleEnergy" in boundary_spec
