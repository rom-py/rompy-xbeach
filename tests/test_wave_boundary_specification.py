"""Tests for wave boundary classes in data/boundary/."""

from pathlib import Path
from rompy_xbeach.types import XBeachDataBlob
from rompy_xbeach.data.boundary import (
    BoundaryStat,
    BoundaryBichrom,
    BoundaryStatTable,
    BoundaryTs1,
    BoundaryTs2,
    BoundaryTsNonh,
    BoundaryOff,
    BoundaryReuse,
)


HERE = Path(__file__).parent


def test_boundary_stat():
    """Test BoundaryStat for stationary waves."""
    boundary = BoundaryStat(
        Hrms=2.0,
        Trep=12.0,
        dir0=285.0,
        m=10,
    )
    assert boundary.model_type == "stat"
    assert boundary.Hrms == 2.0
    assert boundary.Trep == 12.0
    assert boundary.dir0 == 285.0
    assert boundary.m == 10


def test_boundary_stat_get():
    """Test BoundaryStat.get() returns correct XBeach parameters."""
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


def test_boundary_bichrom():
    """Test BoundaryBichrom for bichromatic waves."""
    boundary = BoundaryBichrom(
        Hrms=1.5,
        Trep=10.0,
        Tlong=80.0,
        dir0=270.0,
        m=10,
    )
    assert boundary.model_type == "bichrom"
    assert boundary.Hrms == 1.5
    assert boundary.Tlong == 80.0


def test_boundary_bichrom_get():
    """Test BoundaryBichrom.get() returns correct XBeach parameters."""
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


def test_boundary_ts1(tmp_path):
    """Test BoundaryTs1 for time series at single location."""
    # Create a test file
    test_file = tmp_path / "gen.ezs"
    test_file.write_text("test content")
    
    boundary = BoundaryTs1(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_1"
    assert boundary.id == "ts_1"


def test_boundary_ts1_get(tmp_path):
    """Test BoundaryTs1.get() returns correct XBeach parameters."""
    # Create a test file
    source_file = tmp_path / "source" / "gen.ezs"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("test content")
    
    boundary = BoundaryTs1(
        source=XBeachDataBlob(source=source_file),
    )
    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)
    assert params["wbctype"] == "ts_1"
    assert params["bcfile"] == "bc/gen.ezs"
    # Verify file was copied to bc/ subdirectory
    assert (destdir / "bc" / "gen.ezs").exists()


def test_boundary_ts2(tmp_path):
    """Test BoundaryTs2 for time series at two locations."""
    # Create a test file
    test_file = tmp_path / "gen.ezs"
    test_file.write_text("test content")
    
    boundary = BoundaryTs2(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_2"
    assert boundary.id == "ts_2"


def test_boundary_ts_nonh(tmp_path):
    """Test BoundaryTsNonh for non-hydrostatic time series."""
    # Create a test file
    test_file = tmp_path / "Boun_u.bcf"
    test_file.write_text("test content")
    
    boundary = BoundaryTsNonh(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_ts_nonh"
    assert boundary.id == "ts_nonh"


def test_boundary_stat_table(tmp_path):
    """Test BoundaryStatTable for time-varying parametric waves."""
    # Create a test file
    test_file = tmp_path / "stat_table.txt"
    test_file.write_text("test content")
    
    boundary = BoundaryStatTable(
        source=XBeachDataBlob(source=test_file),
    )
    assert boundary.model_type == "file_stat_table"
    assert boundary.id == "stat_table"


def test_boundary_stat_table_get(tmp_path):
    """Test BoundaryStatTable.get() returns correct XBeach parameters."""
    # Create a test file
    source_file = tmp_path / "source" / "stat_table.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("test content")
    
    boundary = BoundaryStatTable(
        source=XBeachDataBlob(source=source_file),
    )
    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)
    assert params["wbctype"] == "stat_table"
    assert params["bcfile"] == "stat_table.txt"
    # Verify file was copied
    assert (destdir / "stat_table.txt").exists()


def test_boundary_off():
    """Test BoundaryOff for no wave forcing."""
    boundary = BoundaryOff()
    assert boundary.model_type == "off"


def test_boundary_off_get():
    """Test BoundaryOff.get() returns correct XBeach parameters."""
    boundary = BoundaryOff()
    params = boundary.get("/tmp")
    assert params["wbctype"] == "off"
    assert len(params) == 1


def test_boundary_reuse(tmp_path):
    """Test BoundaryReuse with previous_run directory."""
    from rompy_xbeach.types import XBeachDirectoryBlob

    # Create source directory with test files
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "ebcflist.bcf").write_text("test ebcflist content")
    (source_dir / "qbcflist.bcf").write_text("test qbcflist content")

    boundary = BoundaryReuse(
        previous_run=XBeachDirectoryBlob(source=str(source_dir))
    )
    assert boundary.model_type == "reuse"
    assert boundary.id == "reuse"


def test_boundary_reuse_get(tmp_path):
    """Test BoundaryReuse.get() returns correct XBeach parameters."""
    from rompy_xbeach.types import XBeachDirectoryBlob

    # Create source directory with test files
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "ebcflist.bcf").write_text("test ebcflist content")
    (source_dir / "qbcflist.bcf").write_text("test qbcflist content")

    boundary = BoundaryReuse(
        previous_run=XBeachDirectoryBlob(source=str(source_dir))
    )

    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)

    assert params["wbctype"] == "reuse"
    assert "bcfile" not in params  # XBeach knows the file names automatically
    # Verify files were copied
    assert (destdir / "ebcflist.bcf").exists()
    assert (destdir / "qbcflist.bcf").exists()


def test_boundary_stat_with_wave_params():
    """Test BoundaryStat with additional wave boundary parameters."""
    boundary = BoundaryStat(
        Hrms=2.0,
        Trep=12.0,
        dir0=270.0,
        m=10,
        nmax=0.8,
        thetamin=-60,
        thetamax=60,
        dtheta=10,
    )
    params = boundary.get("/tmp")
    assert params["wbctype"] == "stat"
    assert params["Hrms"] == 2.0
    assert params["nmax"] == 0.8
    assert params["thetamin"] == -60
    assert params["thetamax"] == 60
    assert params["dtheta"] == 10


# =====================================================================================
# File-based Spectral Boundary Tests
# =====================================================================================
def test_boundary_file_jons(tmp_path):
    """Test BoundaryFileJons with single bcfile."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileJons

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    bcfile_content = "Hm0 = 2.0\nTp = 12.0\nmainang = 270.0\n"
    (source_dir / "spectrum.txt").write_text(bcfile_content)

    boundary = BoundaryFileJons(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "spectrum.txt")),
    )
    assert boundary.model_type == "file_jons"
    assert boundary.id == "jons"
    assert boundary.filelist is False


def test_boundary_file_jons_get(tmp_path):
    """Test BoundaryFileJons.get() returns correct XBeach parameters."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileJons

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    bcfile_content = "Hm0 = 2.0\nTp = 12.0\nmainang = 270.0\n"
    (source_dir / "spectrum.txt").write_text(bcfile_content)

    boundary = BoundaryFileJons(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "spectrum.txt")),
        nmax=0.8,
    )

    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)

    assert params["wbctype"] == "jons"
    assert params["bcfile"] == "spectrum.txt"
    assert params["nmax"] == 0.8
    assert (destdir / "spectrum.txt").exists()


def test_boundary_file_jons_filelist(tmp_path):
    """Test BoundaryFileJons with FILELIST."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileJons

    # Create source directory with FILELIST and referenced files
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FILELIST
    filelist_content = "FILELIST\n3600 1.0 spec1.txt\n3600 1.0 spec2.txt\n"
    (source_dir / "filelist.txt").write_text(filelist_content)
    
    # Create referenced files
    (source_dir / "spec1.txt").write_text("Hm0 = 2.0\nTp = 12.0\n")
    (source_dir / "spec2.txt").write_text("Hm0 = 2.5\nTp = 11.0\n")

    boundary = BoundaryFileJons(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "filelist.txt")),
        filelist=True,
    )

    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)

    assert params["wbctype"] == "jons"
    assert params["bcfile"] == "filelist.txt"
    # Verify all files were copied
    assert (destdir / "filelist.txt").exists()
    assert (destdir / "spec1.txt").exists()
    assert (destdir / "spec2.txt").exists()


def test_boundary_file_jonstable(tmp_path):
    """Test BoundaryFileJonstable with single bcfile."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileJonstable

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    bcfile_content = "2.0 12.0 270.0 3.3 10.0 3600 1.0\n2.5 11.0 265.0 3.3 10.0 3600 1.0\n"
    (source_dir / "jonstable.txt").write_text(bcfile_content)

    boundary = BoundaryFileJonstable(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "jonstable.txt")),
    )
    assert boundary.model_type == "file_jonstable"
    assert boundary.id == "jonstable"


def test_boundary_file_jonstable_get(tmp_path):
    """Test BoundaryFileJonstable.get() returns correct XBeach parameters."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileJonstable

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    bcfile_content = "2.0 12.0 270.0 3.3 10.0 3600 1.0\n"
    (source_dir / "jonstable.txt").write_text(bcfile_content)

    boundary = BoundaryFileJonstable(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "jonstable.txt")),
        rt=3600.0,
    )

    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)

    assert params["wbctype"] == "jonstable"
    assert params["bcfile"] == "jonstable.txt"
    assert params["rt"] == 3600.0
    assert (destdir / "jonstable.txt").exists()


def test_boundary_file_swan(tmp_path):
    """Test BoundaryFileSwan with single bcfile."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileSwan

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "swan_spectrum.txt").write_text("SWAN spectrum content")

    boundary = BoundaryFileSwan(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "swan_spectrum.txt")),
    )
    assert boundary.model_type == "file_swan"
    assert boundary.id == "swan"
    assert boundary.filelist is False


def test_boundary_file_swan_get(tmp_path):
    """Test BoundaryFileSwan.get() returns correct XBeach parameters."""
    from rompy_xbeach.types import XBeachDataBlob
    from rompy_xbeach.data.boundary import BoundaryFileSwan

    # Create source bcfile
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "swan_spectrum.txt").write_text("SWAN spectrum content")

    boundary = BoundaryFileSwan(
        bcfile_source=XBeachDataBlob(source=str(source_dir / "swan_spectrum.txt")),
        dthetas_xb=10.0,
    )

    destdir = tmp_path / "dest"
    destdir.mkdir(parents=True, exist_ok=True)
    params = boundary.get(destdir)

    assert params["wbctype"] == "swan"
    assert params["bcfile"] == "swan_spectrum.txt"
    assert params["dthetas_xb"] == 10.0
    assert (destdir / "swan_spectrum.txt").exists()
