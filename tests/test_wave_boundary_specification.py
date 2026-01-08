"""Tests for wave boundary specification classes."""

import pytest
from rompy_xbeach.components.boundary.specification import (
    SpectralWaveBoundary,
    NonSpectralWaveBoundary,
    OffWaveBoundary,
    ReuseWaveBoundary,
)
from rompy_xbeach.components.boundary.parameters import (
    SpectralWaveBoundaryConditions,
    NonSpectralWaveBoundaryConditions,
)


def test_spectral_wave_boundary():
    """Test SpectralWaveBoundary with all fields."""
    boundary = SpectralWaveBoundary(
        wbctype="jons",
        bcfile="jonswap.txt",
        wbc=SpectralWaveBoundaryConditions(
            nmax=0.8,
            rt=3600.0,
            dtbc=1.0,
            random=True,
        ),
    )
    assert boundary.model_type == "spectral"
    assert boundary.wbctype == "jons"
    assert boundary.bcfile == "jonswap.txt"
    assert boundary.wbc.nmax == 0.8
    assert boundary.wbc.rt == 3600.0


def test_spectral_wave_boundary_requires_bcfile():
    """Test that SpectralWaveBoundary requires bcfile."""
    with pytest.raises(ValueError, match="Field required"):
        SpectralWaveBoundary(wbctype="jons")


def test_spectral_wave_boundary_without_wbc():
    """Test SpectralWaveBoundary without wbc parameters."""
    boundary = SpectralWaveBoundary(
        wbctype="swan",
        bcfile="swan.sp2",
    )
    assert boundary.wbctype == "swan"
    assert boundary.bcfile == "swan.sp2"
    assert boundary.wbc is None


def test_non_spectral_wave_boundary_stat():
    """Test NonSpectralWaveBoundary for stationary waves (no file needed)."""
    boundary = NonSpectralWaveBoundary(
        wbctype="stat",
        wbc=NonSpectralWaveBoundaryConditions(
            Hrms=2.0,
            Trep=12.0,
            dir0=285.0,
            m=10,
        ),
    )
    assert boundary.model_type == "nonspectral"
    assert boundary.wbctype == "stat"
    assert boundary.bcfile is None
    assert boundary.wbc.Hrms == 2.0
    assert boundary.wbc.Trep == 12.0


def test_non_spectral_wave_boundary_bichrom():
    """Test NonSpectralWaveBoundary for bichromatic waves (no file needed)."""
    boundary = NonSpectralWaveBoundary(
        wbctype="bichrom",
        wbc=NonSpectralWaveBoundaryConditions(
            Hrms=1.5,
            Trep=10.0,
            Tlong=80.0,
            dir0=270.0,
        ),
    )
    assert boundary.wbctype == "bichrom"
    assert boundary.bcfile is None
    assert boundary.wbc.Tlong == 80.0


def test_non_spectral_wave_boundary_ts1_requires_file():
    """Test that ts_1 requires bcfile."""
    with pytest.raises(ValueError, match="requires bcfile"):
        NonSpectralWaveBoundary(wbctype="ts_1")


def test_non_spectral_wave_boundary_ts1_with_file():
    """Test NonSpectralWaveBoundary for ts_1 with file."""
    boundary = NonSpectralWaveBoundary(
        wbctype="ts_1",
        bcfile="bc/gen.ezs",
        wbc=NonSpectralWaveBoundaryConditions(
            Hrms=2.0,
            Trep=12.0,
        ),
    )
    assert boundary.wbctype == "ts_1"
    assert boundary.bcfile == "bc/gen.ezs"


def test_non_spectral_wave_boundary_ts2_requires_file():
    """Test that ts_2 requires bcfile."""
    with pytest.raises(ValueError, match="requires bcfile"):
        NonSpectralWaveBoundary(wbctype="ts_2")


def test_non_spectral_wave_boundary_ts_nonh_requires_file():
    """Test that ts_nonh requires bcfile."""
    with pytest.raises(ValueError, match="requires bcfile"):
        NonSpectralWaveBoundary(wbctype="ts_nonh")


def test_non_spectral_wave_boundary_ts_nonh_with_file():
    """Test NonSpectralWaveBoundary for ts_nonh with file."""
    boundary = NonSpectralWaveBoundary(
        wbctype="ts_nonh",
        bcfile="Boun_u.bcf",
    )
    assert boundary.wbctype == "ts_nonh"
    assert boundary.bcfile == "Boun_u.bcf"


def test_non_spectral_wave_boundary_stat_table_requires_file():
    """Test that stat_table requires bcfile."""
    with pytest.raises(ValueError, match="requires bcfile"):
        NonSpectralWaveBoundary(wbctype="stat_table")


def test_non_spectral_wave_boundary_stat_table_with_file():
    """Test NonSpectralWaveBoundary for stat_table with file."""
    boundary = NonSpectralWaveBoundary(
        wbctype="stat_table",
        bcfile="stat_table.txt",
    )
    assert boundary.wbctype == "stat_table"
    assert boundary.bcfile == "stat_table.txt"


def test_off_wave_boundary():
    """Test OffWaveBoundary."""
    boundary = OffWaveBoundary()
    assert boundary.model_type == "off"
    assert boundary.wbctype == "off"
    assert boundary.bcfile is None
    assert boundary.wbc is None


def test_reuse_wave_boundary():
    """Test ReuseWaveBoundary without file."""
    boundary = ReuseWaveBoundary()
    assert boundary.model_type == "reuse"
    assert boundary.wbctype == "reuse"
    assert boundary.bcfile is None


def test_reuse_wave_boundary_with_file():
    """Test ReuseWaveBoundary with file."""
    boundary = ReuseWaveBoundary(bcfile="path/to/ebcflist.bcf")
    assert boundary.wbctype == "reuse"
    assert boundary.bcfile == "path/to/ebcflist.bcf"


def test_spectral_wave_boundary_serialization():
    """Test serialization of SpectralWaveBoundary."""
    boundary = SpectralWaveBoundary(
        wbctype="jons",
        bcfile="jonswap.txt",
        wbc=SpectralWaveBoundaryConditions(
            nmax=0.8,
            rt=3600.0,
        ),
    )
    data = boundary.model_dump(exclude_none=True)
    assert data["model_type"] == "spectral"
    assert data["wbctype"] == "jons"
    assert data["bcfile"] == "jonswap.txt"
    # wbc parameters are flattened by the serializer
    assert data["nmax"] == 0.8
    assert data["rt"] == 3600.0


def test_non_spectral_wave_boundary_serialization():
    """Test serialization of NonSpectralWaveBoundary."""
    boundary = NonSpectralWaveBoundary(
        wbctype="stat",
        wbc=NonSpectralWaveBoundaryConditions(
            Hrms=2.0,
            Trep=12.0,
        ),
    )
    data = boundary.model_dump(exclude_none=True)
    assert data["model_type"] == "nonspectral"
    assert data["wbctype"] == "stat"
    assert "bcfile" not in data  # None values excluded
    # wbc parameters are flattened by the serializer
    assert data["Hrms"] == 2.0
    assert data["Trep"] == 12.0


def test_spectral_wbctype_validation():
    """Test that only valid spectral wbctype values are accepted."""
    # Valid types
    for wbctype in ["jons", "parametric", "swan", "vardens", "jonstable"]:
        boundary = SpectralWaveBoundary(wbctype=wbctype, bcfile="test.txt")
        assert boundary.wbctype == wbctype

    # Invalid type
    with pytest.raises(ValueError):
        SpectralWaveBoundary(wbctype="invalid", bcfile="test.txt")


def test_non_spectral_wbctype_validation():
    """Test that only valid non-spectral wbctype values are accepted."""
    # Valid types that don't need files
    for wbctype in ["stat", "bichrom"]:
        boundary = NonSpectralWaveBoundary(wbctype=wbctype)
        assert boundary.wbctype == wbctype

    # Valid types that need files
    for wbctype in ["stat_table", "ts_1", "ts_2", "ts_nonh"]:
        boundary = NonSpectralWaveBoundary(wbctype=wbctype, bcfile="test.txt")
        assert boundary.wbctype == wbctype

    # Invalid type
    with pytest.raises(ValueError):
        NonSpectralWaveBoundary(wbctype="invalid")
