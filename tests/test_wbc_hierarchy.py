"""Tests for wave boundary condition parameter hierarchy."""

import pytest
from rompy_xbeach.components.physics.wbc import (
    WaveBoundaryConditions,
    SpectralWaveBoundaryConditions,
    NonSpectralWaveBoundaryConditions,
)
from rompy_xbeach.components.physics import Physics


def test_base_wave_boundary_conditions():
    """Test base WaveBoundaryConditions with general parameters only."""
    wbc = WaveBoundaryConditions(
        nmax=0.8,
        wbcevarreduce=1.0,
        bclwonly=False,
        wbcRemoveStokes=True,
        wbcScaleEnergy=True,
        taper=100.0,
    )
    assert wbc.nmax == 0.8
    assert wbc.wbcevarreduce == 1.0
    assert wbc.bclwonly is False
    assert wbc.wbcRemoveStokes is True
    assert wbc.wbcScaleEnergy is True
    assert wbc.taper == 100.0


def test_spectral_wave_boundary_conditions():
    """Test SpectralWaveBoundaryConditions with both general and spectral parameters."""
    wbc = SpectralWaveBoundaryConditions(
        # General parameters
        nmax=0.8,
        wbcScaleEnergy=True,
        # Spectral parameters
        rt=3600.0,
        dtbc=1.0,
        random=True,
        fcutoff=0.05,
        correcthm0=True,
        wbcversion=3,
    )
    # Check general parameters
    assert wbc.nmax == 0.8
    assert wbc.wbcScaleEnergy is True
    # Check spectral parameters
    assert wbc.rt == 3600.0
    assert wbc.dtbc == 1.0
    assert wbc.random is True
    assert wbc.fcutoff == 0.05
    assert wbc.correcthm0 is True
    assert wbc.wbcversion == 3


def test_non_spectral_wave_boundary_conditions():
    """Test NonSpectralWaveBoundaryConditions with both general and non-spectral parameters."""
    wbc = NonSpectralWaveBoundaryConditions(
        # General parameters (including taper)
        nmax=0.8,
        wbcScaleEnergy=True,
        taper=100.0,
        # Non-spectral parameters
        Hrms=2.0,
        Trep=12.0,
        dir0=285.0,
        m=10,
    )
    # Check general parameters
    assert wbc.nmax == 0.8
    assert wbc.wbcScaleEnergy is True
    assert wbc.taper == 100.0
    # Check non-spectral parameters
    assert wbc.Hrms == 2.0
    assert wbc.Trep == 12.0
    assert wbc.dir0 == 285.0
    assert wbc.m == 10


def test_bichromatic_parameters():
    """Test NonSpectralWaveBoundaryConditions with bichromatic-specific parameters."""
    wbc = NonSpectralWaveBoundaryConditions(
        Hrms=1.5,
        Trep=10.0,
        Tlong=80.0,  # Bichromatic-specific
        dir0=270.0,
        m=10,
    )
    assert wbc.Hrms == 1.5
    assert wbc.Trep == 10.0
    assert wbc.Tlong == 80.0
    assert wbc.dir0 == 270.0


def test_spectral_inherits_from_base():
    """Test that SpectralWaveBoundaryConditions inherits from WaveBoundaryConditions."""
    assert issubclass(SpectralWaveBoundaryConditions, WaveBoundaryConditions)
    wbc = SpectralWaveBoundaryConditions()
    assert isinstance(wbc, WaveBoundaryConditions)
    assert isinstance(wbc, SpectralWaveBoundaryConditions)


def test_non_spectral_inherits_from_base():
    """Test that NonSpectralWaveBoundaryConditions inherits from WaveBoundaryConditions."""
    assert issubclass(NonSpectralWaveBoundaryConditions, WaveBoundaryConditions)
    wbc = NonSpectralWaveBoundaryConditions()
    assert isinstance(wbc, WaveBoundaryConditions)
    assert isinstance(wbc, NonSpectralWaveBoundaryConditions)


def test_physics_accepts_base_wbc():
    """Test that Physics accepts base WaveBoundaryConditions."""
    physics = Physics(
        wbc=WaveBoundaryConditions(
            nmax=0.8,
            wbcScaleEnergy=True,
        )
    )
    assert physics.wbc.nmax == 0.8
    assert physics.wbc.wbcScaleEnergy is True


def test_physics_accepts_spectral_wbc():
    """Test that Physics accepts SpectralWaveBoundaryConditions."""
    physics = Physics(
        wbc=SpectralWaveBoundaryConditions(
            nmax=0.8,
            rt=3600.0,
            dtbc=1.0,
            random=True,
        )
    )
    assert physics.wbc.nmax == 0.8
    assert physics.wbc.rt == 3600.0
    assert physics.wbc.dtbc == 1.0


def test_physics_accepts_non_spectral_wbc():
    """Test that Physics accepts NonSpectralWaveBoundaryConditions."""
    physics = Physics(
        wbc=NonSpectralWaveBoundaryConditions(
            nmax=0.8,
            Hrms=2.0,
            Trep=12.0,
            dir0=285.0,
        )
    )
    assert physics.wbc.nmax == 0.8
    assert physics.wbc.Hrms == 2.0
    assert physics.wbc.Trep == 12.0


def test_spectral_validation_ranges():
    """Test validation ranges for spectral parameters."""
    with pytest.raises(ValueError):
        SpectralWaveBoundaryConditions(rt=1000.0)  # Below minimum 1200.0
    with pytest.raises(ValueError):
        SpectralWaveBoundaryConditions(dtbc=2.5)  # Above maximum 2.0
    with pytest.raises(ValueError):
        SpectralWaveBoundaryConditions(fcutoff=50.0)  # Above maximum 40.0


def test_non_spectral_validation_ranges():
    """Test validation ranges for non-spectral parameters."""
    with pytest.raises(ValueError):
        NonSpectralWaveBoundaryConditions(Hrms=15.0)  # Above maximum 10.0
    with pytest.raises(ValueError):
        NonSpectralWaveBoundaryConditions(Trep=0.5)  # Below minimum 1.0
    with pytest.raises(ValueError):
        NonSpectralWaveBoundaryConditions(m=1)  # Below minimum 2
    with pytest.raises(ValueError):
        NonSpectralWaveBoundaryConditions(Tlong=15.0)  # Below minimum 20.0
    # Note: taper is now in base class, not non-spectral specific


def test_general_parameter_validation():
    """Test validation ranges for general parameters (in base class)."""
    with pytest.raises(ValueError):
        WaveBoundaryConditions(nmax=1.5)  # Above maximum 1.0
    with pytest.raises(ValueError):
        WaveBoundaryConditions(wbcevarreduce=1.5)  # Above maximum 1.0
    with pytest.raises(ValueError):
        WaveBoundaryConditions(swkhmin=-0.02)  # Below minimum -0.01
    with pytest.raises(ValueError):
        WaveBoundaryConditions(taper=1500.0)  # Above maximum 1000.0


def test_serialization_base():
    """Test serialization of base WaveBoundaryConditions."""
    wbc = WaveBoundaryConditions(nmax=0.8, wbcScaleEnergy=True, taper=100.0)
    params = wbc.model_dump(exclude_none=True)
    assert params == {"nmax": 0.8, "wbcScaleEnergy": True, "taper": 100.0}


def test_serialization_spectral():
    """Test serialization of SpectralWaveBoundaryConditions."""
    wbc = SpectralWaveBoundaryConditions(
        nmax=0.8,
        rt=3600.0,
        dtbc=1.0,
        random=True,
    )
    params = wbc.model_dump(exclude_none=True)
    assert params == {
        "nmax": 0.8,
        "rt": 3600.0,
        "dtbc": 1.0,
        "random": True,
    }


def test_serialization_non_spectral():
    """Test serialization of NonSpectralWaveBoundaryConditions."""
    wbc = NonSpectralWaveBoundaryConditions(
        nmax=0.8,
        Hrms=2.0,
        Trep=12.0,
        dir0=285.0,
    )
    params = wbc.model_dump(exclude_none=True)
    assert params == {
        "nmax": 0.8,
        "Hrms": 2.0,
        "Trep": 12.0,
        "dir0": 285.0,
    }
