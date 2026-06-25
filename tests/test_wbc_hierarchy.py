"""Tests for boundary condition parameter hierarchy."""

import pytest
from rompy_xbeach.data.boundary.base import (
    WaveBoundaryParams,
    SpectralWaveBoundaryParams,
)
from rompy_xbeach.data.boundary.nonspectral import (
    BoundaryStat,
    BoundaryBichrom,
)
from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
)


def test_base_wave_boundary_conditions():
    """Test base WaveBoundaryParams with general parameters only."""
    wbc = WaveBoundaryParams(
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
    """Test SpectralWaveBoundaryParams with both general and spectral parameters."""
    wbc = SpectralWaveBoundaryParams(
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
    """Test BoundaryStat with both general and non-spectral parameters."""
    wbc = BoundaryStat(
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
    """Test BoundaryBichrom with bichromatic-specific parameters."""
    wbc = BoundaryBichrom(
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
    """Test that SpectralWaveBoundaryParams inherits from WaveBoundaryParams."""
    assert issubclass(SpectralWaveBoundaryParams, WaveBoundaryParams)
    wbc = SpectralWaveBoundaryParams()
    assert isinstance(wbc, WaveBoundaryParams)
    assert isinstance(wbc, SpectralWaveBoundaryParams)


def test_non_spectral_inherits_from_base():
    """Test that BoundaryStat and BoundaryBichrom inherit from WaveBoundaryParams."""
    assert issubclass(BoundaryStat, WaveBoundaryParams)
    assert issubclass(BoundaryBichrom, WaveBoundaryParams)
    wbc = BoundaryStat(Hrms=1.0, Trep=10.0)
    assert isinstance(wbc, WaveBoundaryParams)


# Note: Physics.wbc field has been removed - wave boundary parameters
# are now handled through Config.wave_boundary or input.wave.wbc


def test_spectral_validation_ranges():
    """Test validation ranges for spectral parameters."""
    with pytest.raises(ValueError):
        SpectralWaveBoundaryParams(rt=1000.0)  # Below minimum 1200.0
    with pytest.raises(ValueError):
        SpectralWaveBoundaryParams(dtbc=2.5)  # Above maximum 2.0
    with pytest.raises(ValueError):
        SpectralWaveBoundaryParams(fcutoff=50.0)  # Above maximum 40.0


def test_non_spectral_validation_ranges():
    """Test validation ranges for non-spectral parameters."""
    with pytest.raises(ValueError):
        BoundaryStat(Hrms=15.0, Trep=10.0)  # Above maximum 10.0
    with pytest.raises(ValueError):
        BoundaryStat(Hrms=1.0, Trep=0.5)  # Below minimum 1.0
    with pytest.raises(ValueError):
        BoundaryStat(Hrms=1.0, Trep=10.0, m=1)  # Below minimum 2
    with pytest.raises(ValueError):
        BoundaryBichrom(Hrms=1.0, Trep=10.0, Tlong=15.0)  # Below minimum 20.0
    # Note: taper is now in base class, not non-spectral specific


def test_general_parameter_validation():
    """Test validation ranges for general parameters (in base class)."""
    with pytest.raises(ValueError):
        WaveBoundaryParams(nmax=1.5)  # Above maximum 1.0
    with pytest.raises(ValueError):
        WaveBoundaryParams(wbcevarreduce=1.5)  # Above maximum 1.0
    with pytest.raises(ValueError):
        WaveBoundaryParams(swkhmin=-0.02)  # Below minimum -0.01
    with pytest.raises(ValueError):
        WaveBoundaryParams(taper=1500.0)  # Above maximum 1000.0


def test_serialization_base():
    """Test serialization of base WaveBoundaryParams."""
    wbc = WaveBoundaryParams(nmax=0.8, wbcScaleEnergy=True, taper=100.0)
    params = wbc.model_dump(exclude_none=True)
    assert params == {"nmax": 0.8, "wbcScaleEnergy": True, "taper": 100.0}


def test_serialization_spectral():
    """Test serialization of SpectralWaveBoundaryParams."""
    wbc = SpectralWaveBoundaryParams(
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
    """Test serialization of BoundaryStat."""
    wbc = BoundaryStat(
        nmax=0.8,
        Hrms=2.0,
        Trep=12.0,
        dir0=285.0,
    )
    params = wbc.model_dump(exclude_none=True)
    assert params == {
        "id": "stat",
        "model_type": "stat",
        "nmax": 0.8,
        "Hrms": 2.0,
        "Trep": 12.0,
        "dir0": 285.0,
        "m": 10,  # default value
    }


# ======================================================================================
# Flow Boundary Conditions Tests
# ======================================================================================


def test_flow_boundary_conditions():
    """Test FlowBoundaryConditions with all boundary types."""
    fbc = FlowBoundaryConditions(
        front="abs_2d",
        back="wall",
        left="neumann",
        right="neumann_v",
        lateralwave="wavecrest",
    )
    assert fbc.front == "abs_2d"
    assert fbc.back == "wall"
    assert fbc.left == "neumann"
    assert fbc.right == "neumann_v"
    assert fbc.lateralwave == "wavecrest"


def test_flow_boundary_conditions_with_numerics():
    """Test FlowBoundaryConditions with numerical parameters."""
    fbc = FlowBoundaryConditions(
        front="abs_1d",
        nc=50,
        highcomp=True,
    )
    assert fbc.front == "abs_1d"
    assert fbc.nc == 50
    assert fbc.highcomp is True


def test_flow_boundary_conditions_serialization():
    """Test FlowBoundaryConditions serialization excludes None values."""
    fbc = FlowBoundaryConditions(
        front="abs_2d",
        left="neumann",
    )
    params = fbc.model_dump(exclude_none=True)
    assert params == {
        "front": "abs_2d",
        "left": "neumann",
    }
    # Verify None values are excluded
    assert "back" not in params
    assert "right" not in params
    assert "lateralwave" not in params
    assert "nc" not in params
    assert "highcomp" not in params


def test_flow_boundary_conditions_validation():
    """Test FlowBoundaryConditions validates boundary type options."""
    # Valid options should work
    fbc = FlowBoundaryConditions(front="nonh_1d")
    assert fbc.front == "nonh_1d"

    fbc = FlowBoundaryConditions(front="waveflume")
    assert fbc.front == "waveflume"

    # Invalid options should fail
    with pytest.raises(ValueError):
        FlowBoundaryConditions(front="invalid")

    with pytest.raises(ValueError):
        FlowBoundaryConditions(left="invalid")


# ======================================================================================
# Tide Boundary Conditions Tests
# ======================================================================================


def test_tide_boundary_conditions():
    """Test TideBoundaryConditions with basic parameters."""
    tbc = TideBoundaryConditions(
        tideloc=0,
        zs0=0.5,
    )
    assert tbc.tideloc == 0
    assert tbc.zs0 == 0.5


def test_tide_boundary_conditions_with_tideloc_2():
    """Test TideBoundaryConditions with tideloc=2 and paulrevere."""
    tbc = TideBoundaryConditions(
        tideloc=2,
        tidetype="velocity",
        paulrevere="land",
    )
    assert tbc.tideloc == 2
    assert tbc.tidetype == "velocity"
    assert tbc.paulrevere == "land"


def test_tide_boundary_conditions_all_tidetypes():
    """Test all tidetype options."""
    for tidetype in ["instant", "velocity", "hybrid"]:
        tbc = TideBoundaryConditions(tidetype=tidetype)
        assert tbc.tidetype == tidetype


def test_tide_boundary_conditions_serialization():
    """Test TideBoundaryConditions serialization excludes None values."""
    tbc = TideBoundaryConditions(
        tideloc=1,
        zs0=0.0,
    )
    params = tbc.model_dump(exclude_none=True)
    assert params == {
        "tideloc": 1,
        "zs0": 0.0,
    }
    # Verify None values are excluded
    assert "tidetype" not in params
    assert "paulrevere" not in params


def test_tide_boundary_conditions_validation():
    """Test TideBoundaryConditions validates options."""
    # Valid tideloc values
    for loc in [0, 1, 2, 4]:
        tbc = TideBoundaryConditions(tideloc=loc)
        assert tbc.tideloc == loc

    # Invalid tideloc should fail
    with pytest.raises(ValueError):
        TideBoundaryConditions(tideloc=3)

    # Invalid tidetype should fail
    with pytest.raises(ValueError):
        TideBoundaryConditions(tidetype="invalid")

    # Invalid paulrevere should fail
    with pytest.raises(ValueError):
        TideBoundaryConditions(paulrevere="invalid")
