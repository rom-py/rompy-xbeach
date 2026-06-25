"""Tests for friction module."""

import pytest
from pathlib import Path
from rompy_xbeach.components.physics.friction import (
    Cf,
    Chezy,
    Manning,
    WhiteColebrook,
    WhiteColebrookGrainsize,
)
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat


# =============================================================================
# Cf (dimensionless friction coefficient) tests
# =============================================================================
def test_cf_with_coefficient():
    """Test Cf formulation with coefficient."""
    friction = Cf(bedfriccoef=0.005)
    params = friction.params
    # When used alone, only the coefficient is serialized
    assert params["bedfriccoef"] == 0.005
    assert "bedfriction" not in params  # Only appears when used in Physics


def test_cf_in_physics():
    """Test Cf formulation in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=Cf(bedfriccoef=0.005))
    params = physics.params
    assert params["bedfriction"] == "cf"
    assert params["bedfriccoef"] == 0.005


def test_cf_with_file(tmp_path):
    """Test Cf formulation with friction file."""
    # Create a dummy friction file
    fric_file = tmp_path / "friction.txt"
    fric_file.write_text("0.005 0.006 0.007\n")

    friction = Cf(bedfricfile={"source": str(fric_file)})
    params = friction.get(tmp_path)

    assert "bedfricfile" in params
    assert Path(tmp_path / params["bedfricfile"]).exists()


# =============================================================================
# Chezy tests
# =============================================================================
def test_chezy_with_coefficient():
    """Test Chezy formulation with coefficient."""
    friction = Chezy(bedfriccoef=55.0)
    params = friction.params
    assert params["bedfriccoef"] == 55.0


def test_chezy_in_physics():
    """Test Chezy formulation in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=Chezy(bedfriccoef=55.0))
    params = physics.params
    assert params["bedfriction"] == "chezy"
    assert params["bedfriccoef"] == 55.0


# =============================================================================
# Manning tests
# =============================================================================
def test_manning_with_coefficient():
    """Test Manning formulation with coefficient."""
    friction = Manning(bedfriccoef=0.02)
    params = friction.params
    assert params["bedfriccoef"] == 0.02


def test_manning_in_physics():
    """Test Manning formulation in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=Manning(bedfriccoef=0.02))
    params = physics.params
    assert params["bedfriction"] == "manning"
    assert params["bedfriccoef"] == 0.02


def test_manning_with_file(tmp_path):
    """Test Manning formulation with friction file."""
    fric_file = tmp_path / "manning.txt"
    fric_file.write_text("0.02 0.025 0.03\n")

    friction = Manning(bedfricfile={"source": str(fric_file)})
    params = friction.get(tmp_path)

    assert "bedfricfile" in params


# =============================================================================
# White-Colebrook tests
# =============================================================================
def test_white_colebrook_with_coefficient():
    """Test White-Colebrook formulation with k_s coefficient."""
    friction = WhiteColebrook(bedfriccoef=0.05)
    params = friction.params
    assert params["bedfriccoef"] == 0.05


def test_white_colebrook_in_physics():
    """Test White-Colebrook formulation in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=WhiteColebrook(bedfriccoef=0.05))
    params = physics.params
    assert params["bedfriction"] == "white-colebrook"
    assert params["bedfriccoef"] == 0.05


# =============================================================================
# White-Colebrook-Grainsize tests
# =============================================================================
def test_white_colebrook_grainsize():
    """Test White-Colebrook grain size formulation."""
    friction = WhiteColebrookGrainsize()
    params = friction.params
    # This formulation doesn't use bedfriccoef
    assert "bedfriccoef" not in params
    assert "bedfricfile" not in params


def test_white_colebrook_grainsize_in_physics():
    """Test White-Colebrook grain size formulation in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=WhiteColebrookGrainsize())
    params = physics.params
    assert params["bedfriction"] == "white-colebrook-grainsize"
    assert "bedfriccoef" not in params


# =============================================================================
# Parameter validation tests
# =============================================================================
def test_bedfriccoef_range_validation():
    """Test bedfriccoef parameter range validation."""
    # Valid values
    Cf(bedfriccoef=0.0)
    Cf(bedfriccoef=0.5)
    Chezy(bedfriccoef=100.0)  # Chezy can be larger

    # Invalid values
    with pytest.raises(ValueError):
        Cf(bedfriccoef=-0.1)


# =============================================================================
# Serialization tests
# =============================================================================
def test_friction_serialization_in_physics():
    """Test that friction models serialize correctly in Physics context."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=Manning(bedfriccoef=0.025))
    params = physics.params

    # Should have bedfriction as the discriminator value
    assert "bedfriction" in params
    assert params["bedfriction"] == "manning"
    # Should not have model_type in params
    assert "model_type" not in params


def test_friction_get_without_file(tmp_path):
    """Test get() method when no file is specified."""
    friction = Chezy(bedfriccoef=50.0)
    params = friction.get(tmp_path)

    assert params["bedfriccoef"] == 50.0
    assert "bedfricfile" not in params

    # No files should be created
    assert len(list(tmp_path.iterdir())) == 0


# =============================================================================
# Additional parameter tests
# =============================================================================
def test_manning_with_mincf():
    """Test Manning formulation with mincf parameter."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=Manning(bedfriccoef=0.02, mincf=0.001))
    params = physics.params
    assert params["bedfriction"] == "manning"
    assert params["bedfriccoef"] == 0.02
    assert params["mincf"] == 0.001


def test_white_colebrook_with_mincf():
    """Test White-Colebrook formulation with mincf parameter."""
    physics = Physics(wavemodel=Surfbeat(), bedfriction=WhiteColebrook(bedfriccoef=0.05, mincf=0.002))
    params = physics.params
    assert params["bedfriction"] == "white-colebrook"
    assert params["bedfriccoef"] == 0.05
    assert params["mincf"] == 0.002


def test_white_colebrook_grainsize_with_xbeachg_params():
    """Test White-Colebrook grain size with XBeach-G specific parameters."""
    physics = Physics(
        wavemodel=Surfbeat(),
        bedfriction=WhiteColebrookGrainsize(
            friction_acceleration="mccall",
            friction_infiltration=True,
            friction_turbulence=True,
        )
    )
    params = physics.params
    assert params["bedfriction"] == "white-colebrook-grainsize"
    assert params["friction_acceleration"] == "mccall"
    assert params["friction_infiltration"] == 1
    assert params["friction_turbulence"] == 1


def test_xbeachg_params_apply_to_all_formulations():
    """Test that XBeach-G friction parameters apply to all formulations."""
    # Test with Manning
    physics_manning = Physics(
        wavemodel=Surfbeat(),
        bedfriction=Manning(
            bedfriccoef=0.02,
            friction_acceleration="nielsen",
            friction_infiltration=True,
            friction_turbulence=True,
        )
    )
    params = physics_manning.params
    assert params["bedfriction"] == "manning"
    assert params["friction_acceleration"] == "nielsen"
    assert params["friction_infiltration"] == 1
    assert params["friction_turbulence"] == 1

    # Test with Chezy
    physics_chezy = Physics(
        wavemodel=Surfbeat(),
        bedfriction=Chezy(
            bedfriccoef=55.0,
            friction_acceleration="mccall",
        )
    )
    params = physics_chezy.params
    assert params["bedfriction"] == "chezy"
    assert params["friction_acceleration"] == "mccall"

    # Test with WhiteColebrook
    physics_wc = Physics(
        wavemodel=Surfbeat(),
        bedfriction=WhiteColebrook(
            bedfriccoef=0.05,
            friction_infiltration=True,
        )
    )
    params = physics_wc.params
    assert params["bedfriction"] == "white-colebrook"
    assert params["friction_infiltration"] == 1
