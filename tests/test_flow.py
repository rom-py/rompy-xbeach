"""Tests for flow module."""

import pytest
from rompy_xbeach.components.physics.friction import HorizontalViscosity


def test_horizontal_viscosity_smagorinsky_default():
    """Test Smagorinsky model with custom constant."""
    visc = HorizontalViscosity(nuh=0.15)
    params = visc.params
    assert params["nuh"] == 0.15
    assert "smag" not in params  # Default, not serialized


def test_horizontal_viscosity_constant():
    """Test constant viscosity (Smagorinsky disabled)."""
    visc = HorizontalViscosity(smag=False, nuh=0.5)
    params = visc.params
    assert params["smag"] == 0
    assert params["nuh"] == 0.5


def test_horizontal_viscosity_with_longshore_enhancement():
    """Test with longshore viscosity enhancement factor."""
    visc = HorizontalViscosity(nuh=0.1, nuhv=2.0)
    params = visc.params
    assert params["nuh"] == 0.1
    assert params["nuhv"] == 2.0


def test_horizontal_viscosity_all_parameters():
    """Test with all parameters specified."""
    visc = HorizontalViscosity(smag=True, nuh=0.12, nuhv=1.5)
    params = visc.params
    assert params["smag"] == 1
    assert params["nuh"] == 0.12
    assert params["nuhv"] == 1.5


def test_horizontal_viscosity_disabled_smagorinsky_no_nuh():
    """Test disabling Smagorinsky without specifying nuh (should work)."""
    visc = HorizontalViscosity(smag=False)
    params = visc.params
    assert params["smag"] == 0
    assert "nuh" not in params


def test_horizontal_viscosity_nuh_range():
    """Test nuh parameter range validation."""
    # Valid values
    HorizontalViscosity(nuh=0.0)
    HorizontalViscosity(nuh=0.5)
    HorizontalViscosity(nuh=1.0)

    # Invalid values
    with pytest.raises(ValueError):
        HorizontalViscosity(nuh=-0.1)
    with pytest.raises(ValueError):
        HorizontalViscosity(nuh=1.1)


def test_horizontal_viscosity_nuhv_range():
    """Test nuhv parameter range validation."""
    # Valid values
    HorizontalViscosity(nuhv=1.0)
    HorizontalViscosity(nuhv=10.0)
    HorizontalViscosity(nuhv=20.0)

    # Invalid values
    with pytest.raises(ValueError):
        HorizontalViscosity(nuhv=0.5)
    with pytest.raises(ValueError):
        HorizontalViscosity(nuhv=21.0)


def test_horizontal_viscosity_empty():
    """Test creating HorizontalViscosity with no parameters (all defaults)."""
    visc = HorizontalViscosity()
    params = visc.params
    assert params == {}
