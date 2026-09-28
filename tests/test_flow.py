"""Tests for flow module."""

import pytest
from rompy_xbeach.components.physics.friction import Viscosity, HorizontalViscosity


def test_viscosity_smagorinsky_default():
    """Test Smagorinsky model with custom constant."""
    visc = Viscosity(nuh=0.15)
    params = visc.params
    assert params["viscosity"] == 1  # Always enabled when using Viscosity class
    assert params["nuh"] == 0.15
    assert "smag" not in params  # Default, not serialized


def test_viscosity_constant():
    """Test constant viscosity (Smagorinsky disabled)."""
    visc = Viscosity(smag=False, nuh=0.5)
    params = visc.params
    assert params["viscosity"] == 1
    assert params["smag"] == 0
    assert params["nuh"] == 0.5


def test_viscosity_with_longshore_enhancement():
    """Test with longshore viscosity enhancement factor."""
    visc = Viscosity(nuh=0.1, nuhv=2.0)
    params = visc.params
    assert params["viscosity"] == 1
    assert params["nuh"] == 0.1
    assert params["nuhv"] == 2.0


def test_viscosity_all_parameters():
    """Test with all parameters specified."""
    visc = Viscosity(smag=True, nuh=0.12, nuhv=1.5)
    params = visc.params
    assert params["viscosity"] == 1
    assert params["smag"] == 1
    assert params["nuh"] == 0.12
    assert params["nuhv"] == 1.5


def test_viscosity_disabled_smagorinsky_no_nuh():
    """Test disabling Smagorinsky without specifying nuh (should work)."""
    visc = Viscosity(smag=False)
    params = visc.params
    assert params["viscosity"] == 1
    assert params["smag"] == 0
    assert "nuh" not in params


def test_viscosity_nuh_range():
    """Test nuh parameter range validation."""
    # Valid values
    Viscosity(nuh=0.0)
    Viscosity(nuh=0.5)
    Viscosity(nuh=1.0)

    # Invalid values
    with pytest.raises(ValueError):
        Viscosity(nuh=-0.1)
    with pytest.raises(ValueError):
        Viscosity(nuh=1.1)


def test_viscosity_nuhv_range():
    """Test nuhv parameter range validation."""
    # Valid values
    Viscosity(nuhv=1.0)
    Viscosity(nuhv=10.0)
    Viscosity(nuhv=20.0)

    # Invalid values
    with pytest.raises(ValueError):
        Viscosity(nuhv=0.5)
    with pytest.raises(ValueError):
        Viscosity(nuhv=21.0)


def test_viscosity_empty():
    """Test creating Viscosity with no parameters (all defaults)."""
    visc = Viscosity()
    params = visc.params
    # Only viscosity=1 is set (the switch), other params use XBeach defaults
    assert params == {"viscosity": 1}


def test_horizontal_viscosity_alias():
    """Test HorizontalViscosity is an alias for Viscosity."""
    assert HorizontalViscosity is Viscosity
