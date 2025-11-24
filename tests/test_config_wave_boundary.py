"""Tests for Config with wave_boundary field."""

import pytest
from pathlib import Path
from rompy_xbeach.config import Config
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


def test_config_accepts_spectral_wave_boundary():
    """Test that Config accepts SpectralWaveBoundary."""
    # Note: We can't fully instantiate Config without grid, bathy, etc.
    # Just test that the field accepts the right type
    from pydantic import ValidationError
    
    # This should work (will fail on other required fields, but that's OK)
    try:
        config = Config(
            wave_boundary=SpectralWaveBoundary(
                wbctype="jons",
                bcfile="jonswap.txt",
                wbc=SpectralWaveBoundaryConditions(nmax=0.8),
            )
        )
    except ValidationError as e:
        # Should fail on missing grid, bathy (required fields)
        # wave_boundary should NOT be in the validation errors
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        assert "wave_boundary" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields


def test_config_accepts_non_spectral_wave_boundary():
    """Test that Config accepts NonSpectralWaveBoundary."""
    from pydantic import ValidationError
    
    try:
        config = Config(
            wave_boundary=NonSpectralWaveBoundary(
                wbctype="stat",
                wbc=NonSpectralWaveBoundaryConditions(
                    Hrms=2.0,
                    Trep=12.0,
                ),
            )
        )
    except ValidationError as e:
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        assert "wave_boundary" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields


def test_config_accepts_off_wave_boundary():
    """Test that Config accepts OffWaveBoundary."""
    from pydantic import ValidationError
    
    try:
        config = Config(
            wave_boundary=OffWaveBoundary()
        )
    except ValidationError as e:
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        assert "wave_boundary" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields


def test_config_wave_boundary_discriminator():
    """Test that wave_boundary uses discriminator correctly."""
    from pydantic import ValidationError
    
    # Should work with correct model_type
    try:
        config = Config(
            wave_boundary={
                "model_type": "spectral",
                "wbctype": "jons",
                "bcfile": "test.txt",
            }
        )
    except ValidationError as e:
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        # Should not have discriminator errors
        assert "wave_boundary" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields


def test_config_input_optional():
    """Test that input field is now optional."""
    from pydantic import ValidationError
    
    # input should be optional now
    try:
        config = Config(
            wave_boundary=OffWaveBoundary()
        )
    except ValidationError as e:
        errors = e.errors()
        error_fields = [err["loc"][0] for err in errors]
        # Should fail on grid, bathy (required), but NOT on input (optional)
        assert "input" not in error_fields
        assert "grid" in error_fields or "bathy" in error_fields
