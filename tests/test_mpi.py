"""Tests for MPI component."""

import pytest
from rompy_xbeach.components.mpi import Mpi


def test_mpi_auto_mode():
    """Test Mpi with auto mode."""
    mpi = Mpi(mpiboundary="auto")
    params = mpi.params
    assert params["mpiboundary"] == "auto"
    assert "mmpi" not in params
    assert "nmpi" not in params


def test_mpi_x_mode():
    """Test Mpi with x-direction subdivision."""
    mpi = Mpi(mpiboundary="x")
    params = mpi.params
    assert params["mpiboundary"] == "x"


def test_mpi_y_mode():
    """Test Mpi with y-direction subdivision."""
    mpi = Mpi(mpiboundary="y")
    params = mpi.params
    assert params["mpiboundary"] == "y"


def test_mpi_manual_mode():
    """Test Mpi with manual domain decomposition."""
    mpi = Mpi(
        mpiboundary="man",
        mmpi=4,
        nmpi=8,
    )
    params = mpi.params
    assert params["mpiboundary"] == "man"
    assert params["mmpi"] == 4
    assert params["nmpi"] == 8


def test_mpi_manual_mode_without_mmpi_raises_error():
    """Test that manual mode without mmpi raises validation error."""
    with pytest.raises(ValueError) as exc_info:
        Mpi(mpiboundary="man", nmpi=8)

    assert "both mmpi and nmpi must be specified" in str(exc_info.value)


def test_mpi_manual_mode_without_nmpi_raises_error():
    """Test that manual mode without nmpi raises validation error."""
    with pytest.raises(ValueError) as exc_info:
        Mpi(mpiboundary="man", mmpi=4)

    assert "both mmpi and nmpi must be specified" in str(exc_info.value)


def test_mpi_manual_mode_without_both_raises_error():
    """Test that manual mode without mmpi and nmpi raises validation error."""
    with pytest.raises(ValueError) as exc_info:
        Mpi(mpiboundary="man")

    assert "both mmpi and nmpi must be specified" in str(exc_info.value)


def test_mpi_with_mmpi_nmpi_in_non_manual_mode():
    """Test that mmpi and nmpi can be specified in non-manual modes (they're just ignored)."""
    mpi = Mpi(
        mpiboundary="auto",
        mmpi=4,
        nmpi=8,
    )
    params = mpi.params
    assert params["mpiboundary"] == "auto"
    assert params["mmpi"] == 4
    assert params["nmpi"] == 8


def test_mpi_validation_ranges():
    """Test Mpi parameter validation ranges."""
    # Valid ranges
    mpi = Mpi(mpiboundary="man", mmpi=1, nmpi=100)
    assert mpi.mmpi == 1
    assert mpi.nmpi == 100

    # Invalid mmpi (too low)
    with pytest.raises(ValueError):
        Mpi(mpiboundary="man", mmpi=0, nmpi=4)

    # Invalid mmpi (too high)
    with pytest.raises(ValueError):
        Mpi(mpiboundary="man", mmpi=101, nmpi=4)

    # Invalid nmpi (too low)
    with pytest.raises(ValueError):
        Mpi(mpiboundary="man", mmpi=4, nmpi=0)

    # Invalid nmpi (too high)
    with pytest.raises(ValueError):
        Mpi(mpiboundary="man", mmpi=4, nmpi=101)


def test_mpi_empty():
    """Test Mpi with no parameters."""
    mpi = Mpi()
    params = mpi.params
    assert len(params) == 0
