"""Tests for the XBeach Physics component."""

import pytest
import logging
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Stationary, Surfbeat, Nonh


# =====================================================================================
# Basic instantiation tests
# =====================================================================================
def test_physics_default():
    """Test Physics with default values."""
    physics = Physics()
    assert physics.model_type == "physics"
    assert physics.wavemodel is None
    assert physics.advection is None
    assert physics.flow is None
    assert physics.morphology is None


def test_physics_with_wavemodel():
    """Test Physics with wavemodel specified."""
    physics = Physics(wavemodel=Stationary())
    assert isinstance(physics.wavemodel, Stationary)

    physics = Physics(wavemodel=Surfbeat())
    assert isinstance(physics.wavemodel, Surfbeat)

    physics = Physics(wavemodel=Nonh(), swave=False)
    assert isinstance(physics.wavemodel, Nonh)


def test_physics_with_boolean_switches():
    """Test Physics with boolean switches."""
    physics = Physics(
        morphology=True,
        sedtrans=True,
        flow=True,
        swave=True,
    )
    assert physics.morphology is True
    assert physics.sedtrans is True
    assert physics.flow is True
    assert physics.swave is True


def test_physics_disable_processes():
    """Test Physics with processes disabled."""
    physics = Physics(
        morphology=False,
        sedtrans=False,
        wind=False,
    )
    assert physics.morphology is False
    assert physics.sedtrans is False
    assert physics.wind is False


# =====================================================================================
# Params property tests
# =====================================================================================
def test_params_empty():
    """Test params property with empty Physics."""
    physics = Physics()
    params = physics.params
    assert len(params) == 0


def test_params_with_wavemodel():
    """Test params property with wavemodel."""
    physics = Physics(wavemodel=Surfbeat())
    params = physics.params
    assert params["wavemodel"] == "surfbeat"


def test_params_with_boolean_switches():
    """Test params property with boolean switches."""
    physics = Physics(
        morphology=True,
        sedtrans=True,
        flow=False,
        wind=False,
    )
    params = physics.params
    assert params["morphology"] == 1
    assert params["sedtrans"] == 1
    assert params["flow"] == 0
    assert params["wind"] == 0


def test_params_excludes_none():
    """Test that params excludes None values."""
    physics = Physics(
        morphology=True,
        sedtrans=None,
        flow=None,
    )
    params = physics.params
    assert "morphology" in params
    assert "sedtrans" not in params
    assert "flow" not in params


def test_params_all_boolean_fields():
    """Test params with all boolean fields set."""
    physics = Physics(
        advection=True,
        avalanching=True,
        cyclic=False,
        flow=True,
        gwflow=False,
        lwave=True,
        morphology=True,
        nonh=False,
        q3d=False,
        sedtrans=True,
        setbathy=False,
        ships=False,
        single_dir=True,
        snells=False,
        swave=True,
        swrunup=False,
        vegetation=False,
        viscosity=True,
        wind=True,
    )
    params = physics.params

    # All should be serialized to int
    assert params["advection"] == 1
    assert params["avalanching"] == 1
    assert params["cyclic"] == 0
    assert params["flow"] == 1
    assert params["gwflow"] == 0
    assert params["lwave"] == 1
    assert params["morphology"] == 1
    assert params["nonh"] == 0
    assert params["q3d"] == 0
    assert params["sedtrans"] == 1
    assert params["setbathy"] == 0
    assert params["ships"] == 0
    assert params["single_dir"] == 1
    assert params["snells"] == 0
    assert params["swave"] == 1
    assert params["swrunup"] == 0
    assert params["vegetation"] == 0
    assert params["viscosity"] == 1
    assert params["wind"] == 1


# =====================================================================================
# get() method tests
# =====================================================================================
def test_get_method_without_destdir():
    """Test get() method with destdir."""
    from pathlib import Path
    physics = Physics(
        morphology=True,
        sedtrans=True,
        wavemodel=Surfbeat(),
    )
    params = physics.get(Path("/tmp"))
    assert params["morphology"] == 1
    assert params["sedtrans"] == 1
    assert params["wavemodel"] == "surfbeat"


def test_get_method_with_destdir(tmp_path):
    """Test get() method with destdir (should be ignored)."""
    physics = Physics(
        morphology=True,
        flow=True,
    )
    params = physics.get(tmp_path)
    assert params["morphology"] == 1
    assert params["flow"] == 1

    # No files should be created in destdir
    assert len(list(tmp_path.iterdir())) == 0


def test_get_method_preserves_all_params():
    """Test that get() method preserves all parameters."""
    from pathlib import Path
    physics = Physics(
        wavemodel=Nonh(),
        morphology=True,
        sedtrans=True,
        flow=True,
        swave=False,  # Must be False when wavemodel=Nonh
        wind=False,
        nonh=True,
    )
    params = physics.get(Path("/tmp"))
    assert params["wavemodel"] == "nonh"
    assert params["morphology"] == 1
    assert params["sedtrans"] == 1
    assert params["flow"] == 1
    assert params["swave"] == 0
    assert params["wind"] == 0
    assert params["nonh"] == 1


# =====================================================================================
# Complex configuration tests
# =====================================================================================
def test_physics_morphological_simulation():
    """Test Physics configuration for morphological simulation."""
    physics = Physics(
        wavemodel=Surfbeat(),
        morphology=True,
        sedtrans=True,
        avalanching=True,
        flow=True,
        swave=True,
        lwave=True,
    )
    params = physics.params

    assert params["wavemodel"] == "surfbeat"
    assert params["morphology"] == 1
    assert params["sedtrans"] == 1
    assert params["avalanching"] == 1
    assert params["flow"] == 1
    assert params["swave"] == 1
    assert params["lwave"] == 1


def test_physics_hydrodynamic_only_simulation():
    """Test Physics configuration for hydrodynamic-only simulation."""
    physics = Physics(
        wavemodel=Surfbeat(),
        morphology=False,
        sedtrans=False,
        flow=True,
        swave=True,
    )
    params = physics.params

    assert params["wavemodel"] == "surfbeat"
    assert params["morphology"] == 0
    assert params["sedtrans"] == 0
    assert params["flow"] == 1
    assert params["swave"] == 1


def test_physics_nonhydrostatic_simulation():
    """Test Physics configuration for non-hydrostatic simulation."""
    physics = Physics(
        wavemodel=Nonh(),
        nonh=True,
        swave=False,  # Must be explicitly False when wavemodel=Nonh
        flow=True,
        morphology=False,
        sedtrans=False,
    )
    params = physics.params

    assert params["wavemodel"] == "nonh"
    assert params["nonh"] == 1
    assert params["swave"] == 0
    assert params["flow"] == 1
    assert params["morphology"] == 0
    assert params["sedtrans"] == 0


def test_physics_stationary_simulation():
    """Test Physics configuration for stationary simulation."""
    physics = Physics(
        wavemodel=Stationary(),
        flow=False,
        morphology=False,
        sedtrans=False,
    )
    params = physics.params

    assert params["wavemodel"] == "stationary"
    assert params["flow"] == 0
    assert params["morphology"] == 0
    assert params["sedtrans"] == 0


def test_physics_with_vegetation():
    """Test Physics configuration with vegetation."""
    physics = Physics(
        vegetation=True,
        flow=True,
        swave=True,
    )
    params = physics.params

    assert params["vegetation"] == 1
    assert params["flow"] == 1
    assert params["swave"] == 1


def test_physics_with_groundwater():
    """Test Physics configuration with groundwater flow."""
    physics = Physics(
        gwflow=True,
        flow=True,
        morphology=True,
    )
    params = physics.params

    assert params["gwflow"] == 1
    assert params["flow"] == 1
    assert params["morphology"] == 1


def test_physics_with_ships():
    """Test Physics configuration with ship waves."""
    physics = Physics(
        ships=True,
        flow=True,
        swave=True,
    )
    params = physics.params

    assert params["ships"] == 1
    assert params["flow"] == 1
    assert params["swave"] == 1


# =====================================================================================
# Edge cases
# =====================================================================================
def test_physics_minimal_configuration():
    """Test Physics with minimal configuration."""
    physics = Physics()
    params = physics.params

    # Should be empty dict
    assert params == {}


def test_physics_single_parameter():
    """Test Physics with single parameter."""
    physics = Physics(morphology=True)
    params = physics.params
    assert len(params) == 1
    assert params["morphology"] == 1


def test_physics_mixed_true_false():
    """Test Physics with mixed True/False values."""
    physics = Physics(
        morphology=True,
        sedtrans=False,
        flow=True,
        wind=False,
        swave=True,
        avalanching=False,
    )
    params = physics.params

    assert params["morphology"] == 1
    assert params["sedtrans"] == 0
    assert params["flow"] == 1
    assert params["wind"] == 0
    assert params["swave"] == 1
    assert params["avalanching"] == 0


# =====================================================================================
# Serialization tests
# =====================================================================================
def test_bool_serialization_to_int():
    """Test that boolean values are correctly serialized to integers."""
    physics = Physics(
        morphology=True,
        sedtrans=False,
    )
    params = physics.params

    # Check that booleans are converted to integers
    assert params["morphology"] == 1
    assert params["sedtrans"] == 0
    assert isinstance(params["morphology"], int)
    assert isinstance(params["sedtrans"], int)


def test_wavemodel_serialization():
    """Test that wavemodel component is serialized to string."""
    test_cases = [
        (Stationary(), "stationary"),
        (Surfbeat(), "surfbeat"),
        (Nonh(), "nonh"),
    ]
    for component, expected_str in test_cases:
        physics = Physics(
            wavemodel=component, swave=False if isinstance(component, Nonh) else None
        )
        params = physics.params
        assert isinstance(params["wavemodel"], str)
        assert params["wavemodel"] == expected_str


# =====================================================================================
# Validation tests
# =====================================================================================
def test_wavemodel_invalid_value():
    """Test that invalid wavemodel values raise an error."""
    with pytest.raises(Exception):  # Will raise validation error
        Physics(wavemodel="invalid")


def test_boolean_fields_accept_bool_only():
    """Test that boolean fields accept boolean values."""
    # Valid boolean values
    physics = Physics(morphology=True)
    assert physics.morphology is True

    physics = Physics(morphology=False)
    assert physics.morphology is False

    # Pydantic v2 allows type coercion for booleans
    # Integer 1 is coerced to True
    physics = Physics(morphology=1)
    assert physics.morphology is True

    # Integer 0 is coerced to False
    physics = Physics(morphology=0)
    assert physics.morphology is False


# =====================================================================================
# Validator tests for default enabled processes
# =====================================================================================
def test_log_default_enabled_processes(caplog):
    """Test that DEBUG messages are logged for default-enabled processes not set."""
    with caplog.at_level(logging.DEBUG):
        Physics()

    # Check that DEBUG messages are logged for all default-enabled processes
    default_enabled_params = [
        "advection",
        "avalanching",
        "flow",
        "lwave",
        "sedtrans",
        "single_dir",
        "swave",
        "viscosity",
        "wind",
    ]

    for param in default_enabled_params:
        assert param in caplog.text
        assert "not explicitly set" in caplog.text
        assert "will be ENABLED by XBeach default" in caplog.text


def test_no_log_when_default_enabled_process_is_set(caplog):
    """Test that no DEBUG message is logged when default-enabled process is explicitly set."""
    with caplog.at_level(logging.DEBUG):
        Physics(sedtrans=True, flow=False, swave=True)

    # sedtrans, flow, and swave should not trigger DEBUG logs since they're explicitly set
    # But other default-enabled params should still log
    assert (
        "sedtrans" not in caplog.text
        or "sedtrans) not explicitly set" not in caplog.text
    )
    assert "flow" not in caplog.text or "flow) not explicitly set" not in caplog.text
    assert "swave" not in caplog.text or "swave) not explicitly set" not in caplog.text

    # Other default-enabled params should still log
    assert "advection" in caplog.text
    assert "avalanching" in caplog.text


def test_no_log_for_default_disabled_processes(caplog):
    """Test that no DEBUG messages are logged for processes that default to disabled."""
    with caplog.at_level(logging.DEBUG):
        Physics()

    # These parameters default to 0 (disabled) in XBeach, so no DEBUG should be logged
    default_disabled_params = [
        "cyclic",
        "gwflow",
        "morphology",
        "nonh",
        "q3d",
        "setbathy",
        "ships",
        "snells",
        "swrunup",
        "vegetation",
    ]

    for param in default_disabled_params:
        # These should not appear in the "not explicitly set" messages
        assert f"{param}) not explicitly set" not in caplog.text


# =====================================================================================
# Cross-parameter validation tests
# =====================================================================================
def test_nonh_with_swave_true_raises_error():
    """Test that setting nonh=True with swave=True raises a validation error."""
    with pytest.raises(ValueError) as exc_info:
        Physics(nonh=True, swave=True)

    assert "swave' cannot be True when non-hydrostatic mode is enabled" in str(
        exc_info.value
    )
    assert "Set swave=False explicitly" in str(exc_info.value)


def test_nonh_with_swave_none_raises_error():
    """Test that setting nonh=True without setting swave raises a validation error."""
    with pytest.raises(ValueError) as exc_info:
        Physics(nonh=True)

    assert "swave' must be explicitly set to False when non-hydrostatic" in str(
        exc_info.value
    )
    assert "XBeach would enable swave by default" in str(exc_info.value)
    assert "Please set swave=False explicitly" in str(exc_info.value)


def test_nonh_with_swave_false_is_valid():
    """Test that setting nonh=True with swave=False is valid."""
    physics = Physics(nonh=True, swave=False)
    assert physics.nonh is True
    assert physics.swave is False

    params = physics.params
    assert params["nonh"] == 1
    assert params["swave"] == 0


def test_swave_true_without_nonh_is_valid():
    """Test that swave=True is valid when nonh is not True."""
    # nonh=False
    physics1 = Physics(swave=True, nonh=False)
    assert physics1.swave is True
    assert physics1.nonh is False

    # nonh=None (default)
    physics2 = Physics(swave=True)
    assert physics2.swave is True
    assert physics2.nonh is None


# =====================================================================================
# Parameter component tests
# =====================================================================================
def test_nonh_with_nhq3d_parameter():
    """Test that Nonh component can specify nhq3d parameter."""
    physics = Physics(wavemodel=Nonh(nhq3d=True), swave=False)
    params = physics.params

    assert params["wavemodel"] == "nonh"
    assert params["nhq3d"] == 1
    assert params["swave"] == 0


def test_nonh_without_nhq3d_parameter():
    """Test that Nonh component without nhq3d doesn't include it in params."""
    physics = Physics(wavemodel=Nonh(), swave=False)
    params = physics.params

    assert params["wavemodel"] == "nonh"
    assert "nhq3d" not in params
    assert params["swave"] == 0
