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
        flow=True,
        swave=True,
        wind=True,
    )
    assert physics.flow is True
    assert physics.swave is True
    assert physics.wind is True


def test_physics_disable_processes():
    """Test Physics with processes disabled."""
    physics = Physics(
        wind=False,
        swave=False,
        flow=False,
    )
    assert physics.wind is False
    assert physics.swave is False
    assert physics.flow is False


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
        swave=True,
        lwave=True,
        flow=False,
        wind=False,
    )
    params = physics.params
    assert params["swave"] == 1
    assert params["lwave"] == 1
    assert params["flow"] == 0
    assert params["wind"] == 0


def test_params_excludes_none():
    """Test that params excludes None values."""
    physics = Physics(
        swave=True,
        lwave=None,
        flow=None,
    )
    params = physics.params
    assert "swave" in params
    assert "lwave" not in params
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
        setbathy=False,
        ships=False,
        single_dir=True,
        snells=False,
        swave=True,
        swrunup=False,
        viscosity=True,
        wci=False,
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
    assert params["setbathy"] == 0
    assert params["ships"] == 0
    assert params["single_dir"] == 1
    assert params["snells"] == 0
    assert params["swave"] == 1
    assert params["swrunup"] == 0
    assert params["viscosity"] == 1
    assert params["wci"] == 0
    assert params["wind"] == 1


# =====================================================================================
# get() method tests
# =====================================================================================
def test_get_method_without_destdir():
    """Test get() method without destdir."""
    from pathlib import Path

    physics = Physics(
        flow=True,
        swave=True,
        wavemodel=Surfbeat(),
    )
    params = physics.get(Path("/tmp"))
    assert params["flow"] == 1
    assert params["swave"] == 1
    assert params["wavemodel"] == "surfbeat"


def test_get_method_with_destdir(tmp_path):
    """Test get() method with destdir (should be ignored)."""
    physics = Physics(
        wind=True,
        flow=True,
    )
    params = physics.get(tmp_path)
    assert params["wind"] == 1
    assert params["flow"] == 1

    # No files should be created in destdir
    assert len(list(tmp_path.iterdir())) == 0


def test_get_method_preserves_all_params():
    """Test that get() method preserves all parameters."""
    from pathlib import Path

    physics = Physics(
        wavemodel=Nonh(),
        flow=True,
        swave=False,  # Must be False when wavemodel=Nonh
        wind=False,
        roller=False,
    )
    params = physics.get(Path("/tmp"))
    assert params["wavemodel"] == "nonh"
    assert params["flow"] == 1
    assert params["swave"] == 0
    assert params["wind"] == 0
    assert params["roller"] == 0


# =====================================================================================
# Complex configuration tests
# =====================================================================================
def test_physics_morphological_simulation():
    """Test Physics configuration for morphological simulation."""
    physics = Physics(
        wavemodel=Surfbeat(),
        avalanching=True,
        flow=True,
        swave=True,
        lwave=True,
    )
    params = physics.params

    assert params["wavemodel"] == "surfbeat"
    assert params["avalanching"] == 1
    assert params["flow"] == 1
    assert params["swave"] == 1
    assert params["lwave"] == 1


def test_physics_hydrodynamic_only_simulation():
    """Test Physics configuration for hydrodynamic-only simulation."""
    physics = Physics(
        wavemodel=Surfbeat(),
        flow=True,
        swave=True,
    )
    params = physics.params

    assert params["wavemodel"] == "surfbeat"
    assert params["flow"] == 1
    assert params["swave"] == 1


def test_physics_nonhydrostatic_simulation():
    """Test Physics configuration for non-hydrostatic simulation.

    Note: The legacy 'nonh' parameter is deprecated in XBeach. Use wavemodel=Nonh()
    which outputs 'wavemodel = nonh' in params.txt.
    """
    physics = Physics(
        wavemodel=Nonh(),
        swave=False,  # Must be explicitly False when wavemodel=Nonh
        flow=True,
    )
    params = physics.params

    assert params["wavemodel"] == "nonh"
    assert params["swave"] == 0
    assert params["flow"] == 1


def test_physics_stationary_simulation():
    """Test Physics configuration for stationary simulation."""
    physics = Physics(
        wavemodel=Stationary(),
        flow=False,
    )
    params = physics.params

    assert params["wavemodel"] == "stationary"
    assert params["flow"] == 0


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
    )
    params = physics.params

    assert params["gwflow"] == 1
    assert params["flow"] == 1


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
    physics = Physics(flow=True)
    params = physics.params
    assert len(params) == 1
    assert params["flow"] == 1


def test_physics_mixed_true_false():
    """Test Physics with mixed True/False values."""
    physics = Physics(
        flow=True,
        wind=False,
        swave=True,
        avalanching=False,
    )
    params = physics.params

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
        flow=True,
        wind=False,
    )
    params = physics.params

    # Check that booleans are converted to integers
    assert params["flow"] == 1
    assert params["wind"] == 0
    assert isinstance(params["flow"], int)
    assert isinstance(params["wind"], int)


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
    physics = Physics(flow=True)
    assert physics.flow is True

    physics = Physics(flow=False)
    assert physics.flow is False

    # Pydantic v2 allows type coercion for booleans
    # Integer 1 is coerced to True
    physics = Physics(flow=1)
    assert physics.flow is True

    # Integer 0 is coerced to False
    physics = Physics(flow=0)
    assert physics.flow is False


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
        "single_dir",
        "swave",
        "viscosity",
        "wci",
        "wind",
    ]

    for param in default_enabled_params:
        assert param in caplog.text
        assert "not explicitly set" in caplog.text
        assert "will be ENABLED by XBeach default" in caplog.text


def test_no_log_when_default_enabled_process_is_set(caplog):
    """Test that no DEBUG message is logged when default-enabled process is explicitly set."""
    with caplog.at_level(logging.DEBUG):
        Physics(flow=False, swave=True)

    # flow and swave should not trigger DEBUG logs since they're explicitly set
    # But other default-enabled params should still log
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
def test_nonh_wavemodel_with_swave_true_logs_warning(caplog):
    """Test that using Nonh wavemodel with swave=True logs a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        physics = Physics(wavemodel=Nonh(), swave=True)

    assert "swave' should not be True when using Nonh wavemodel" in caplog.text
    assert "XBeach requires swave=0" in caplog.text
    # Model is still created
    assert isinstance(physics.wavemodel, Nonh)
    assert physics.swave is True


def test_nonh_wavemodel_with_swave_none_logs_warning(caplog):
    """Test that using Nonh wavemodel without setting swave logs a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        physics = Physics(wavemodel=Nonh())

    assert "swave' should be explicitly set to False when using Nonh" in caplog.text
    assert "XBeach enables swave by default" in caplog.text
    # Model is still created
    assert isinstance(physics.wavemodel, Nonh)
    assert physics.swave is None


def test_nonh_wavemodel_with_swave_false_no_warning(caplog):
    """Test that using Nonh wavemodel with swave=False does not log a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        physics = Physics(wavemodel=Nonh(), swave=False)

    assert "swave" not in caplog.text
    assert isinstance(physics.wavemodel, Nonh)
    assert physics.swave is False

    params = physics.params
    assert params["wavemodel"] == "nonh"
    assert params["swave"] == 0


def test_swave_true_without_nonh_wavemodel_no_warning(caplog):
    """Test that swave=True without Nonh wavemodel does not log a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        # Surfbeat wavemodel
        physics1 = Physics(wavemodel=Surfbeat(), swave=True)
        # No wavemodel specified (default)
        physics2 = Physics(swave=True)

    assert "swave" not in caplog.text
    assert physics1.swave is True
    assert isinstance(physics1.wavemodel, Surfbeat)
    assert physics2.swave is True
    assert physics2.wavemodel is None


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


def test_nonh_with_advanced_parameters():
    """Test Nonh component with advanced non-hydrostatic parameters from Table 42."""
    physics = Physics(
        wavemodel=Nonh(
            Topt=10.0,
            breakviscfac=1.5,
            breakvisclen=1.0,
            maxbrsteep=0.4,
            secbrsteep=0.2,
            nhbreaker=2,
            solver="tridiag",
            solver_acc=0.005,
            solver_maxit=30,
        ),
        swave=False,
    )
    params = physics.params

    assert params["wavemodel"] == "nonh"
    assert params["Topt"] == 10.0
    assert params["breakviscfac"] == 1.5
    assert params["breakvisclen"] == 1.0
    assert params["maxbrsteep"] == 0.4
    assert params["secbrsteep"] == 0.2
    assert params["nhbreaker"] == 2
    assert params["solver"] == "tridiag"
    assert params["solver_acc"] == 0.005
    assert params["solver_maxit"] == 30
    assert params["swave"] == 0
