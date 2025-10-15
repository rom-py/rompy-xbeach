"""Tests for extended Physics component parameters."""

import pytest
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.boundary import WaveBoundaryConditions
from rompy_xbeach.components.physics.constants import Coriolis, PhysicalConstants
from rompy_xbeach.components.physics.flow import (
    FlowNumerics,
    HorizontalViscosity,
    WaveCurrentInteraction,
)
from rompy_xbeach.components.physics.numerics import (
    NonHydrostaticNumerics,
    WaveNumerics,
)


def test_horizontal_viscosity():
    """Test HorizontalViscosity model."""
    visc = HorizontalViscosity(
        smag=True,
        nuh=0.1,
        nuhv=2.0,
        gamma_turb=1.5,
    )
    params = visc.params
    assert params["smag"] == 1
    assert params["nuh"] == 0.1
    assert params["nuhv"] == 2.0
    assert params["gamma_turb"] == 1.5


def test_wave_current_interaction():
    """Test WaveCurrentInteraction model."""
    wci = WaveCurrentInteraction(
        cats=10.0,
        hwci=0.2,
        hwcimax=50.0,
    )
    params = wci.params
    assert params["cats"] == 10.0
    assert params["hwci"] == 0.2
    assert params["hwcimax"] == 50.0


def test_flow_numerics():
    """Test FlowNumerics model."""
    flow_num = FlowNumerics(
        eps=0.01,
        eps_sd=0.3,
        hmin=0.1,
        deltahmin=0.2,
        oldhmin=False,
        umin=0.05,
        secorder=True,
        oldhu=False,
    )
    params = flow_num.params
    assert params["eps"] == 0.01
    assert params["eps_sd"] == 0.3
    assert params["hmin"] == 0.1
    assert params["deltahmin"] == 0.2
    assert params["oldhmin"] == 0
    assert params["umin"] == 0.05
    assert params["secorder"] == 1
    assert params["oldhu"] == 0


def test_wave_numerics():
    """Test WaveNumerics model."""
    wave_num = WaveNumerics(
        scheme="warmbeam",
        maxiter=100,
        maxerror=0.0001,
        wavint=300.0,
    )
    params = wave_num.params
    assert params["scheme"] == "warmbeam"
    assert params["maxiter"] == 100
    assert params["maxerror"] == 0.0001
    assert params["wavint"] == 300.0


def test_wave_boundary_conditions():
    """Test WaveBoundaryConditions model."""
    wave_bc = WaveBoundaryConditions(
        nmax=0.7,
        wbcevarreduce=0.8,
        bclwonly=True,
        swkhmin=0.01,
        wbcRemoveStokes=False,
        wbcScaleEnergy=True,
        cyclicdiradjust=False,
    )
    params = wave_bc.params
    assert params["nmax"] == 0.7
    assert params["wbcevarreduce"] == 0.8
    assert params["bclwonly"] == 1
    assert params["swkhmin"] == 0.01
    assert params["wbcRemoveStokes"] == 0
    assert params["wbcScaleEnergy"] == 1
    assert params["cyclicdiradjust"] == 0


def test_nonhydrostatic_numerics():
    """Test NonHydrostaticNumerics model."""
    nh_num = NonHydrostaticNumerics(
        solver="tridiag",
        solver_acc=0.01,
        solver_maxit=50,
        solver_urelax=0.9,
        Topt=12.0,
        dispc=1.0,
        kdmin=0.01,
        nhlay=0.5,
        maxbrsteep=0.5,
        secbrsteep=0.3,
        reformsteep=0.15,
        nhbreaker=2,
    )
    params = nh_num.params
    assert params["solver"] == "tridiag"
    assert params["solver_acc"] == 0.01
    assert params["solver_maxit"] == 50
    assert params["solver_urelax"] == 0.9
    assert params["Topt"] == 12.0
    assert params["dispc"] == 1.0
    assert params["kdmin"] == 0.01
    assert params["nhlay"] == 0.5
    assert params["maxbrsteep"] == 0.5
    assert params["secbrsteep"] == 0.3
    assert params["reformsteep"] == 0.15
    assert params["nhbreaker"] == 2


def test_physical_constants():
    """Test PhysicalConstants model."""
    constants = PhysicalConstants(
        g=9.81,
        rho=1025.0,
        depthscale=1.0,
    )
    params = constants.params
    assert params["g"] == 9.81
    assert params["rho"] == 1025.0
    assert params["depthscale"] == 1.0


def test_coriolis():
    """Test Coriolis model."""
    coriolis = Coriolis(
        lat=-33.0,
        wearth=0.04167,
    )
    params = coriolis.params
    assert params["lat"] == -33.0
    assert params["wearth"] == 0.04167


def test_physics_with_all_new_components():
    """Test Physics with all new component fields."""
    physics = Physics(
        viscosity_params=HorizontalViscosity(
            smag=True,
            nuh=0.1,
            nuhv=1.5,
            gamma_turb=1.0,
        ),
        wci_params=WaveCurrentInteraction(
            cats=5.0,
            hwci=0.15,
            hwcimax=80.0,
        ),
        flow_numerics=FlowNumerics(
            eps=0.005,
            hmin=0.05,
            deltahmin=0.1,
            umin=0.01,
        ),
        wave_numerics=WaveNumerics(
            scheme="warmbeam",
            maxiter=500,
            maxerror=0.0005,
        ),
        wave_boundary=WaveBoundaryConditions(
            nmax=0.8,
            wbcScaleEnergy=True,
        ),
        constants=PhysicalConstants(
            g=9.81,
            rho=1025.0,
        ),
        coriolis=Coriolis(
            lat=-33.5,
        ),
    )

    # Use get() method which flattens nested components
    params = physics.get(destdir="/tmp")

    # Check viscosity params
    assert params["smag"] == 1
    assert params["nuh"] == 0.1
    assert params["nuhv"] == 1.5
    assert params["gamma_turb"] == 1.0

    # Check WCI params
    assert params["cats"] == 5.0
    assert params["hwci"] == 0.15
    assert params["hwcimax"] == 80.0

    # Check flow numerics
    assert params["eps"] == 0.005
    assert params["hmin"] == 0.05
    assert params["deltahmin"] == 0.1
    assert params["umin"] == 0.01

    # Check wave numerics
    assert params["scheme"] == "warmbeam"
    assert params["maxiter"] == 500
    assert params["maxerror"] == 0.0005

    # Check wave boundary
    assert params["nmax"] == 0.8
    assert params["wbcScaleEnergy"] == 1

    # Check constants
    assert params["g"] == 9.81
    assert params["rho"] == 1025.0

    # Check Coriolis
    assert params["lat"] == -33.5


def test_physics_nonhydrostatic_numerics():
    """Test Physics with non-hydrostatic numerics."""
    physics = Physics(
        nonh=True,
        swave=False,  # Required when nonh=True
        nonhydrostatic_numerics=NonHydrostaticNumerics(
            solver="tridiag",
            maxbrsteep=0.4,
            nhbreaker=2,
        ),
    )

    # Use get() method which flattens nested components
    params = physics.get(destdir="/tmp")
    assert params["nonh"] == 1
    assert params["swave"] == 0
    assert params["solver"] == "tridiag"
    assert params["maxbrsteep"] == 0.4
    assert params["nhbreaker"] == 2


def test_validation_ranges():
    """Test that validation ranges work correctly."""
    # Test valid ranges
    visc = HorizontalViscosity(nuh=0.5, nuhv=10.0, gamma_turb=1.5)
    assert visc.nuh == 0.5

    # Test invalid ranges
    with pytest.raises(ValueError):
        HorizontalViscosity(nuh=2.0)  # > 1.0

    with pytest.raises(ValueError):
        WaveCurrentInteraction(cats=100.0)  # > 50.0

    with pytest.raises(ValueError):
        FlowNumerics(eps=0.5)  # > 0.1

    with pytest.raises(ValueError):
        PhysicalConstants(g=10.0)  # > 9.9

    with pytest.raises(ValueError):
        Coriolis(lat=100.0)  # > 90.0
