# Physics Component

The `Physics` component controls physical processes in XBeach: wave propagation, flow computation, bed friction, viscosity, and more.

## Overview

!!! important "wavemodel is required"
    `Physics` requires a `wavemodel` — XBeach will not run without one. All other
    fields are optional and fall back to XBeach's built-in defaults when omitted.

```python
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat, Roelvink1
from rompy_xbeach.components.physics.friction import Viscosity, Cf
from rompy_xbeach.components.physics.vegetation import Vegetation

physics = Physics(
    wavemodel=Surfbeat(
        breaktype=Roelvink1(gamma=0.55),
    ),
    bedfriction=Cf(bedfriccoef=0.01),
    viscosity=Viscosity(nuh=0.1),
    vegetation=True,
    swave=True,
    flow=True,
)
```

## Process Switches

These boolean fields enable/disable major processes:

| Field | XBeach Parameter | Description | Default |
|-------|-----------------|-------------|---------|
| [`swave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.swave) | `swave` | Short wave action balance | 1 |
| [`lwave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.lwave) | `lwave` | Long wave propagation | 1 |
| [`flow`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.flow) | `flow` | Flow computation | 1 |
| [`sedtrans`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.sedtrans) | `sedtrans` | Sediment transport | 1 |
| [`morphology`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.morphology) | `morphology` | Morphological updating | 1 |
| [`avalanching`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.avalanching) | `avalanching` | Avalanching | 1 |
| [`wind`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wind) | `wind` | Wind forcing | 0 |
| [`vegetation`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.vegetation) | `vegetation` | Vegetation effects | 0 |
| [`ships`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.ships) | `ships` | Ship-induced waves | 0 |

## Wave Models

The `wavemodel` field accepts one of three types:

### Stationary

For stationary wave conditions (no wave groups):

```python
from rompy_xbeach.components.physics.wavemodel import Stationary, Baldock

physics = Physics(
    wavemodel=Stationary(
        breaktype=Baldock(gamma=0.78),
    ),
)
```

**Breaker options for Stationary:**

- `Baldock` — Baldock et al. (1998)
- `Janssen` — Janssen & Battjes (2007)

### Surfbeat

For infragravity wave modelling (default):

```python
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat, Roelvink1

physics = Physics(
    wavemodel=Surfbeat(
        breaktype=Roelvink1(gamma=0.55, alpha=1.0, n=10),
        single_dir=False,
    ),
    wci=True,  # Wave-current interaction
)
```

**Breaker options for Surfbeat:**

- `Roelvink1` — Roelvink (1993)
- `Roelvink2` — Roelvink (1993) with Hb/h dependency
- `RoelvinkDaly` — Roelvink-Daly (2012)

### Nonh (Non-hydrostatic)

For phase-resolving wave modelling:

```python
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Nonh

physics = Physics(
    wavemodel=Nonh(
        nhbreaker=1,
        solver="tridiag",
        Topt=0.5,
        kdmin=0.01,
    ),
)
```

**Key Nonh parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| [`nhbreaker`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.nhbreaker) | Non-hydrostatic breaker model | 0 |
| [`solver`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.solver) | Pressure solver (tridiag/sip) | tridiag |
| [`Topt`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.Topt) | Optimal timestep factor | 0.5 |
| [`kdmin`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.kdmin) | Minimum kd for dispersion | 0.01 |

## Bed Friction

```python
from rompy_xbeach.components.physics.friction import Chezy, Manning, Cf

# Using Chezy coefficient
friction = Chezy(bedfriccoef=55)

# Using Manning coefficient
friction = Manning(bedfriccoef=0.02)

# Using friction factor
friction = Cf(bedfriccoef=0.01)
```

**Friction formulations:**

| Value | Description |
|-------|-------------|
| `chezy` | Chezy formulation |
| `cf` | Friction factor |
| `manning` | Manning formulation |
| `white-colebrook` | White-Colebrook formulation |
| `white-colebrook-grainsize` | White-Colebrook with grain size |

## Viscosity

```python
from rompy_xbeach.components.physics.friction import Viscosity

# Constant viscosity
viscosity = Viscosity(nuh=0.1)

# Smagorinsky model
viscosity = Viscosity(smag=True)

# Calibration factor
viscosity = Viscosity(nuhfac=1.0)
```

## Vegetation

```python
from rompy_xbeach.components.physics.vegetation import Vegetation

vegetation = Vegetation(
    nsec=2,        # Number of vertical sections
    ah=[1.5],      # Vegetation height per section (m)
    bv=[0.01],     # Stem diameter per section (m)
    Nv=[100],      # Stem density per section (stems/m²)
    Cd=[1.0],      # Drag coefficient per section
)
```

## Wind

```python
from rompy_xbeach.components.physics.wind import Wind

# Enable with custom drag coefficient
wind = Wind(Cd=0.002)

# Or just enable with defaults
physics = Physics(wind=True)
```

## Numerics

### Wave Numerics

```python
from rompy_xbeach.components.physics.numerics import WaveNumerics

wave_numerics = WaveNumerics(
    scheme="warmbeam",  # Numerical scheme
    wavint=60,          # Wave integration interval (s)
    maxerror=0.001,     # Convergence criterion
    maxiter=500,        # Maximum iterations
)
```

### Flow Numerics

```python
from rompy_xbeach.components.physics.numerics import FlowNumerics

flow_numerics = FlowNumerics(
    cfl=0.7,           # CFL criterion
    eps=0.005,         # Threshold depth (m)
    hmin=0.2,          # Minimum water depth (m)
    secorder=False,    # Second-order advection
)
```

## Physical Constants

```python
from rompy_xbeach.components.physics.constants import PhysicalConstants

constants = PhysicalConstants(
    rho=1025,    # Water density (kg/m³)
    rhoa=1.25,   # Air density (kg/m³)
    g=9.81,      # Gravitational acceleration (m/s²)
    lat=-33.0,   # Latitude for Coriolis
)
```

## Complete Example

```python
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat, Roelvink1
from rompy_xbeach.components.physics.friction import Viscosity, Chezy
from rompy_xbeach.components.physics.numerics import WaveNumerics, FlowNumerics
from rompy_xbeach.components.physics.constants import PhysicalConstants

physics = Physics(
    # Wave model
    wavemodel=Surfbeat(
        breaktype=Roelvink1(gamma=0.55, alpha=1.0),
        single_dir=False,
    ),
    # Wave-current interaction
    wci=True,
    # Friction
    bedfriction=Chezy(bedfriccoef=55),
    # Viscosity
    viscosity=Viscosity(nuh=0.1),
    # Numerics
    wave_numerics=WaveNumerics(scheme="warmbeam"),
    flow_numerics=FlowNumerics(cfl=0.7),
    # Constants
    constants=PhysicalConstants(rho=1025),
    # Process switches
    swave=True,
    flow=True,
)
```

## API Reference

See [Physics API Reference](../api-reference/components.md#physics) for the complete API documentation.
