# Physics Component

The `Physics` component controls physical processes in XBeach: wave propagation, flow computation, bed friction, viscosity, and more.

## Overview

```python
from rompy_xbeach.components import Physics
from rompy_xbeach.components.physics import (
    Surfbeat,
    Roelvink1,
    BedFriction,
    Viscosity,
    Vegetation,
)

physics = Physics(
    wavemodel=Surfbeat(
        break_type=Roelvink1(gamma=0.55),
    ),
    bedfriction=BedFriction(bedfriccoef=0.01),
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
| `swave` | `swave` | Short wave action balance | 1 |
| `lwave` | `lwave` | Long wave propagation | 1 |
| `flow` | `flow` | Flow computation | 1 |
| `sedtrans` | `sedtrans` | Sediment transport | 1 |
| `morphology` | `morphology` | Morphological updating | 1 |
| `avalanching` | `avalanching` | Avalanching | 1 |
| `wind` | `wind` | Wind forcing | 0 |
| `vegetation` | `vegetation` | Vegetation effects | 0 |
| `ships` | `ships` | Ship-induced waves | 0 |

## Wave Models

The `wavemodel` field accepts one of three types:

### Stationary

For stationary wave conditions (no wave groups):

```python
from rompy_xbeach.components.physics import Stationary, Baldock

physics = Physics(
    wavemodel=Stationary(
        break_type=Baldock(gamma=0.78),
    ),
)
```

**Breaker options for Stationary:**

- `Baldock` — Baldock et al. (1998)
- `Janssen` — Janssen & Battjes (2007)

### Surfbeat

For infragravity wave modelling (default):

```python
from rompy_xbeach.components.physics import Surfbeat, Roelvink1

physics = Physics(
    wavemodel=Surfbeat(
        break_type=Roelvink1(gamma=0.55, alpha=1.0, n=10),
        single_dir=False,
        wci=True,  # Wave-current interaction
    ),
)
```

**Breaker options for Surfbeat:**

- `Roelvink1` — Roelvink (1993)
- `Roelvink2` — Roelvink (1993) with Hb/h dependency
- `RoelvinkDaly` — Roelvink-Daly (2012)

### Nonh (Non-hydrostatic)

For phase-resolving wave modelling:

```python
from rompy_xbeach.components.physics import Nonh

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
| `nhbreaker` | Non-hydrostatic breaker model | 0 |
| `solver` | Pressure solver (tridiag/sip) | tridiag |
| `Topt` | Optimal timestep factor | 0.5 |
| `kdmin` | Minimum kd for dispersion | 0.01 |

## Bed Friction

```python
from rompy_xbeach.components.physics import BedFriction

# Using Chezy coefficient
friction = BedFriction(
    bedfriction="chezy",
    bedfriccoef=55,
)

# Using Manning coefficient
friction = BedFriction(
    bedfriction="manning",
    bedfriccoef=0.02,
)

# Spatially varying friction
friction = BedFriction(
    bedfriction="chezy",
    bedfricfile=dict(
        source="/path/to/friction.dep",
    ),
)
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
from rompy_xbeach.components.physics import Viscosity

# Constant viscosity
viscosity = Viscosity(nuh=0.1)

# Smagorinsky model
viscosity = Viscosity(smag=True)

# Calibration factor
viscosity = Viscosity(nuhfac=1.0)
```

## Vegetation

```python
from rompy_xbeach.components.physics import Vegetation

vegetation = Vegetation(
    nsec=2,      # Number of vertical sections
    ah=1.5,      # Vegetation height (m)
    bv=0.01,     # Stem diameter (m)
    Nv=100,      # Stem density (stems/m²)
    Cd=1.0,      # Drag coefficient
)
```

## Wind

```python
from rompy_xbeach.components.physics import Wind

# Enable with custom drag coefficient
wind = Wind(Cd=0.002)

# Or just enable with defaults
physics = Physics(wind=True)
```

## Numerics

### Wave Numerics

```python
from rompy_xbeach.components.physics import WaveNumerics

wave_numerics = WaveNumerics(
    scheme="warmbeam",  # Numerical scheme
    wavint=60,          # Wave integration interval (s)
    maxerror=0.001,     # Convergence criterion
    maxiter=500,        # Maximum iterations
)
```

### Flow Numerics

```python
from rompy_xbeach.components.physics import FlowNumerics

flow_numerics = FlowNumerics(
    cfl=0.7,           # CFL criterion
    eps=0.005,         # Threshold depth (m)
    hmin=0.2,          # Minimum water depth (m)
    secorder=False,    # Second-order advection
)
```

## Physical Constants

```python
from rompy_xbeach.components.physics import PhysicalConstants

constants = PhysicalConstants(
    rho=1025,    # Water density (kg/m³)
    rhoa=1.25,   # Air density (kg/m³)
    g=9.81,      # Gravitational acceleration (m/s²)
    lat=-33.0,   # Latitude for Coriolis
)
```

## Complete Example

```python
from rompy_xbeach.components import Physics
from rompy_xbeach.components.physics import (
    Surfbeat,
    Roelvink1,
    BedFriction,
    Viscosity,
    WaveNumerics,
    FlowNumerics,
    PhysicalConstants,
)

physics = Physics(
    # Wave model
    wavemodel=Surfbeat(
        break_type=Roelvink1(gamma=0.55, alpha=1.0),
        single_dir=False,
        wci=True,
    ),
    # Friction
    bedfriction=BedFriction(
        bedfriction="chezy",
        bedfriccoef=55,
    ),
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
    sedtrans=True,
)
```

## API Reference

See [Physics API Reference](../api-reference/components.md#physics) for the complete API documentation.
