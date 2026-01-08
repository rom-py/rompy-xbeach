# Boundary Components

XBeach requires boundary conditions for flow, tide/surge, and waves. Rompy-xbeach provides components for each.

## Overview

There are two ways to specify boundaries:

1. **Data interfaces** (`input.wave`, `input.tide`) — Generate boundary files from data sources
2. **Parameter components** — Specify parameters directly for pre-existing files

This page covers the parameter components. For data-driven boundaries, see [Data Interfaces](../data-interfaces/index.md).

## Flow Boundaries

Control how flow behaves at domain edges:

```python
from rompy_xbeach.components.boundary.parameters import FlowBoundaryConditions

flow_bc = FlowBoundaryConditions(
    front="abs_2d",      # Offshore boundary
    back="wall",         # Onshore boundary
    left="neumann",      # Left lateral boundary
    right="neumann",     # Right lateral boundary
)
```

### Boundary Types

**Front (offshore) boundary:**

| Value | Description |
|-------|-------------|
| `abs_1d` | 1D absorbing-generating |
| `abs_2d` | 2D absorbing-generating |
| `wall` | Reflective wall |
| `wlevel` | Water level boundary |
| `nonh_1d` | Non-hydrostatic 1D |
| `waveflume` | Wave flume |

**Back (onshore) boundary:**

| Value | Description |
|-------|-------------|
| `wall` | Reflective wall |
| `abs_1d` | 1D absorbing |
| `abs_2d` | 2D absorbing |
| `wlevel` | Water level |

**Left/Right (lateral) boundaries:**

| Value | Description |
|-------|-------------|
| `neumann` | Zero-gradient |
| `wall` | Reflective wall |
| `no_advec` | No advection |

### Additional Parameters

```python
flow_bc = FlowBoundaryConditions(
    front="abs_2d",
    back="wall",
    left="neumann",
    right="neumann",
    lateralwave="neumann",  # Lateral wave boundary
)
```

## Tide Boundaries

Control tide/surge forcing:

```python
from rompy_xbeach.components.boundary.parameters import TideBoundaryConditions

tide_bc = TideBoundaryConditions(
    tideloc=2,           # Number of tide locations
    tidetype="velocity", # Tide boundary type
    zs0=0.5,             # Initial water level (m)
    paulrevere="land",   # Sea/land specification
)
```

### Tide Locations

| `tideloc` | Description |
|-----------|-------------|
| 0 | No tide boundary |
| 1 | Single tide location (uniform) |
| 2 | Two corners (interpolated) |
| 4 | Four corners |

### Tide Types

| `tidetype` | Description |
|------------|-------------|
| `instant` | Instantaneous water level |
| `velocity` | Velocity-based |
| `hybrid` | Hybrid approach |

## Wave Boundaries

For manual wave boundary specification (when not using data interfaces):

### Spectral Boundaries

```python
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary
from rompy_xbeach.components.boundary.parameters import SpectralWaveBoundaryConditions

wave_bc = SpectralWaveBoundary(
    wbctype="jonstable",
    bcfile="jonswap.txt",
    wbc=SpectralWaveBoundaryConditions(
        # Directional settings
        thetamin=-90,
        thetamax=90,
        dtheta=10,
    ),
)
```

**Spectral boundary types (`wbctype`):**

| Value | Description |
|-------|-------------|
| `jons` | Single JONSWAP spectrum |
| `swan` | SWAN 2D spectrum |
| `vardens` | Variance density spectrum |
| `jonstable` | Time-varying JONSWAP table |

### Non-Spectral Boundaries

```python
from rompy_xbeach.components.boundary.specification import NonSpectralWaveBoundary

wave_bc = NonSpectralWaveBoundary(
    wbctype="stat",
    Hrms=1.0,
    Tp=10.0,
    dir0=270.0,
    s=20.0,
)
```

**Non-spectral boundary types:**

| Value | Description |
|-------|-------------|
| `stat` | Stationary parametric |
| `bichrom` | Bichromatic waves |
| `ts_1` | Time series (1 location) |
| `ts_2` | Time series (2 locations) |
| `ts_nonh` | Non-hydrostatic time series |

### Directional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `thetamin` | Minimum wave direction (°) | -90 |
| `thetamax` | Maximum wave direction (°) | 90 |
| `dtheta` | Directional resolution (°) | 10 |
| `thetanaut` | Nautical convention | False |

### Special Boundaries

```python
from rompy_xbeach.components.boundary.specification import OffWaveBoundary, ReuseWaveBoundary

# No wave boundary
wave_bc = OffWaveBoundary()

# Reuse existing boundary files
wave_bc = ReuseWaveBoundary()
```

## Hotstart

Initialize from a previous simulation:

```python
from rompy_xbeach.components.hotstart import Hotstart

hotstart = Hotstart(
    hotstartfileno=0,    # Which hotstart file set to use
    source=dict(
        source="/path/to/hotstart/dir",
    ),
)
```

Or simply enable with files already in place:

```python
config = Config(
    hotstart=True,  # Files must be in run directory
)
```

## Complete Example

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
)
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary

config = Config(
    grid=grid,
    bathy=bathy,
    
    # Flow boundaries
    flow_boundary=FlowBoundaryConditions(
        front="abs_2d",
        back="wall",
        left="neumann",
        right="neumann",
    ),
    
    # Tide boundary
    tide_boundary=TideBoundaryConditions(
        tideloc=2,
        zs0=0.0,
    ),
    
    # Wave boundary (manual)
    wave_boundary=SpectralWaveBoundary(
        wbctype="jonstable",
        bcfile="waves.txt",
    ),
)
```

## YAML Configuration

```yaml
flow_boundary:
  front: abs_2d
  back: wall
  left: neumann
  right: neumann

tide_boundary:
  tideloc: 2
  zs0: 0.0

wave_boundary:
  model_type: spectral
  wbctype: jonstable
  bcfile: waves.txt
  dtbc: 1.0
```

## API Reference

See [Boundaries API Reference](../api-reference/components.md#boundaries) for the complete API documentation.
