# Boundary Conditions

XBeach requires boundary conditions for flow and tide/surge at domain edges. This page covers the `flow_boundary` and `tide_boundary` components in Config.

For **wave boundary conditions**, see [Data Interfaces > Wave Boundaries](../data-interfaces/boundaries.md).

## Flow Boundaries

Control how flow behaves at domain edges using `Config.flow_boundary`:

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
| `abs_1d` | 1D absorbing-generating (default) |
| `abs_2d` | 2D absorbing-generating |
| `wall` | Reflective wall |
| `wlevel` | Water level specified |
| `nonh_1d` | Non-hydrostatic 1D |
| `waveflume` | Wave flume mode |

**Back (onshore) boundary:**

| Value | Description |
|-------|-------------|
| `wall` | Reflective wall (default) |
| `abs_1d` | 1D absorbing |
| `abs_2d` | 2D absorbing |
| `wlevel` | Water level specified |

**Left/Right (lateral) boundaries:**

| Value | Description |
|-------|-------------|
| `neumann` | Zero-gradient (default) |
| `wall` | Reflective wall |
| `no_advec` | No advection |

## Tide Boundaries

Configure tide and surge boundary conditions using `Config.tide_boundary`:

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

## Complete Example

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
)

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
    
    # Wave boundaries are specified via input.wave
    # See Data Interfaces > Wave Boundaries
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
```

## API Reference

- [FlowBoundaryConditions](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions)
- [TideBoundaryConditions](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundaryConditions)
