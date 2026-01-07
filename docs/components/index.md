# Components Overview

Rompy-xbeach organises XBeach parameters into **components** — Pydantic models that group related parameters together. This provides:

- **Discoverability** — Find parameters by category
- **Validation** — Catch invalid values early
- **Type safety** — IDE autocomplete and type checking
- **Documentation** — Each field has descriptions and valid ranges

## Component Hierarchy

```
Config
├── physics           # Physical processes and numerics
├── sediment          # Sediment transport and morphology
├── output            # Output configuration
├── flow_boundary     # Flow boundary conditions
├── tide_boundary     # Tide/surge parameters
├── wave_boundary     # Manual wave boundary specification
├── hotstart          # Hotstart initialisation
└── mpi               # MPI parallelisation
```

## Quick Reference

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| [Physics](physics.md) | Wave models, friction, viscosity | `wavemodel`, `bedfriction`, `swave`, `flow` |
| [Sediment](sediment.md) | Transport, morphology, bed composition | `morfac`, `D50`, `form`, `turb` |
| [Output](output.md) | Output format and intervals | `outputformat`, `tintg`, `tintm` |
| [Boundaries](boundaries.md) | Flow, tide, wave boundaries | `front`, `back`, `tideloc`, `wbctype` |

## Using Components

### Direct Instantiation

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat
from rompy_xbeach.components.physics.friction import Cf
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.morphology import Morphology

config = Config(
    grid=...,  # Required
    bathy=..., # Required
    physics=Physics(
        wavemodel=Surfbeat(),
        bedfriction=Cf(bedfriccoef=0.01),
    ),
    sediment=Sediment(
        morphology=Morphology(morfac=10),
    ),
)
```

### From YAML

```yaml
physics:
  wavemodel:
    model_type: surfbeat
  bedfriction:
    bedfriccoef: 0.01

sediment:
  morphology:
    morfac: 10
```

## Component Patterns

### Aggregator Components

Top-level components like `Physics` and `Sediment` are **aggregators** — they group sub-components:

```python
class Physics(XBeachBaseModel):
    wavemodel: Optional[Wavemodel] = None
    bedfriction: Optional[Union[bool, BedFriction]] = None
    viscosity: Optional[Union[bool, Viscosity]] = None
    # ... plus direct parameters like swave, flow, etc.
```

### Leaf Components

Sub-components like `Morphology` contain actual XBeach parameters:

```python
class Morphology(XBeachBaseModel):
    morfac: Optional[float] = Field(default=None, ge=0)
    morstart: Optional[float] = Field(default=None, ge=0)
    morstop: Optional[float] = Field(default=None, ge=0)
    # ...
```

### Process Switches

Many features use `Union[bool, Component]`:

```python
# Enable with defaults
physics = Physics(vegetation=True)

# Enable with custom parameters
from rompy_xbeach.components.physics.vegetation import Vegetation
physics = Physics(vegetation=Vegetation(nsec=2))

# Disable
physics = Physics(vegetation=False)
```

### Discriminated Unions

Some fields accept multiple types, distinguished by `model_type`:

```python
# Wavemodel can be Stationary, Surfbeat, or Nonh
wavemodel: Stationary | Surfbeat | Nonh

# Each has different parameters
Surfbeat(breaktype=Roelvink1(gamma=0.55))
Nonh(nhbreaker=1, solver="tridiag")
```

## Serialisation

Components implement a `get(destdir)` method that returns XBeach parameters:

```python
morph = Morphology(morfac=10, morstart=3600)
params = morph.get("/path/to/run")
# Returns: {"morfac": 10, "morstart": 3600}
```

The `Config` class aggregates all component outputs into the final `params.txt`.

## Finding Parameters

If you know the XBeach parameter name, see the [Parameter Reference](../user-guide/parameter-reference.md) for its location.

If you're exploring, browse the component pages:

- [Physics](physics.md) — Wave models, friction, viscosity, numerics
- [Sediment](sediment.md) — Transport, morphology, bed composition
- [Output](output.md) — Output format, intervals, variables
- [Boundaries](boundaries.md) — Flow, tide, wave boundaries
