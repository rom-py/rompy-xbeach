# Rompy-XBeach Architecture

This document describes the architecture of rompy-xbeach, explaining how XBeach model parameters are organised into Python components and why this structure was chosen.

## Overview

Rompy-xbeach is a Python interface for the [XBeach](https://xbeach.readthedocs.io/) coastal morphodynamic model. It provides:

- **Type-safe configuration** — Pydantic models validate parameters before running XBeach
- **Data interfaces** — Automatic generation of boundary condition files from various data sources
- **Structured organisation** — Related parameters grouped into logical components

The core challenge is that XBeach uses a flat `params.txt` file with ~250 key-value pairs, while Python benefits from hierarchical organisation. Rompy-xbeach bridges this gap by providing structured input that serialises to the flat format XBeach expects.

## The Config Class

The `Config` class is the main entry point. It orchestrates all XBeach parameters and generates the `params.txt` file.

```python
from rompy_xbeach.config import Config
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.base import XBeachBathy
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat
from rompy_xbeach.components.physics.friction import Cf
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.morphology import Morphology
from rompy_xbeach.components.output import Output

config = Config(
    grid=RegularGrid(...),
    bathy=XBeachBathy(...),
    physics=Physics(
        wavemodel=Surfbeat(),
        bedfriction=Cf(bedfriccoef=0.01),
    ),
    sediment=Sediment(
        morphology=Morphology(morfac=10),
    ),
    output=Output(outputformat="netcdf"),
)
```

When called, `Config` serialises all components to a flat dictionary that becomes `params.txt`:

```
wavemodel   = surfbeat
bedfriccoef = 0.01
morfac      = 10
outputformat = netcdf
```

## Component Hierarchy

Parameters are organised into components based on their function:

```
Config
├── grid              # Grid definition (generates x, y files)
├── bathy             # Bathymetry (generates depth file)
├── input             # Data-driven boundary conditions
│   ├── wave          # Wave forcing (generates boundary files)
│   ├── tide          # Tide forcing (generates zs0file)
│   └── wind          # Wind forcing (generates wind file)
├── physics           # Physical process parameters
│   ├── wavemodel     # Wave model selection and parameters
│   ├── bedfriction   # Bed friction formulation
│   ├── viscosity     # Horizontal viscosity
│   ├── vegetation    # Vegetation effects
│   ├── wind          # Wind drag parameters
│   ├── wave_numerics # Wave solver numerics
│   ├── flow_numerics # Flow solver numerics
│   └── constants     # Physical constants
├── sediment          # Sediment transport and morphology
│   ├── transport     # Transport formulation
│   ├── morphology    # Morphological acceleration
│   ├── bed_update    # Bed level update controls
│   ├── bed_composition # Grain sizes, porosity
│   └── groundwater   # Groundwater flow
├── output            # Output configuration
├── flow_boundary     # Flow boundary conditions
├── tide_boundary     # Tide/surge boundary parameters
├── wave_boundary     # Manual wave boundary specification
├── hotstart          # Hotstart initialisation
└── mpi               # MPI parallelisation
```

## Why Components?

### 1. Maintainability

With ~250 parameters, a single flat class would be unwieldy. Components group related parameters, making the codebase easier to navigate and maintain.

### 2. Validation

Components enable context-aware validation. For example:

- `Morphology` can validate that `morfac > 0`
- `BedComposition` can validate that `D90 > D50`
- `Nonh` wavemodel can warn if `swave=True` (incompatible)

### 3. Discoverability

IDE autocomplete guides users through the hierarchy:

```python
config.sediment.  # Shows: transport, morphology, bed_update, ...
config.sediment.morphology.  # Shows: morfac, morstart, morstop, ...
```

### 4. Mutual Exclusivity

Some XBeach parameters only apply to certain modes. Components enforce this:

```python
# Surfbeat-specific parameters are only available on Surfbeat
physics=Physics(wavemodel=Surfbeat(breaktype=Roelvink1()))

# Non-hydrostatic parameters are only available on Nonh
physics=Physics(wavemodel=Nonh(nhbreaker=1, solver="tridiag"))
```

Attempting to set `nhbreaker` on `Surfbeat` is a type error — the parameter doesn't exist on that class.

## Key Patterns

### Process Switches: `Union[bool, Component]`

Many XBeach features can be enabled with defaults or customised:

```python
# Enable with XBeach defaults
physics=Physics(vegetation=True)

# Enable with custom parameters
from rompy_xbeach.components.physics.vegetation import Vegetation
physics=Physics(vegetation=Vegetation(nsec=2, ah=[1.5], Cd=[1.0]))

# Disable explicitly
physics=Physics(vegetation=False)

# Omit entirely (uses XBeach default, typically disabled)
physics=Physics()
```

This pattern applies to: `vegetation`, `viscosity`, `wind`, `bedfriction`, `ships`, `hotstart`.

### Discriminated Unions for Wave Models

The `wavemodel` field uses Pydantic's discriminated unions:

```python
wavemodel: Stationary | Surfbeat | Nonh
```

Each variant has different parameters:

| Wavemodel | Key Parameters |
|-----------|----------------|
| `Stationary` | `breaktype` (Baldock, Janssen) |
| `Surfbeat` | `breaktype` (Roelvink variants), `single_dir`, `wci` |
| `Nonh` | `nhbreaker`, `solver`, `kdmin`, `Topt` |

### Breaker Formulations

Breaker types are further specialised:

```python
# Roelvink (1993) - only for Surfbeat
Surfbeat(breaktype=Roelvink1(gamma=0.55, alpha=1.0))

# Roelvink-Daly (2012) - only for Surfbeat  
Surfbeat(breaktype=RoelvinkDaly(gamma=0.55))

# Baldock (1998) - only for Stationary
Stationary(breaktype=Baldock(gamma=0.78, alpha=1.0))
```

This prevents invalid combinations that XBeach would reject at runtime.

### Data Interfaces vs Parameter Components

There's an important distinction:

**Data interfaces** (`input.wave`, `input.tide`, `input.wind`) fetch external data, generate files, and produce parameters:

```python
input=Input(
    wave=BoundaryJonstable(
        source=SourceDataset(...),  # External data source
        hm0_var="hs",               # Variable mapping
        tp_var="tp",
    )
)
# Generates: jonswap.txt, bcfile entries, and wbctype/dtbc/etc parameters
```

**Parameter components** (`wave_boundary`, `flow_boundary`) specify parameters directly:

```python
wave_boundary=SpectralWaveBoundary(
    wbctype="jonstable",
    bcfile="jonswap.txt",  # Pre-existing file
    dtbc=1.0,
)
```

Use data interfaces when generating boundary conditions from data sources. Use parameter components when working with pre-existing boundary files.

## Finding Parameters

### By Component

If you know the category, navigate the hierarchy:

```python
# Morphology parameters
config.sediment.morphology.morfac
config.sediment.morphology.morstart

# Wave numerics
config.physics.wave_numerics.scheme
config.physics.wave_numerics.wavint

# Output settings
config.output.outputformat
config.output.tintg
```

### By XBeach Name

Most field names match XBeach parameter names exactly:

| XBeach Parameter | Rompy Field |
|-----------------|-------------|
| `morfac` | `sediment.morphology.morfac` |
| `D50` | `sediment.bed_composition.D50` |
| `bedfriccoef` | `physics.bedfriction.bedfriccoef` |
| `front` | `flow_boundary.front` |
| `tideloc` | `tide_boundary.tideloc` |
| `hotstart` | `hotstart` (bool) or `hotstart.hotstart` (component) |

Exceptions exist where a field controls internal behaviour rather than an XBeach parameter (e.g., `source` fields on data blobs).

## Serialisation

Components implement a `get(destdir)` method that returns a dictionary of XBeach parameters:

```python
morphology = Morphology(morfac=10, morstart=3600)
params = morphology.get("/path/to/run")
# Returns: {"morfac": 10, "morstart": 3600}
```

The `Config.__call__()` method aggregates all component outputs:

```python
config = Config(...)
params = config(runtime)  # Calls get() on all components
# params is a flat dict ready for params.txt
```

Boolean values are converted to integers (XBeach convention):

```python
sedtrans=True   → sedtrans = 1
vegetation=True → vegetation = 1
```

## YAML Configuration

Configs can be defined in YAML for reproducibility:

```yaml
model_type: xbeach
grid:
  model_type: regular
  dx: 5
  dy: 10
  # ...

physics:
  wavemodel:
    model_type: surfbeat
    breaktype:
      model_type: roelvink1
      gamma: 0.55
  bedfriction:
    bedfriccoef: 0.01

sediment:
  morphology:
    morfac: 10
    morstart: 3600

output:
  outputformat: netcdf
  tintg: 3600
```

The `model_type` field enables Pydantic to deserialise the correct class for discriminated unions.

## Best Practices

### 1. Start Simple

Begin with minimal configuration and add components as needed:

```python
config = Config(
    grid=grid,
    bathy=bathy,
    input=Input(wave=wave_boundary),
)
# Uses XBeach defaults for physics, sediment, output
```

### 2. Use Components for Non-Default Behaviour

Only specify components when you need to change defaults:

```python
# Only customise what you need
physics=Physics(
    wavemodel=Surfbeat(),  # Explicit surfbeat (default is stationary)
    bedfriction=BedFriction(bedfriccoef=0.02),  # Custom friction
    # viscosity, vegetation, etc. use defaults
)
```

### 3. Leverage Type Hints

IDEs provide autocomplete and type checking:

```python
from rompy_xbeach.components.physics import Surfbeat, Roelvink1

# IDE shows available parameters
wavemodel = Surfbeat(
    breaktype=Roelvink1(gamma=0.55),  # IDE shows Roelvink1 params
)
```

### 4. Validate Early

Pydantic validates on construction. Run your config setup before submitting jobs:

```python
try:
    config = Config(...)
except ValidationError as e:
    print(e)  # Shows which parameter failed and why
```

## Summary

Rompy-xbeach organises XBeach's flat parameter space into a structured hierarchy:

- **Components** group related parameters for maintainability
- **Discriminated unions** enforce valid parameter combinations
- **Data interfaces** generate boundary files from external sources
- **Serialisation** flattens the hierarchy back to `params.txt`

The learning curve is knowing which component contains which parameter, but IDE support and consistent naming (matching XBeach parameter names) minimise this friction.
