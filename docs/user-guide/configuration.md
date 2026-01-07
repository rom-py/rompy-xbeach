# Configuration

This guide covers how to configure XBeach simulations using rompy-xbeach.

## The Config Class

The `Config` class is the central configuration object. It accepts components for different aspects of the model:

```python
from rompy_xbeach.config import Config

config = Config(
    grid=...,           # Grid definition
    bathy=...,          # Bathymetry
    input=...,          # Data-driven boundary conditions
    physics=...,        # Physical processes
    sediment=...,       # Sediment transport and morphology
    output=...,         # Output configuration
    flow_boundary=...,  # Flow boundary conditions
    tide_boundary=...,  # Tide boundary parameters
    wave_boundary=...,  # Manual wave boundary specification
    hotstart=...,       # Hotstart initialisation
    mpi=...,            # MPI parallelisation
)
```

All fields are optional and have sensible defaults.

## Configuration Methods

### Python Objects

Direct instantiation with full IDE support:

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.morphology import Morphology

config = Config(
    physics=Physics(
        wavemodel=Surfbeat(),
    ),
    sediment=Sediment(
        morphology=Morphology(morfac=10),
    ),
)
```

### YAML Files

Declarative configuration for reproducibility:

```yaml
model_type: xbeach

physics:
  wavemodel:
    model_type: surfbeat
  sedtrans: true

sediment:
  morphology:
    morfac: 10
```

```python
import yaml
from rompy_xbeach.config import Config

with open("config.yml") as f:
    config = Config(**yaml.safe_load(f))
```

### Dictionary

Useful for programmatic configuration:

```python
config = Config(**{
    "physics": {
        "wavemodel": {"model_type": "surfbeat"},
        "sedtrans": True,
    },
    "sediment": {
        "morphology": {"morfac": 10},
    },
})
```

## The `model_type` Discriminator

When using YAML or dictionaries, the `model_type` field tells Pydantic which class to instantiate:

```yaml
wavemodel:
  model_type: surfbeat  # Creates Surfbeat instance
  breaktype:
    model_type: roelvink1  # Creates Roelvink1 instance
    gamma: 0.55
```

This enables type-safe deserialisation of discriminated unions.

## Process Switches

Many XBeach features use the `Union[bool, Component]` pattern:

```python
# Enable with defaults
physics = Physics(vegetation=True)

# Enable with custom parameters
from rompy_xbeach.components.physics.vegetation import Vegetation
physics = Physics(vegetation=Vegetation(nsec=2, ah=[1.5]))

# Disable explicitly
physics = Physics(vegetation=False)

# Omit (uses XBeach default)
physics = Physics()
```

This pattern applies to:

| Field | Component | XBeach Parameter |
|-------|-----------|------------------|
| `vegetation` | `Vegetation` | `vegetation` |
| `viscosity` | `Viscosity` | `nuh`, `smag`, etc. |
| `wind` | `Wind` | `wind`, `Cd` |
| `bedfriction` | `BedFriction` | `bedfriction`, `bedfriccoef` |
| `ships` | `Ships` | `ships` |
| `hotstart` | `Hotstart` | `hotstart`, `hotstartfileno` |

## Validation

Pydantic validates configuration at construction time:

```python
from rompy_xbeach.components.sediment import Morphology

# Valid
morph = Morphology(morfac=10)

# Invalid - raises ValidationError
morph = Morphology(morfac=-1)  # morfac must be >= 0
```

### Range Validation

Most numeric parameters have valid ranges:

```python
from rompy_xbeach.components.physics.wavemodel import Roelvink1

# gamma must be between 0.4 and 0.9
breaker = Roelvink1(gamma=0.55)  # OK
breaker = Roelvink1(gamma=1.5)   # ValidationError
```

### Cross-Field Validation

Some validations involve multiple fields:

```python
from rompy_xbeach.components.sediment.composition import BedComposition

# D90 must be greater than D50
bed = BedComposition(D50=[0.0002], D90=[0.0003])  # OK
bed = BedComposition(D50=[0.0003], D90=[0.0002])  # ValidationError
```

### Mutual Exclusivity

The wave boundary source can only be specified once:

```python
from rompy_xbeach.config import Config, DataInterface

# This raises ValidationError
config = Config(
    input=DataInterface(wave=...),  # Data-driven
    wave_boundary=...,              # Manual - can't have both!
)
```

## Serialisation

When the config is called, it serialises to a flat dictionary:

```python
# During model generation
params = config(runtime)

# params is a dict like:
{
    "wavemodel": "surfbeat",
    "morfac": 10,
    "bedfriccoef": 0.01,
    # ... all XBeach parameters
}
```

This dictionary is written to `params.txt`.

### Boolean Conversion

Python booleans are converted to XBeach integers:

```python
Physics(sedtrans=True)   # → sedtrans = 1
Physics(sedtrans=False)  # → sedtrans = 0
```

### None Values

`None` values are excluded from output, allowing XBeach defaults:

```python
Physics(swave=None)  # swave not written, XBeach uses default
```

## Accessing Parameters

After configuration, you can inspect the parameters:

```python
config = Config(...)

# Get the params dict (after calling)
params = config(runtime)

# Or access the stored params
print(config.params)
```

## Environment Variables

Some paths can use environment variables:

```python
from rompy_xbeach.data.base import XBeachBathy

bathy = XBeachBathy(
    source=dict(
        model_type="xyz:crs",
        filename="${DATA_DIR}/bathymetry.xyz",
    ),
    interpolator=dict(model_type="regular_grid"),
)
```

## Next Steps

- [Data Interfaces](data-interfaces.md) — Configure boundary conditions from data
- [Parameter Reference](parameter-reference.md) — Find specific XBeach parameters
- [Components](../components/index.md) — Detailed component documentation
