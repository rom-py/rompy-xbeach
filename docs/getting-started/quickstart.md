# Quickstart

This guide walks you through creating a basic XBeach simulation with rompy-xbeach.

## Basic Workflow

1. Define a grid
2. Provide bathymetry
3. Configure physics and sediment parameters
4. Set up boundary conditions
5. Generate the model and run

## Minimal Example

```python
from datetime import datetime
from rompy_xbeach import Config, XBeachModel
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.bathy import StaticBathy
from rompy_xbeach.components import Physics, Output

# 1. Define the grid
grid = RegularGrid(
    ori=dict(x=0, y=0),
    alfa=0,
    dx=10,
    dy=20,
    nx=100,
    ny=50,
)

# 2. Provide bathymetry (from a file or data source)
bathy = StaticBathy(
    source=dict(
        model_type="xyz:crs",
        filename="bathymetry.xyz",
    ),
    interpolator=dict(model_type="regular_grid"),
)

# 3. Configure the model
config = Config(
    grid=grid,
    bathy=bathy,
    physics=Physics(
        wavemodel="surfbeat",  # or Surfbeat() for more control
    ),
    output=Output(
        outputformat="netcdf",
        tintg=3600,  # Global output every hour
    ),
)

# 4. Create and run the model
model = XBeachModel(
    run_id="my_simulation",
    period=dict(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    ),
    config=config,
)

# Generate input files
model.generate()

# Run XBeach (requires xbeach executable)
model.run()
```

## Using YAML Configuration

For reproducibility, define your configuration in YAML:

```yaml
# config.yml
model_type: xbeach

grid:
  model_type: regular
  ori:
    x: 0
    y: 0
  alfa: 0
  dx: 10
  dy: 20
  nx: 100
  ny: 50

bathy:
  model_type: static
  source:
    model_type: "xyz:crs"
    filename: bathymetry.xyz
  interpolator:
    model_type: regular_grid

physics:
  wavemodel:
    model_type: surfbeat
    break_type:
      model_type: roelvink1
      gamma: 0.55

sediment:
  morphology:
    morfac: 10

output:
  outputformat: netcdf
  tintg: 3600
```

Load and use:

```python
import yaml
from rompy_xbeach import Config

with open("config.yml") as f:
    config_dict = yaml.safe_load(f)

config = Config(**config_dict)
```

## Adding Wave Boundary Conditions

### From Data Source

```python
from rompy_xbeach.data import Input
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

config = Config(
    grid=grid,
    bathy=bathy,
    input=Input(
        wave=BoundaryStationSpectraJonstable(
            source=dict(
                model_type="dataset",
                uri="wave_spectra.nc",
            ),
            # Variable mappings
            hm0_var="hs",
            tp_var="tp",
            dir_var="dir",
        ),
    ),
)
```

### Manual Specification

For pre-existing boundary files:

```python
from rompy_xbeach.components.boundary import SpectralWaveBoundary

config = Config(
    grid=grid,
    bathy=bathy,
    wave_boundary=SpectralWaveBoundary(
        wbctype="jonstable",
        bcfile="jonswap.txt",
        dtbc=1.0,
    ),
)
```

## Customising Physics

```python
from rompy_xbeach.components import Physics
from rompy_xbeach.components.physics import (
    Surfbeat,
    Roelvink1,
    BedFriction,
    Viscosity,
)

physics = Physics(
    # Wave model with custom breaker
    wavemodel=Surfbeat(
        break_type=Roelvink1(gamma=0.55, alpha=1.0),
        single_dir=False,
    ),
    # Bed friction
    bedfriction=BedFriction(
        bedfriction="chezy",
        bedfriccoef=55,
    ),
    # Horizontal viscosity
    viscosity=Viscosity(nuh=0.1),
    # Enable processes
    flow=True,
    sedtrans=True,
)
```

## Customising Sediment and Morphology

```python
from rompy_xbeach.components import Sediment
from rompy_xbeach.components.sediment import (
    Morphology,
    SedimentTransport,
    BedComposition,
)

sediment = Sediment(
    transport=SedimentTransport(
        form="vanthiel_vanrijn",
        waveform="vanthiel",
    ),
    morphology=Morphology(
        morfac=10,
        morstart=3600,  # Start morphology after 1 hour
    ),
    bed_composition=BedComposition(
        D50=0.0002,  # 200 microns
        D90=0.0003,
        por=0.4,
    ),
)
```

## Next Steps

- [Architecture](../user-guide/architecture.md) — Understand the component structure
- [Configuration](../user-guide/configuration.md) — Detailed configuration options
- [Components](../components/index.md) — Reference for all components
- [Parameter Reference](../user-guide/parameter-reference.md) — Find specific XBeach parameters
