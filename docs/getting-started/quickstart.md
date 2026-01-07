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
from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy_xbeach.config import Config
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.base import XBeachBathy
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.output import Output

# 1. Define the grid
grid = RegularGrid(
    ori=dict(x=0, y=0),
    alfa=0,
    dx=10,
    dy=20,
    nx=100,
    ny=50,
)

# 2. Provide bathymetry (from a GeoTIFF file)
from rompy_xbeach.source import SourceGeotiff

bathy = XBeachBathy(
    source=SourceGeotiff(filename="bathymetry.tif"),
)

# 3. Configure the model
from rompy_xbeach.components.physics.wavemodel import Surfbeat

config = Config(
    grid=grid,
    bathy=bathy,
    physics=Physics(
        wavemodel=Surfbeat(),
    ),
    output=Output(
        outputformat="netcdf",
        tintg=3600,  # Global output every hour
    ),
)

# 4. Create and run the model
model = ModelRun(
    run_id="my_simulation",
    period=TimeRange(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    ),
    config=config,
)

# Generate input files
model.generate()

# Run XBeach (requires xbeach executable)
# model.run()
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
    model_type: geotiff
    filename: bathymetry.tif

physics:
  wavemodel:
    model_type: surfbeat
    breaktype:
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
from rompy_xbeach.config import Config

with open("config.yml") as f:
    config_dict = yaml.safe_load(f)

config = Config(**config_dict)
```

## Adding Wave Boundary Conditions

### From Data Source

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(
            source=dict(
                model_type="wavespectra:crs",
                uri="wave_spectra.nc",
            ),
        ),
    ),
)
```

### Manual Specification

For pre-existing boundary files:

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary

config = Config(
    grid=grid,
    bathy=bathy,
    wave_boundary=SpectralWaveBoundary(
        wbctype="jonstable",
        bcfile="jonswap.txt",
    ),
)
```

## Customising Physics

```python
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat, Roelvink1
from rompy_xbeach.components.physics.friction import Viscosity, Chezy

physics = Physics(
    # Wave model with custom breaker
    wavemodel=Surfbeat(breaktype=Roelvink1(gamma=0.55, alpha=1.0)),
    # Bed friction (use Chezy with coefficient)
    bedfriction=Chezy(bedfriccoef=55),
    # Horizontal viscosity
    viscosity=Viscosity(nuh=0.1),
    # Enable processes
    flow=True,
)
```

## Customising Sediment and Morphology

```python
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.morphology import Morphology
from rompy_xbeach.components.sediment.transport import SedimentTransport
from rompy_xbeach.components.sediment.composition import BedComposition

sediment = Sediment(
    sedtrans=SedimentTransport(
        form="vanthiel_vanrijn",
        waveform="vanthiel",
    ),
    morphology=Morphology(
        morfac=10,
        morstart=3600,  # Start morphology after 1 hour
    ),
    bed_composition=BedComposition(
        D50=[0.0002],  # 200 microns (list for each sediment class)
        D90=[0.0003],
        por=0.4,
    ),
)
```

## Next Steps

- [Architecture](../user-guide/architecture.md) — Understand the component structure
- [Configuration](../user-guide/configuration.md) — Detailed configuration options
- [Components](../components/index.md) — Reference for all components
- [Parameter Reference](../user-guide/parameter-reference.md) — Find specific XBeach parameters
