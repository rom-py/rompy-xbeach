# Data Interfaces

Rompy-xbeach provides multiple ways to configure boundary conditions, each suited to different workflows.

## Three Approaches to Boundary Configuration

### 1. Data Interfaces (Automatic Generation)

**Use when:** You have raw data (NetCDF, THREDDS, buoy data) and want rompy-xbeach to generate XBeach input files automatically.

Data interfaces handle:

- **Data retrieval** — Fetching from local files, remote servers, or databases
- **Processing** — Interpolation, unit conversion, format transformation
- **File generation** — Creating XBeach-compatible input files (e.g., `jonswap.txt`)
- **Data-dependent parameters** — Setting `bcfile`, `dtbc`, time specifications, etc.

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

# Data interface generates files AND sets data-dependent parameters
wave = BoundaryStationSpectraJonstable(
    source=dict(model_type="dataset", uri="wave_spectra.nc"),
    hm0_var="hs",
    tp_var="tp",
)
```

### 2. Manual Specification (Pre-existing Files)

**Use when:** You already have XBeach boundary files and just need to reference them.

```python
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary

# Reference existing files directly
wave_boundary = SpectralWaveBoundary(
    wbctype="jonstable",
    bcfile="my_existing_jonswap.txt",  # Pre-existing file
)
```

### 3. Boundary Parameters (Data-Independent Settings)

**Use when:** You need to configure boundary behaviour that doesn't depend on data files.

These parameters control how XBeach handles boundaries regardless of data source:

```python
from rompy_xbeach.components.boundary.parameters import FlowBoundaryConditions, TideBoundaryConditions

# Flow boundary behaviour (not data-dependent)
flow_bc = FlowBoundaryConditions(
    front="abs_2d",
    back="wall",
    left="neumann",
    right="neumann",
)

# Tide boundary settings
tide_bc = TideBoundaryConditions(
    tideloc=2,
    tidetype="velocity",
)
```

## Combining Approaches

In practice, you often combine these:

```python
from rompy_xbeach.config import Config, DataInterface

config = Config(
    # Data interface: generates wave boundary files from data
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(...),
    ),
    # Boundary parameters: data-independent settings
    flow_boundary=FlowBoundaryConditions(front="abs_2d", back="wall"),
    tide_boundary=TideBoundaryConditions(tideloc=2),
)
```

Or with pre-existing files:

```python
config = Config(
    # Manual specification: use existing boundary files
    wave_boundary=SpectralWaveBoundary(wbctype="jonstable", bcfile="waves.txt"),
    # Boundary parameters: still needed for flow/tide behaviour
    flow_boundary=FlowBoundaryConditions(front="abs_2d", back="wall"),
)
```

!!! warning "Don't mix data interfaces and manual specification for the same boundary"
    You cannot specify both `input.wave` and `wave_boundary`. Choose one approach per boundary type.

---

## Data Interface Details

The following sections detail the data interface approach for automatic file generation.

## The DataInterface Class

The `DataInterface` class groups all data-driven boundary conditions:

```python
from rompy_xbeach.config import DataInterface

input_config = DataInterface(
    wave=...,   # Wave boundary conditions
    tide=...,   # Tide/water level forcing
    wind=...,   # Wind forcing
)
```

## Wave Boundaries

### From Spectral Data (JONSWAP Table)

The most common approach for spectral wave boundaries:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

wave = BoundaryStationSpectraJonstable(
    source=dict(
        model_type="wavespectra",
        uri="wave_spectra.nc",
    ),
)
```

This generates a `jonswap.txt` file and sets:

- `wbctype = jonstable`
- `bcfile = jonswap.txt`
- `dtbc` (boundary update interval)

### From Parametric Data (JONSWAP)

For time-invariant or single-condition boundaries:

```python
from rompy_xbeach.data.boundary import BoundaryStationParamJons

wave = BoundaryStationParamJons(
    source=dict(
        model_type="dataset",
        uri="wave_params.nc",
    ),
    hm0="hs",
    tp="tp",
    mainang="dir",
)
```

### From 2D Spectra (SWAN format)

For full 2D spectral boundaries:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraSwan

wave = BoundaryStationSpectraSwan(
    source=dict(
        model_type="wavespectra",
        uri="swan_spectra.nc",
    ),
)
```

### Variable Mapping

For parametric boundaries, map your data variables to XBeach parameters:

| Interface Field | XBeach Meaning |
|----------------|----------------|
| `hm0` | Significant wave height variable name or constant |
| `tp` | Peak period variable name or constant |
| `mainang` | Mean wave direction variable name or constant |
| `gammajsp` | JONSWAP gamma variable name or constant |
| `dspr` | Directional spread variable name or constant |

## Tide/Water Level

### From Tidal Constituents

Using harmonic constituents (e.g., from oceantide):

```python
from rompy_xbeach.data.waterlevel import TideConsGrid

tide = TideConsGrid(
    source=dict(
        model_type="oceantide",
        uri="tidal_constituents.nc",
    ),
    tideloc=2,  # Number of tide locations (corners)
)
```

### From Time Series

Direct water level time series:

```python
from rompy_xbeach.data.waterlevel import WaterLevelStation

tide = WaterLevelStation(
    source=dict(
        model_type="dataset",
        uri="water_levels.nc",
    ),
    variables=["zs"],  # Water level variable
    tideloc=1,
)
```

This generates `zs0file` and sets:

- `tideloc` — Number of tide boundary locations
- `tidelen` — Length of tide time series

## Wind Forcing

### Spatially Uniform Wind

```python
from rompy_xbeach.data.wind import WindStation, WindVector

wind = WindStation(
    source=dict(
        model_type="dataset",
        uri="wind.nc",
    ),
    wind_vars=WindVector(u="u10", v="v10"),
)
```

### Spatially Varying Wind

```python
from rompy_xbeach.data.wind import WindGrid, WindVector

wind = WindGrid(
    source=dict(
        model_type="dataset",
        uri="wind_grid.nc",
    ),
    wind_vars=WindVector(u="u10", v="v10"),
)
```

## Data Sources

### Local Files

```python
source=dict(
    model_type="dataset",
    uri="/path/to/data.nc",
)
```

### Remote (THREDDS/OpenDAP)

```python
source=dict(
    model_type="dataset",
    uri="https://thredds.server.com/data.nc",
)
```

### Intake Catalogs

```python
source=dict(
    model_type="intake",
    catalog_uri="catalog.yml",
    dataset="wave_data",
)
```

### XYZ Files (for bathymetry)

```python
from rompy_xbeach.source import SourceXYZ

source=SourceXYZ(
    filename="bathymetry.xyz",
    crs="EPSG:4326",
    res=10.0,  # Resolution for gridding
)
```

## Time Handling

Data interfaces automatically handle time:

- Extract data for the simulation period
- Interpolate to required time steps
- Handle timezone conversions

The simulation period comes from the `ModelRun`:

```python
from datetime import datetime
from rompy.model import ModelRun
from rompy.core.time import TimeRange

model = ModelRun(
    run_id="my_simulation",
    period=TimeRange(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    ),
    config=config,
)
```

## Spatial Interpolation

For gridded data, interpolation to the XBeach grid is automatic:

```python
from rompy_xbeach.data.base import XBeachBathy

bathy = XBeachBathy(
    source=dict(
        model_type="dataset",
        uri="bathymetry.nc",
    ),
    interpolator=dict(
        model_type="scipy_regular_grid",
    ),
)
```

## Manual Boundary Specification

If you have pre-existing boundary files, use the parameter components instead:

```python
from rompy_xbeach.config import Config
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary

config = Config(
    # Don't use input.wave
    wave_boundary=SpectralWaveBoundary(
        wbctype="jonstable",
        bcfile="my_jonswap.txt",  # Pre-existing file
    ),
)
```

!!! warning "Don't mix data interfaces and manual specification"
    You cannot specify both `input.wave` and `wave_boundary`. Choose one approach.

## Example: Complete Data-Driven Setup

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable
from rompy_xbeach.data.waterlevel import TideConsGrid
from rompy_xbeach.data.wind import WindStation, WindVector

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(
            source=dict(
                model_type="wavespectra",
                uri="https://thredds.example.com/waves.nc",
            ),
        ),
        tide=TideConsGrid(
            source=dict(
                model_type="oceantide",
                uri="tides.nc",
            ),
            tideloc=2,
        ),
        wind=WindStation(
            source=dict(
                model_type="dataset",
                uri="wind.nc",
            ),
            wind_vars=WindVector(u="u10", v="v10"),
        ),
    ),
)
```

## Next Steps

- [Parameter Reference](parameter-reference.md) — Find boundary-related parameters
- [Boundaries Component](../components/boundaries.md) — Manual boundary specification
