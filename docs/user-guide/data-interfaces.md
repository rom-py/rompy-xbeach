# Data Interfaces

Rompy-xbeach provides data interfaces that automatically generate XBeach boundary condition files from various data sources. This is one of the most powerful features, allowing you to connect XBeach to NetCDF files, THREDDS servers, and other data sources.

## Overview

Data interfaces handle:

1. **Data retrieval** — Fetching data from local files, remote servers, or databases
2. **Processing** — Interpolation, unit conversion, format transformation
3. **File generation** — Creating XBeach-compatible input files
4. **Parameter setting** — Automatically setting related XBeach parameters

## The Input Class

The `Input` class groups all data-driven boundary conditions:

```python
from rompy_xbeach.data import Input

input_config = Input(
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
        model_type="dataset",
        uri="wave_spectra.nc",
    ),
    # Map your variable names to expected names
    hm0_var="hs",      # Significant wave height
    tp_var="tp",       # Peak period
    dir_var="dir",     # Mean direction
    spread_var="spr",  # Directional spread (optional)
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
    hm0_var="hs",
    tp_var="tp",
    dir_var="dir",
)
```

### From 2D Spectra (SWAN format)

For full 2D spectral boundaries:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraSwan

wave = BoundaryStationSpectraSwan(
    source=dict(
        model_type="dataset",
        uri="swan_spectra.nc",
    ),
    freq_var="frequency",
    dir_var="direction",
    efth_var="efth",  # Energy density
)
```

### Variable Mapping

Data interfaces use `*_var` suffixes to map your data variables:

| Interface Field | XBeach Meaning |
|----------------|----------------|
| `hm0_var` | Significant wave height variable name |
| `tp_var` | Peak period variable name |
| `dir_var` | Mean wave direction variable name |
| `spread_var` | Directional spread variable name |
| `freq_var` | Frequency array variable name |
| `efth_var` | Energy density variable name |

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
    var="zs",  # Water level variable
    tideloc=1,
)
```

This generates `zs0file` and sets:

- `tideloc` — Number of tide boundary locations
- `tidelen` — Length of tide time series

## Wind Forcing

### Spatially Uniform Wind

```python
from rompy_xbeach.data.wind import WindStation

wind = WindStation(
    source=dict(
        model_type="dataset",
        uri="wind.nc",
    ),
    u_var="u10",  # U-component
    v_var="v10",  # V-component
)
```

### Spatially Varying Wind

```python
from rompy_xbeach.data.wind import WindGrid

wind = WindGrid(
    source=dict(
        model_type="dataset",
        uri="wind_grid.nc",
    ),
    u_var="u10",
    v_var="v10",
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
source=dict(
    model_type="xyz:crs",
    filename="bathymetry.xyz",
    crs="EPSG:4326",
)
```

## Time Handling

Data interfaces automatically handle time:

- Extract data for the simulation period
- Interpolate to required time steps
- Handle timezone conversions

The simulation period comes from the `ModelRun`:

```python
model = XBeachModel(
    period=dict(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    ),
    config=config,
)
```

## Spatial Interpolation

For gridded data, interpolation to the XBeach grid is automatic:

```python
bathy = StaticBathy(
    source=dict(
        model_type="dataset",
        uri="bathymetry.nc",
    ),
    interpolator=dict(
        model_type="regular_grid",
    ),
)
```

## Manual Boundary Specification

If you have pre-existing boundary files, use the parameter components instead:

```python
from rompy_xbeach.components.boundary import SpectralWaveBoundary

config = Config(
    # Don't use input.wave
    wave_boundary=SpectralWaveBoundary(
        wbctype="jonstable",
        bcfile="my_jonswap.txt",  # Pre-existing file
        dtbc=1.0,
    ),
)
```

!!! warning "Don't mix data interfaces and manual specification"
    You cannot specify both `input.wave` and `wave_boundary`. Choose one approach.

## Example: Complete Data-Driven Setup

```python
from rompy_xbeach import Config
from rompy_xbeach.data import Input
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable
from rompy_xbeach.data.waterlevel import TideConsGrid
from rompy_xbeach.data.wind import WindStation

config = Config(
    grid=grid,
    bathy=bathy,
    input=Input(
        wave=BoundaryStationSpectraJonstable(
            source=dict(
                model_type="dataset",
                uri="https://thredds.example.com/waves.nc",
            ),
            hm0_var="hs",
            tp_var="tp",
            dir_var="dir",
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
            u_var="u10",
            v_var="v10",
        ),
    ),
)
```

## Next Steps

- [Parameter Reference](parameter-reference.md) — Find boundary-related parameters
- [Boundaries Component](../components/boundaries.md) — Manual boundary specification
