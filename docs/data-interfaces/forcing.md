# Forcing

Rompy-xbeach provides data interfaces for wind and tide forcing. These handle the extraction, interpolation, and file generation required to drive XBeach simulations with external forcing data.

## Overview

XBeach supports several types of forcing:

| Forcing Type | XBeach File | Data Interface Classes |
|--------------|-------------|------------------------|
| **Wind** | `wind.txt` | [`WindStation`](../api-reference/data.md#rompy_xbeach.data.wind.WindStation), [`WindGrid`](../api-reference/data.md#rompy_xbeach.data.wind.WindGrid), `WindPoint` |
| **Tide (constituents)** | `zs0file.txt` | [`TideConsGrid`](../api-reference/data.md#rompy_xbeach.data.waterlevel.TideConsGrid), `TideConsPoint` |
| **Water level (timeseries)** | `zs0file.txt` | [`WaterLevelStation`](../api-reference/data.md#rompy_xbeach.data.waterlevel.WaterLevelStation), `WaterLevelGrid`, `WaterLevelPoint` |

## Wind Forcing

### Data Structure Types

| Class | Source Data | Use Case |
|-------|-------------|----------|
| [`WindStation`](../api-reference/data.md#rompy_xbeach.data.wind.WindStation) | Multi-point stations | Regional model output with station locations |
| [`WindGrid`](../api-reference/data.md#rompy_xbeach.data.wind.WindGrid) | Spatially gridded | Reanalysis or forecast grids |
| `WindPoint` | Single-point timeseries | Local measurements, CSV files |

### Wind Variables

Wind can be specified as vectors (u, v components) or scalars (speed, direction):

```python
from rompy_xbeach.data.wind import WindVector, WindScalar

# Vector components (u, v)
wind_vars = WindVector(u="uwnd", v="vwnd")

# Scalar (speed, direction)
wind_vars = WindScalar(spd="wspd", dir="wdir")
```

### WindStation

For multi-point station data with spatial selection:

```python
from rompy_xbeach.data.wind import WindStation, WindVector

wind = WindStation(
    source=source,
    coords=dict(
        s="station",    # Station dimension
        x="longitude",  # X coordinate
        y="latitude",   # Y coordinate
    ),
    wind_vars=WindVector(u="uwnd", v="vwnd"),
    sel_method="idw",   # "idw" or "nearest"
    sel_method_kwargs=dict(tolerance=0.2),
    location="centre",  # "centre" or "offshore"
)

params = wind.get(destdir=destdir, grid=grid, time=times)
```

| Parameter | Description |
|-----------|-------------|
| `sel_method` | Selection method: `"idw"` (inverse distance weighting) or `"nearest"` |
| `location` | Where to extract: `"centre"` (grid centre) or `"offshore"` (offshore boundary) |

### WindGrid

For spatially gridded data:

```python
from rompy_xbeach.data.wind import WindGrid, WindVector

wind = WindGrid(
    source=source,
    coords=dict(x="longitude", y="latitude"),
    wind_vars=WindVector(u="u10", v="v10"),
    sel_method="sel",      # "sel" or "interp"
    sel_method_kwargs=dict(method="nearest"),
)

params = wind.get(destdir=destdir, grid=grid, time=times)
```

### WindPoint

For single-point timeseries (no spatial selection):

```python
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.data.wind import WindPoint, WindScalar

source = SourceTimeseriesCSV(
    filename="wind.csv",
    tcol="time",
)

wind = WindPoint(
    source=source,
    wind_vars=WindScalar(spd="wspd", dir="wdir"),
)

params = wind.get(destdir=destdir, grid=grid, time=times)
```

### Wind Output Format

The generated `wind.txt` file contains:

```
tsec    wspd    wdir
0       8.5     270
3600    9.2     265
7200    8.8     268
...
```

Where:
- `tsec` — Time in seconds from simulation start
- `wspd` — Wind speed (m/s)
- `wdir` — Wind direction (degrees, nautical convention)

## Tide Forcing

### From Constituents

Generate tide elevation timeseries from harmonic constituents using `oceantide`:

#### TideConsGrid

For gridded constituent data (e.g., OTIS, FES):

```python
from rompy_xbeach.source import SourceCRSOceantide
from rompy_xbeach.data.waterlevel import TideConsGrid

source = SourceCRSOceantide(
    reader="read_otis_binary",
    kwargs=dict(
        gfile="grid_file",
        hfile="elevation_file",
        ufile="transport_file",
    ),
    crs="EPSG:4326",
)

tide = TideConsGrid(
    source=source,
    coords=dict(x="lon", y="lat"),
)

params = tide.get(destdir=destdir, grid=grid, time=times)
```

#### TideConsPoint

For single-point constituent data (e.g., from tide gauge analysis):

```python
from rompy_xbeach.source import SourceTideConsPointCSV
from rompy_xbeach.data.waterlevel import TideConsPoint

source = SourceTideConsPointCSV(
    filename="tide_constituents.csv",
    acol="amplitude",    # Amplitude column
    pcol="phase",        # Phase column
    ccol="constituent",  # Constituent name column
)

tide = TideConsPoint(source=source)

params = tide.get(destdir=destdir, grid=grid, time=times)
```

The CSV should have format:
```csv
constituent,amplitude,phase
M2,0.45,123.5
S2,0.18,156.2
K1,0.12,45.8
...
```

### From Water Level Timeseries

For pre-computed water level timeseries:

#### WaterLevelStation

```python
from rompy_xbeach.data.waterlevel import WaterLevelStation

tide = WaterLevelStation(
    source=source,
    coords=dict(s="station", x="lon", y="lat"),
    variables=["zs"],  # Water level variable name
    tideloc=1,         # Number of tide boundary locations
)
```

#### WaterLevelPoint

```python
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.data.waterlevel import WaterLevelPoint

source = SourceTimeseriesCSV(
    filename="ssh.csv",
    tcol="time",
)

tide = WaterLevelPoint(
    source=source,
    variables=["ssh"],
)

params = tide.get(destdir=destdir, grid=grid, time=times)
```

### Tide Output Format

The generated `zs0file.txt` contains:

```
tsec    zs
0       0.45
3600    0.32
7200    0.15
...
```

Where:
- `tsec` — Time in seconds from simulation start
- `zs` — Water surface elevation (m)

## Source Objects

### For Wind/Tide Grids

```python
from rompy_xbeach.source import SourceCRSDataset

source = SourceCRSDataset(
    uri="forcing.nc",
    crs="EPSG:4326",
    x_dim="longitude",
    y_dim="latitude",
)
```

### For Oceantide

```python
from rompy_xbeach.source import SourceCRSOceantide

source = SourceCRSOceantide(
    reader="read_otis_binary",  # or other oceantide readers
    kwargs=dict(...),
    crs="EPSG:4326",
)
```

### For CSV Timeseries

```python
from rompy.core.source import SourceTimeseriesCSV

source = SourceTimeseriesCSV(
    filename="data.csv",
    tcol="time",  # Time column name
)
```

### For Tide Constituents CSV

```python
from rompy_xbeach.source import SourceTideConsPointCSV

source = SourceTideConsPointCSV(
    filename="constituents.csv",
    acol="amplitude",
    pcol="phase",
    ccol="constituent",
)
```

## Integration with Config

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.wind import WindStation, WindVector
from rompy_xbeach.data.waterlevel import TideConsGrid

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=wave_boundary,
        wind=WindStation(
            source=wind_source,
            wind_vars=WindVector(u="u10", v="v10"),
        ),
        tide=TideConsGrid(
            source=tide_source,
        ),
    ),
)
```

## Example: Complete Forcing Setup

```python
from pathlib import Path
from rompy.core.time import TimeRange
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.grid import RegularGrid, GeoPoint
from rompy_xbeach.source import SourceTideConsPointCSV
from rompy_xbeach.data.wind import WindPoint, WindScalar
from rompy_xbeach.data.waterlevel import TideConsPoint

# 1. Define grid
grid = RegularGrid(
    ori=GeoPoint(x=115.594239, y=-32.641104, crs="EPSG:4326"),
    alfa=347.0,
    dx=10, dy=15, nx=230, ny=220,
    crs="EPSG:28350",
)

# 2. Define time range
times = TimeRange(start="2024-01-01T00", end="2024-01-02T00", interval="1h")

# 3. Wind from CSV
wind_source = SourceTimeseriesCSV(filename="wind.csv", tcol="time")
wind = WindPoint(
    source=wind_source,
    wind_vars=WindScalar(spd="wspd", dir="wdir"),
)

# 4. Tide from constituents
tide_source = SourceTideConsPointCSV(
    filename="tide_cons.csv",
    acol="amplitude",
    pcol="phase",
    ccol="constituent",
)
tide = TideConsPoint(source=tide_source)

# 5. Generate files
destdir = Path("./xbeach_run")
destdir.mkdir(exist_ok=True)

wind_params = wind.get(destdir=destdir, grid=grid, time=times)
tide_params = tide.get(destdir=destdir, grid=grid, time=times)

print(f"Wind file: {wind_params['windfile']}")
print(f"Tide file: {tide_params['zs0file']}")
```

## Bulk Forcing

For simple scenarios where forcing is defined by single-point timeseries (common in nearshore modelling where spatial variation is minimal), use the "Point" classes:

- `WindPoint` — Wind from CSV or DataFrame
- `WaterLevelPoint` — Water level from CSV or DataFrame
- `TideConsPoint` — Tide from constituent table
- `BoundaryPointParamJons` — Waves from parameter timeseries
- `BoundaryPointParamJonstable` — Waves from parameter timeseries

These are ideal for:

- Local measurements from a single station
- Simplified forcing scenarios
- Quick model setup and testing

```python
# All forcing from CSV files
config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryPointParamJonstable(
            source=SourceTimeseriesCSV(filename="waves.csv"),
            hm0="hs", tp="tp", mainang="dir",
        ),
        wind=WindPoint(
            source=SourceTimeseriesCSV(filename="wind.csv"),
            wind_vars=WindScalar(spd="wspd", dir="wdir"),
        ),
        tide=WaterLevelPoint(
            source=SourceTimeseriesCSV(filename="tide.csv"),
            variables=["ssh"],
        ),
    ),
)
```

## Troubleshooting

### No Data at Grid Location

**Symptom**: Empty or constant values in output files.

**Cause**: Source data doesn't cover the grid location.

**Fix**: Check source data extent and increase selection tolerance:

```python
wind = WindStation(
    source=source,
    sel_method_kwargs=dict(tolerance=1.0),
)
```

### Time Zone Issues

**Symptom**: Forcing appears shifted in time.

**Cause**: Mismatch between source data timezone and simulation time.

**Fix**: Ensure consistent timezone handling in your source data.

### Direction Convention

**Symptom**: Wind/waves from wrong direction.

**Cause**: Different direction conventions (meteorological vs oceanographic).

**Fix**: XBeach uses nautical convention (direction FROM). Check your source data convention.

## See Also

- [Wave Boundaries](boundaries.md) — Wave boundary generation
- [Examples: Forcing Demo](../examples/forcing-demo.ipynb) — Interactive notebook
