# Wave Boundaries

Rompy-xbeach provides comprehensive support for generating XBeach wave boundary conditions from external data sources. The boundary system handles multiple data structures (grids, stations, points), various XBeach boundary types (JONSWAP, JONSTABLE, SWAN), and automatic file generation.

## Two Approaches

There are two ways to specify wave boundaries:

### 1. Manual Specification (`Config.wave_boundary`)

Use pre-existing boundary files or simple parametric conditions:

```python
from rompy_xbeach.components.boundary.specification import SpectralWaveBoundary

config = Config(
    grid=grid,
    bathy=bathy,
    wave_boundary=SpectralWaveBoundary(
        wbctype="jons",
        bcfile="jonswap.txt",  # Pre-existing file
    ),
)
```

### 2. Data-Driven Generation (`Config.input.wave`)

Automatically generate boundary files from data sources:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJons

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJons(
            source=source,
            # ...
        ),
    ),
)
```

!!! warning "Cannot Mix Approaches"
    Use either `wave_boundary` OR `input.wave`, not both.

## Data-Driven Boundary Classes

### Naming Convention

Boundary classes follow a consistent naming pattern:

```
Boundary{DataStructure}{DataType}{BcType}
```

| Component | Options | Description |
|-----------|---------|-------------|
| **DataStructure** | `Station`, `Grid`, `Point` | Spatial structure of source data |
| **DataType** | `Spectra`, `Param` | Full spectra or integrated parameters |
| **BcType** | `Jons`, `Jonstable`, `Swan` | XBeach boundary type |

### Available Classes

| Class | Source Data | XBeach Type |
|-------|-------------|-------------|
| [`BoundaryStationSpectraJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraJons) | Station 2D spectra | `jons` |
| [`BoundaryStationSpectraJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraJonstable) | Station 2D spectra | `jonstable` |
| [`BoundaryStationSpectraSwan`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraSwan) | Station 2D spectra | `swan` |
| [`BoundaryStationParamJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationParamJons) | Station parameters | `jons` |
| [`BoundaryStationParamJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationParamJonstable) | Station parameters | `jonstable` |
| [`BoundaryPointParamJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryPointParamJons) | Point timeseries | `jons` |
| [`BoundaryPointParamJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryPointParamJonstable) | Point timeseries | `jonstable` |

## XBeach Boundary Types

### JONSWAP (`jons`)

Parametric JONSWAP spectrum defined by Hm0, Tp, direction, and spreading:

```python
from rompy_xbeach.data.boundary import BoundaryStationParamJons

wbdata = BoundaryStationParamJons(
    source=source,
    hm0="hs",           # Variable name for wave height
    tp="tp",            # Variable name for peak period
    mainang="dir",      # Variable name for direction
    gammajsp="gamma",   # Variable name for peak enhancement (optional)
    dspr="spread",      # Variable name for directional spreading (optional)
)
```

**Output**: Individual JONSWAP files for each timestep, referenced by a filelist.

### JONSWAP Table (`jonstable`)

Time-varying JONSWAP parameters in a single table file:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

wbdata = BoundaryStationSpectraJonstable(source=source)
```

**Output**: Single `jonswap.txt` file with columns for Hm0, Tp, direction, etc.

### SWAN 2D Spectra (`swan`)

Full 2D directional spectra in SWAN format:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraSwan

wbdata = BoundaryStationSpectraSwan(source=source)
```

**Output**: SWAN-format spectral files preserving full directional information.

## Data Structures

### Station Data

Multi-point data with a station/site dimension. Data is selected at the location nearest to the grid's offshore boundary:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJons

wbdata = BoundaryStationSpectraJons(
    source=source,
    coords=dict(
        x="lon",      # Longitude coordinate
        y="lat",      # Latitude coordinate
        s="site",     # Station dimension
        t="time",     # Time dimension (optional, uses default)
    ),
    sel_method="idw",  # Selection method: "idw" or "nearest"
    sel_method_kwargs=dict(tolerance=0.5),
)
```

### Grid Data

Spatially gridded data. Data is interpolated or selected at the offshore boundary:

```python
wbdata = BoundaryGridSpectraJons(
    source=source,
    coords=dict(x="longitude", y="latitude"),
    sel_method="interp",  # or "sel"
)
```

### Point Data

Single-point timeseries with no spatial selection:

```python
from rompy_xbeach.data.boundary import BoundaryPointParamJons

wbdata = BoundaryPointParamJons(
    source=source,
    hm0="phs1",
    tp="ptp1",
    mainang="pdp1",
)
```

## Source Objects

### Wavespectra Source

For spectral data compatible with the `wavespectra` library:

```python
from rompy_xbeach.source import SourceCRSWavespectra

source = SourceCRSWavespectra(
    uri="wave_spectra.nc",
    crs="EPSG:4326",
)
```

### Dataset Source

For NetCDF files with integrated parameters:

```python
from rompy_xbeach.source import SourceCRSDataset

source = SourceCRSDataset(
    uri="wave_params.nc",
    crs="EPSG:4326",
)
```

### CSV Timeseries

For point data from CSV files:

```python
from rompy.core.source import SourceTimeseriesCSV

source = SourceTimeseriesCSV(
    filename="wave_params.csv",
    tcol="time",
)
```

## File Generation Options

### Single vs Multiple Files

Control whether to generate one file or multiple files:

```python
# Multiple files (one per timestep) - DEFAULT
wbdata = BoundaryStationSpectraJons(
    source=source,
    filelist=True,  # Creates jons-filelist.txt + individual files
)

# Single file at simulation start
wbdata = BoundaryStationSpectraJons(
    source=source,
    filelist=False,  # Creates single jonswap file
)
```

### Time Parameters

```python
wbdata = BoundaryStationSpectraJons(
    source=source,
    dtbc=2.0,    # Timestep for boundary conditions (s)
    fnyq=0.3,    # Nyquist frequency (Hz)
)
```

## Variable Mapping

### From Spectra

When using spectral data, parameters are computed automatically:

```python
# Spectra source - parameters derived from spectra
wbdata = BoundaryStationSpectraJonstable(source=source_spectra)
```

### From Parameters

When using integrated parameters, map variable names:

```python
wbdata = BoundaryStationParamJons(
    source=source,
    hm0="phs1",        # Wave height variable
    tp="ptp1",         # Peak period variable
    mainang="pdp1",    # Direction variable
    gammajsp="ppe1",   # Peak enhancement (optional)
    dspr="pspr1",      # Directional spreading (optional)
)
```

### Mixed: Variables and Constants

You can mix data variables with constant values:

```python
wbdata = BoundaryStationParamJons(
    source=source,
    hm0="phs1",        # From data
    tp="ptp1",         # From data
    mainang="pdp1",    # From data
    gammajsp=3.3,      # Constant value
    dspr=20.0,         # Constant value (degrees)
)
```

## Wave Boundary Parameters (`wbc`)

Both manual and data-driven boundaries can use additional parameters:

```python
from rompy_xbeach.components.boundary.parameters import SpectralWaveBoundaryConditions

wbdata = BoundaryStationParamJons(
    source=source,
    hm0="phs1",
    tp="ptp1",
    mainang="pdp1",
    wbc=SpectralWaveBoundaryConditions(
        nmax=0.8,              # Ratio of max/mean frequency
        rt=3600.0,             # Duration of wave record (s)
        dtbc=1.0,              # Time step in boundary file (s)
        wbcevarreduce=1.0,     # Variance reduction factor
        bclwonly=False,        # Use only long waves
        wbcRemoveStokes=True,  # Remove Stokes drift
        wbcScaleEnergy=True,   # Scale energy to match Hm0
        correcthm0=True,       # Correct Hm0
    ),
)
```

## The `get()` Method

All boundary classes implement `get()` to generate files:

```python
wbc = wbdata.get(
    destdir=Path("./run"),
    grid=grid,
    time=TimeRange(start="2024-01-01", end="2024-01-02"),
)

# Returns a WaveBoundary object with params
print(wbc.params)
# {'wbctype': 'jons', 'bcfile': 'jons-filelist.txt', ...}
```

### What `get()` Does

1. Opens the source data
2. Selects/interpolates at the offshore boundary location
3. Slices to the simulation time period
4. Computes spectral parameters (if from spectra)
5. Writes XBeach-compatible boundary files
6. Returns parameters for `params.txt`

## Integration with Config

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(
            source=source,
            wbc=SpectralWaveBoundaryConditions(
                nmax=0.8,
                rt=3600.0,
            ),
        ),
    ),
)
```

## Example: Complete Workflow

```python
from pathlib import Path
from rompy.core.time import TimeRange
from rompy_xbeach.grid import RegularGrid, GeoPoint
from rompy_xbeach.source import SourceCRSWavespectra
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable
from rompy_xbeach.components.boundary.parameters import SpectralWaveBoundaryConditions

# 1. Define grid
grid = RegularGrid(
    ori=GeoPoint(x=115.594239, y=-32.641104, crs="EPSG:4326"),
    alfa=347.0,
    dx=10, dy=15, nx=230, ny=220,
    crs="EPSG:28350",
)

# 2. Define source
source = SourceCRSWavespectra(
    uri="wave_spectra.nc",
    crs="EPSG:4326",
)

# 3. Create boundary object
wbdata = BoundaryStationSpectraJonstable(
    source=source,
    coords=dict(x="lon", y="lat", s="site"),
    wbc=SpectralWaveBoundaryConditions(
        nmax=0.8,
        rt=3600.0,
        dtbc=1.0,
    ),
)

# 4. Generate files
destdir = Path("./xbeach_run")
times = TimeRange(start="2024-01-01T00", end="2024-01-02T00")

wbc = wbdata.get(destdir=destdir, grid=grid, time=times)

# 5. Check output
print(f"Boundary type: {wbc.wbctype}")
print(f"Boundary file: {wbc.bcfile}")
print(f"Parameters: {wbc.params}")
```

## Troubleshooting

### No Data at Offshore Location

**Symptom**: Empty or NaN values in boundary files.

**Cause**: Source data doesn't cover the grid's offshore boundary.

**Fix**: Check source data extent and grid positioning, or increase selection tolerance:

```python
wbdata = BoundaryStationSpectraJons(
    source=source,
    sel_method_kwargs=dict(tolerance=1.0),  # Increase tolerance
)
```

### Time Mismatch

**Symptom**: Boundary files don't cover simulation period.

**Cause**: Source data time range doesn't overlap with simulation period.

**Fix**: Verify source data time coverage:

```python
# Check source data times
ds = source.open()
print(ds.time.values)
```

### Direction Convention

**Symptom**: Waves coming from wrong direction.

**Cause**: Different direction conventions (from vs to, nautical vs cartesian).

**Fix**: Check and adjust direction mapping in your source data.

## See Also

- [Forcing](forcing.md) — Wind and tide forcing
- [Examples: Wave Boundary Demo](../examples/wave-boundary-demo.ipynb) — Interactive notebook
