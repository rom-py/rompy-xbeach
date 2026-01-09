# Wave Boundaries

Wave boundary conditions in rompy-xbeach are specified through `Config.input.wave`. This unified interface supports:

- **Data-driven boundaries** — Generate boundary files from external data sources (spectra, parameters)
- **File-based boundaries** — Use pre-existing boundary files
- **Parametric boundaries** — Simple stationary wave conditions without files
- **Special boundaries** — No waves or reuse from previous simulations

All wave boundary classes are in the `rompy_xbeach.data.boundary` module.

## Class Naming Convention

Data-driven spectral boundary classes follow a consistent naming pattern:

```
Boundary{SourceType}{DataType}{OutputFormat}
```

Where:

- **SourceType**: `Station`, `Grid`, or `Point` — how the source data is structured
- **DataType**: `Spectra` or `Param` — whether source contains 2D spectra or bulk parameters
- **OutputFormat**: `Jons`, `Jonstable`, or `Swan` — the XBeach boundary file format

Examples:

| Class Name | Source | Data | Output |
|------------|--------|------|--------|
| `BoundaryStationSpectraJonstable` | Station (multi-point) | 2D Spectra | JONSTABLE |
| `BoundaryGridParamJons` | Grid (spatial) | Parameters | JONSWAP |
| `BoundaryPointParamJonstable` | Point (single) | Parameters | JONSTABLE |

## Quick Start

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable

# Data-driven: generate JONSTABLE from wave spectra
config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(
            source=spectra_source,
            hm0_var="hs",
            tp_var="tp",
            mainang_var="dir",
        ),
    ),
)
```

## Boundary Class Categories

### 1. Data-Driven Spectral Boundaries

Generate spectral boundary files from external data sources. These classes read wave data, extract/interpolate to the model boundary, and write XBeach-compatible files.

| Class | Source Type | Output Format |
|-------|-------------|---------------|
| [`BoundaryStationSpectraJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraJons) | Station spectra | Single JONSWAP |
| [`BoundaryStationSpectraJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraJonstable) | Station spectra | Time-varying JONSTABLE |
| [`BoundaryStationSpectraSwan`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationSpectraSwan) | Station spectra | SWAN 2D spectrum |
| [`BoundaryStationParamJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationParamJons) | Station parameters | Single JONSWAP |
| [`BoundaryStationParamJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStationParamJonstable) | Station parameters | Time-varying JONSTABLE |
| [`BoundaryGridParamJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryGridParamJons) | Gridded parameters | Single JONSWAP |
| [`BoundaryGridParamJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryGridParamJonstable) | Gridded parameters | Time-varying JONSTABLE |
| [`BoundaryPointParamJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryPointParamJons) | Point timeseries | Single JONSWAP |
| [`BoundaryPointParamJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryPointParamJonstable) | Point timeseries | Time-varying JONSTABLE |

### 2. File-Based Spectral Boundaries

Use pre-existing boundary files. These classes fetch files from a source location and configure XBeach to use them.

| Class | File Type | Description |
|-------|-----------|-------------|
| [`BoundaryFileJons`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryFileJons) | JONSWAP | Single or multiple JONSWAP files |
| [`BoundaryFileJonstable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryFileJonstable) | JONSTABLE | Time-varying JONSWAP table |
| [`BoundaryFileSwan`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryFileSwan) | SWAN | SWAN 2D spectrum files |

### 3. Non-Spectral Boundaries

Simple parametric boundaries or time series from files.

| Class | XBeach `wbctype` | Description |
|-------|------------------|-------------|
| [`BoundaryStat`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStat) | `stat` | Stationary parametric (no file needed) |
| [`BoundaryBichrom`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryBichrom) | `bichrom` | Bichromatic waves (no file needed) |
| [`BoundaryStatTable`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryStatTable) | `stat_table` | Time-varying parametric from file |
| [`BoundaryTs1`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryTs1) | `ts_1` | Time series at single location |
| [`BoundaryTs2`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryTs2) | `ts_2` | Time series at two locations |
| [`BoundaryTsNonh`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryTsNonh) | `ts_nonh` | Non-hydrostatic time series |

### 4. Special Boundaries

| Class | XBeach `wbctype` | Description |
|-------|------------------|-------------|
| [`BoundaryOff`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryOff) | `off` | No wave forcing |
| [`BoundaryReuse`](../api-reference/data.md#rompy_xbeach.data.boundary.BoundaryReuse) | `reuse` | Reuse files from previous simulation |

---

## Data-Driven Boundaries

### From Wave Spectra

When you have 2D wave spectra (frequency × direction), use the `Spectra` classes:

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable
from rompy_xbeach.source import SourceCRSWavespectra

# Source: wave spectra from NetCDF
source = SourceCRSWavespectra(
    uri="wave_spectra.nc",
    reader="read_ww3",  # wavespectra reader
    crs=4326,
)

# Generate time-varying JONSTABLE boundary
wave = BoundaryStationSpectraJonstable(
    source=source,
    location="offshore",  # Extract at offshore boundary
    filelist=True,        # Generate FILELIST with multiple files
)
```

### From Wave Parameters

When you have bulk wave parameters (Hm0, Tp, direction), use the `Param` classes:

```python
from rompy_xbeach.data.boundary import BoundaryGridParamJonstable
from rompy_xbeach.source import SourceCRSDataset

# Source: gridded wave parameters
source = SourceCRSDataset(
    uri="wave_params.nc",
    crs=4326,
)

# Generate JONSTABLE from parameters
wave = BoundaryGridParamJonstable(
    source=source,
    hm0_var="hs",           # Variable name for Hm0
    tp_var="tp",            # Variable name for Tp
    mainang_var="dir",      # Variable name for direction
    gammajsp_var="gamma",   # Optional: JONSWAP gamma
    dspr_var="spr",         # Optional: directional spreading
    location="offshore",
    filelist=True,
)
```

### Source Types

| Source Type | Class | Use Case |
|-------------|-------|----------|
| **Station** | `BoundaryStation*` | Multi-point data, nearest station selected |
| **Grid** | `BoundaryGrid*` | Gridded data, interpolated to boundary |
| **Point** | `BoundaryPoint*` | Single-point timeseries |

### Location Options

The `location` field controls where data is extracted:

| Value | Description |
|-------|-------------|
| `offshore` | Middle of the offshore (seaward) boundary |
| `centre` | Centre of the model grid |
| `grid` | All grid points (for spatially-varying boundaries) |

---

## File-Based Boundaries

Use pre-existing boundary files with the `BoundaryFile*` classes:

```python
from rompy_xbeach.data.boundary import BoundaryFileJonstable
from rompy_xbeach.types import XBeachDataBlob

# Single JONSTABLE file
wave = BoundaryFileJonstable(
    bcfile_source=XBeachDataBlob(source="jonstable.txt"),
)

# Multiple files via FILELIST
wave = BoundaryFileJonstable(
    bcfile_source=XBeachDataBlob(source="filelist.txt"),
    filelist=True,  # Source file is a FILELIST
)
```

### FILELIST Format

When `filelist=True`, the source file should be in XBeach FILELIST format:

```
FILELIST
1800 0.2 jonswap1.inp
1800 0.2 jonswap2.inp
3600 0.2 jonswap3.inp
```

Each line after `FILELIST` contains: `<duration> <timestep> <filename>`

The `FilelistMixin` automatically:
1. Parses the FILELIST file
2. Fetches all referenced boundary files
3. Copies them to the run directory

---

## Non-Spectral Boundaries

### Stationary Parametric

For simple, constant wave conditions without files:

```python
from rompy_xbeach.data.boundary import BoundaryStat

wave = BoundaryStat(
    Hrms=1.0,    # Root-mean-square wave height [m]
    Tp=10.0,     # Peak period [s]
    dir0=270.0,  # Mean direction [deg]
    s=20.0,      # Directional spreading [-]
)
```

### Bichromatic Waves

For laboratory-style bichromatic wave conditions:

```python
from rompy_xbeach.data.boundary import BoundaryBichrom

wave = BoundaryBichrom(
    Hrms=0.5,
    Tp=8.0,
    dir0=270.0,
    s=1000.0,    # Narrow spreading
)
```

### Time Series from Files

```python
from rompy_xbeach.data.boundary import BoundaryTs1
from rompy_xbeach.types import XBeachDataBlob

wave = BoundaryTs1(
    source=XBeachDataBlob(source="bc_gen.ezs"),
)
```

---

## Special Boundaries

### No Wave Forcing

```python
from rompy_xbeach.data.boundary import BoundaryOff

wave = BoundaryOff()
```

### Reuse Previous Simulation

```python
from rompy_xbeach.data.boundary import BoundaryReuse
from rompy_xbeach.types import XBeachDirectoryBlob

wave = BoundaryReuse(
    previous_run=XBeachDirectoryBlob(source="/path/to/previous/run"),
)
```

---

## XBeach Boundary Types Reference

| `wbctype` | Description | Rompy-xbeach Classes |
|-----------|-------------|---------------------|
| `jons` | Single JONSWAP spectrum | `BoundaryStationSpectraJons`, `BoundaryStationParamJons`, `BoundaryGridParamJons`, `BoundaryPointParamJons`, `BoundaryFileJons` |
| `jonstable` | Time-varying JONSWAP table | `BoundaryStationSpectraJonstable`, `BoundaryStationParamJonstable`, `BoundaryGridParamJonstable`, `BoundaryPointParamJonstable`, `BoundaryFileJonstable` |
| `swan` | SWAN 2D spectrum | `BoundaryStationSpectraSwan`, `BoundaryFileSwan` |
| `stat` | Stationary parametric | `BoundaryStat` |
| `bichrom` | Bichromatic | `BoundaryBichrom` |
| `stat_table` | Time-varying parametric | `BoundaryStatTable` |
| `ts_1` | Time series (1 location) | `BoundaryTs1` |
| `ts_2` | Time series (2 locations) | `BoundaryTs2` |
| `ts_nonh` | Non-hydrostatic time series | `BoundaryTsNonh` |
| `off` | No waves | `BoundaryOff` |
| `reuse` | Reuse previous | `BoundaryReuse` |

---

## Wave Boundary Parameters

Wave boundary parameters are specified directly on the boundary data classes via `Config.input.wave`, not through separate component objects. This design ensures that spectral-specific parameters (like `thetamin`, `dtheta`) are only available on spectral boundary classes, preventing invalid parameter combinations.

### Parameter Inheritance

All wave boundary classes inherit from one of two base parameter classes:

| Base Class | Inheriting Classes | Key Parameters |
|------------|-------------------|----------------|
| [`WaveBoundaryParams`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | All boundary classes | `rt`, `dtbc`, `random`, `sprdthr` |
| [`SpectralWaveBoundaryParams`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Spectral classes only | Above + `thetamin`, `thetamax`, `dtheta`, `thetanaut`, `correctHm0` |

This inheritance structure means:

- **Spectral boundaries** (JONS, JONSTABLE, SWAN) have access to directional grid parameters
- **Non-spectral boundaries** (stat, bichrom, ts_*) do not expose spectral parameters, avoiding configuration errors

### Common Parameters (All Boundaries)

| Parameter | Description | Default |
|-----------|-------------|--------|
| `rt` | Duration of wave time series [s] | None |
| `dtbc` | Timestep for boundary condition [s] | 1.0 |
| `random` | Random seed for wave generation | True |
| `sprdthr` | Threshold for directional spreading | 0.08 |

### Spectral Parameters (Spectral Boundaries Only)

| Parameter | Description | Default |
|-----------|-------------|--------|
| `thetamin` | Minimum wave direction [deg] | -90 |
| `thetamax` | Maximum wave direction [deg] | 90 |
| `dtheta` | Directional resolution [deg] | 10 |
| `thetanaut` | Use nautical convention | False |
| `correctHm0` | Correct Hm0 for directional spreading | True |

Example:

```python
wave = BoundaryStationSpectraJonstable(
    source=source,
    # Spectral parameters (only available on spectral classes)
    thetamin=-60,
    thetamax=60,
    dtheta=5,
    thetanaut=True,
    # Common parameters
    dtbc=0.5,
)
```

---

## Complete Example

```python
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.boundary import BoundaryStationSpectraJonstable
from rompy_xbeach.source import SourceCRSWavespectra
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.bathy import XBeachBathy

# Define grid
grid = RegularGrid(...)

# Define bathymetry
bathy = XBeachBathy(...)

# Define wave source
wave_source = SourceCRSWavespectra(
    uri="https://thredds.server/waves.nc",
    reader="read_ww3",
    crs=4326,
)

# Create config with wave boundary
config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=BoundaryStationSpectraJonstable(
            source=wave_source,
            location="offshore",
            filelist=True,
            thetamin=-90,
            thetamax=90,
            dtheta=10,
        ),
    ),
)

# Generate XBeach files
config.generate(destdir="./run")
```

---

## API Reference

See [Wave Boundaries API](../api-reference/data.md#wave-boundaries) for complete class documentation.
