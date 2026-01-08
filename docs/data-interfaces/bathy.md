# Bathymetry

The `XBeachBathy` class handles bathymetry data processing for XBeach simulations. It provides powerful capabilities for interpolating source data onto the model grid, extending the domain offshore, and handling coordinate system transformations.

## Why a Bathymetry Abstraction?

Bathymetry data rarely comes in the exact format XBeach needs:

- **Different CRS** — Source data may be in WGS84 while the grid uses a projected CRS
- **Different resolution** — Source data resolution may not match the model grid
- **Insufficient extent** — Nearshore surveys often don't extend far enough offshore
- **Missing data** — Gaps in coverage need interpolation

`XBeachBathy` addresses all these challenges automatically.

## Basic Usage

```python
from rompy_xbeach.data.bathy import XBeachBathy
from rompy_xbeach.source import SourceGeotiff

# Define the bathymetry source
source = SourceGeotiff(filename="bathymetry.tif")

# Create the bathymetry object
bathy = XBeachBathy(
    source=source,
    posdwn=False,  # Depths are negative (below sea level)
)

# Generate XBeach files
xfile, yfile, depfile, grid = bathy.get(
    destdir=Path("./run"),
    grid=grid,
)
```

## Source Types

`XBeachBathy` accepts any source object that provides gridded data with CRS information. Common choices for bathymetry:

| Source | Best For |
|--------|----------|
| `SourceGeotiff` | GeoTIFF raster files (most common) |
| `SourceXYZ` | XYZ point cloud / survey data |
| `SourceCRSFile` | NetCDF, Zarr, or other xarray-compatible files |
| `SourceCRSDataset` | Existing xarray Dataset objects |

```python
from rompy_xbeach.source import SourceGeotiff, SourceXYZ

# GeoTIFF (simplest - CRS embedded in file)
source = SourceGeotiff(filename="bathymetry.tif")

# XYZ point data (requires gridding)
source = SourceXYZ(
    filename="survey.xyz",
    crs="EPSG:4326",
    res=0.0005,
)
```

For detailed documentation on all source types and their parameters, see [Sources](sources.md).

## Interpolation

### How It Works

1. Source data is opened with CRS information
2. Data is reprojected to the grid CRS if needed
3. Regular grid interpolation maps source data to model grid points
4. Missing values are handled according to settings

### Interpolator Configuration

The interpolator is configurable through an interface that allows different interpolation backends. Currently, a scipy-based interpolator is available, but the interface can be extended to support other interpolation methods if needed.

```python
from rompy_xbeach.interpolate import RegularGridInterpolator

interpolator = RegularGridInterpolator(
    kwargs=dict(
        method="linear",      # Interpolation method
        fill_value=None,      # Extrapolation handling
    ),
)

bathy = XBeachBathy(
    source=source,
    interpolator=interpolator,
)
```

The `RegularGridInterpolator` uses `scipy.interpolate.RegularGridInterpolator` under the hood. Available methods:

| Method | Description |
|--------|-------------|
| `"linear"` | Bilinear interpolation (default) |
| `"nearest"` | Nearest neighbour |
| `"cubic"` | Cubic interpolation |

### Handling Missing Data

```python
bathy = XBeachBathy(
    source=source,
    interpolate_na=True,   # Interpolate NaN values (default)
    # interpolate_na=False # Keep NaN values
)
```

## Depth Convention

XBeach supports both depth conventions via the `posdwn` parameter:

- **`posdwn=True`** — Positive down: depths below sea level are positive values
- **`posdwn=False`** — Negative down: depths below sea level are negative values

The `posdwn` parameter in `XBeachBathy` should match your **source data** convention:

```python
# Source has negative depths (below sea level = negative)
bathy = XBeachBathy(source=source, posdwn=False)

# Source has positive depths (below sea level = positive)
bathy = XBeachBathy(source=source, posdwn=True)
```

This setting is also written to `params.txt` so XBeach interprets the bathymetry correctly.

!!! warning "Check Your Convention"
    Incorrect `posdwn` setting will invert your bathymetry, placing deep water at the coast.

## Offshore Extension

Nearshore surveys often don't extend far enough offshore for wave modelling. `XBeachBathy` can automatically extend the bathymetry seaward.

### Linear Extension

Extends the offshore boundary with a constant slope until reaching a target depth:

```python
from rompy_xbeach.data.bathy import SeawardExtensionLinear

extension = SeawardExtensionLinear(
    depth=25,    # Target depth (m)
    slope=0.1,   # Slope of extension (rise/run)
)

bathy = XBeachBathy(
    source=source,
    extension=extension,
)
```

The extension:

1. Takes the depth at the offshore boundary
2. Extends seaward with the specified slope
3. Stops when reaching the target depth
4. Adds new grid rows to accommodate the extension

```python
# Gentler slope = longer extension
extension = SeawardExtensionLinear(depth=25, slope=0.02)

# Steeper slope = shorter extension
extension = SeawardExtensionLinear(depth=25, slope=0.2)
```

### Grid Modification

When extension is applied, the grid is modified:

```python
# Original grid
print(f"Original shape: {grid.shape}")  # (230, 220)

# After extension
xfile, yfile, depfile, extended_grid = bathy.get(destdir, grid)
print(f"Extended shape: {extended_grid.shape}")  # (280, 220)
```

The returned `extended_grid` has additional rows for the offshore extension.

## Lateral Extension

Extend the grid laterally (alongshore) by replicating edge values:

```python
bathy = XBeachBathy(
    source=source,
    left=10,   # Extend left boundary by 10 cells
    right=10,  # Extend right boundary by 10 cells
)
```

This is useful when:

- Source data doesn't fully cover the alongshore extent
- You want buffer zones at lateral boundaries

### Combined Extension

Offshore and lateral extensions can be combined:

```python
bathy = XBeachBathy(
    source=source,
    extension=SeawardExtensionLinear(depth=25, slope=0.02),
    left=10,
    right=10,
)
```

## Output Files

The `get()` method generates three XBeach input files:

| File | Description |
|------|-------------|
| `x.txt` | X-coordinates of grid points |
| `y.txt` | Y-coordinates of grid points |
| `dep.txt` | Depth values at grid points |

```python
xfile, yfile, depfile, grid = bathy.get(destdir, grid)

print(xfile)   # Path to x.txt
print(yfile)   # Path to y.txt
print(depfile) # Path to dep.txt
```

## Reading Output Data

Use the xarray accessor to read generated files:

```python
import xarray as xr

# Read bathymetry into xarray Dataset
dset = xr.Dataset.xbeach.from_xbeach(depfile, grid)

# Plot
dset.xbeach.plot_model_bathy(grid, posdwn=False)
```

## CRS Handling

### Automatic Reprojection

Source data is automatically reprojected to match the grid CRS:

```python
# Source in WGS84
source = SourceGeotiff(filename="bathy_wgs84.tif")  # EPSG:4326

# Grid in projected CRS
grid = RegularGrid(
    ori=GeoPoint(x=115.5, y=-32.6, crs="EPSG:4326"),
    crs="EPSG:28350",  # MGA Zone 50
    # ...
)

# Bathymetry is automatically reprojected
bathy = XBeachBathy(source=source)
bathy.get(destdir, grid)  # Handles CRS transformation
```

### Mixed CRS Scenarios

| Source CRS | Grid CRS | Handling |
|------------|----------|----------|
| WGS84 | Projected | Auto-reproject source to grid CRS |
| Projected | Same projected | Direct interpolation |
| Projected A | Projected B | Auto-reproject source to grid CRS |

## Integration with Config

```python
from rompy_xbeach.config import Config

config = Config(
    grid=grid,
    bathy=XBeachBathy(
        source=SourceGeotiff(filename="bathymetry.tif"),
        posdwn=False,
        extension=SeawardExtensionLinear(depth=25, slope=0.1),
    ),
    # ...
)
```

When `Config` is called, it:

1. Generates bathymetry files via `bathy.get()`
2. Updates grid parameters if extension was applied
3. Includes file paths in `params.txt`

## Example: Complete Workflow

```python
from pathlib import Path
from rompy_xbeach.grid import RegularGrid, GeoPoint
from rompy_xbeach.data.bathy import XBeachBathy, SeawardExtensionLinear
from rompy_xbeach.source import SourceGeotiff
from rompy_xbeach.interpolate import RegularGridInterpolator

# 1. Define the grid
grid = RegularGrid(
    ori=GeoPoint(x=115.594239, y=-32.641104, crs="EPSG:4326"),
    alfa=347.0,
    dx=10, dy=15,
    nx=230, ny=220,
    crs="EPSG:28350",
)

# 2. Configure bathymetry with extension
bathy = XBeachBathy(
    source=SourceGeotiff(filename="bathymetry.tif"),
    posdwn=False,
    interpolator=RegularGridInterpolator(
        kwargs=dict(method="linear", fill_value=None)
    ),
    extension=SeawardExtensionLinear(depth=25, slope=0.05),
    left=5,
    right=5,
)

# 3. Generate files
destdir = Path("./xbeach_run")
destdir.mkdir(exist_ok=True)

xfile, yfile, depfile, extended_grid = bathy.get(destdir, grid)

# 4. Verify
print(f"Original grid: {grid.shape}")
print(f"Extended grid: {extended_grid.shape}")
print(f"Files created: {list(destdir.glob('*.txt'))}")

# 5. Visualise
import xarray as xr
dset = xr.Dataset.xbeach.from_xbeach(depfile, extended_grid)
dset.xbeach.plot_model_bathy(extended_grid, posdwn=False)
```

## Troubleshooting

### Inverted Bathymetry

**Symptom**: Deep water appears at the coast, shallow water offshore.

**Cause**: Incorrect `posdwn` setting or grid orientation.

**Fix**: 
```python
# Check your depth convention
bathy = XBeachBathy(source=source, posdwn=False)  # Try toggling
```

### Missing Data at Boundaries

**Symptom**: NaN values at grid edges.

**Cause**: Source data doesn't fully cover the grid extent.

**Fix**:
```python
# Use lateral extension
bathy = XBeachBathy(source=source, left=10, right=10)

# Or use extrapolation in interpolator
interpolator = RegularGridInterpolator(
    kwargs=dict(method="nearest", fill_value=None)
)
```

### CRS Mismatch Errors

**Symptom**: Interpolation produces unexpected results or errors.

**Cause**: Source CRS not correctly specified.

**Fix**:
```python
# Explicitly set source CRS
source = SourceCRSDataset(
    uri="bathymetry.nc",
    crs="EPSG:4326",  # Specify explicitly
)
```

## See Also

- [Grid](grid.md) — Grid definition and CRS handling
- [Examples: Bathymetry Demo](../examples/bathy-demo.ipynb) — Interactive notebook with detailed examples
