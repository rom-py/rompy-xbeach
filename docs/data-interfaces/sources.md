# Source Objects

Source objects are the foundation of all data interfaces in rompy-xbeach. They provide a unified way to load data from various formats while ensuring proper coordinate reference system (CRS) handling.

## Overview

Source objects in rompy-xbeach extend the [core sources](https://github.com/rom-py/rompy/blob/main/src/rompy/core/source.py){target="_blank"} from rompy with **CRS awareness**. Every source's `open()` method returns an xarray Dataset with:

- The `rio` accessor available (via rioxarray)
- The `crs` attribute properly set

This enables automatic coordinate transformations when data needs to be reprojected to match the model grid.

## Available Sources

### SourceGeotiff

For GeoTIFF files, which typically contain embedded CRS information:

```python
from rompy_xbeach.source import SourceGeotiff

source = SourceGeotiff(
    filename="bathymetry.tif",
    band=1,  # Band to read (default: 1)
)
```

GeoTIFF is the simplest option when available — no CRS specification needed as it's read from the file metadata.

### SourceXYZ

For point-cloud data in XYZ format (CSV/text files with x, y, z columns):

```python
from rompy_xbeach.source import SourceXYZ

source = SourceXYZ(
    filename="bathymetry.xyz",
    crs="EPSG:4326",
    res=0.0005,                             # Resolution for gridding
    xcol="easting",                         # Column name for x
    ycol="northing",                        # Column name for y
    zcol="elevation",                       # Column name for z
    read_csv_kwargs=dict(sep="\t"),         # pandas.read_csv options
    griddata_kwargs=dict(method="linear"),  # scipy.interpolate.griddata options
)
```

This source automatically interpolates scattered points onto a regular grid using scipy's `griddata`.

### SourceCRSFile

For any file format supported by xarray (NetCDF, Zarr, etc.):

```python
from rompy_xbeach.source import SourceCRSFile

source = SourceCRSFile(
    uri="bathymetry.nc",
    crs="EPSG:4326",
    x_dim="longitude",  # Name of x dimension in the file
    y_dim="latitude",   # Name of y dimension in the file
    kwargs=dict(engine="netcdf4"),  # xarray.open_dataset options
)
```

### SourceCRSIntake

For data catalogued with [Intake](https://intake.readthedocs.io/):

```python
from rompy_xbeach.source import SourceCRSIntake

source = SourceCRSIntake(
    catalog_uri="catalog.yaml",
    dataset_id="bathymetry",
    crs="EPSG:4326",
    x_dim="x",
    y_dim="y",
)
```

### SourceCRSDataset

For using an existing xarray Dataset object:

```python
from rompy_xbeach.source import SourceCRSDataset
import xarray as xr

# Load data however you need
ds = xr.open_dataset("data.nc")

# Wrap it in a source with CRS info
source = SourceCRSDataset(
    obj=ds,
    crs="EPSG:4326",
)
```

This is useful when you need custom preprocessing before passing data to rompy-xbeach.

### SourceCRSWavespectra

For wave spectra data using [wavespectra](https://github.com/wavespectra/wavespectra){target="_blank"} readers:

```python
from rompy_xbeach.source import SourceCRSWavespectra

source = SourceCRSWavespectra(
    reader="read_era5",
    uri="era5_waves.nc",
    crs="EPSG:4326",  # Default: 4326
    x_dim="lon",      # Default: lon
    y_dim="lat",      # Default: lat
)
```

### SourceCRSOceantide

For tidal constituent data using [oceantide](https://github.com/oceanum/oceantide):

```python
from rompy_xbeach.source import SourceCRSOceantide

source = SourceCRSOceantide(
    reader="read_otis_netcdf",
    kwargs=dict(filename="tides.nc"),
    crs="EPSG:4326",  # Default: 4326
    x_dim="lon",      # Default: lon
    y_dim="lat",      # Default: lat
)
```

## CRS Handling

The key feature of rompy-xbeach sources is automatic CRS management. When a data interface (like `XBeachBathy`) uses a source:

1. **Source opens data** with CRS metadata attached
2. **Data interface checks** if source CRS matches grid CRS
3. **Automatic reprojection** occurs if CRS differs

```python
from rompy_xbeach.grid import RegularGrid, GeoPoint
from rompy_xbeach.data.bathy import XBeachBathy
from rompy_xbeach.source import SourceGeotiff

# Grid in projected coordinates (metres)
grid = RegularGrid(
    ori=GeoPoint(x=115.59, y=-32.64, crs="EPSG:4326"),
    crs="EPSG:28350",  # MGA Zone 50
    dx=10, dy=10,
    nx=100, ny=100,
    alfa=0,
)

# Source data in geographic coordinates (degrees)
source = SourceGeotiff(filename="bathy_wgs84.tif")  # CRS: EPSG:4326

# XBeachBathy handles the reprojection automatically
bathy = XBeachBathy(source=source)
xfile, yfile, depfile, grid = bathy.get(destdir="./run", grid=grid)
```

## Quick Reference

| Source | Use Case | CRS Required |
|--------|----------|--------------|
| `SourceGeotiff` | GeoTIFF raster files | No (embedded) |
| `SourceXYZ` | Point cloud / XYZ text files | Yes |
| `SourceCRSFile` | NetCDF, Zarr, etc. | Yes |
| `SourceCRSIntake` | Intake catalogues | Yes |
| `SourceCRSDataset` | Existing xarray Dataset | Yes |
| `SourceCRSWavespectra` | Wave spectra files | Yes (default: 4326) |
| `SourceCRSOceantide` | Tidal constituent files | Yes (default: 4326) |

## See Also

- [Grid](grid.md) — How grids handle CRS and coordinate transformations
- [Bathymetry](bathy.md) — Using sources with `XBeachBathy`
- [Wave Boundaries](boundaries.md) — Using sources for wave data
- [Forcing](forcing.md) — Using sources for wind and tide data
