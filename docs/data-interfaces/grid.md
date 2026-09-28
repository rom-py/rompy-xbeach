# Grid

The `RegularGrid` class defines the computational domain for XBeach simulations. It provides a powerful abstraction that handles coordinate reference systems (CRS), reprojections, and grid generation.

## Why a Grid Abstraction?

XBeach uses a rotated regular grid where:

- The **x-axis** points offshore (cross-shore direction)
- The **y-axis** points alongshore
- The **origin** is at one corner of the domain
- The grid is rotated by angle `alfa` from east

This coordinate system differs from typical geographic or projected coordinates, requiring careful handling of:

1. **Origin placement** — The origin must be positioned correctly relative to the coastline
2. **Rotation** — The grid must be oriented with x pointing offshore
3. **CRS transformations** — Source data may be in different coordinate systems

## Grid Definition

### Basic Parameters

```python
from rompy_xbeach.grid import RegularGrid, GeoPoint

grid = RegularGrid(
    ori=GeoPoint(x=115.5875, y=-32.646, crs="EPSG:4326"),  # Origin in WGS84
    alfa=344.0,      # Rotation angle (degrees from east)
    dx=10,           # Cross-shore spacing (m)
    dy=20,           # Alongshore spacing (m)
    nx=270,          # Number of cross-shore points
    ny=305,          # Number of alongshore points
    crs="EPSG:28350", # Grid CRS (projected)
)
```

| Parameter | Description |
|-----------|-------------|
| `ori` | Grid origin as a `GeoPoint` with its own CRS |
| `alfa` | Angle of x-axis from east (degrees, counter-clockwise) |
| `dx` | Grid spacing in x-direction (cross-shore) |
| `dy` | Grid spacing in y-direction (alongshore) |
| `nx` | Number of grid points in x-direction |
| `ny` | Number of grid points in y-direction |
| `crs` | Coordinate reference system for the grid |

### The GeoPoint Class

The origin is defined using `GeoPoint`, which allows specifying coordinates in any CRS:

```python
from rompy_xbeach.grid import GeoPoint

# Origin in WGS84 (lat/lon)
ori = GeoPoint(x=115.5875, y=-32.646, crs="EPSG:4326")

# Origin in projected coordinates
ori = GeoPoint(x=381500, y=6385000, crs="EPSG:28350")
```

`GeoPoint` supports reprojection:

```python
# Reproject to a different CRS
ori_projected = ori.reproject("EPSG:28350")
print(ori_projected)  # GeoPoint(x=381234.5, y=6384567.8, crs='EPSG:28350')

# Reproject back
ori_wgs84 = ori_projected.reproject("EPSG:4326")
```

## CRS Handling

### Why CRS Matters

Coastal modelling often involves data from multiple sources with different coordinate systems:

- **Bathymetry** may be in WGS84 (EPSG:4326)
- **Wave data** may use a regional projection
- **XBeach grid** typically uses a local projected CRS for accurate distances

Rompy-xbeach handles these transformations automatically:

```python
# Origin defined in WGS84
grid = RegularGrid(
    ori=GeoPoint(x=115.5875, y=-32.646, crs="EPSG:4326"),
    alfa=344.0,
    dx=10, dy=20, nx=270, ny=305,
    crs="EPSG:28350",  # Grid in MGA Zone 50
)

# The origin is automatically reprojected to the grid CRS
print(grid.x0, grid.y0)  # Coordinates in EPSG:28350
```

### Projected vs Geographic CRS

For coastal modelling, **projected CRS** is strongly recommended:

| CRS Type | Pros | Cons |
|----------|------|------|
| **Projected** (e.g., UTM) | Accurate distances, angles | Limited geographic extent |
| **Geographic** (WGS84) | Global coverage | Distorted distances at high latitudes |

```python
# Recommended: Use projected CRS for the grid
grid = RegularGrid(
    ori=GeoPoint(x=115.5875, y=-32.646, crs="EPSG:4326"),
    crs="EPSG:28350",  # MGA Zone 50 (Australia)
    # ...
)
```

## Grid Properties

The `RegularGrid` provides many useful properties:

### Coordinates

```python
# Origin in grid CRS
grid.x0, grid.y0

# Full coordinate arrays (2D)
grid.x  # Shape: (nx, ny)
grid.y  # Shape: (nx, ny)

# Grid shape
grid.shape  # (nx, ny)
```

### Boundaries

```python
# Boundary coordinates (x, y tuples)
grid.front    # Offshore boundary (first row)
grid.back     # Land boundary (last row)
grid.left     # Left lateral boundary
grid.right    # Right lateral boundary

# Centre points
grid.offshore  # Centre of offshore boundary
grid.centre    # Centre of entire grid
```

### Cartopy Integration

```python
# Cartopy transform for plotting
grid.transform

# Stereographic projection centred on grid
grid.projection

# GeoDataFrame with grid cells
grid.gdf
```

## Grid Orientation

### Understanding `alfa`

The `alfa` parameter defines the rotation of the x-axis from east:

- `alfa=0` — x-axis points east
- `alfa=90` — x-axis points north
- `alfa=180` — x-axis points west
- `alfa=270` — x-axis points south

For XBeach, the x-axis should point **offshore** (perpendicular to the coastline).

### Positioning the Origin

The origin should be placed at the **offshore corner** of the domain where:

- The first row of the grid is the offshore boundary
- Waves enter from the offshore boundary

!!! warning "Common Mistake"
    If the origin is placed incorrectly, the bathymetry will be inverted (deep water at the coast).

```python
# Correct: Origin at offshore corner, x pointing onshore
grid = RegularGrid(
    ori=GeoPoint(x=115.5875, y=-32.646, crs="EPSG:4326"),
    alfa=344.0,  # Adjusted so x points toward shore
    # ...
)
```

## Visualisation

The `plot()` method provides comprehensive visualisation:

```python
# Basic plot with coastlines
ax = grid.plot()

# With grid mesh overlay
ax = grid.plot(show_mesh=True, mesh_step=10)

# Custom projection
import cartopy.crs as ccrs
ax = grid.plot(projection=ccrs.PlateCarree())

# Highlight boundaries
ax = grid.plot(
    show_origin=True,      # Red dot at origin
    show_offshore=True,    # Red line at offshore boundary
)
```

### Plot Options

| Parameter | Description |
|-----------|-------------|
| `scale` | GSHHS coastline resolution: `"c"`, `"l"`, `"i"`, `"h"`, `"f"` |
| `show_mesh` | Overlay grid mesh |
| `mesh_step` | Downsample mesh for clarity |
| `show_origin` | Highlight origin point |
| `show_offshore` | Highlight offshore boundary |
| `projection` | Cartopy projection for the plot |
| `buffer` | Extent buffer around grid (in grid units) |

## Integration with Other Components

### With Bathymetry

```python
from rompy_xbeach.data.bathy import XBeachBathy

bathy = XBeachBathy(source=source)

# Bathymetry is interpolated onto the grid
xfile, yfile, depfile, updated_grid = bathy.get(
    destdir=Path("./run"),
    grid=grid,
)
```

### With Boundaries

```python
from rompy_xbeach.data.boundary import BoundaryStationSpectraJons

# Wave data is extracted at the offshore boundary centre
wbc = boundary_data.get(
    destdir=Path("./run"),
    grid=grid,  # Uses grid.offshore for location
    time=times,
)
```

### With Config

```python
from rompy_xbeach.config import Config

config = Config(
    grid=grid,
    bathy=bathy,
    # ...
)

# Grid parameters are written to params.txt
# nx, ny, dx, dy, alfa, xori, yori, etc.
```

## Example: Complete Grid Setup

```python
from rompy_xbeach.grid import RegularGrid, GeoPoint

# Define grid for a beach facing northwest
grid = RegularGrid(
    # Origin in WGS84 (offshore corner)
    ori=GeoPoint(x=115.594239, y=-32.641104, crs="EPSG:4326"),
    
    # Rotation: x-axis points toward shore (347° from east)
    alfa=347.0,
    
    # Grid spacing in metres
    dx=10,   # 10m cross-shore
    dy=15,   # 15m alongshore
    
    # Grid dimensions
    nx=230,  # 2.3km cross-shore extent
    ny=220,  # 3.3km alongshore extent
    
    # Projected CRS for accurate distances
    crs="EPSG:28350",
)

# Verify the setup
print(f"Grid shape: {grid.shape}")
print(f"Origin (projected): ({grid.x0:.1f}, {grid.y0:.1f})")
print(f"Offshore centre: {grid.offshore}")

# Visualise
ax = grid.plot(scale="f", show_mesh=True, mesh_step=20)
```

## See Also

- [Bathymetry](bathy.md) — How bathymetry integrates with the grid
- [Examples: Grid Demo](../examples/grid-demo.ipynb) — Interactive notebook with detailed examples
