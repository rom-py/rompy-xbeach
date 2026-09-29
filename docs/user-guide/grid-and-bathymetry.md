# Grid and bathymetry

XBeach runs on a rectilinear grid whose x-axis points from the offshore boundary towards land. In rompy-xbeach, [`RegularGrid`][rompy_xbeach.grid.RegularGrid] places and sizes that grid in a projected coordinate system, and [`XBeachBathy`][rompy_xbeach.data.bathy.XBeachBathy] interpolates bathymetry from a data source onto it, optionally extending the grid offshore and sideways. This page covers both and the files and parameters they write.

```python exec="on" session="grid"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
OUT_DIR = tempfile.mkdtemp()
```

## Defining the grid

| Field | Meaning |
|---|---|
| `ori` | Origin: the corner of the grid on the offshore boundary, as a [`GeoPoint`][rompy_xbeach.grid.GeoPoint] or a dictionary with `x`, `y` and `crs` |
| `alfa` | Angle of the x-axis, in degrees counter-clockwise from east |
| `dx`, `dy` | Cell size along x (cross-shore) and y (alongshore), in `crs` units |
| `nx`, `ny` | Number of grid **points** along x and y |
| `crs` | Coordinate reference system the grid is built in, normally a projected one in metres |

The x-axis runs cross-shore from the origin towards land, so `alfa` is the direction the shore normal points onshore. On a west-facing coast the x-axis points roughly east: `alfa=347` rotates it 13° clockwise from east.

```python exec="on" source="above" result="text" session="grid"
from rompy_xbeach.grid import RegularGrid

grid = RegularGrid(
    ori={"x": 115.594239, "y": -32.641104, "crs": "EPSG:4326"},
    alfa=347.0,
    dx=10.0,
    dy=15.0,
    nx=230,
    ny=220,
    crs="EPSG:28350",
)
print(grid)
```

CRSs can be given as an EPSG string (`"EPSG:28350"`), an EPSG code (`28350`), or a `pyproj`, `cartopy` or `rasterio` CRS object.

### The origin in any coordinate system

The origin can be given in any CRS; the grid reprojects it into its own `crs`. A [`GeoPoint`][rompy_xbeach.grid.GeoPoint] can also be reprojected directly:

```python exec="on" source="above" result="text" session="grid"
from rompy_xbeach.grid import GeoPoint

origin = GeoPoint(x=115.594239, y=-32.641104, crs=4326)
print(origin.reproject(28350))
print(f"Grid origin in the grid CRS: x0={grid.x0:.1f}, y0={grid.y0:.1f}")
```

### Grid geometry

The grid provides its coordinates and named boundaries. Arrays are shaped `(ny, nx)`: each row is a cross-shore profile, from the offshore boundary to land.

| Attribute | Meaning |
|---|---|
| `x`, `y`, `shape` | Coordinates of every grid point in the grid CRS |
| `front` | Offshore boundary (first column), where waves enter |
| `back` | Landward boundary (last column) |
| `left`, `right` | Lateral boundaries (last and first row) |
| `offshore` | Midpoint of the offshore boundary, where wave boundary data is taken |
| `centre` | Centre of the grid, where wind and water level data are taken by default |
| `bbox()` | Bounding box, optionally with a `buffer` |

```python exec="on" source="above" result="text" session="grid"
print("Shape (ny, nx):   ", grid.shape)
print("Offshore midpoint:", grid.offshore)
print("Centre:           ", grid.centre)
```

### Checking the orientation

Always plot a new grid. `plot()` draws the grid on a map, with the origin as a red dot and the offshore boundary as a red line, which must face the open sea. `scale` adds a GSHHS coastline at resolution `"c"`, `"l"`, `"i"`, `"h"` or `"f"` (downloaded by cartopy on first use):

```python
ax = grid.plot(scale="f")
```

A grid with its origin on the landward side and `alfa` turned by 180° covers the same area, but its offshore boundary is on land and XBeach would force waves from the beach. Other options (`show_mesh`, `projection`, `ax`, styling) are shown in the [grid plotting example](https://rom-py.github.io/rompy-notebooks/notebooks/xbeach/examples/grid_plotting_and_export/).

### Exporting the grid

`to_file()` writes the grid cells with a GeoPandas driver and stores the grid definition in the file. A KML file opens in Google Earth, and [`RegularGrid.from_file`][rompy_xbeach.grid.RegularGrid.from_file] rebuilds the same grid from it:

```python exec="on" source="above" result="text" session="grid"
from pathlib import Path

grid_file = Path(OUT_DIR) / "grid.kml"
grid.to_file(grid_file, driver="KML")
print(RegularGrid.from_file(grid_file) == grid)
```

### Grid parameters

`params` holds what the grid writes to `params.txt`. XBeach counts cells, so its `nx` and `ny` are one less than the number of points. `vardx = 0` tells XBeach the grid has constant spacing and is defined by its origin and rotation, so no `xfile` or `yfile` is needed.

```python exec="on" source="above" result="text" session="grid"
for key, value in grid.params.items():
    print(f"{key} = {value}")
```

### Expanding a grid

`expand()` returns a larger grid with cells added on any side: `front` (offshore), `back`, `left` and `right`. The origin moves when cells are added at the `front` or `right`. You rarely call it directly; `XBeachBathy` uses it for its extensions.

```python exec="on" source="above" result="text" session="grid"
extended = grid.expand(front=30, left=10, right=10)
print(f"{grid.nx} x {grid.ny} points -> {extended.nx} x {extended.ny} points")
```

## Bathymetry

[`XBeachBathy`][rompy_xbeach.data.bathy.XBeachBathy] reads a data source, reprojects it to the grid CRS, fills gaps, interpolates it onto the grid and applies the extensions. Any gridded source works, including GeoTIFF, NetCDF, XYZ point clouds and intake catalogues; see [Data sources](sources.md).

```python exec="on" source="above" result="text" session="grid"
from rompy_xbeach.data.bathy import XBeachBathy
from rompy_xbeach.source import SourceGeotiff

bathy = XBeachBathy(source=SourceGeotiff(filename="tests/data/bathy.tif"), posdwn=False)
xfile, yfile, depfile, model_grid = bathy.get(destdir=OUT_DIR, grid=grid)
print([f.name for f in (xfile, yfile, depfile)], model_grid.shape)
```

`get()` returns the files it wrote and the grid actually used, which differs from the input grid when an extension is applied. Inside a model, `Config` calls it for you and takes the grid parameters from the returned grid.

| Option | Default | Effect |
|---|---|---|
| `source` | required | Where the data comes from |
| `variables` | `"data"` | The variable to read; a single one |
| `posdwn` | `True` | `True` if the values are depths (positive down), `False` for elevations (positive up) |
| `interpolator` | linear | How the data is interpolated onto the grid |
| `interpolate_na` | `True` | Fill gaps (NaN) in the source along x and then y before interpolating |
| `extension` | none | Seaward extension to a uniform offshore depth |
| `left`, `right` | `0` | Rows added on each lateral side |

### Depth convention: `posdwn`

`posdwn` must match your data. It is written to XBeach as `posdwn = 1` or `posdwn = -1`, so the depth file keeps the sign convention of the source, and the seaward extension uses it to extend in the right direction.

```python exec="on" source="above" result="text" session="grid"
print(XBeachBathy(source=SourceGeotiff(filename="tests/data/bathy.tif"), posdwn=False).params)
```

### Interpolation

The default [`RegularGridInterpolator`][rompy_xbeach.interpolate.RegularGridInterpolator] passes its `kwargs` to `scipy.interpolate.RegularGridInterpolator`, for example `method` (`"linear"`, `"nearest"`, `"cubic"`, ...). By default scipy raises an error if the model grid reaches outside the data; `bounds_error=False` with `fill_value=None` extrapolates instead:

```python
from rompy_xbeach.interpolate import RegularGridInterpolator

bathy = XBeachBathy(
    source=SourceGeotiff(filename="tests/data/bathy.tif"),
    posdwn=False,
    interpolator=RegularGridInterpolator(
        kwargs={"method": "linear", "bounds_error": False, "fill_value": None}
    ),
)
```

`interpolate_na_kwargs` is passed to xarray's `interpolate_na` when gaps are filled, for example `{"method": "nearest"}` or a `max_gap`.

### Seaward extension

XBeach expects a uniform depth along the offshore boundary, deep enough for the incoming waves, and survey data often does not reach that far. [`SeawardExtensionLinear`][rompy_xbeach.data.bathy.SeawardExtensionLinear] adds cells offshore so the bed slopes linearly from the offshore edge of the data down to `depth`:

- the number of cells added is set by the shallowest point on the original offshore boundary: `(depth - shallowest depth) / slope`, divided by `dx`;
- each row then slopes from `depth` to its own depth at the original boundary, so the new offshore boundary has the uniform `depth`.

A gentler `slope` gives a longer extension. The defaults are `depth=25` and `slope=0.3`.

```python exec="on" source="above" result="text" session="grid"
from rompy_xbeach.data.bathy import SeawardExtensionLinear

for slope in (0.1, 0.05):
    bathy = XBeachBathy(
        source=SourceGeotiff(filename="tests/data/bathy.tif"),
        posdwn=False,
        extension=SeawardExtensionLinear(depth=25.0, slope=slope),
    )
    _, _, _, model_grid = bathy.get(destdir=OUT_DIR, grid=grid)
    print(f"slope={slope}: {model_grid.nx - grid.nx} cells added offshore")
```

If the data is already deeper than `depth` at the offshore boundary, no cells are added and a warning asks whether `posdwn` is set correctly; a wrong `posdwn` is the usual cause.

### Lateral extension

`left` and `right` add rows on each lateral side by repeating the edge profiles, which moves the lateral boundaries and their artefacts away from the area of interest. The two sides can differ.

```python exec="on" source="above" result="text" session="grid"
bathy = XBeachBathy(
    source=SourceGeotiff(filename="tests/data/bathy.tif"), posdwn=False, left=20, right=5
)
_, _, _, model_grid = bathy.get(destdir=OUT_DIR, grid=grid)
print(f"{grid.ny} rows -> {model_grid.ny} rows")
```

### Checking the bathymetry

The `xbeach` accessor on xarray datasets reads a depth file back onto its grid and plots it in real and model coordinates, with the cross-shore profiles and slopes:

```python
import xarray as xr

dset = xr.Dataset.xbeach.from_xbeach(depfile, model_grid)
dset.xbeach.plot_model_bathy(model_grid, posdwn=False)
```

## Files and parameters written

In a model, the grid and bathymetry write:

| Written | Content |
|---|---|
| `bathy.txt` | Depths on the (extended) grid, `(ny, nx)` values, referenced by `depfile` |
| `xdata.txt`, `ydata.txt` | Coordinates of the (extended) grid points; for reference only, since `vardx = 0` does not use them |
| `posdwn` | `1` or `-1` from `XBeachBathy.posdwn` |
| `vardx`, `nx`, `ny`, `dx`, `dy`, `xori`, `yori`, `alfa`, `projection` | From the grid, after any extension |

```python exec="on" source="above" result="text" session="grid"
bathy = XBeachBathy(
    source=SourceGeotiff(filename="tests/data/bathy.tif"),
    posdwn=False,
    extension=SeawardExtensionLinear(depth=25.0, slope=0.05),
    left=5,
    right=5,
)
_, _, depfile, model_grid = bathy.get(destdir=OUT_DIR, grid=grid)
params = {**bathy.params, **model_grid.params, "depfile": depfile.name}
for key, value in params.items():
    print(f"{key} = {value}")
```

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["grid", "bathymetry"]))
```
