# Tide and water levels

XBeach takes time-varying water levels from a `zs0file`, a table of time and water level applied at the model boundary. rompy-xbeach writes this file from tidal constituents, from water level data such as sea-surface height from an ocean model, or from both added together. A water level interface goes in `Config.input.tide`. Its `get(destdir, grid, period)` method writes the file and returns `zs0file`, `tideloc` and `tidelen` for `params.txt`.

```python exec="on" session="water-levels"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
destdir = Path(tempfile.mkdtemp())
```

The examples below use this grid, a one-day run period, and `destdir`, a temporary folder the files are written to:

```python exec="on" source="above" session="water-levels"
from rompy.core.time import TimeRange
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
period = TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h")
```

## Choosing a class

| Data you have | Class |
|---|---|
| Gridded tidal constituents (a tide model) | [`TideConsGrid`][rompy_xbeach.data.waterlevel.TideConsGrid] |
| Constituents at one site (harmonic analysis of a gauge) | [`TideConsPoint`][rompy_xbeach.data.waterlevel.TideConsPoint] |
| Gridded water levels | [`WaterLevelGrid`][rompy_xbeach.data.waterlevel.WaterLevelGrid] |
| Water levels at stations | [`WaterLevelStation`][rompy_xbeach.data.waterlevel.WaterLevelStation] |
| One water level timeseries (CSV, gauge) | [`WaterLevelPoint`][rompy_xbeach.data.waterlevel.WaterLevelPoint] |
| Surge or sea-surface height plus tide | [`CombinedWaterLevel`][rompy_xbeach.data.waterlevel.CombinedWaterLevel] |
| A constant water level | none: set `zs0` with `tideloc=0` in [`TideBoundaryConditions`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions] |

Grid and station data are sampled at the grid centre by default. `location="offshore"` samples the middle of the offshore boundary instead. Station data are interpolated between the nearest sites by inverse distance weighting (`sel_method="idw"`), or taken from the closest site with `sel_method="nearest"`. Gridded data are taken from the nearest cell. Point data are used as they are.

## Tide from constituents

[`TideConsGrid`][rompy_xbeach.data.waterlevel.TideConsGrid] predicts the tide with [oceantide](https://github.com/oceanum/oceantide) from a gridded tide model, read through a [`SourceCRSOceantide`][rompy_xbeach.source.SourceCRSOceantide]. Here the source is a regional OTIS model. `freq` sets the time step of the prediction (default `1h`):

```python exec="on" source="above" result="text" session="water-levels"
from rompy_xbeach.data.waterlevel import TideConsGrid
from rompy_xbeach.source import SourceCRSOceantide

cons_dir = Path("tests/data/swaus_tide_cons")
tide = TideConsGrid(
    source=SourceCRSOceantide(
        reader="read_otis_binary",
        kwargs={
            "gfile": cons_dir / "grid_m2s2n2k2k1o1p1q1mmmf",
            "hfile": cons_dir / "h_m2s2n2k2k1o1p1q1mmmf",
            "ufile": cons_dir / "u_m2s2n2k2k1o1p1q1mmmf",
        },
        crs=4326,
    ),
    coords={"x": "lon", "y": "lat"},
    freq="30min",
)
params = tide.get(destdir, grid, period)
print(params)
```

The `zs0file` has the time in seconds since the start of the run and the water level in metres:

```python exec="on" source="above" result="text" session="water-levels"
print("\n".join((destdir / params["zs0file"]).read_text().splitlines()[:5]))
```

[`TideConsPoint`][rompy_xbeach.data.waterlevel.TideConsPoint] takes the amplitude (m) and phase (degrees) of each constituent at one site, from a CSV file read by [`SourceTideConsPointCSV`][rompy_xbeach.source.SourceTideConsPointCSV]:

```python exec="on" source="above" result="text" session="water-levels"
from rompy_xbeach.data.waterlevel import TideConsPoint
from rompy_xbeach.source import SourceTideConsPointCSV

print(Path("tests/data/tide_cons_station.csv").read_text())
tide = TideConsPoint(
    source=SourceTideConsPointCSV(filename="tests/data/tide_cons_station.csv")
)
print(tide.get(destdir, grid, period))
```

## Water level data

The `WaterLevel...` classes read one water level variable, named in `variables`, and write it at the source time steps within the run period. If the period does not start or end on a source time, the data are interpolated to those times.

Gridded data, here sea-surface height from an ocean model:

```python exec="on" source="above" result="text" session="water-levels"
from rompy_xbeach.data.waterlevel import WaterLevelGrid
from rompy_xbeach.source import SourceCRSFile

ssh_grid = WaterLevelGrid(
    source=SourceCRSFile(
        uri="tests/data/ssh_gridded.nc", crs=4326, x_dim="lon", y_dim="lat"
    ),
    coords={"x": "lon", "y": "lat"},
    variables=["ssh"],
)
params = ssh_grid.get(destdir, grid, period)
print(params)
print("\n".join((destdir / params["zs0file"]).read_text().splitlines()[:5]))
```

Station data, with a `site` dimension and coordinate variables:

```python exec="on" source="above" result="text" session="water-levels"
from rompy_xbeach.data.waterlevel import WaterLevelStation

ssh_station = WaterLevelStation(
    source=SourceCRSFile(
        uri="tests/data/ssh_stations.nc", crs=4326, x_dim="lon", y_dim="lat"
    ),
    coords={"s": "site", "x": "lon", "y": "lat"},
    variables=["ssh"],
)
print(ssh_station.get(destdir, grid, period))
```

A single timeseries, read with rompy's [`SourceTimeseriesCSV`][rompy.core.source.SourceTimeseriesCSV]:

```python exec="on" source="above" result="text" session="water-levels"
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.data.waterlevel import WaterLevelPoint

ssh_point = WaterLevelPoint(
    source=SourceTimeseriesCSV(filename="tests/data/ssh.csv", tcol="time"),
    variables=["ssh"],
)
print(ssh_point.get(destdir, grid, period))
```

## Surge plus tide

Ocean model sea-surface height often excludes the tide. [`CombinedWaterLevel`][rompy_xbeach.data.waterlevel.CombinedWaterLevel] predicts the tide from a `TideConsGrid` or `TideConsPoint` at its `freq`, interpolates a `WaterLevel...` series onto those times and adds the two:

```python exec="on" source="above" result="text" session="water-levels"
from rompy_xbeach.data.waterlevel import CombinedWaterLevel

tide = TideConsPoint(
    source=SourceTideConsPointCSV(filename="tests/data/tide_cons_station.csv"),
    freq="30min",
)
combined = CombinedWaterLevel(tide=tide, waterlevel=ssh_grid)
params = combined.get(destdir, grid, period)
print(params)
print("\n".join((destdir / params["zs0file"]).read_text().splitlines()[:5]))
```

## `tideloc` and the tide boundary settings

XBeach can apply water level signals at 1, 2 or 4 corners of the domain (`tideloc`). The water level interfaces write a single signal, `tideloc = 1`, which XBeach applies along the offshore boundary. `tideloc` is a field of each interface, but it only accepts 1:

```python exec="on" source="above" result="text" session="water-levels"
try:
    WaterLevelPoint(
        source=SourceTimeseriesCSV(filename="tests/data/ssh.csv", tcol="time"),
        variables=["ssh"],
        tideloc=2,
    )
except NotImplementedError as error:
    print(error)
```

The other tide boundary settings, such as `tidetype`, `paulrevere` or `zs0`, are set with [`TideBoundaryConditions`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions] in `Config.tide_boundary`. Its [`tideloc`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.tideloc] is written after the forcing parameters, so if it is set it replaces the value written by `input.tide`.

!!! warning "Runs without water level forcing"
    XBeach defaults to `tideloc = 2`, which expects a `zs0file`. Runs without water level forcing need `TideBoundaryConditions(tideloc=0, zs0=...)`, which keeps the water level constant at `zs0`.

## Using water levels in a model

Any of the classes above goes in `input.tide`:

```python
config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(wave=wave, tide=combined),
    physics=physics,
)
```

In YAML the class is chosen by `model_type`, for example `tide_cons_grid`, `water_level_station` or `combined_water_level`:

```python exec="on" source="above" result="text" session="water-levels"
import yaml
from rompy_xbeach.config import DataInterface

config_yaml = """
tide:
  model_type: combined_water_level
  tide:
    model_type: tide_cons_point
    source:
      model_type: tide_cons_point_csv
      filename: tests/data/tide_cons_station.csv
    freq: 30min
  waterlevel:
    model_type: water_level_point
    source:
      model_type: csv
      filename: tests/data/ssh.csv
      tcol: time
    variables: [ssh]
"""
data = DataInterface(**yaml.safe_load(config_yaml))
print(type(data.tide).__name__)
```

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["tides", "forcing"]))
```
