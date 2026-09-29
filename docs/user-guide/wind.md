# Wind

XBeach applies wind as a single timeseries of speed and direction, uniform over the domain, read from a `windfile`. rompy-xbeach extracts that timeseries from gridded, station or single-point wind data. A wind interface goes in `Config.input.wind`. Its `get(destdir, grid, period)` method writes the file and returns `windfile` for `params.txt`. Wind also has to be switched on in [`Physics`](#switching-wind-on).

```python exec="on" session="wind"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
destdir = Path(tempfile.mkdtemp())
```

The examples below use this grid, a 12-hour run period, and `destdir`, a temporary folder the files are written to:

```python exec="on" source="above" session="wind"
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
period = TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h")
```

## Choosing a class

| Data you have | Class | Selection |
|---|---|---|
| Gridded fields (reanalysis, weather model) | [`WindGrid`][rompy_xbeach.data.wind.WindGrid] | Nearest grid cell (`sel_method="sel"`), or `sel_method="interp"` |
| Many sites, each with its own coordinates | [`WindStation`][rompy_xbeach.data.wind.WindStation] | Inverse distance weighting between sites (`sel_method="idw"`), or `sel_method="nearest"` |
| One timeseries (weather station, CSV) | [`WindPoint`][rompy_xbeach.data.wind.WindPoint] | None, used as it is |

Grid and station winds are taken at the grid centre by default. `location="offshore"` uses the middle of the offshore boundary instead, which can matter when winds change quickly towards the coast.

## Wind variables

`wind_vars` names the wind variables in the dataset, in one of two forms:

- [`WindVector`][rompy_xbeach.data.wind.WindVector] names the u and v components (`u`, `v`). They are converted to speed and the direction the wind comes from.
- [`WindScalar`][rompy_xbeach.data.wind.WindScalar] names the speed and direction variables (`spd`, `dir`). The direction must be nautical, where the wind comes from, as XBeach expects; it is written unchanged.

## Gridded winds

[`WindGrid`][rompy_xbeach.data.wind.WindGrid] reads gridded fields, here ERA5 10 m winds. `coords` names the spatial dimensions of the dataset:

```python exec="on" source="above" result="text" session="wind"
from rompy_xbeach.data.wind import WindGrid, WindVector
from rompy_xbeach.source import SourceCRSFile

wind = WindGrid(
    source=SourceCRSFile(uri="tests/data/era5-20230101.nc", crs=4326),
    coords={"x": "longitude", "y": "latitude"},
    wind_vars=WindVector(u="u10", v="v10"),
)
params = wind.get(destdir, grid, period)
print(params)
```

The `windfile` has the time in seconds since the start of the run, the speed (m/s) and the nautical direction the wind comes from (degrees). It holds the source time steps within the run period; if the period does not start or end on a source time, the data are interpolated to those times.

```python exec="on" source="above" result="text" session="wind"
print("\n".join((destdir / params["windfile"]).read_text().splitlines()[:5]))
```

## Winds at stations

[`WindStation`][rompy_xbeach.data.wind.WindStation] reads data with a site dimension and coordinate variables, for example winds at the output sites of a wave model. The site dimension is named in `coords`:

```python exec="on" source="above" result="text" session="wind"
from rompy_xbeach.data.wind import WindStation

wind = WindStation(
    source=SourceCRSFile(uri="tests/data/smc-params-20230101.nc", crs=4326),
    coords={"s": "seapoint"},
    wind_vars=WindVector(u="uwnd", v="vwnd"),
    sel_method="nearest",
    location="offshore",
)
params = wind.get(destdir, grid, period)
print("\n".join((destdir / params["windfile"]).read_text().splitlines()[:5]))
```

Options for the wavespectra selection functions, such as `tolerance` or `max_sites`, go in `sel_method_kwargs`.

## A single timeseries

[`WindPoint`][rompy_xbeach.data.wind.WindPoint] uses a timeseries as it is, here a CSV file with speed and direction columns read with rompy's [`SourceTimeseriesCSV`][rompy.core.source.SourceTimeseriesCSV]:

```python exec="on" source="above" result="text" session="wind"
from rompy.core.source import SourceTimeseriesCSV
from rompy_xbeach.data.wind import WindPoint, WindScalar

wind = WindPoint(
    source=SourceTimeseriesCSV(filename="tests/data/wind.csv", tcol="time"),
    wind_vars=WindScalar(spd="wspd", dir="wdir"),
)
params = wind.get(destdir, grid, period)
print("\n".join((destdir / params["windfile"]).read_text().splitlines()[:5]))
```

## Switching wind on

!!! warning
    XBeach 1.24 ignores the wind file unless wind is switched on in the physics (`wind = 1`). Add `wind=True` to [`Physics`][rompy_xbeach.components.physics.physics.Physics] whenever you set `input.wind`.

`Physics(wind=True)` switches wind on with XBeach's default settings. The [`Wind`][rompy_xbeach.components.physics.wind.Wind] component also switches it on and sets the drag coefficient [`Cd`][rompy_xbeach.components.physics.wind.Wind.Cd] of the wind stress, τ = ρ<sub>a</sub> C<sub>d</sub> |W| W (XBeach default 0.002):

```python exec="on" source="above" result="text" session="wind"
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat
from rompy_xbeach.components.physics.wind import Wind

print(Physics(wavemodel=Surfbeat(), wind=True).get(destdir))
print(Physics(wavemodel=Surfbeat(), wind=Wind(Cd=0.0015)).get(destdir))
```

## Using wind in a model

```python
config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(wave=wave, wind=wind),
    physics=Physics(wavemodel=Surfbeat(), wind=True),
)
```

In YAML the class is chosen by `model_type` (`wind_grid`, `wind_station` or `wind_point`), and the variables by `wind_vector` or `wind_scalar`:

```python exec="on" source="above" result="text" session="wind"
import yaml
from rompy_xbeach.config import DataInterface

config_yaml = """
wind:
  model_type: wind_grid
  source:
    model_type: file
    uri: tests/data/era5-20230101.nc
    crs: 4326
  coords:
    x: longitude
    y: latitude
  wind_vars:
    model_type: wind_vector
    u: u10
    v: v10
"""
data = DataInterface(**yaml.safe_load(config_yaml))
print(type(data.wind).__name__)
```

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["wind", "forcing"]))
```
