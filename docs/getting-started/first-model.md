# Your first model

This page builds a small XBeach model of a beach south of Perth, Western Australia, writes its workspace and runs it. Every step on this page is executed when the docs are built, so the output shown is what the code produces. [Tutorial lesson 1](https://rom-py.github.io/rompy-notebooks/notebooks/xbeach/tutorial/01_first_model/) covers the same model as a notebook, with plots.

```python exec="on" session="first"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
OUT_DIR = tempfile.mkdtemp()
```

## 1. Run period

[`TimeRange`][rompy.core.time.TimeRange] comes from rompy and defines the simulation period. XBeach's `tstop` and the time range of any forcing data are derived from it.

```python exec="on" source="above" result="text" session="first"
from rompy.core.time import TimeRange

period = TimeRange(start="2023-01-01T00:00", end="2023-01-01T00:30", interval="10m")
print(period)
```

## 2. Grid

A [`RegularGrid`][rompy_xbeach.grid.RegularGrid] is placed by its origin, its rotation `alfa` (degrees counter-clockwise from east) and its cell size and count. The origin is on the offshore boundary and x points onshore. The origin can be given in any coordinate system; the grid itself is built in the projected `crs`.

```python exec="on" source="above" result="text" session="first"
from rompy_xbeach.grid import RegularGrid

grid = RegularGrid(
    ori={"x": 115.594239, "y": -32.641104, "crs": "EPSG:4326"},
    alfa=347.0,
    dx=20.0,
    dy=30.0,
    nx=115,
    ny=110,
    crs="EPSG:28350",
)
print(grid.params)
```

## 3. Bathymetry

[`XBeachBathy`][rompy_xbeach.data.bathy.XBeachBathy] reads a data source, here a GeoTIFF of elevations (positive up, so `posdwn=False`), and interpolates it onto the grid. The linear seaward extension deepens the offshore edge to a uniform depth, which XBeach needs at the wave boundary.

```python exec="on" source="above" session="first"
from rompy_xbeach.data.bathy import SeawardExtensionLinear, XBeachBathy
from rompy_xbeach.source import SourceGeotiff

bathy = XBeachBathy(
    source=SourceGeotiff(filename="tests/data/bathy.tif"),
    posdwn=False,
    extension=SeawardExtensionLinear(depth=15.0, slope=0.05),
)
```

## 4. Waves

The simplest wave boundary, [`BoundaryParams`][rompy_xbeach.data.boundary.nonspectral.BoundaryParams], applies a constant wave height, period and direction. `dir0` is nautical: the direction waves come from, clockwise from north. XBeach needs a directional grid whenever short waves are modelled; by default its angles are relative to the grid x-axis, so -90 to 90 degrees covers all waves travelling towards the shore.

```python exec="on" source="above" session="first"
from rompy_xbeach.data.boundary import BoundaryParams

waves = BoundaryParams(
    Hrms=1.0, Trep=10.0, dir0=270.0, thetamin=-90.0, thetamax=90.0, dtheta=15.0
)
```

## 5. Physics and boundaries

[`Physics`][rompy_xbeach.components.physics.physics.Physics] has one required choice, the wave model. The stationary model is the fastest and is enough for a first run. Without tide forcing, `tideloc=0` tells XBeach to keep the constant water level `zs0`.

```python exec="on" source="above" session="first"
from rompy_xbeach.components.boundary.parameters import TideBoundaryConditions
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Stationary

physics = Physics(wavemodel=Stationary())
tide_boundary = TideBoundaryConditions(tideloc=0, zs0=0.0)
```

## 6. The model configuration

[`Config`][rompy_xbeach.config.Config] gathers the pieces. Forcing goes in `input`; the other fields group related XBeach parameters. The output here asks for maps of wave height, water level and bed level every 5 minutes.

```python exec="on" source="above" session="first"
from rompy_xbeach.components.output import Output
from rompy_xbeach.config import Config, DataInterface

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(wave=waves),
    physics=physics,
    tide_boundary=tide_boundary,
    output=Output(globalvars=["H", "zs", "zb"], tintg=300.0),
)
```

## 7. Generate the workspace

rompy's [`ModelRun`][rompy.model.ModelRun] combines the configuration with the run period and an output directory. Calling it writes the XBeach workspace into `output_dir/run_id`.

```python exec="on" source="above" result="text" session="first"
from pathlib import Path

from rompy.model import ModelRun

modelrun = ModelRun(run_id="first_model", period=period, output_dir=OUT_DIR, config=config)
workspace = Path(modelrun())
print(sorted(p.name for p in workspace.iterdir()))
```

The workspace holds the grid and bathymetry files and `params.txt`, which lists every parameter set by the objects above (the header lines starting with `%` are left out here):

```python exec="on" source="above" result="text" session="first"
params = (workspace / "params.txt").read_text()
print("\n".join(line for line in params.splitlines() if not line.startswith("%")))
```

XBeach's `nx` and `ny` count grid cells, one fewer than the points in the grid, which is why step 2 printed `nx = 114` for 115 points. Here `nx` is 116 because the seaward extension added cells offshore of the original grid. `xori` and `yori` moved with it.

## 8. Run XBeach

The easiest way to run XBeach is the public Docker image, through rompy's [`DockerConfig`][rompy.backends.config.DockerConfig] backend:

```python
from rompy.backends import DockerConfig

backend = DockerConfig(image="ghcr.io/rom-py/xbeach:trunk-r6147", executable="xbeach")
modelrun.run(backend, workspace_dir=workspace)
```

XBeach writes its results to `xboutput.nc` in the workspace. With your own XBeach installation, run `xbeach` from inside the workspace folder instead. [Running XBeach](../user-guide/running.md) covers backends, MPI and checking a run.

## Next steps

- The [tutorial](../tutorial.md) goes through each part of the model in turn.
- [How rompy-xbeach works](../user-guide/how-it-works.md) explains how these objects become `params.txt`.
- The [User guide](../user-guide/configuration.md) covers each part of the model in detail.
