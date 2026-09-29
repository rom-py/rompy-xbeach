# rompy-xbeach

rompy-xbeach sets up and runs the [XBeach](https://xbeach.readthedocs.io/) coastal model from Python or YAML. It is the XBeach plugin of [rompy](https://rom-py.github.io/rompy/), which provides the parts every model shares: the run period, data sources, model runs and backends.

With rompy-xbeach you describe an XBeach model as validated Python objects, and it writes the XBeach workspace for you:

- **The grid and bathymetry** from GeoTIFF, XYZ or gridded data, interpolated to the model grid, with seaward and lateral extensions.
- **Wave boundaries** from parameters, spectra or existing files, in any of the XBeach boundary types.
- **Tide, water level and wind** forcing from time series, gridded data or tide constituents.
- **Model settings**: physics, sediment, output, flow and tide boundaries, MPI and hotstart. Each setting is a typed field named after its XBeach parameter, and invalid combinations are rejected before XBeach runs.

```python
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat
from rompy_xbeach.config import Config, DataInterface

config = Config(
    grid=grid,                        # a RegularGrid
    bathy=bathy,                      # an XBeachBathy
    input=DataInterface(wave=waves),  # wave, tide and wind forcing
    physics=Physics(wavemodel=Surfbeat()),
)
run = ModelRun(
    run_id="storm",
    period=TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h"),
    output_dir="runs",
    config=config,
)
run()  # writes params.txt and the input files to runs/storm
```

[Your first model](getting-started/first-model.md) builds this example in full.

## Where to go next

| If you want to | Go to |
|---|---|
| Install rompy-xbeach and XBeach | [Installation](getting-started/installation.md) |
| Build and run a model in five minutes | [Your first model](getting-started/first-model.md) |
| Learn step by step, with notebooks | [Tutorial](tutorial.md) |
| Understand how the configuration maps to XBeach | [How rompy-xbeach works](user-guide/how-it-works.md) |
| Set up a specific part of the model | The [User guide](user-guide/configuration.md) |
| Find the field for an XBeach parameter | [Parameter index](reference/parameters.md) |
| Look up a class | [Reference](reference/index.md) |

rompy-xbeach is one of the rompy model plugins, with [rompy-swan](https://rom-py.github.io/rompy-swan/) and [rompy-schism](https://rom-py.github.io/rompy-schism/). The [rompy docs](https://rom-py.github.io/rompy/) explain the ideas they share.
