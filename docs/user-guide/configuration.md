# Configuration

An XBeach model is one [`Config`][rompy_xbeach.config.Config] object. It can be built in Python or written as YAML; both are validated by the same classes, so a YAML file is checked as thoroughly as Python code. This page lists what a `Config` holds, shows the same model in Python and YAML, and covers generating it with rompy's [`ModelRun`][rompy.model.ModelRun] and the `rompy` command line.

```python exec="on" session="configuration"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
OUT_DIR = tempfile.mkdtemp()
```

## What a `Config` holds

| Field | Required | Default | Holds |
|---|---|---|---|
| `grid` | yes | | [`RegularGrid`][rompy_xbeach.grid.RegularGrid]: position, rotation and size of the grid |
| `bathy` | yes | | [`XBeachBathy`][rompy_xbeach.data.bathy.XBeachBathy]: depths interpolated from a data source |
| `physics` | yes | | [`Physics`][rompy_xbeach.components.physics.physics.Physics]: the wave model (required) and processes |
| `input` | no | `None` | [`DataInterface`][rompy_xbeach.config.DataInterface]: `wave`, `wind` and `tide` forcing, each optional |
| `tide_boundary` | no | `None` | [`TideBoundaryConditions`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions]: `tideloc`, `zs0`, ... |
| `flow_boundary` | no | `None` | [`FlowBoundaryConditions`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions]: `front`, `back`, `left`, `right` |
| `sediment` | no | `Sediment()` | [`Sediment`][rompy_xbeach.components.sediment.sediment.Sediment]: transport, morphology and bed |
| `output` | no | `Output()` | [`Output`][rompy_xbeach.components.output.Output]: variables, locations and intervals |
| `mpi` | no | `Mpi()` | [`Mpi`][rompy_xbeach.components.mpi.Mpi]: domain decomposition for parallel runs |
| `hotstart` | no | `None` | `True` or a [`Hotstart`][rompy_xbeach.components.hotstart.Hotstart]: start from a previous run |
| `tunits` | no | from the run period | XBeach `tunits`, written as `seconds since <period start>` when not set |

The run period is not part of the `Config`: it belongs to the `ModelRun`, and `Config` derives `tstop` and `tunits` from it.

The defaults for `sediment`, `output` and `mpi` write almost nothing: `Sediment()` and `Mpi()` leave every parameter to XBeach, and `Output()` only sets `outputformat = netcdf` (XBeach itself defaults to Fortran binary output). Setting `sediment=None` or `mpi=None` leaves the component out entirely.

## Building a configuration in Python

Each field takes an object of the class listed above. The model below is the one from [Your first model](../getting-started/first-model.md):

```python exec="on" source="above" session="configuration"
from rompy_xbeach.components.boundary.parameters import TideBoundaryConditions
from rompy_xbeach.components.output import Output
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Stationary
from rompy_xbeach.config import Config, DataInterface
from rompy_xbeach.data.bathy import SeawardExtensionLinear, XBeachBathy
from rompy_xbeach.data.boundary import BoundaryParams
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.source import SourceGeotiff

config = Config(
    grid=RegularGrid(
        ori={"x": 115.594239, "y": -32.641104, "crs": "EPSG:4326"},
        alfa=347.0,
        dx=20.0,
        dy=30.0,
        nx=115,
        ny=110,
        crs="EPSG:28350",
    ),
    bathy=XBeachBathy(
        source=SourceGeotiff(filename="tests/data/bathy.tif"),
        posdwn=False,
        extension=SeawardExtensionLinear(depth=15.0, slope=0.05),
    ),
    input=DataInterface(
        wave=BoundaryParams(
            Hrms=1.0, Trep=10.0, dir0=270.0, thetamin=-90.0, thetamax=90.0, dtheta=15.0
        ),
    ),
    physics=Physics(wavemodel=Stationary()),
    tide_boundary=TideBoundaryConditions(tideloc=0, zs0=0.0),
    output=Output(globalvars=["H", "zs", "zb"], tintg=300.0),
)
```

Nested objects can also be given as dictionaries, as `ori` is here. Pydantic turns them into the right class, using `model_type` where a field accepts several classes (see [below](#choosing-between-classes-model_type)).

To make variants of a configuration, for a sensitivity test or a parameter sweep, copy it and replace a field. The original is unchanged:

```python exec="on" source="above" result="text" session="configuration"
from rompy_xbeach.components.physics.wavemodel import Surfbeat

surfbeat = config.model_copy(update={"physics": Physics(wavemodel=Surfbeat())})
print(type(config.physics.wavemodel).__name__, type(surfbeat.physics.wavemodel).__name__)
```

`model_copy` does not validate the new value, so a copy can hold a combination that `Config` would reject, such as a stationary model with a JONSWAP boundary. Build a new `Config` instead when the change touches settings that depend on each other.

## Generating the workspace

A [`ModelRun`][rompy.model.ModelRun] adds the run period ([`TimeRange`][rompy.core.time.TimeRange]), a run id and an output folder. Calling it writes the XBeach workspace into `output_dir/run_id`:

```python exec="on" source="above" result="text" session="configuration"
from pathlib import Path

from rompy.core.time import TimeRange
from rompy.model import ModelRun

modelrun = ModelRun(
    run_id="python_model",
    period=TimeRange(start="2023-01-01T00:00", end="2023-01-01T01:00", interval="10m"),
    output_dir=OUT_DIR,
    config=config,
)
workspace = Path(modelrun())
print(sorted(p.name for p in workspace.iterdir()))
```

After generation, `config.params` holds the flat dictionary that was rendered into `params.txt`:

```python exec="on" source="above" result="text" session="configuration"
print({key: config.params[key] for key in ("tstop", "tunits", "wavemodel", "wbctype", "nx")})
```

[How rompy-xbeach works](how-it-works.md) describes the steps between the objects and `params.txt`.

## The same model as YAML

A YAML file holds the fields of a `ModelRun`: `run_id`, `output_dir`, `period` and `config`. Each nested object has the same fields as its Python class:

```python exec="on" result="yaml" session="configuration"
# Hidden: write the YAML file shown below into the temporary folder.
YAML_TEXT = """\
run_id: yaml_model
output_dir: output
delete_existing: true

period:
  start: 2023-01-01T00:00
  end: 2023-01-01T01:00
  interval: 10m

config:
  model_type: xbeach
  grid:
    ori: {x: 115.594239, y: -32.641104, crs: "EPSG:4326"}
    alfa: 347.0
    dx: 20.0
    dy: 30.0
    nx: 115
    ny: 110
    crs: "EPSG:28350"
  bathy:
    source:
      model_type: geotiff
      filename: tests/data/bathy.tif
    posdwn: false
    extension:
      model_type: linear
      depth: 15.0
      slope: 0.05
  input:
    wave:
      model_type: params
      Hrms: 1.0
      Trep: 10.0
      dir0: 270.0
      thetamin: -90.0
      thetamax: 90.0
      dtheta: 15.0
  physics:
    wavemodel:
      model_type: stationary
  tide_boundary:
    tideloc: 0
    zs0: 0.0
  output:
    globalvars: [H, zs, zb]
    tintg: 300.0
"""
CONFIG_FILE = Path(OUT_DIR) / "model.yml"
CONFIG_FILE.write_text(YAML_TEXT)
print(YAML_TEXT)
```

Relative paths, such as the bathymetry file and `output_dir`, are resolved from the folder Python or the `rompy` command runs in.

### Loading and validating YAML in Python

Loading the file into a `ModelRun` validates everything and builds the same objects as the Python code above:

```python exec="on" source="above" result="text" session="configuration"
import yaml

conf = yaml.safe_load(CONFIG_FILE.read_text())
modelrun = ModelRun(**conf)

print(type(modelrun.config.input.wave).__name__)
print(type(modelrun.config.physics.wavemodel).__name__)
```

Mistakes are reported with the path to the offending field. Here a wave height is given as text:

```python exec="on" source="above" result="text" session="configuration"
from pydantic import ValidationError

bad = yaml.safe_load(CONFIG_FILE.read_text())
bad["config"]["input"]["wave"]["Hrms"] = "large"
try:
    ModelRun(**bad)
except ValidationError as error:
    for err in error.errors():
        print(".".join(str(loc) for loc in err["loc"]), "->", err["msg"])
```

Misspelt field names are rejected with a suggestion, for example `Unknown field 'phyiscs'. Did you mean 'physics'?`.

To load only the model part of a file, pass the `config` section to `Config`: `Config(**conf["config"])`.

### Choosing between classes: `model_type`

Where a field accepts more than one class, `model_type` names the one to use. It is required for these fields and optional elsewhere. In a `ModelRun` file, `config` itself needs `model_type: xbeach`, because rompy chooses the model from it; a dictionary passed straight to `Config` does not.

| Field | Example `model_type` values |
|---|---|
| `config` | `xbeach` |
| `bathy.source`, forcing `source` | `geotiff`, `xyz`, `file`, `intake`, `wavespectra`, `oceantide`, `csv` |
| `bathy.extension` | `linear`; `base` for no extension |
| `bathy.interpolator` | `scipy_regular_grid` |
| `input.wave` | `params`, `station_spectra_jons`, `station_spectra_swan`, `grid_param_jons`, ... |
| `input.wind` | `wind_grid`, `wind_station`, `wind_point` |
| `input.tide` | `tide_cons_grid`, `water_level_station`, `water_level_point`, ... |
| `physics.wavemodel` | `stationary`, `surfbeat`, `nonh` |
| `physics.bedfriction` | `cf`, `chezy`, `manning`, `white-colebrook`, ... |

Each class's `model_type` is shown with its fields in the [reference](../reference/index.md). An unknown value lists the valid ones:

```python exec="on" source="above" result="text" session="configuration"
bad = yaml.safe_load(CONFIG_FILE.read_text())
bad["config"]["physics"]["wavemodel"]["model_type"] = "stationry"
try:
    ModelRun(**bad)
except ValidationError as error:
    print(error.errors()[0]["msg"])
```

### Generating from Python

A validated `ModelRun` is generated as before. The output folder is set to the temporary folder here:

```python exec="on" session="configuration"
# Hidden: write into the temporary folder rather than ./output.
conf["output_dir"] = OUT_DIR
modelrun = ModelRun(**conf)
```

```python exec="on" source="above" result="text" session="configuration"
workspace = Path(modelrun())
print(workspace.name, sorted(p.name for p in workspace.iterdir()))
```

## The `rompy` command line

rompy installs a `rompy` command that works on the same YAML file, with no Python code. Run it from the folder the relative paths refer to.

| Command | What it does |
|---|---|
| `rompy validate model.yml` | Loads and validates the file; exits with an error if anything is wrong |
| `rompy generate model.yml` | Writes the workspace; `--output-dir` overrides `output_dir` |
| `rompy run model.yml --backend-config backend.yml` | Generates the workspace and runs XBeach with a backend |
| `rompy schema` | Prints the JSON schema of the configuration |

```bash
rompy validate model.yml && echo "Configuration is valid"
rompy generate model.yml
```

`rompy run` also takes `--dry-run` (generate only) and `--skip-generate` (run a workspace generated earlier). The backend file and the options for running are described in [Running XBeach](running.md).

When the command line reads a file, it also:

- replaces `${VAR}` with the value of the environment variable `VAR`, for example `filename: ${DATA_DIR}/bathy.tif`, and stops with an error if the variable is not set;
- supports `!include other.yml` to compose a configuration from several files.

These two features belong to the command line's file loader; `yaml.safe_load` in Python does not apply them.

## YAML and Python objects

YAML to objects is the supported direction: keep the YAML file as the source of the configuration and load it with `ModelRun(**yaml.safe_load(...))`. The reverse, writing a `Config` built in Python back to YAML, is not supported at present. `model_dump()` on a component returns its XBeach parameters rather than its fields (for example `wavemodel: surfbeat` instead of `wavemodel: {model_type: surfbeat}`), and coordinate reference systems are dumped as pyproj objects, so the result cannot be loaded back as a `Config`.

A model created from a dictionary keeps its inputs, which `dump_inputs_dict()` returns:

```python exec="on" source="above" result="text" session="configuration"
inputs = modelrun.config.dump_inputs_dict()
print(inputs["physics"], inputs["bathy"]["extension"])
```

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["configuration", "declarative", "yaml", "cli"]))
```
