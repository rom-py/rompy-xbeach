# How rompy-xbeach works

XBeach reads one flat parameter file, `params.txt`, plus input files for the grid, the bathymetry and the forcing. rompy-xbeach builds these from a tree of validated objects. This page explains that tree, which parts come from rompy, and how the objects become `params.txt`.

## The configuration tree

```text
ModelRun                      rompy: run id, period, output directory; generates and runs
└── config: Config            rompy-xbeach: the XBeach model
    ├── grid                  RegularGrid: where the model is
    ├── bathy                 XBeachBathy: depths on the grid, from a data source
    ├── input                 DataInterface: forcing data
    │   ├── wave              a wave boundary (parameters, spectra or files)
    │   ├── tide              water levels or tide constituents
    │   └── wind              wind time series or fields
    ├── physics               Physics: wave model and processes (required)
    ├── sediment              Sediment: transport, morphology, bed
    ├── flow_boundary         FlowBoundaryConditions: front, back, left, right
    ├── tide_boundary         TideBoundaryConditions: tideloc, zs0, ...
    ├── output                Output: variables, locations, times
    ├── mpi                   Mpi: domain decomposition
    └── hotstart              Hotstart: start from a previous run
```

The objects fall into two groups:

- **Data interfaces** (`grid`, `bathy`, `input`) read data, write input files into the workspace and set the parameters that point to them, such as `depfile`, `bcfile` or `zs0file`.
- **Components** (`physics`, `sediment`, the boundaries, `output`, `mpi`, `hotstart`) hold model settings. Their fields are named after the XBeach parameters they set.

## What comes from rompy

rompy-xbeach builds on [rompy](https://rom-py.github.io/rompy/), which provides the parts shared by all models:

| rompy provides | Used in rompy-xbeach for |
|---|---|
| [`ModelRun`][rompy.model.ModelRun] | Generating the workspace and running XBeach |
| [`TimeRange`][rompy.core.time.TimeRange] | The run period: `tstop`, `tunits` and the time range of the forcing |
| [`BaseConfig`][rompy.core.config.BaseConfig] | The base of [`Config`][rompy_xbeach.config.Config], with the `template` and `checkout` fields |
| [`DataGrid`][rompy.core.data.DataGrid] | The base of the data interfaces, with the `source`, `filter`, `crop_data`, `buffer` and `time_buffer` fields |
| rompy sources, e.g. [`SourceFile`][rompy.core.source.SourceFile] | Reading data; rompy-xbeach adds a coordinate reference system |
| Backends, e.g. [`DockerConfig`][rompy.backends.config.DockerConfig] | Running XBeach locally, in Docker or on a cluster |
| The `rompy` command line | Generating and running models from YAML |

The [reference](../reference/index.md) lists the fields each class inherits from rompy together with its own.

## From objects to `params.txt`

A component's fields map one-to-one to XBeach parameters, and `get()` returns them as a flat dictionary:

```python exec="on" source="above" result="text"
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.friction import Manning
from rompy_xbeach.components.physics.wavemodel import Roelvink1, Surfbeat

physics = Physics(
    wavemodel=Surfbeat(breaktype=Roelvink1(gamma=0.55)),
    bedfriction=Manning(bedfriccoef=0.02),
    wind=True,
)
print(physics.get(destdir="."))
```

Three rules turn fields into parameters:

- **A field set to `None` is not written**, and XBeach uses its own default. Most fields default to `None`; their descriptions give the XBeach default.
- **A choice between variants writes the variant's name.** `wavemodel=Surfbeat()` writes `wavemodel = surfbeat`, and the variant's own fields (here `break` and `gamma`) are written alongside. The variants are pydantic models told apart by their `model_type`, so only valid combinations can be built.
- **Booleans are written as 0 or 1.** Some processes accept either a boolean or a component: `vegetation=True` switches vegetation on with XBeach's defaults, while `vegetation=Vegetation(...)` also sets its parameters.

File fields, such as `bedfricfile`, copy the file into the workspace and write its name.

## Generating the workspace

Calling a [`ModelRun`][rompy.model.ModelRun] generates the workspace. rompy calls the configuration with the run details, and [`Config`][rompy_xbeach.config.Config] then:

1. sets `tstop` and `tunits` from the run period;
2. calls each data interface, which writes its files (for example the wave boundary files) and returns their parameters;
3. writes the bathymetry and grid files and their parameters;
4. adds the parameters of each component;
5. renders `params.txt` from these parameters with the template.

Each step writes into the same workspace folder, `output_dir/run_id`, which is ready for XBeach to run.

## Validation

Mistakes are caught when the objects are created, before XBeach runs:

- **Types and ranges.** Each field checks its type and allowed range, and unknown fields are rejected with a suggestion:

    ```python exec="on" source="above" result="text"
    from rompy_xbeach.components.physics.wavemodel import Surfbeat

    try:
        Surfbeat(breakype="roelvink1")
    except Exception as err:
        print(err)
    ```

- **Required choices.** `Physics` needs a wave model, and `Config` needs a grid, bathymetry and physics.
- **Combinations XBeach refuses.** `Config` checks settings that depend on each other, for example that the wave boundary type suits the wave model:

    ```python exec="on" source="above" result="text"
    from rompy_xbeach.components.physics import Physics
    from rompy_xbeach.components.physics.wavemodel import Stationary
    from rompy_xbeach.config import Config, DataInterface
    from rompy_xbeach.data.bathy import XBeachBathy
    from rompy_xbeach.data.boundary import BoundaryParams
    from rompy_xbeach.grid import RegularGrid
    from rompy_xbeach.source import SourceGeotiff

    try:
        Config(
            grid=RegularGrid(ori={"x": 0, "y": 0}, alfa=0, dx=10, dy=10, nx=10, ny=10, crs=28350),
            bathy=XBeachBathy(source=SourceGeotiff(filename="tests/data/bathy.tif")),
            input=DataInterface(wave=BoundaryParams(Hrms=1, Trep=10, Tlong=80)),
            physics=Physics(wavemodel=Stationary()),
        )
    except Exception as err:
        print(err)
    ```
