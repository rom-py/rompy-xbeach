# Hotstart and chained runs

XBeach can save the model state during a run and start a later run from it. This splits a long simulation into shorter runs, or starts a run from a spun-up state instead of still water. In rompy-xbeach, the first run writes the state through [`Output`][rompy_xbeach.components.output.Output], and the next run reads it through the `hotstart` field of [`Config`][rompy_xbeach.config.Config], which copies the files from the previous run's workspace.

```python exec="on" session="hotstart"
# Hidden setup: quiet logging, and a fake previous run holding two saved states.
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
PREVIOUS = Path(tempfile.mkdtemp())
for number in (1, 2):
    for variable in ("zs", "zb", "uu", "vv"):
        (PREVIOUS / f"hotstart_{variable}{number:06d}.dat").touch()
WORKSPACE = Path(tempfile.mkdtemp())
```

## Writing hotstart files

`writehotstart=True` saves the model state every `tinth` seconds; without `tinth` it is saved only at the end of the run. Each save is a set of files `hotstart_<variable><number>.dat`, one per state variable, numbered from 1. The minimum set is `zs`, `zb`, `uu` and `vv`; more are written depending on the processes switched on.

```python exec="on" source="above" result="text" session="hotstart"
from rompy_xbeach.components.output import Output

output = Output(globalvars=["zs"], tintg=60.0, writehotstart=True, tinth=600.0)
print(output.get(destdir="."))
```

A 20-minute run with `tinth=600` writes states 1 (at 10 minutes) and 2 (at 20 minutes).

## Starting from a hotstart

[`Hotstart`][rompy_xbeach.components.hotstart.Hotstart] chooses the saved state with `hotstartfileno` and, with `previous_run`, the workspace to copy it from. `previous_run` is an [`XBeachDirectoryBlob`][rompy_xbeach.types.XBeachDirectoryBlob]: a folder, local or remote (for example `s3://...`), that must exist when the object is created. When the workspace is generated, the files of the chosen state, `hotstart_*<number>.dat`, are copied into it:

```python exec="on" source="above" result="text" session="hotstart"
from rompy_xbeach.components.hotstart import Hotstart
from rompy_xbeach.types import XBeachDirectoryBlob

hotstart = Hotstart(hotstartfileno=2, previous_run=XBeachDirectoryBlob(source=str(PREVIOUS)))
print(hotstart.get(destdir=WORKSPACE))
print(sorted(p.name for p in WORKSPACE.iterdir()))
```

In the `Config`, `hotstart` takes:

| Value | Effect |
|---|---|
| `None` (default) | No hotstart |
| `True` | Writes `hotstart = 1` only: XBeach reads state 0 from files already in the workspace, and nothing is copied |
| `Hotstart(...)` | Writes `hotstart` and `hotstartfileno`, and copies the state from `previous_run` if given |

In YAML:

```yaml
hotstart:
  hotstartfileno: 2
  previous_run:
    source: output/first_run
```

## A chained run

A chained run is two `ModelRun`s that share the grid, bathymetry and model settings. The second starts when the first ends and reads the first run's last saved state. Running needs XBeach (see [Running XBeach](running.md)):

```python
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy_xbeach.components.hotstart import Hotstart
from rompy_xbeach.components.output import Output
from rompy_xbeach.config import Config
from rompy_xbeach.types import XBeachDirectoryBlob

common = dict(grid=grid, bathy=bathy, input=forcing, physics=physics, tide_boundary=tide_boundary)

# First run: 00:00 to 00:20, saving the state every 10 minutes (states 1 and 2)
first = ModelRun(
    run_id="first_run",
    period=TimeRange(start="2023-01-01T00:00", end="2023-01-01T00:20", interval="10m"),
    output_dir="output",
    config=Config(
        **common,
        output=Output(globalvars=["zs"], tintg=60.0, writehotstart=True, tinth=600.0),
    ),
)
first_workspace = first()
first.run(backend, workspace_dir=first_workspace)

# Second run: 00:20 to 00:40, starting from state 2, the end of the first run
second = ModelRun(
    run_id="second_run",
    period=TimeRange(start="2023-01-01T00:20", end="2023-01-01T00:40", interval="10m"),
    output_dir="output",
    config=Config(
        **common,
        output=Output(globalvars=["zs"], tintg=60.0),
        hotstart=Hotstart(
            hotstartfileno=2,
            previous_run=XBeachDirectoryBlob(source=str(first_workspace)),
        ),
    ),
)
second.run(backend, workspace_dir=second())
```

The second `Hotstart` can only be created once the first run has finished, since `previous_run` must exist. The water level in the second run then continues from the end of the first, without a second spin-up.

### What must match between runs

- **Grid and bathymetry.** Hotstart files hold arrays on the grid of the first run, so the grid and any bathymetry extensions must be identical.
- **Model settings.** Keep the same wave model and processes, and the same MPI layout if MPI is used.
- **Forcing.** Give the second run forcing for its own period: the data interfaces cut the forcing to each run's period. To continue exactly the same wave time series, reuse the first run's boundary files (see [Waves](waves.md)).

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["hotstart"]))
```
