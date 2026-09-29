# Running XBeach

rompy-xbeach writes an XBeach workspace; running XBeach on it is done by rompy's run backends. [`LocalConfig`][rompy.backends.config.LocalConfig] runs a command on your machine, and [`DockerConfig`][rompy.backends.config.DockerConfig] runs XBeach in a container, so no local XBeach build is needed. This page covers both, parallel runs with MPI, running from the command line and checking that a run worked.

The examples use a `modelrun` built as in [Your first model](../getting-started/first-model.md).

## Generate once, then run

`ModelRun.run(backend, workspace_dir=...)` runs XBeach in an existing workspace. Pass the folder returned by generation, so the files are not generated a second time:

```python
workspace = modelrun()                       # write the workspace
modelrun.run(backend, workspace_dir=workspace)
```

Without `workspace_dir`, `run()` generates the workspace first. It returns `True` if the command finished without error.

## With your own XBeach installation

`LocalConfig` runs a shell command in the workspace folder. Use it when XBeach is installed on your machine or cluster node:

```python
from rompy.backends import LocalConfig

backend = LocalConfig(command="xbeach", timeout=3600)
modelrun.run(backend, workspace_dir=workspace)
```

`env_vars` adds environment variables, and `stream_output=True` prints XBeach's output as it runs.

## With Docker

The public image `ghcr.io/rom-py/xbeach` contains XBeach built with NetCDF output and MPI. `DockerConfig` mounts the workspace into the container and runs `executable` there:

```python
from rompy.backends import DockerConfig

backend = DockerConfig(image="ghcr.io/rom-py/xbeach:trunk-r6147", executable="xbeach")
modelrun.run(backend, workspace_dir=workspace)
```

The tag selects the XBeach version:

| Tag | XBeach |
|---|---|
| `1.24.6057-halloween-beta` | Release v1.24.6057 "Halloween" (BETA) |
| `trunk-r<revision>`, e.g. `trunk-r6147` | The SVN trunk at that revision |
| `latest` | The newest `trunk-r<revision>` image |

Pin a release or trunk tag so runs stay reproducible; `latest` moves when a new trunk revision is published. The image has three commands:

| Command | What runs |
|---|---|
| `xbeach` | The MPI build when started by `mpirun` with 2 or more processes, the serial build otherwise |
| `xbeach-serial` | The serial build |
| `xbeach-mpi` | The MPI build, which refuses to run on a single process |

!!! warning "Output files are owned by root"
    The container runs as root, so the files XBeach writes into the workspace are owned by root on Linux. Remove or change them with `sudo`, or run the container yourself with `--user` (see [below](#docker-without-rompy)).

## In parallel with MPI

XBeach splits the domain between MPI processes. The [`Mpi`][rompy_xbeach.components.mpi.Mpi] component in the `Config` chooses how, through `mpiboundary`:

| `mpiboundary` | Split |
|---|---|
| `auto` | XBeach chooses the split with the shortest internal boundaries (the XBeach default) |
| `x` | Cross-shore, each subdomain spanning the full alongshore width |
| `y` | Alongshore, each subdomain spanning the full cross-shore length |
| `man` | Manual: `mmpi` domains cross-shore by `nmpi` alongshore, both required |

```python exec="on" source="above" result="text"
from rompy_xbeach.components.mpi import Mpi

print(Mpi(mpiboundary="man", mmpi=2, nmpi=3).get(destdir="."))
```

The number of processes is set when XBeach is launched, not in the `Config`. With `mpirun -n N`, XBeach uses one process for output and computes on **N - 1**: `-n 4` gives three subdomains, and `-n 2` runs the computation on a single process. `DockerConfig` sets N with `cpu` and the launcher with `mpiexec`:

```python
from rompy_xbeach.components.mpi import Mpi

parallel = ModelRun(
    run_id="mpi",
    period=modelrun.period,
    output_dir=modelrun.output_dir,
    config=modelrun.config.model_copy(update={"mpi": Mpi(mpiboundary="auto")}),
)
workspace = parallel()
backend = DockerConfig(
    image="ghcr.io/rom-py/xbeach:trunk-r6147", executable="xbeach", mpiexec="mpirun", cpu=4
)
parallel.run(backend, workspace_dir=workspace)
```

With your own MPI build of XBeach, use `LocalConfig(command="mpirun -n 4 xbeach")`.

## From the command line

`rompy run` generates and runs a YAML model configuration (see [Configuration](configuration.md#the-same-model-as-yaml)) with a backend described in a second YAML file. The backend file holds the fields of the backend class plus its `type` (`local`, `docker` or `slurm`):

```yaml
# docker.yml
type: docker
image: ghcr.io/rom-py/xbeach:trunk-r6147
executable: xbeach
timeout: 3600
```

```bash
rompy run model.yml --backend-config docker.yml
```

For MPI, add `mpiexec: mpirun` and `cpu: 4` to the backend file. `rompy run --dry-run` only generates the workspace, `--skip-generate` runs a workspace generated earlier, and `rompy backends create --backend-type docker` writes a template backend file.

### Docker without rompy

The image also runs a generated workspace directly. From inside the workspace folder:

```bash
docker run --rm -v "$PWD":/data ghcr.io/rom-py/xbeach:trunk-r6147 xbeach
docker run --rm -v "$PWD":/data ghcr.io/rom-py/xbeach:trunk-r6147 mpirun -n 4 xbeach
```

On Linux, add `--user "$(id -u):$(id -g)"` to keep your own ownership of the output files.

## Checking a run

XBeach writes into the workspace:

| File | Content |
|---|---|
| `XBlog.txt` | The run log: parameters read, warnings, progress and the end of the run |
| `XBerror.txt`, `XBwarning.txt` | Errors and warnings; empty after a clean run |
| `xboutput.nc` | The output (NetCDF, the rompy-xbeach default; the name is set by `Output.ncfilename`) |

A run that stops early usually says why in `XBerror.txt` or at the end of `XBlog.txt`, for example `File 'E_reuse.bcf' not found. Terminating simulation`. A finished run ends its log with `End of program xbeach`. XBeach also notes in `XBlog.txt` each parameter it did not use (`Unknown, unused or multiple statements of parameter ...`), which is worth checking after changing XBeach version. With MPI, the log reports `MPI version, running on N processes` (the computing processes) and the processor grid.

```python
from pathlib import Path

import xarray as xr

log = (Path(workspace) / "XBlog.txt").read_text()
print(log.splitlines()[-5:])

ds = xr.open_dataset(Path(workspace) / "xboutput.nc")
```

[Troubleshooting](troubleshooting.md) lists common problems and their fixes.

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["execution", "backends", "mpi"]))
```
