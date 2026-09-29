# Output

[`Output`][rompy_xbeach.components.output.Output] chooses what XBeach writes and when: instantaneous and time-averaged maps, time series at fixed points, run-up gauges, the output format and file, and hotstart files. It is the `output` field of [`Config`][rompy_xbeach.config.Config], which defaults to `Output()`: NetCDF output with no variables selected. The variable lists are written with the counts XBeach needs (`nglobalvar`, `nmeanvar`, `npointvar`, `npoints`, `nrugauge`), which are filled in for you.

```python exec="on" session="output"
# Hidden setup: quiet logging, a temporary folder and a helper that prints the
# parameters as they appear in params.txt (lists one item per line).
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
TMP = Path(tempfile.mkdtemp())


def show(params):
    for key, value in params.items():
        if isinstance(value, list):
            print("\n".join(str(item) for item in value))
        else:
            print(f"{key} = {value}")
```

## Output types

| Type | Variables | Locations | Interval |
|---|---|---|---|
| Instantaneous maps | [`globalvars`][rompy_xbeach.components.output.Output.globalvars] | Whole grid | `tintg` |
| Time-averaged maps: mean, variance, minimum and maximum | [`meanvars`][rompy_xbeach.components.output.Output.meanvars] | Whole grid | `tintm` |
| Point time series | [`pointvars`][rompy_xbeach.components.output.Output.pointvars] | [`points`][rompy_xbeach.components.output.Output.points] | `tintp` |
| Run-up gauges | [`pointvars`][rompy_xbeach.components.output.Output.pointvars] | [`rugauges`][rompy_xbeach.components.output.Output.rugauges] | `tintp` |

Hourly means of wave height and water level, and bed level snapshots every 30 minutes:

```python exec="on" source="above" result="text" session="output"
from rompy_xbeach.components.output import Output

output = Output(
    meanvars=["H", "zs"],
    tintm=3600.0,
    globalvars=["zb"],
    tintg=1800.0,
)
show(output.get(destdir=TMP))
```

`show` is a small helper that prints each parameter on its own line, and each item of a list on its own line, as in `params.txt`.

## Output variables

Variable names are those of XBeach, listed in [`OutputVarsEnum`][rompy_xbeach.types.OutputVarsEnum] and in the [XBeach output variables](https://xbeach.readthedocs.io/en/latest/output_variables.html). Unknown names are rejected, and so are duplicates in a list:

```python exec="on" source="above" result="text" session="output"
from pydantic import ValidationError

try:
    Output(globalvars=["H", "Hs"])
except ValidationError as err:
    print(err.errors()[0]["msg"][:120], "...")
```

```python exec="on" source="above" result="text" session="output"
from rompy_xbeach.types import OutputVarsEnum

print(len(OutputVarsEnum), "variables, for example:", [v.value for v in OutputVarsEnum][:12])
```

XBeach accepts at most 20 global, 15 mean and 50 point variables, and 50 points and 50 run-up gauges. rompy-xbeach logs a warning beyond these limits.

## Points and run-up gauges

Point and gauge locations are `(x, y)` pairs in the grid's coordinate reference system (`RegularGrid.crs`), and XBeach uses the nearest grid point. Run-up gauges follow the moving waterline along the cross-shore grid line nearest to the given location. Both use the variables in `pointvars`; for run-up gauges XBeach also adds `xw`, `yw` and `zs`.

Taking the coordinates from the grid arrays, shaped `(ny, nx)`, avoids placing points outside the grid. Here three points along the middle cross-shore line, and a gauge on the same line:

```python exec="on" source="above" result="text" session="output"
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
iy = grid.ny // 2
points = [(round(float(grid.x[iy, ix]), 1), round(float(grid.y[iy, ix]), 1)) for ix in (20, 60, 100)]

output = Output(points=points, rugauges=[points[0]], pointvars=["zs", "H", "u"], tintp=1.0)
show(output.get(destdir=TMP))
```

Point variables without any location, or locations without variables, produce no point output and are logged as warnings. `nrugdepth` and `rugdepth` set the depth used to find the last wet point of a gauge.

## Output times

All times are in seconds from the start of the run.

| Field | Sets | XBeach default |
|---|---|---|
| [`tstart`][rompy_xbeach.components.output.Output.tstart] | Start of all output | 0 |
| [`tintg`][rompy_xbeach.components.output.Output.tintg] | Interval of instantaneous maps, from `tstart` | 900 s |
| [`tintm`][rompy_xbeach.components.output.Output.tintm] | Averaging interval of time-averaged maps; the first is written at `tstart + tintm` | The whole output period |
| [`tintp`][rompy_xbeach.components.output.Output.tintp] | Interval of point and run-up gauge output, from `tstart` | 1 s |
| [`tinth`][rompy_xbeach.components.output.Output.tinth] | Interval of hotstart files, with `writehotstart` | End of the run |

With morphological acceleration (`morfac` with `morfacopt` on), output times are in morphological time; see [Sediment and morphology](sediment.md#morphological-acceleration).

```python exec="on" source="above" result="text" session="output"
output = Output(globalvars=["zb", "zs"], tstart=1800.0, tintg=600.0)
show(output.get(destdir=TMP))
```

### Output at irregular times

Instead of a fixed interval, output can be written at times read from a file: `tsglobal` for maps, `tsmean` for time-averaged maps and `tspoints` for point and run-up gauge output. The file has the number of times on its first line and one time per line after it. It is given as an [`XBeachDataBlob`][rompy_xbeach.types.XBeachDataBlob], copied into the workspace and written by name. When both are set, the file takes precedence over the interval, and rompy-xbeach logs a warning.

```python exec="on" source="above" result="text" session="output"
from rompy_xbeach.types import XBeachDataBlob

times = [0, 600, 1200, 3600, 7200]
(TMP / "global_times.txt").write_text(f"{len(times)}\n" + "\n".join(map(str, times)) + "\n")

output = Output(globalvars=["zb"], tsglobal=XBeachDataBlob(source=TMP / "global_times.txt"))
show(output.get(destdir=TMP / "workspace"))
```

## Format and file

| Field | Options | Default |
|---|---|---|
| [`outputformat`][rompy_xbeach.components.output.Output.outputformat] | `netcdf`, `fortran` (binary `.dat` files) or `debug` (both) | `netcdf` in rompy-xbeach |
| [`outputprecision`][rompy_xbeach.components.output.Output.outputprecision] | `single` or `double`, for NetCDF output | `double` |
| [`ncfilename`][rompy_xbeach.components.output.Output.ncfilename] | NetCDF file name | `xboutput.nc` |

`outputformat = netcdf` is always written unless you set the field to `None`. The Docker image's XBeach is built with NetCDF; an XBeach built without it stops with `netcdf`.

```python exec="on" source="above" result="text" session="output"
output = Output(ncfilename="storm.nc", outputprecision="single")
show(output.get(destdir=TMP))
```

Other fields:

- `projection` stores a coordinate reference system string in the NetCDF file as metadata; it does not affect the results.
- `rotate` (XBeach default on) rotates vector output by the grid angle `alfa`.
- `remdryoutput` (XBeach default on for NetCDF) removes dry points from the output of water levels and related variables.
- `timings` switches XBeach's progress output to the screen.

## Hotstart files

`writehotstart=True` saves the model state so that another run can continue from it, every `tinth` seconds or at the end of the run. See [Hotstart and chained runs](hotstart.md).

```python exec="on" source="above" result="text" session="output"
output = Output(writehotstart=True, tinth=3600.0)
show(output.get(destdir=TMP))
```

`Output.get()` keeps switches such as `writehotstart` as `True` or `False`; `Config` converts all booleans to 0 or 1 when it writes `params.txt`.

## In YAML

Under the `output` key of the model config, points as pairs:

```python exec="on" source="above" result="text" session="output"
import yaml

output = Output(
    **yaml.safe_load(
        """
        globalvars: [zb, zs, H]
        tintg: 600.0
        pointvars: [zs, H]
        points:
          - [383500.0, 6389500.0]
        tintp: 2.0
        """
    )
)
show(output.get(destdir=TMP))
```

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["outputs"]))
```
