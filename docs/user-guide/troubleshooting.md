# Troubleshooting

Common problems when setting up and running XBeach with rompy-xbeach, grouped by where they show up: when the configuration is built, when the workspace is generated, and when XBeach runs. Each entry gives the cause and the fix, with a link to the page that covers the topic.

## When the configuration is built

### `Unknown field 'phyiscs'. Did you mean 'physics'?`

A field name is misspelt or does not exist on that class. The message suggests the closest field name. Field names follow the XBeach parameter names; check the class in the [reference](../reference/index.md). → [Configuration](configuration.md)

### `Input tag '...' found using 'model_type' does not match any of the expected tags`

The `model_type` of a nested object is misspelt, or belongs to a class the field does not accept. The message lists the valid values. If the message says `Unable to extract tag using discriminator 'model_type'`, the YAML leaves `model_type` out where a field accepts several classes. → [Configuration](configuration.md#choosing-between-classes-model_type)

### `Field required` for `physics` or `wavemodel`

`Config` requires `grid`, `bathy` and `physics`, and [`Physics`][rompy_xbeach.components.physics.physics.Physics] requires a `wavemodel`: `Stationary()`, `Surfbeat()` or `Nonh()`. → [Physics](physics.md)

### `wbctype=... cannot be used with the ... wave model`

`Config` rejects wave boundary types that XBeach refuses for the chosen wave model:

| Boundary | Allowed wave models |
|---|---|
| `BoundaryParams` with `Tlong` (bichromatic) | surfbeat |
| JONSWAP (`...Jons`, `wbctype = parametric`), SWAN spectra (`...Swan`), `BoundaryReuse` | surfbeat, nonh |
| `BoundaryTs1`, `BoundaryTs2` | surfbeat |
| `BoundaryTsNonh` | nonh |

Change the wave model or the boundary class. → [Waves](waves.md)

### `NotImplementedError: Only tideloc=1 is currently implemented`

The water level and tide interfaces write a single time series on the offshore boundary (`tideloc = 1`) and accept no other value. → [Tide and water levels](water-levels.md)

## When the workspace is generated

### `Interpolating grid failed, the source data extends over ...`

The model grid, or its extension, reaches outside the bathymetry data. Use data that covers the grid, or let the interpolator extrapolate with `RegularGridInterpolator(kwargs={"bounds_error": False, "fill_value": None})`. → [Grid and bathymetry](grid-and-bathymetry.md#interpolation)

### Warning: `The offshore depth (...) is greater than the extended depth (...)`

The seaward extension found the data already deeper than its target `depth`, so it adds no cells. The usual cause is a wrong `posdwn`: elevations (negative under water) need `posdwn=False`. Otherwise lower the extension `depth`, or leave the extension out. → [Grid and bathymetry](grid-and-bathymetry.md#seaward-extension)

### `time range ... outside of source time range ...`

The run period is not fully covered by the forcing data. Shorten the period or use data that covers it; `time_buffer` only adds source time steps that exist. → [Data sources](sources.md#cropping-to-the-run-period)

### `Station selection with sel_method='idw' returned only missing values`

Inverse distance weighting found no neighbouring sites within its tolerance, which happens with a single-site dataset. Use `sel_method="nearest"`, or adjust `sel_method_kwargs` (`tolerance`, `max_sites`). → [Data sources](sources.md#interpolating-to-the-location-sel_method)

### A variable or coordinate is not found in the source

`coords` defaults to `longitude`, `latitude` and `time`, and `variables` to each class's defaults. Set them to the names in your dataset, for example `coords={"x": "lon", "y": "lat"}`, and inspect the object's `ds` property to see what it reads. → [Data sources](sources.md#choosing-which-data-is-used)

### File paths are not found when running from YAML

Relative paths in a YAML file are resolved from the folder Python or the `rompy` command runs in, not from the folder of the YAML file. Run from the folder the paths refer to, or use absolute paths or `${VAR}` environment variables. → [Configuration](configuration.md#the-rompy-command-line)

## When XBeach runs

### Waves enter from the beach, or the model blows up at the offshore boundary

The grid is oriented the wrong way. The origin must be on the offshore boundary with the x-axis pointing onshore, and `alfa` is measured in degrees counter-clockwise from east. Plot the grid and check that the red offshore boundary faces the sea. → [Grid and bathymetry](grid-and-bathymetry.md#checking-the-orientation)

### Short waves are missing or come from the wrong direction

The directional grid (`thetamin`, `thetamax` and `dtheta`, or `dtheta_s` for spectral boundaries) is required whenever short waves are modelled. Its angles are relative to the grid x-axis unless `thetanaut = 1`, so `thetamin=-90, thetamax=90` covers every direction travelling onshore. → [Waves](waves.md)

### Wind has no effect

XBeach v1.24 defaults to `wind = 0`, so wind forcing in `input.wind` is ignored unless the process is switched on with `Physics(wind=True)`. → [Wind](wind.md)

### The water level is not the constant `zs0` without tide forcing

XBeach defaults to `tideloc = 2`. For a run without tide or water level forcing, set `TideBoundaryConditions(tideloc=0, zs0=...)` to keep a constant water level. When `input.tide` is set, it writes `tideloc = 1`; a `tideloc` in `tide_boundary` replaces that value in `params.txt`. → [Flow and tide boundaries](boundary-conditions.md)

### `Unknown, unused or multiple statements of parameter ...` in `XBlog.txt`

XBeach did not use a parameter in `params.txt`. This is expected for parameters that only apply to other settings, and a sign of a version mismatch when XBeach does not know the parameter at all. Check the XBeach version of your executable or Docker tag. → [Running XBeach](running.md#checking-a-run)

### `BoundaryStatTable` runs fail

`BoundaryStatTable` writes `wbctype = stat_table`, which XBeach does not accept, so it cannot be used yet. For time-varying parametric waves use a JONSWAP table boundary (`...Jonstable`). → [Waves](waves.md)

### MPI runs use one process fewer than requested

This is how XBeach works: with `mpirun -n N`, one process writes output and N - 1 compute. `XBlog.txt` reports `MPI version, running on N-1 processes`. Ask for one process more than the number of subdomains you want. `xbeach-mpi` in the Docker image refuses to run on a single process; `xbeach` falls back to the serial build. → [Running XBeach](running.md#in-parallel-with-mpi)

### Output files in the workspace are owned by root

The Docker container runs as root, so on Linux the files XBeach writes are owned by root. Remove or change them with `sudo`, or run the image yourself with `docker run --user "$(id -u):$(id -g)"`. → [Running XBeach](running.md#docker-without-rompy)

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["execution", "configuration"]))
```
