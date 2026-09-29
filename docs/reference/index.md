# API reference

The reference is generated from the rompy-xbeach source code. Each class lists all its fields, including those it inherits from [rompy](https://rom-py.github.io/rompy/reference/), with their types, defaults and descriptions. Fields named after an XBeach parameter write that parameter to `params.txt`. The [parameter index](parameters.md) finds the field for a given parameter.

| Page | What it covers |
|---|---|
| [Config](config.md) | `Config`, the model configuration, and `DataInterface`, which holds the forcing |
| [Grid](grid.md) | `RegularGrid` and `GeoPoint` |
| [Bathymetry](bathy.md) | `XBeachBathy`, seaward extensions and the interpolator |
| [Sources](sources.md) | Readers for GeoTIFF, XYZ, datasets and files with a CRS, and tide constituents |
| [Data interfaces](data.md) | Base classes for point, station and grid forcing; forcing files |
| [Wave boundaries](wave-boundaries.md) | Parametric, spectral and special wave boundaries |
| [Water levels](water-levels.md) | Water level time series and tide constituents |
| [Wind](wind.md) | Wind from points, stations or grids |
| [Physics](physics.md) | `Physics`, wave models, bed friction, numerics and other processes |
| [Sediment](sediment.md) | `Sediment`, transport, morphology, bed composition, groundwater |
| [Flow and tide boundaries](boundary-conditions.md) | Flow and tide boundary condition settings |
| [Output](output.md) | Output variables, locations and times |
| [Hotstart and MPI](hotstart-mpi.md) | Starting from a previous run, parallel runs |
| [Base types](types.md) | Base classes, file fields and enumerations |
| [Parameter index](parameters.md) | Every XBeach parameter and where to set it |
