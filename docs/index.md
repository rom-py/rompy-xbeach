# Rompy-XBeach

**Python interface for the XBeach coastal morphodynamic model**

Rompy-xbeach provides a type-safe, Pythonic way to configure and run [XBeach](https://xbeach.readthedocs.io/) simulations. It is part of the [rompy](https://github.com/rom-py/rompy) ecosystem for regional ocean modelling.

## Features

- **Type-safe configuration** — Pydantic models validate parameters before running XBeach, catching errors early with clear messages
- **Data interfaces** — Automatic generation of boundary condition files from various data sources (NetCDF, THREDDS, local files)
- **Structured organisation** — Related parameters grouped into logical components for better discoverability
- **YAML support** — Define configurations declaratively for reproducibility
- **IDE support** — Full autocomplete and type hints in modern editors

## Quick Example

```python
from rompy_xbeach import Config
from rompy_xbeach.components import Physics, Sediment, Output
from rompy_xbeach.components.physics import Surfbeat, BedFriction
from rompy_xbeach.components.sediment import Morphology

config = Config(
    grid=grid,
    bathy=bathy,
    physics=Physics(
        wavemodel=Surfbeat(),
        bedfriction=BedFriction(bedfriccoef=0.01),
    ),
    sediment=Sediment(
        morphology=Morphology(morfac=10),
    ),
    output=Output(outputformat="netcdf", tintg=3600),
)
```

This generates a valid `params.txt` file:

```
wavemodel    = surfbeat
bedfriccoef  = 0.01
morfac       = 10
outputformat = netcdf
tintg        = 3600.0
```

## Why Rompy-XBeach?

XBeach uses a flat `params.txt` file with ~250 parameters. While flexible, this can be:

- **Error-prone** — Typos in parameter names silently ignored
- **Hard to discover** — Which parameters exist? What are valid values?
- **Difficult to validate** — Invalid combinations only fail at runtime

Rompy-xbeach addresses these by:

1. **Validating at construction** — Invalid values raise clear errors immediately
2. **Grouping related parameters** — Find morphology settings under `sediment.morphology`
3. **Enforcing constraints** — Can't set non-hydrostatic parameters on a surfbeat model
4. **Providing defaults** — Sensible XBeach defaults with documentation

## Documentation Structure

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install rompy-xbeach and run your first simulation

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn the architecture and how to configure XBeach models

    [:octicons-arrow-right-24: Architecture](user-guide/architecture.md)

-   :material-puzzle:{ .lg .middle } **Components**

    ---

    Detailed reference for each component (Physics, Sediment, Output, etc.)

    [:octicons-arrow-right-24: Components](components/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation generated from source code

    [:octicons-arrow-right-24: API Reference](api-reference/config.md)

-   :material-notebook:{ .lg .middle } **Examples**

    ---

    Interactive Jupyter notebooks demonstrating rompy-xbeach features

    [:octicons-arrow-right-24: Examples](examples/index.md)

-   :material-account-group:{ .lg .middle } **Developer**

    ---

    Contributing guidelines and development setup

    [:octicons-arrow-right-24: Contributing](developer/contributing.md)

</div>

## Part of the Rompy Ecosystem

Rompy-xbeach is a plugin for [rompy](https://github.com/rom-py/rompy), the regional ocean modelling framework. Other model plugins include:

- [rompy-swan](https://github.com/rom-py/rompy-swan) — SWAN spectral wave model
- [rompy-schism](https://github.com/rom-py/rompy-schism) — SCHISM unstructured grid model

## Links

- [XBeach Documentation](https://xbeach.readthedocs.io/)
- [XBeach Release & Source](https://oss.deltares.nl/web/xbeach/release-and-source)
- [Rompy Core](https://rom-py.github.io/rompy/)
