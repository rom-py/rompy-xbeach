# Flow and tide boundaries

Two components set how the flow and the water level behave at the edges of the grid: [`FlowBoundaryConditions`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions] (the `flow_boundary` field of [`Config`][rompy_xbeach.config.Config]) chooses the boundary type on each side, and [`TideBoundaryConditions`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions] (`tide_boundary`) sets how water levels are applied. Both are optional, and a field left as `None` keeps the XBeach default. Wave boundaries are set separately, through `input.wave` (see [Waves](waves.md)), and water level time series through `input.tide` (see [Tide and water levels](water-levels.md)).

```python exec="on" session="boundaries"
# Hidden setup: quiet logging.
from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
```

## The sides of the grid

The grid origin is on the offshore boundary and x points onshore, so the sides are named from the sea:

| Field | Side |
|---|---|
| `front` | Offshore boundary, at x = 0 |
| `back` | Landward boundary, at the last x |
| `right` | Lateral boundary at y = 0 |
| `left` | Lateral boundary at the last y |

## Flow boundaries

| Field | Options | XBeach default |
|---|---|---|
| [`front`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.front] | `abs_2d`, `abs_1d`, `wall`, `wlevel`, `nonh_1d`, `waveflume` | `abs_2d` on 2D grids, `abs_1d` on 1D grids (see below) |
| [`back`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.back] | `abs_2d`, `abs_1d`, `wall`, `wlevel` | `abs_2d` on 2D grids, `abs_1d` on 1D grids |
| [`left`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.left], [`right`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.right] | `neumann`, `wall`, `no_advec`, `neumann_v`, `abs_1d` | `neumann` |

- `abs_2d` and `abs_1d` are absorbing-generating (weakly reflective) boundaries that impose the incoming waves and let outgoing long waves leave; `abs_2d` also handles waves leaving at an angle.
- `wall` blocks all flow through the boundary.
- `nonh_1d` is the offshore boundary for the non-hydrostatic model, and `waveflume` a wave maker for laboratory flumes.
- `neumann` assumes no alongshore gradient in water level; `neumann_v` copies the velocity from the adjacent cell and `no_advec` keeps only the advective terms.

XBeach overrides `front` for two wave models: it always uses `abs_1d` with `Stationary` and `nonh_1d` with `Nonh`, and logs a warning when it changes the value.

!!! warning "`wlevel`"
    XBeach 1.24 stops at start-up with "wlevel no longer supported" when `front` or `back` is `wlevel`.

With no fields set, nothing is written:

```python exec="on" source="above" result="text" session="boundaries"
from rompy_xbeach.components.boundary.parameters import FlowBoundaryConditions

print(FlowBoundaryConditions().get(destdir=None))
```

A beach backed by a dune or seawall, closed on the landward side:

```python exec="on" source="above" result="text" session="boundaries"
flow_boundary = FlowBoundaryConditions(
    front="abs_2d", back="wall", left="neumann", right="neumann"
)
print(flow_boundary.get(destdir=None))
```

A laboratory flume, with a wave maker offshore and walls on the sides:

```python exec="on" source="above" result="text" session="boundaries"
flow_boundary = FlowBoundaryConditions(
    front="waveflume", back="abs_1d", left="wall", right="wall"
)
print(flow_boundary.get(destdir=None))
```

Values outside the allowed options are rejected:

```python exec="on" source="above" result="text" session="boundaries"
from pydantic import ValidationError

try:
    FlowBoundaryConditions(back="absorbing")
except ValidationError as err:
    print(err.errors()[0]["msg"])
```

### Other flow boundary settings

| Field | Sets | XBeach default |
|---|---|---|
| [`lateralwave`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.lateralwave] | Short-wave energy at the lateral boundaries: `neumann`, `wavecrest` or `cyclic` | `neumann` |
| [`nc`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.nc] | Number of cells over which the mean current at the offshore boundary is smoothed | ny + 1 |
| [`highcomp`][rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.highcomp] | High-order compensation terms at the boundary | off |

With `neumann`, oblique waves can leave a shadow zone along the lateral boundaries. `wavecrest` assumes no gradient along the wave crests, which reduces it, and `cyclic` connects the two sides for alongshore-uniform tests.

```python exec="on" source="above" result="text" session="boundaries"
flow_boundary = FlowBoundaryConditions(lateralwave="wavecrest")
print(flow_boundary.get(destdir=None))
```

## Water level boundaries

[`tideloc`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.tideloc] sets how many water level signals XBeach applies:

| `tideloc` | Water level |
|---|---|
| `0` | Constant level `zs0` everywhere, no water level file |
| `1` | One time series along the offshore boundary |
| `2` | Two time series, placed according to `paulrevere` |
| `4` | One time series at each corner |

With `tideloc=2`, [`paulrevere`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.paulrevere] chooses where the two signals go: `land` (the XBeach default) applies one to the offshore corners and one to the landward corners, `sea` one to each offshore corner.

### Runs without tide forcing

With `tideloc` greater than 0, XBeach reads the water levels from a `zs0file`. XBeach 1.24 defaults to `tideloc=2` unless `zs0` is set in `params.txt`, so a run without water level forcing should set `tideloc=0` and the constant level [`zs0`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.zs0] explicitly:

```python exec="on" source="above" result="text" session="boundaries"
from rompy_xbeach.components.boundary.parameters import TideBoundaryConditions

tide_boundary = TideBoundaryConditions(tideloc=0, zs0=0.3)
print(tide_boundary.get(destdir=None))
```

### How the water level is imposed

[`tidetype`][rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.tidetype] sets how the water level signal enters the model: `velocity` (the XBeach default) through the flow velocities at the boundary, `instant` directly on the water level, or `hybrid`, a combination of both.

```python exec="on" source="above" result="text" session="boundaries"
tide_boundary = TideBoundaryConditions(tidetype="instant")
print(tide_boundary.get(destdir=None))
```

## With water level forcing

The water level and tide interfaces in `input.tide` write the `zs0file` themselves, together with `tideloc`, which they only support as 1 (one signal along the offshore boundary), and `tidelen`. See [Tide and water levels](water-levels.md).

`Config` adds the `tide_boundary` parameters after those of `input.tide`, so a `tideloc` or `zs0` set in `tide_boundary` replaces the value written by the forcing. With water level forcing, leave `tideloc` unset in `tide_boundary` and use it only for settings such as `tidetype`.

## In YAML

Both components are plain mappings under `flow_boundary` and `tide_boundary` in the model config:

```python exec="on" source="above" result="text" session="boundaries"
import yaml

config = yaml.safe_load(
    """
    flow_boundary:
      front: abs_2d
      back: wall
      lateralwave: wavecrest
    tide_boundary:
      tideloc: 0
      zs0: 0.3
    """
)
print(FlowBoundaryConditions(**config["flow_boundary"]).get(destdir=None))
print(TideBoundaryConditions(**config["tide_boundary"]).get(destdir=None))
```

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["boundary-conditions", "tides"]))
```
