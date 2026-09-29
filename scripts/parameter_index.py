"""Generate the XBeach parameter index for the docs (run by markdown-exec).

Every XBeach parameter that rompy-xbeach can write, with the class and field that set
it. Components name their fields after XBeach parameters, so the index is built from
the component classes, the wave boundary parameter classes and, for the parameters
rompy-xbeach derives from the grid and data interfaces, from an example workspace.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import tempfile
import typing
from pathlib import Path

from pydantic import BaseModel

import rompy_xbeach.components
from rompy_xbeach.data.boundary import base as boundary_base
from rompy_xbeach.data.boundary import nonspectral
from rompy_xbeach.types import XBeachBaseModel

# Fields that hold settings of rompy-xbeach itself, not XBeach parameters.
NOT_PARAMETERS = {"model_type", "id"}

# Wave boundary classes whose own fields are written to params.txt.
BOUNDARY_CLASSES = [
    boundary_base.WaveBoundaryParams,
    boundary_base.SpectralWaveBoundaryParams,
    nonspectral.BoundaryParams,
]


def _classes(annotation) -> list[type]:
    """The classes in a field annotation (unions, Optional and Annotated unpacked)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _classes(typing.get_args(annotation)[0])
    if origin is not None:
        return [c for arg in typing.get_args(annotation) for c in _classes(arg)]
    return [annotation] if isinstance(annotation, type) else []


def _is_component(annotation) -> bool:
    """True if the field only holds components (no value XBeach reads directly)."""
    classes = [c for c in _classes(annotation) if c is not type(None)]
    return bool(classes) and all(issubclass(c, BaseModel) for c in classes)


def _component_classes() -> list[type]:
    classes = {}
    for info in pkgutil.walk_packages(rompy_xbeach.components.__path__, "rompy_xbeach.components."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, XBeachBaseModel)
                and obj.__module__ == module.__name__
            ):
                classes[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(classes.values())


def _description(field) -> str:
    text = field.description or ""
    return re.sub(r"\s+", " ", text).replace("|", "\\|").strip()


def _collect() -> dict[str, list[tuple[str, str, str]]]:
    """Map each parameter to (class path, field, description) entries."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    for cls in _component_classes() + BOUNDARY_CLASSES:
        own = set(getattr(cls, "__annotations__", {}))
        for name, field in cls.model_fields.items():
            if name in NOT_PARAMETERS or name not in own:
                continue
            description = _description(field)
            if _is_component(field.annotation):
                # A discriminated union writes the chosen variant: `wavemodel = surfbeat`
                if not field.discriminator:
                    continue
                values = [
                    c.model_fields["model_type"].default
                    for c in _classes(field.annotation)
                    if c is not type(None) and "model_type" in c.model_fields
                ]
                description = f"{description} One of: {', '.join(f'`{v}`' for v in values)}."
            path = f"{cls.__module__}.{cls.__qualname__}"
            # The parameter is the alias when the XBeach name is not a valid field name
            index.setdefault(field.alias or name, []).append((path, name, description))
    return index


def _example_params() -> dict[str, str]:
    """Parameters written in an example workspace (grid, bathymetry, waves, time)."""
    from rompy.core.time import TimeRange
    from rompy.model import ModelRun

    from rompy_xbeach.components.boundary.parameters import TideBoundaryConditions
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
            alfa=347.0, dx=20.0, dy=30.0, nx=115, ny=110, crs="EPSG:28350",
        ),
        bathy=XBeachBathy(
            source=SourceGeotiff(filename="tests/data/bathy.tif"),
            posdwn=False,
            extension=SeawardExtensionLinear(depth=15.0, slope=0.05),
        ),
        input=DataInterface(
            wave=BoundaryParams(Hrms=1.0, Trep=10.0, dir0=270.0, thetamin=-90.0, thetamax=90.0, dtheta=15.0)
        ),
        physics=Physics(wavemodel=Stationary()),
        tide_boundary=TideBoundaryConditions(tideloc=0, zs0=0.0),
    )
    with tempfile.TemporaryDirectory() as tmp:
        run = ModelRun(
            run_id="index",
            period=TimeRange(start="2023-01-01T00:00", end="2023-01-01T00:30", interval="10m"),
            output_dir=tmp,
            config=config,
        )
        params = (Path(run()) / "params.txt").read_text()
    out = {}
    for line in params.splitlines():
        if "=" in line and not line.lstrip().startswith("%"):
            key, value = (s.strip() for s in line.split("=", 1))
            out[key] = value
    return out


def parameter_table() -> str:
    """Markdown table of parameters set through component and boundary fields."""
    index = _collect()
    lines = ["| Parameter | Set with | Description |", "|---|---|---|"]
    for name in sorted(index, key=str.lower):
        entries = index[name]
        where = "<br>".join(
            f"[`{path.rsplit('.', 1)[1]}.{field}`][{path}.{field}]" for path, field, _ in entries
        )
        descriptions = {d for _, _, d in entries if d}
        lines.append(f"| `{name}` | {where} | {' / '.join(sorted(descriptions))} |")
    return "\n".join(lines)


def derived_table() -> str:
    """Markdown table of parameters rompy-xbeach writes from the run, grid and data."""
    index = _collect()
    params = _example_params()
    lines = ["| Parameter | Example value |", "|---|---|"]
    for name in sorted(params, key=str.lower):
        if name not in index:
            lines.append(f"| `{name}` | `{params[name]}` |")
    return "\n".join(lines)
