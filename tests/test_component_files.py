"""Every file field of a component is fetched into the workspace by get()."""

import importlib
import pkgutil

import pytest

import rompy_xbeach
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.friction import Manning
from rompy_xbeach.components.physics.wavemodel import (
    Roelvink2,
    ShortWaveFriction,
    Surfbeat,
)
from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob

# Import every module so all component classes are registered as subclasses
for module in pkgutil.walk_packages(rompy_xbeach.__path__, "rompy_xbeach."):
    importlib.import_module(module.name)


def subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from subclasses(subclass)


def file_fields(cls):
    return [
        name
        for name, field in cls.model_fields.items()
        if "XBeachDataBlob" in str(field.annotation)
    ]


# Extra values some components need alongside a file field to be valid
REQUIRED = {"BedUpdate": {"setbathyfile": {"nsetbathy": 1}}}

CASES = sorted(
    {
        (cls.__name__, cls, field)
        for cls in subclasses(XBeachBaseModel)
        for field in file_fields(cls)
    },
    key=lambda case: (case[0], case[2]),
)


def test_components_with_file_fields_found():
    """The discovery below covers the known components with file fields."""
    names = {name for name, _, _ in CASES}
    assert {"Manning", "Vegetation", "Morphology", "Output"} <= names


@pytest.mark.parametrize(
    "cls,field",
    [(cls, field) for _, cls, field in CASES],
    ids=[f"{n}.{f}" for n, _, f in CASES],
)
def test_file_field_is_fetched(tmp_path, cls, field):
    """get() copies the file into destdir and writes its name under the field name."""
    source = tmp_path / "source" / f"{field}.txt"
    source.parent.mkdir()
    source.write_text("data")
    destdir = tmp_path / "run"
    destdir.mkdir()

    extra = REQUIRED.get(cls.__name__, {}).get(field, {})
    component = cls(**{field: XBeachDataBlob(source=source)}, **extra)
    params = component.get(destdir)
    assert params[field] == source.name
    assert (destdir / source.name).is_file()


def test_nested_file_fields_are_fetched(tmp_path):
    """Files deep inside discriminated components reach the workspace."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "friction.txt").write_text("data")
    (source / "wavfric.txt").write_text("data")
    destdir = tmp_path / "run"
    destdir.mkdir()

    physics = Physics(
        wavemodel=Surfbeat(
            breaktype=Roelvink2(
                wavfric=ShortWaveFriction(
                    fwfile=XBeachDataBlob(source=source / "wavfric.txt")
                )
            )
        ),
        bedfriction=Manning(bedfricfile=XBeachDataBlob(source=source / "friction.txt")),
    )
    params = physics.get(destdir)
    assert params["bedfricfile"] == "friction.txt"
    assert params["fwfile"] == "wavfric.txt"
    assert sorted(p.name for p in destdir.iterdir()) == ["friction.txt", "wavfric.txt"]
