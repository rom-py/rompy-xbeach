# Contributing

Contributions are welcome: bug reports and ideas as [GitHub issues](https://github.com/rom-py/rompy-xbeach/issues), and changes as pull requests.

## Development setup

```bash
git clone https://github.com/rom-py/rompy-xbeach.git
cd rompy-xbeach
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,extra]"
```

## Tests and style

```bash
pytest                   # the test suite
ruff check src tests     # lint: the rules in ruff.toml are checked in CI
ruff format src tests    # format
```

Test fixtures live in `tests/data/`. Test each component's parameters, each validator in both directions (accepted and rejected), and the files a data interface writes.

## Adding a parameter or component

rompy-xbeach components map directly to XBeach parameters (see [How rompy-xbeach works](../user-guide/how-it-works.md)). When adding one:

- **Put it where XBeach groups it**: physics, sediment, output, boundaries, and so on.
- **Name the field after the XBeach parameter.** If the parameter is not a valid Python name, set it as the field's `alias` (e.g. `breaktype` with `alias="break"`).
- **Default to `None`** so that XBeach's own default applies, and give the XBeach default in the description: `description="... (XBeach default: 0.8)"`. Use `ge`, `le`, `gt`, `lt` and `Literal` for the valid values.
- **Use a discriminated union for a choice between variants.** Each variant has a `model_type` literal equal to the value XBeach expects; the union field writes it (e.g. `wavemodel = surfbeat`).
- **Use `bool | Component`** for a process that can be switched on with defaults or configured.
- **Use [`XBeachDataBlob`][rompy_xbeach.types.XBeachDataBlob] for a file.** `XBeachBaseModel.get()` copies it into the workspace and writes the file name under the field name. Components do not override `get()` to handle files.
- **Check combinations in `Config`, do not rewrite them.** A setting that depends on another component is checked by a validator on [`Config`][rompy_xbeach.config.Config], which raises an error saying what to use instead.

```python
from typing import Optional

from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class MyComponent(XBeachBaseModel):
    """What the component configures, in XBeach terms."""

    my_param: Optional[float] = Field(
        default=None,
        description="What it does (XBeach default: 1.0)",
        ge=0.0,
        le=10.0,
    )
```

The [parameter index](../reference/parameters.md) and the reference pick up new fields automatically.

## Documentation

The docs are built with [ProperDocs](https://github.com/ProperDocs/properdocs) (a maintained fork of MkDocs 1.6), Material and mkdocstrings, using the shared configuration from [rompy-docs](https://github.com/rom-py/rompy-docs):

```bash
pip install -e ".[docs,extra]"
properdocs serve -f mkdocs.yml              # live preview
properdocs build --strict -f mkdocs.yml     # what CI runs
rompy-docs check -f mkdocs.yml              # compare with the shared configuration
```

- **The reference is generated** from the docstrings and field descriptions. Write docstrings in numpy style.
- **Examples in docstrings and pages run when the docs are built.** Write them as markdown-exec blocks; a failing example fails the strict build. Use a plain `python` block only for examples that need a model run or remote data:

    ````markdown
    ```python exec="on" source="above" result="text" session="physics"
    from rompy_xbeach.components.physics import Physics
    from rompy_xbeach.components.physics.wavemodel import Surfbeat

    print(Physics(wavemodel=Surfbeat()).get("."))
    ```
    ````

- **Link classes** with `` [`Physics`][rompy_xbeach.components.physics.physics.Physics] ``. rompy classes resolve to the rompy site.
- **Notebooks** live in [rompy-notebooks](https://github.com/rom-py/rompy-notebooks). The Tutorial and Examples pages list them from the inventory that site publishes.
- To build against an unpublished rompy site, set `ROMPY_INVENTORY` to its `objects.inv` (a URL, or `file://` path).

## Pull requests

1. Create a branch from `output`.
2. Make the change, with tests and docs.
3. Run the tests, `ruff check` and the strict docs build.
4. Add the change to the [changelog](changelog.md).
5. Open a pull request.
