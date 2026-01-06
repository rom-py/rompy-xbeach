# Contributing

Thank you for your interest in contributing to rompy-xbeach!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/rom-py/rompy-xbeach.git
cd rompy-xbeach
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

4. Run tests:

```bash
pytest tests/
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_physics.py -v
pytest tests/test_config.py::test_specific_function -v
```

## Adding New Components

When adding new XBeach parameters or components:

1. **Find the right location** — Use the component hierarchy (physics, sediment, output, etc.)

2. **Follow existing patterns** — Look at similar components for structure

3. **Add validation** — Use Pydantic's `Field` with `ge`, `le`, `gt`, `lt` for ranges

4. **Document fields** — Every field needs a `description` with XBeach default noted

5. **Add tests** — Test validation, serialisation, and integration

### Example: Adding a New Parameter

```python
from typing import Optional
from pydantic import Field
from rompy_xbeach.types import XBeachBaseModel

class MyComponent(XBeachBaseModel):
    """Description of the component."""
    
    my_param: Optional[float] = Field(
        default=None,
        description="Description of parameter (XBeach default: 1.0)",
        ge=0.0,  # Greater than or equal
        le=10.0,  # Less than or equal
    )
```

### Example: Adding a Process Switch

For parameters that enable a feature with optional configuration:

```python
from typing import Optional, Union
from pydantic import Field

class Physics(XBeachBaseModel):
    my_feature: Optional[Union[bool, MyFeatureConfig]] = Field(
        default=None,
        description="Enable my feature",
    )
```

## Documentation

Documentation uses MkDocs with Material theme.

### Build locally:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

### Add new pages:

1. Create markdown file in `docs/`
2. Add to `nav` section in `mkdocs.yml`

### API documentation:

Use mkdocstrings directives:

```markdown
::: rompy_xbeach.components.physics.Physics
    options:
      show_root_heading: true
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run linting: `ruff check src/`
6. Commit with clear message
7. Push and create PR

## Questions?

Open an issue on GitHub or reach out to the maintainers.
