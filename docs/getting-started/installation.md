# Installation

## Requirements

- Python 3.9 or later
- [rompy](https://github.com/rom-py/rompy) core library
- XBeach executable (for running simulations)

## Install from PyPI

```bash
pip install rompy-xbeach
```

## Install from Source

For development or the latest features:

```bash
git clone https://github.com/rom-py/rompy-xbeach.git
cd rompy-xbeach
pip install -e ".[dev]"
```

## Verify Installation

```python
import rompy_xbeach
print(rompy_xbeach.__version__)
```

## XBeach Installation

Rompy-xbeach generates XBeach input files but does not include the XBeach executable. You need to install XBeach separately:

- **Pre-built binaries**: Available from [XBeach Release & Source](https://oss.deltares.nl/web/xbeach/release-and-source)
- **Build from source**: See [XBeach documentation](https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#compilation-of-xbeach)

Ensure the `xbeach` executable is in your system PATH, or specify its location when running simulations.

## Optional Dependencies

For additional data source support:

```bash
pip install rompy-xbeach[extra]
```

This includes:

- `rompy-binary-datasources` — Support for binary data formats
