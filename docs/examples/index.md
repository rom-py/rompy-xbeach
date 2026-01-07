# Examples

Interactive Jupyter notebooks demonstrating rompy-xbeach features.

!!! note "Notebook Status"
    Some notebooks may need updates to reflect recent component changes. 
    They are provided for reference and will be updated progressively.

## Available Examples

### Core Concepts

| Notebook | Description |
|----------|-------------|
| [Grid](grid-demo.ipynb) | Grid creation and configuration |
| [Bathymetry](bathy-demo.ipynb) | Bathymetry data handling |
| [Wave Boundary](wave-boundary-demo.ipynb) | Wave boundary condition setup |

### Components

| Notebook | Description |
|----------|-------------|
| [Physics](physics-demo.ipynb) | Physics component configuration |
| [Output](output-demo.ipynb) | Output configuration |
| [Forcing](forcing-demo.ipynb) | Wind and tide forcing |

### Complete Workflows

| Notebook | Description |
|----------|-------------|
| [Procedural Example](example_procedural.ipynb) | Full model setup using Python objects |
| [Declarative Example](example_declarative.ipynb) | Full model setup using YAML configuration |

## Running Notebooks Locally

To run these notebooks yourself:

```bash
# Clone the notebooks repository
git clone -b xbeach https://github.com/rom-py/rompy-notebooks.git

# Navigate to XBeach notebooks
cd rompy-notebooks/notebooks/xbeach

# Start Jupyter
jupyter lab
```

## Source Repository

These notebooks are maintained in a separate repository to keep rompy-xbeach lightweight:

- **GitHub**: [rom-py/rompy-notebooks](https://github.com/rom-py/rompy-notebooks/tree/xbeach/notebooks/xbeach) (xbeach branch)
