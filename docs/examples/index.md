# Examples

Interactive Jupyter notebooks demonstrating rompy-xbeach features.

## Getting Started

Start here to learn the full rompy-xbeach workflow:

| Notebook | Description |
|----------|-------------|
| [Procedural Workflow](example-procedural.ipynb) | Step-by-step model setup using Python code |
| [Declarative Workflow](example-declarative.ipynb) | Model setup using YAML configuration |

These notebooks provide a complete overview of setting up an XBeach model, from grid
definition through to generating the model workspace.

---

## Data Interface Tutorials

Learn how to prepare input data for XBeach models:

| Notebook | Description |
|----------|-------------|
| [Grid](data-interfaces/tutorial-grid.ipynb) | Define computational grids with various configurations |
| [Sources](data-interfaces/tutorial-source.ipynb) | Load data from NetCDF, GeoTIFF, and other formats |
| [Bathymetry](data-interfaces/tutorial-bathy.ipynb) | Interpolate and extend bathymetry onto the model grid |
| [Wave Boundary](data-interfaces/tutorial-wave-boundary.ipynb) | Generate wave boundary conditions from spectral or parametric data |
| [Forcing](data-interfaces/tutorial-forcing.ipynb) | Set up wind and tide forcing from geolocated sources |
| [Timeseries Forcing](data-interfaces/tutorial-timeseries-forcing.ipynb) | Use CSV/DataFrame timeseries for forcing |

---

## Component Tutorials

Configure XBeach physics and output parameters:

| Notebook | Description |
|----------|-------------|
| [Physics](components/tutorial_01_physics.ipynb) | Wave breaking, friction, and flow numerics |
| [Sediment](components/tutorial_02_sediment.ipynb) | Sediment transport and morphology |
| [Output](components/tutorial_03_output.ipynb) | Output variables and timing |
| [Boundary Conditions](components/tutorial_04_boundary-conditions.ipynb) | Flow boundary condition types |
| [Hotstart](components/tutorial_05_hotstart.ipynb) | Restart simulations from saved state |
| [MPI](components/tutorial_06_mpi.ipynb) | Parallel execution configuration |

---

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
