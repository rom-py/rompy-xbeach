"""Flow and tide boundary condition components.

This subpackage contains flow and tide boundary condition parameters.
Wave boundary conditions are now in data/boundary/ subpackage.
"""

from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
)

__all__ = [
    "FlowBoundaryConditions",
    "TideBoundaryConditions",
]
