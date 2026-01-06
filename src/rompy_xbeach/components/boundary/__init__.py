"""Boundary condition components.

This subpackage contains all boundary condition related code:
- parameters.py: Boundary parameter classes (Flow, Tide, Wave boundary conditions)
- specification.py: Wave boundary specification classes (WaveBoundary hierarchy)
"""

from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
    WaveBoundaryConditions,
    SpectralWaveBoundaryConditions,
    NonSpectralWaveBoundaryConditions,
)
from rompy_xbeach.components.boundary.specification import (
    SpectralWaveBoundary,
    NonSpectralWaveBoundary,
    OffWaveBoundary,
    ReuseWaveBoundary,
)

__all__ = [
    # Parameter classes
    "FlowBoundaryConditions",
    "TideBoundaryConditions",
    "WaveBoundaryConditions",
    "SpectralWaveBoundaryConditions",
    "NonSpectralWaveBoundaryConditions",
    # Specification classes
    "SpectralWaveBoundary",
    "NonSpectralWaveBoundary",
    "OffWaveBoundary",
    "ReuseWaveBoundary",
]
