"""Boundary condition components.

This subpackage contains all boundary condition related code:
- parameters.py: Boundary parameter classes (WaveBoundaryConditions, FlowBoundaryConditions)
- specification.py: Wave boundary specification classes (WaveBoundary hierarchy)
"""

from rompy_xbeach.components.boundary.parameters import (
    WaveBoundaryConditions,
    SpectralWaveBoundaryConditions,
    NonSpectralWaveBoundaryConditions,
    FlowBoundaryConditions,
)
from rompy_xbeach.components.boundary.specification import (
    SpectralWaveBoundary,
    NonSpectralWaveBoundary,
    OffWaveBoundary,
    ReuseWaveBoundary,
)

__all__ = [
    # Parameter classes
    "WaveBoundaryConditions",
    "SpectralWaveBoundaryConditions",
    "NonSpectralWaveBoundaryConditions",
    "FlowBoundaryConditions",
    # Specification classes
    "SpectralWaveBoundary",
    "NonSpectralWaveBoundary",
    "OffWaveBoundary",
    "ReuseWaveBoundary",
]
