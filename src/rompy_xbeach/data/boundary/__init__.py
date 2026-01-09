"""XBeach wave boundary condition classes.

This package provides wave boundary condition classes for XBeach simulations.
All boundary classes follow a consistent interface with a `get()` method that
returns a dictionary of XBeach parameters.

Spectral Boundaries (from external data):
- BoundaryStationSpectraJons: JONSWAP from station spectra
- BoundaryStationParamJons: JONSWAP from station parameters
- BoundaryPointParamJons: JONSWAP from point parameters
- BoundaryGridParamJons: JONSWAP from gridded parameters
- BoundaryStationSpectraJonstable: JONSTABLE from station spectra
- BoundaryStationParamJonstable: JONSTABLE from station parameters
- BoundaryPointParamJonstable: JONSTABLE from point parameters
- BoundaryGridParamJonstable: JONSTABLE from gridded parameters
- BoundaryStationSpectraSwan: SWAN from station spectra

Non-Spectral Boundaries:
- BoundaryStat: Stationary parametric waves (no file needed)
- BoundaryBichrom: Bichromatic waves (no file needed)
- BoundaryStatTable: Time-varying parametric from file
- BoundaryTs1: Time series at single location from file
- BoundaryTs2: Time series at two locations from file
- BoundaryTsNonh: Non-hydrostatic time series from file

Special Boundaries:
- BoundaryOff: No wave forcing
- BoundaryReuse: Reuse previous simulation files
"""

from rompy_xbeach.data.boundary.base import (
    WaveBoundaryParams,
    SpectralWaveBoundaryParams,
    NonSpectralWaveBoundaryParams,
    BoundaryBase,
    BoundaryBaseGrid,
    BoundaryBaseStation,
    BoundaryBasePoint,
    SpectraMixin,
    ParamMixin,
    FilelistMixin,
    dspr_to_s,
    s_to_dspr,
)

from rompy_xbeach.data.boundary.spectral import (
    BoundaryStationSpectraJons,
    BoundaryStationParamJons,
    BoundaryPointParamJons,
    BoundaryGridParamJons,
    BoundaryStationSpectraJonstable,
    BoundaryStationParamJonstable,
    BoundaryPointParamJonstable,
    BoundaryGridParamJonstable,
    BoundaryStationSpectraSwan,
)

from rompy_xbeach.data.boundary.nonspectral import (
    BoundaryStat,
    BoundaryBichrom,
    BoundaryStatTable,
    BoundaryTs1,
    BoundaryTs2,
    BoundaryTsNonh,
)

from rompy_xbeach.data.boundary.special import (
    BoundaryOff,
    BoundaryReuse,
)


__all__ = [
    # Base classes and utilities
    "WaveBoundaryParams",
    "SpectralWaveBoundaryParams",
    "NonSpectralWaveBoundaryParams",
    "BoundaryBase",
    "BoundaryBaseGrid",
    "BoundaryBaseStation",
    "BoundaryBasePoint",
    "SpectraMixin",
    "ParamMixin",
    "FilelistMixin",
    "dspr_to_s",
    "s_to_dspr",
    # JONS classes
    "BoundaryStationSpectraJons",
    "BoundaryStationParamJons",
    "BoundaryPointParamJons",
    "BoundaryGridParamJons",
    # JONSTABLE classes
    "BoundaryStationSpectraJonstable",
    "BoundaryStationParamJonstable",
    "BoundaryPointParamJonstable",
    "BoundaryGridParamJonstable",
    # SWAN classes
    "BoundaryStationSpectraSwan",
    # Non-spectral classes
    "BoundaryStat",
    "BoundaryBichrom",
    "BoundaryStatTable",
    "BoundaryTs1",
    "BoundaryTs2",
    "BoundaryTsNonh",
    # Special classes
    "BoundaryOff",
    "BoundaryReuse",
]
