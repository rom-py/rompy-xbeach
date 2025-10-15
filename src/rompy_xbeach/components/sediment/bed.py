"""XBeach bed composition parameter configurations.

This module contains models for bed composition, layering, and bed update parameters.
"""

from typing import Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class BedComposition(XBeachBaseModel):
    """Bed composition and layering parameters.
    
    Controls the vertical structure of the bed when using multiple sediment
    fractions and bed layers. Parameters determine when layers split or merge
    based on their thickness relative to nominal values.
    
    The variable thickness layer (specified by nd_var) can grow or shrink
    during the simulation to accommodate sediment deposition or erosion.
    """
    
    frac_dz: Optional[float] = Field(
        default=None,
        description=(
            "Relative thickness to split time step for bed updating "
            "(XBeach default: 0.7)"
        ),
        ge=0.5,
        le=0.98,
    )
    split: Optional[float] = Field(
        default=None,
        description=(
            "Split threshold for variable sediment layer as ratio to nominal "
            "thickness. When layer exceeds split * nominal thickness, it splits "
            "(XBeach default: 1.01)"
        ),
        ge=1.005,
        le=1.1,
    )
    merge: Optional[float] = Field(
        default=None,
        description=(
            "Merge threshold for variable sediment layer as ratio to nominal "
            "thickness. When layer falls below merge * nominal thickness, it merges "
            "(XBeach default: 0.01)"
        ),
        ge=0.005,
        le=0.1,
    )
    nd_var: Optional[int] = Field(
        default=None,
        description=(
            "Index of layer with variable thickness (1-indexed) "
            "(XBeach default: 2)"
        ),
        ge=1,
    )


class Quasi3D(XBeachBaseModel):
    """Quasi-3D sediment transport parameters.
    
    Controls the vertical structure of flow and sediment concentration when
    quasi-3D mode is enabled (q3d=1). The vertical domain is divided into
    sigma layers with specified distribution.
    
    The quasi-3D model resolves vertical variations in horizontal velocity
    and sediment concentration, improving accuracy for situations with strong
    vertical gradients.
    
    References
    ----------
    Van Thiel de Vries, J. S. M. (2009). Dune erosion during storm surges.
    PhD thesis, Delft University of Technology.
    """
    
    kmax: Optional[int] = Field(
        default=None,
        description=(
            "Number of sigma layers in quasi-3D model. kmax=1 means no vertical "
            "structure of flow and suspensions (XBeach default: 100)"
        ),
        ge=1,
        le=1000,
    )
    sigfac: Optional[float] = Field(
        default=None,
        description=(
            "Dsig scales with log(sigfac). Controls vertical layer distribution "
            "(XBeach default: 1.3)"
        ),
        ge=0.0,
        le=10.0,
    )
    deltar: Optional[float] = Field(
        default=None,
        description=(
            "Estimated ripple height for roughness calculations "
            "(XBeach default: 0.025 m)"
        ),
        ge=0.001,
        le=1.0,
    )
    rwave: Optional[float] = Field(
        default=None,
        description=(
            "User-defined wave roughness adjustment factor "
            "(XBeach default: 2.0)"
        ),
        ge=0.1,
        le=10.0,
    )
    vonkar: Optional[float] = Field(
        default=None,
        description=(
            "Von Karman constant "
            "(XBeach default: 0.4)"
        ),
        ge=0.01,
        le=1.0,
    )
    vicmol: Optional[float] = Field(
        default=None,
        description=(
            "Molecular viscosity "
            "(XBeach default: 1e-06 m²/s)"
        ),
        ge=0.0,
        le=0.001,
    )
