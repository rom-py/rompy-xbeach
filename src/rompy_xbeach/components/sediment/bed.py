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
            "Index of layer with variable thickness (1-indexed) (XBeach default: 2)"
        ),
        ge=1,
    )
