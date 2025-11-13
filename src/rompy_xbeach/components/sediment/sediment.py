"""XBeach sediment and morphology configuration."""

import logging
from typing import Literal, Optional, Union

from pydantic import Field

from rompy_xbeach.components.sediment.bed import BedComposition
from rompy_xbeach.components.sediment.morphology import (
    Avalanching,
    Morphology,
    PrescribedBathymetry,
)
from rompy_xbeach.components.sediment.transport import (
    Quasi3D,
    SedimentTransport,
    TransportNumerics,
)
from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


class Sediment(XBeachBaseModel):
    """XBeach sediment transport and morphology configuration.

    This component groups all sediment-related parameters including:
    - Sediment transport formulations and processes
    - Morphological evolution (morfac, time windows)
    - Avalanching
    - Bed composition and layering
    - Quasi-3D sediment transport
    - Prescribed bathymetry evolution

    All fields default to None, meaning XBeach's default values will be used
    unless explicitly specified.

    Examples
    --------
    >>> from rompy_xbeach.components.sediment import Sediment
    >>> from rompy_xbeach.components.sediment.transport import SedimentTransport
    >>> from rompy_xbeach.components.sediment.morphology import Morphology
    >>>
    >>> sediment = Sediment(
    ...     transport=SedimentTransport(
    ...         form="vanthiel_vanrijn",
    ...         facua=0.15,
    ...         bdslpeffmag="roelvink_total",
    ...     ),
    ...     morphology=Morphology(
    ...         morfac=10.0,
    ...         morstart=0.0,
    ...         morstop=3600.0,
    ...     ),
    ... )

    See Also
    --------
    rompy_xbeach.components.sediment.transport : Sediment transport parameters
    rompy_xbeach.components.sediment.morphology : Morphology parameters
    rompy_xbeach.components.sediment.bed : Bed composition parameters

    References
    ----------
    Roelvink, D., & Reniers, A. (2011). A guide to modeling coastal morphology.
    Advances in Coastal and Ocean Engineering, Vol. 12.
    """

    model_type: Literal["sediment"] = Field(
        default="sediment",
        description="Model type discriminator",
    )

    sedtrans: Optional[Union[bool, SedimentTransport]] = Field(
        default=None,
        description=(
            "Sediment transport parameters from XBeach Table 36, including "
            "formulations, bed slope effects, process switches, and calibration factors"
        ),
    )
    numerics: Optional[TransportNumerics] = Field(
        default=None,
        description=(
            "Sediment transport numerical parameters "
            "(cmax, sourcesink, thetanum, dtlimts, oldTsmin)"
        ),
    )
    morphology: Optional[Union[bool, Morphology]] = Field(
        default=None,
        description=(
            "Morphological evolution parameters "
            "(morfac, morfacopt, morstart, morstop, lsgrad, struct, ne_layer)"
        ),
    )
    avalanching: Optional[Avalanching] = Field(
        default=None,
        description=("Avalanching parameters (dryslp, wetslp, hswitch, dzmax)"),
    )
    prescribed_bathy: Optional[PrescribedBathymetry] = Field(
        default=None,
        description=(
            "Prescribed bathymetry evolution parameters (nsetbathy, setbathyfile)"
        ),
    )

    # Bed composition
    bed_composition: Optional[BedComposition] = Field(
        default=None,
        description=(
            "Bed composition and layering parameters (frac_dz, split, merge, nd_var)"
        ),
    )
    q3d: Optional[Union[bool, Quasi3D]] = Field(
        default=None,
        description="Turn on quasi-3D sediment transport (XBeach default: 0)",
    )
