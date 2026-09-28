"""XBeach wave-current interaction parameter configurations.

This module contains models for wave-current interaction parameters.

"""

import logging
from typing import Optional, Literal
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


class WaveCurrentInteraction(XBeachBaseModel):
    """Wave-current interaction parameters.

    Wave-current interaction (WCI) accounts for the feedback of currents on wave
    propagation. When enabled, currents affect the wave celerity and direction
    through Doppler shifting.

    The WCI computation is limited to specific depth ranges (hwci to hwcimax) to
    avoid numerical issues in very shallow or deep water.

    References
    ----------
    Dingemans et al. (1987). Water wave propagation over uneven bottoms.

    """

    wci: Literal[True] = Field(
        default=True,
        description="Enable wave-current interaction",
    )
    cats: Optional[float] = Field(
        default=None,
        description=(
            "Current averaging time scale for wave-current interaction, "
            "in terms of mean wave periods (XBeach default: 4.0)"
        ),
        ge=1.0,
        le=50.0,
    )
    hwci: Optional[float] = Field(
        default=None,
        description=(
            "Minimum depth until which wave-current interaction is used "
            "(XBeach default: 0.1 m)"
        ),
        ge=0.001,
        le=1.0,
    )
    hwcimax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum depth until which wave-current interaction is used "
            "(XBeach default: 100.0 m)"
        ),
        ge=0.01,
        le=100.0,
    )
