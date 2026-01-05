"""XBeach sediment transport parameter configurations.

This module contains models for sediment transport formulations and related parameters.
"""

from typing import Literal, Optional

import logging

from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

from rompy_xbeach.types import XBeachBaseModel


class SedimentTransport(XBeachBaseModel):
    """Sediment transport parameters (XBeach Table 36).

    This class consolidates all sediment transport parameters from XBeach's
    Table 36, including:
    - Sediment transport formulations (form, waveform)
    - Wave asymmetry and skewness effects (facAs, facSk, facua)
    - Bed slope effects (bdslpeff*, facsl, reposeangle)
    - Process switches (sws, lws, lwt, turb, turbadv)
    - Bed and suspended load calibration (bed, sus, bulk)
    - Advanced calibration parameters (Tsmin, tsfac, facDc, etc.)

    All fields default to None, meaning XBeach's default values will be used
    unless explicitly specified.

    References
    ----------
    Roelvink, D., & Reniers, A. (2011). A guide to modeling coastal morphology.
    Soulsby, R. (1997). Dynamics of marine sands.
    Van Rijn, L. C. (1993). Principles of sediment transport in rivers, estuaries
    and coastal seas.
    Van Thiel de Vries, J. S. M. (2009). Dune erosion during storm surges.
    """

    sedtrans: Literal[True] = Field(
        default=True,
        description="Enable sediment transport (always True when using SedimentTransport class)",
    )
    BRfac: Optional[float] = Field(
        default=None,
        description=("Calibration factor for surface slope (XBeach default: 1.0)"),
        ge=0.0,
        le=1.0,
    )
    Tbfac: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for bore interval: tbore = Tbfac * tbore "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    Tsmin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum adaptation time scale in advection-diffusion equation "
            "(XBeach default: 0.5 s)"
        ),
        ge=0.01,
        le=10.0,
    )
    bdslpeffdir: Optional[Literal["none", "talmon"]] = Field(
        default=None,
        description=(
            "Modify the direction of sediment transport based on bed slope "
            "(XBeach default: none)"
        ),
    )
    bdslpeffdirfac: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor in the modification of the direction "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=2.0,
    )
    bdslpeffini: Optional[Literal["none", "total", "bed"]] = Field(
        default=None,
        description=(
            "Modify the critical shields parameter based on bed slope "
            "(XBeach default: none)"
        ),
    )
    bdslpeffmag: Optional[
        Literal[
            "none", "roelvink_total", "roelvink_bed", "soulsby_total", "soulsby_bed"
        ]
    ] = Field(
        default=None,
        description=(
            "Modify the magnitude of sediment transport based on bed slope "
            "(XBeach default: roelvink_total)"
        ),
    )
    bed: Optional[float] = Field(
        default=None,
        description=("Calibration factor for bed transports (XBeach default: 1.0)"),
        ge=0.0,
        le=1.0,
    )
    bermslope: Optional[float] = Field(
        default=None,
        description=(
            "Swash zone slope for (semi-)reflective beaches (XBeach default: 0.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    betad: Optional[float] = Field(
        default=None,
        description=(
            "Dissipation parameter for long wave breaking turbulence "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=10.0,
    )
    bulk: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to compute bulk transport rather than bed and suspended "
            "load separately (XBeach default: 0)"
        ),
    )
    ci: Optional[float] = Field(
        default=None,
        description=("Mass coefficient in shields inertia term (XBeach default: 1.0)"),
        ge=0.5,
        le=1.5,
    )
    cm: Optional[float] = Field(
        default=None,
        description=("Mass coefficient in shields inertia term (XBeach default: 1.5)"),
        ge=0.0,
        le=3.0,
    )
    dilatancy: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to reduce critical shields number due to dilatancy "
            "(XBeach default: 0)"
        ),
    )
    facAs: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for time-averaged flows due to wave asymmetry "
            "(XBeach default: 0.2)"
        ),
        ge=0.0,
        le=1.0,
    )
    facDc: Optional[float] = Field(
        default=None,
        description=(
            "Option to control sediment diffusion coefficient (XBeach default: 1.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    facSk: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for time-averaged flows due to wave skewness "
            "(XBeach default: 0.15)"
        ),
        ge=0.0,
        le=1.0,
    )
    facsl: Optional[float] = Field(
        default=None,
        description=("Factor for bed slope effect (XBeach default: 1.6)"),
        ge=0.0,
        le=1.6,
    )
    facua: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for time-averaged flows due to wave skewness "
            "and asymmetry (alias for setting both facAs and facSk) "
            "(XBeach default: 0.1)"
        ),
        ge=0.0,
        le=1.0,
    )
    fallvelred: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to reduce fall velocity for high concentrations (XBeach default: 0)"
        ),
    )
    form: Optional[Literal["soulsby_vanrijn", "vanthiel_vanrijn", "vanrijn1993"]] = (
        Field(
            default=None,
            description=(
                "Equilibrium sediment concentration formulation "
                "(XBeach default: vanthiel_vanrijn)"
            ),
        )
    )
    jetfac: Optional[float] = Field(
        default=None,
        description=(
            "Option to mimic turbulence production near revetments "
            "(XBeach default: 0.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    lws: Optional[bool] = Field(
        default=None,
        description=("Switch to enable long wave stirring (XBeach default: 1)"),
    )
    lwt: Optional[bool] = Field(
        default=None,
        description=("Switch to enable long wave turbulence (XBeach default: 0)"),
    )
    phit: Optional[float] = Field(
        default=None,
        description=(
            "Phase lag angle in Nielsen transport equation "
            "(XBeach default: 25.0 degrees)"
        ),
        ge=0.0,
        le=90.0,
    )
    pormax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum porosity used in the expression of Van Rhee (XBeach default: 0.5)"
        ),
        ge=0.3,
        le=0.6,
    )
    reposeangle: Optional[float] = Field(
        default=None,
        description=("Angle of internal friction (XBeach default: 30.0 degrees)"),
        ge=0.0,
        le=45.0,
    )
    rheeA: Optional[float] = Field(
        default=None,
        description=("A parameter in the Van Rhee expression (XBeach default: 0.75)"),
        ge=0.75,
        le=2.0,
    )
    smax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum shields parameter for equilibrium sediment concentration "
            "according to Diane Foster (XBeach default: -1.0, no limit)"
        ),
        ge=-1.0,
        le=3.0,
    )
    sus: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for suspension transports (XBeach default: 1.0)"
        ),
        ge=0.0,
    )
    sws: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable short wave and roller stirring and undertow "
            "(XBeach default: 1)"
        ),
    )
    tsfac: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient determining ts = tsfac * h/ws in sediment source term "
            "(XBeach default: 0.1)"
        ),
        ge=0.01,
        le=1.0,
    )
    turb: Optional[Literal["none", "wave_averaged", "bore_averaged"]] = Field(
        default=None,
        description=(
            "Switch to include short wave turbulence (XBeach default: wave_averaged)"
        ),
    )
    turbadv: Optional[Literal["none", "lagrangian", "eulerian"]] = Field(
        default=None,
        description=(
            "Switch to activate turbulence advection model (XBeach default: none)"
        ),
    )
    waveform: Optional[Literal["ruessink_vanrijn", "vanthiel"]] = Field(
        default=None,
        description=(
            "Wave shape model for asymmetry and skewness (XBeach default: vanthiel)"
        ),
    )
    z0: Optional[float] = Field(
        default=None,
        description=(
            "Zero flow velocity level in Soulsby and Van Rijn (1997) "
            "sediment concentration (XBeach default: 0.006 m)"
        ),
        ge=0.0001,
        le=0.05,
    )

    @model_validator(mode="after")
    def no_bore_averaged_with_ruessink_vanrijn(self) -> "SedimentTransport":
        """Bore-averaged turbulence cannot be used with ruessink_vanrijn waveform.

        The Ruessink et al. (2012) formulation does not determine an exact wave shape,
        so the bore interval cannot be calculated. Bore-averaged short-wave turbulence
        requires the bore interval to be computed from the wave shape.

        See: https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#wave-shape
        """
        if self.turb == "bore_averaged" and self.waveform == "ruessink_vanrijn":
            logger.warning(
                "Bore-averaged turbulence (turb='bore_averaged') cannot be combined with "
                "the Ruessink et al. (2012) wave form (waveform='ruessink_vanrijn'). "
                "The Ruessink formulation does not determine an exact wave shape, which is "
                "required to calculate the bore interval for bore-averaged turbulence. "
                "Consider using waveform='vanthiel' or turb='wave_averaged' instead."
            )
        return self


class TransportNumerics(XBeachBaseModel):
    """Sediment transport numerical parameters.

    These parameters control numerical aspects of sediment transport calculations,
    including maximum concentration limits and numerical scheme selection.
    """

    cmax: Optional[float] = Field(
        default=None,
        description=("Maximum allowed sediment concentration (XBeach default: 0.1)"),
        ge=0.0,
        le=1.0,
    )
    dtlimts: Optional[float] = Field(
        default=None,
        description=(
            "Factor of the timestep to determine the numerical limiter of the "
            "adaptation time (XBeach default: 1.0)"
        ),
        ge=0.0,
        le=20.0,
    )
    oldTsmin: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to apply the Tsmin parameter instead of dtlimts (XBeach default: 0)"
        ),
    )
    sourcesink: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable source-sink terms to calculate bed level change "
            "rather than suspended transport gradients (XBeach default: 0)"
        ),
    )
    thetanum: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient determining whether upwind (1) or central scheme (0.5) "
            "is used (XBeach default: 1.0)"
        ),
        ge=0.5,
        le=1.0,
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
    q3d: Literal[True] = Field(
        default=True,
        description="Turn on quasi-3D sediment transport (XBeach default: 0)",
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
    kmax: Optional[int] = Field(
        default=None,
        description=(
            "Number of sigma layers in quasi-3D model. kmax=1 means no vertical "
            "structure of flow and suspensions (XBeach default: 100)"
        ),
        ge=1,
        le=1000,
    )
    rwave: Optional[float] = Field(
        default=None,
        description=(
            "User-defined wave roughness adjustment factor (XBeach default: 2.0)"
        ),
        ge=0.1,
        le=10.0,
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
    vicmol: Optional[float] = Field(
        default=None,
        description=("Molecular viscosity (XBeach default: 1e-06 m²/s)"),
        ge=0.0,
        le=0.001,
    )
    vonkar: Optional[float] = Field(
        default=None,
        description=("Von Karman constant (XBeach default: 0.4)"),
        ge=0.01,
        le=1.0,
    )
