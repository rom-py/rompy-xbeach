"""XBeach bed friction parameter configurations.

This module contains models for bed friction formulations used to calculate
bed shear stress in the shallow water equations.

"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class FrictionModifiers(XBeachBaseModel):
    """Mixin class for XBeach-G friction modification parameters.

    These parameters apply to all friction formulations and modify the
    bed shear stress calculation through acceleration, infiltration, and
    turbulence effects.
    """

    fwcutoff: Optional[float] = Field(
        default=None,
        description=(
            "Depth greater than which the bed friction factor is not applied "
            "(XBeach default: 1000.0 m)"
        ),
        ge=0.0,
        le=1000.0,
    )
    friction_acceleration: Optional[Literal["none", "mccall", "nielsen"]] = Field(
        default=None,
        description=(
            "Turn on or off the effect of acceleration on bed roughness. "
            "Applies to all friction formulations (XBeach default: none). "
            "Options: none, mccall, nielsen"
        ),
    )
    friction_infiltration: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on or off the effect of infiltration on bed roughness "
            "following Conley and Inman. Applies to all friction formulations "
            "(XBeach default: 0)"
        ),
    )
    friction_turbulence: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on or off the effect of turbulence on bed roughness "
            "following Reniers and van Thiel. Applies to all friction formulations "
            "(XBeach default: 0)"
        ),
    )


class BedFriction(FrictionModifiers):
    """Base class for bed friction formulations with common parameters.

    Most bed friction formulations allow specification of a friction coefficient
    either as a single value or spatially varying via a file. Additional parameters
    control depth cutoffs and XBeach-G specific friction modifications that apply
    to all friction formulations.

    """

    bedfriccoef: Optional[float] = Field(
        default=None,
        description="Bed friction coefficient (XBeach default: 0.01)",
        ge=0.0,
    )
    bedfricfile: Optional[XBeachDataBlob] = Field(
        default=None,
        description=(
            "Bed friction file with spatially varying friction coefficients. "
            "If specified, overrides bedfriccoef."
        ),
    )

    def get(self, destdir: str | Path) -> dict:
        """Fetch external friction file if specified, and return the params dict.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for fetching files.

        Returns
        -------
        dict
            Parameters dictionary with file paths updated to workspace directory.
        """
        # Get base params (XBeachDataBlob fields are automatically excluded by serializer)
        params = super().get(destdir)

        # Fetch DataBlob file and add the fetched file path
        if self.bedfricfile:
            params["bedfricfile"] = self.bedfricfile.get(destdir).name

        return params

    @model_validator(mode="after")
    def check_mutually_exclusive(self):
        if self.bedfriccoef is not None and self.bedfricfile is not None:
            raise ValueError("Only one of bedfriccoef or bedfricfile can be specified.")
        return self


class Cf(BedFriction):
    """Dimensionless friction coefficient formulation.

    Uses a constant dimensionless friction coefficient (c_f) to calculate
    bed shear stress.
    """

    model_type: Literal["cf"] = Field(
        default="cf",
        description="Model type discriminator",
    )


class Chezy(BedFriction):
    """Chezy bed friction formulation.

    Uses the Chezy coefficient (C) to calculate bed shear stress.
    A typical Chezy value is in the order of 55 m^(1/2)/s.
    """

    model_type: Literal["chezy"] = Field(
        default="chezy",
        description="Model type discriminator",
    )


class Manning(BedFriction):
    """Manning bed friction formulation.

    Uses the Manning coefficient (n) to calculate bed shear stress.
    Manning can be seen as a depth-dependent Chezy value.
    A typical Manning value is in the order of 0.02 s/m^(1/3).
    """

    model_type: Literal["manning"] = Field(
        default="manning",
        description="Model type discriminator",
    )
    mincf: Optional[float] = Field(
        default=None,
        description=(
            "Minimum dimensionless friction coefficient for Manning formulation "
            "(XBeach default: 0)"
        ),
        ge=0.0,
        le=1.0,
    )
    maxcf: Optional[float] = Field(
        default=None,
        description=(
            "Maximum dimensionless friction coefficient for Manning formulation "
            "(XBeach default: no limit)"
        ),
        ge=0.0,
    )


class WhiteColebrook(BedFriction):
    """White-Colebrook bed friction formulation.

    Uses the geometrical roughness of Nikuradse (k_s) to calculate bed shear stress.
    The White-Colebrook formulation has a log relation with the water depth.
    A typical k_s value is in the order of 0.01 - 0.15 m.
    """

    model_type: Literal["white-colebrook"] = Field(
        default="white-colebrook",
        description="Model type discriminator",
    )
    mincf: Optional[float] = Field(
        default=None,
        description=(
            "Minimum dimensionless friction coefficient for White-Colebrook formulation "
            "(XBeach default: 0)"
        ),
        ge=0.0,
        le=1.0,
    )
    maxcf: Optional[float] = Field(
        default=None,
        description=(
            "Maximum dimensionless friction coefficient for White-Colebrook formulation "
            "(XBeach default: no limit)"
        ),
        ge=0.0,
    )


class WhiteColebrookGrainsize(FrictionModifiers):
    """White-Colebrook grain size bed friction formulation.

    Based on the relation between the D90 of the top bed layer and the
    geometrical roughness of Nikuradse. The user does not have to specify
    a value for the bed friction coefficient as it is computed from the
    sediment grain size distribution.

    This formulation is the XBeach-G default. Like other friction formulations,
    it supports friction limits (mincf/maxcf) and inherits XBeach-G friction
    modifiers (fwcutoff, acceleration, infiltration, turbulence) from FrictionModifiers.

    Note: This formulation does NOT use bedfriccoef or bedfricfile as it computes
    friction from the sediment grain size (D90).
    """

    model_type: Literal["white-colebrook-grainsize"] = Field(
        default="white-colebrook-grainsize",
        description="Model type discriminator",
    )
    mincf: Optional[float] = Field(
        default=None,
        description=(
            "Minimum dimensionless friction coefficient for grain size formulation "
            "(XBeach default: 0)"
        ),
        ge=0.0,
        le=1.0,
    )
    maxcf: Optional[float] = Field(
        default=None,
        description=(
            "Maximum dimensionless friction coefficient for grain size formulation "
            "(XBeach default: no limit)"
        ),
        ge=0.0,
    )
