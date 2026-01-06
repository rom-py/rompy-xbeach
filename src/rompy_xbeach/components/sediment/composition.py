"""XBeach bed composition parameter configurations.

This module contains models for bed composition parameters including grain size
distributions, sediment density, porosity, and layer thickness.
"""

from typing import Optional, List
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel


class BedComposition(XBeachBaseModel):
    """Bed composition parameters for sediment properties.

    These parameters define the physical properties of the sediment bed including
    grain size distribution, density, porosity, and layer structure. They are
    critical for morphological simulations.

    XBeach supports multiple sediment classes (ngd > 1) for simulating graded
    sediments. When using multiple classes, D50 and D90 become lists with one
    value per class.

    The bed is divided into layers with thickness defined by dzg1 (top), dzg2
    (variable middle), and dzg3 (bottom). The variable layer (dzg2) can grow
    or shrink during simulation.

    Note
    ----
    For gravel beaches (XBeach-G), use larger grain sizes:
    - D50: 0.002-0.08 m (default 0.01 m)
    - D90: 0.003-0.12 m (default 0.015 m)

    For sandy beaches, use smaller grain sizes:
    - D50: 0.00005-0.0008 m (default 0.0002 m)
    - D90: 0.0001-0.0015 m (default 0.0003 m)

    References
    ----------
    XBeach manual: Bed composition parameters section.
    """

    ngd: Optional[int] = Field(
        default=None,
        description=(
            "Number of sediment classes. Use ngd > 1 for graded sediments "
            "(XBeach default: 1)"
        ),
        ge=1,
        le=20,
    )
    nd: Optional[int] = Field(
        default=None,
        description=("Number of computational layers in the bed (XBeach default: 3)"),
        ge=3,
        le=1000,
    )
    D50: Optional[List[float]] = Field(
        default=None,
        description=(
            "Median grain diameter (m) for each sediment class. "
            "For sandy beaches: ~0.0002 m (XBeach default). "
            "For gravel beaches: ~0.01 m. "
            "Provide a list if ngd > 1."
        ),
    )
    D90: Optional[List[float]] = Field(
        default=None,
        description=(
            "90th percentile grain diameter (m) for each sediment class. "
            "For sandy beaches: ~0.0003 m (XBeach default). "
            "For gravel beaches: ~0.015 m. "
            "Provide a list if ngd > 1."
        ),
    )
    D15: Optional[List[float]] = Field(
        default=None,
        description=(
            "15th percentile grain diameter (m) for each sediment class. "
            "Only used when dilatancy=1 (XBeach default: 0.00015 m)"
        ),
    )
    rhos: Optional[float] = Field(
        default=None,
        description=("Sediment density (kg/m³) (XBeach default: 2650.0)"),
        ge=2400.0,
        le=2800.0,
    )
    por: Optional[float] = Field(
        default=None,
        description=("Bed porosity (volume fraction of voids) (XBeach default: 0.4)"),
        ge=0.3,
        le=0.5,
    )
    dzg1: Optional[float] = Field(
        default=None,
        description=(
            "Thickness of top sediment layer (m) (XBeach default: 0.1 m). "
            "Also sets default for dzg2 and dzg3 if not specified."
        ),
        ge=0.01,
        le=1.0,
    )
    dzg2: Optional[float] = Field(
        default=None,
        description=(
            "Nominal thickness of variable (middle) sediment layer (m). "
            "This layer can grow/shrink during simulation (XBeach default: dzg1)"
        ),
        ge=0.01,
        le=1.0,
    )
    dzg3: Optional[float] = Field(
        default=None,
        description=("Thickness of bottom sediment layer (m) (XBeach default: dzg1)"),
        ge=0.01,
        le=1.0,
    )
    sedcal: Optional[List[float]] = Field(
        default=None,
        description=(
            "Sediment transport calibration factor for each sediment class "
            "(XBeach default: 1.0 for each class)"
        ),
    )
    ucrcal: Optional[List[float]] = Field(
        default=None,
        description=(
            "Critical velocity calibration factor for each sediment class "
            "(XBeach default: 1.0 for each class)"
        ),
    )
    ws_nonh: Optional[float] = Field(
        default=None,
        description=(
            "Sediment fall velocity (m/s). Only used in non-hydrostatic mode "
            "(wavemodel=nonh). If not specified, XBeach computes this internally "
            "from D50 using Ahrens (2000) (XBeach default: 0.0, computed)"
        ),
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_grain_size_lists(self) -> "BedComposition":
        """Validate that grain size lists match ngd if specified."""
        ngd = self.ngd or 1

        for field_name in ["D50", "D90", "D15", "sedcal", "ucrcal"]:
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != ngd:
                    raise ValueError(
                        f"{field_name} must have {ngd} values (one per sediment class), "
                        f"but got {len(value)} values. Set ngd={len(value)} or adjust {field_name}."
                    )
        return self

    @model_validator(mode="after")
    def validate_d90_greater_than_d50(self) -> "BedComposition":
        """Validate that D90 > D50 for each sediment class."""
        if self.D50 is not None and self.D90 is not None:
            for i, (d50, d90) in enumerate(zip(self.D50, self.D90)):
                if d90 <= d50:
                    raise ValueError(
                        f"D90[{i}] ({d90}) must be greater than D50[{i}] ({d50})"
                    )
        return self
