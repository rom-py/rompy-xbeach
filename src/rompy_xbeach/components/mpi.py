"""XBeach MPI parallelisation parameter configurations.

This module contains models for MPI (Message Passing Interface) parallelisation
settings that control domain decomposition for parallel execution.
"""

from typing import Literal, Optional

from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel


class Mpi(XBeachBaseModel):
    """MPI parallelisation parameters (XBeach Table 45).

    Controls how the model domain is subdivided into sub-models for parallel
    execution on multiple cores. Each sub-model is computed on a separate core,
    which increases computational speed. Sub-models only exchange information
    over their boundaries when necessary.

    The domain subdivision strategy is controlled by `mpiboundary`:
    - **auto**: Subdivides domain to minimise internal boundary length
    - **x**: Subdivides in cross-shore direction (full alongshore extent per domain)
    - **y**: Subdivides in alongshore direction (full cross-shore extent per domain)
    - **man**: Manual subdivision using `mmpi` and `nmpi` values

    Note
    ----
    The number of sub-models is determined by the MPI wrapper (e.g., MPICH2 or
    OpenMPI), not by XBeach itself.

    References
    ----------
    XBeach manual: MPI section for parallelisation details.
    """

    mmpi: Optional[int] = Field(
        default=None,
        description=(
            "Number of domains in cross-shore direction when manually specifying "
            "MPI domains (XBeach default: 2)"
        ),
        ge=1,
        le=100,
    )
    mpiboundary: Optional[Literal["auto", "x", "y", "man"]] = Field(
        default=None,
        description=(
            "Strategy for MPI domain boundaries: auto (shortest boundary), "
            "x (cross-shore subdivision), y (alongshore subdivision), "
            "or man (manual using mmpi/nmpi) (XBeach default: auto)"
        ),
    )
    nmpi: Optional[int] = Field(
        default=None,
        description=(
            "Number of domains in alongshore direction when manually specifying "
            "MPI domains (XBeach default: 4)"
        ),
        ge=1,
        le=100,
    )

    @model_validator(mode="after")
    def validate_manual_mode(self) -> "Mpi":
        """Validate that mmpi and nmpi are specified when using manual mode."""
        if self.mpiboundary == "man":
            if self.mmpi is None or self.nmpi is None:
                raise ValueError(
                    "When mpiboundary='man', both mmpi and nmpi must be specified "
                    "to define the manual domain decomposition."
                )
        return self
