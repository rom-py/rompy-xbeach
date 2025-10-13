"""XBeach vegetation parameter configurations.

This module contains the Vegetation model used by the Physics.vegetation field.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class Vegetation(XBeachBaseModel):
    """Vegetation model configuration.

    When used in Physics.vegetation field, this enables vegetation modeling (vegetation=1)
    and allows specification of vegetation-specific parameters.
    """

    _is_boolean_switch = True

    model_type: Literal[True] = Field(
        default=True,
        description="Model type discriminator - set to True to enable vegetation",
    )
    nveg: Optional[int] = Field(
        default=None,
        description="Number of vegetation species",
        ge=1,
    )
    veggiefile: Optional[XBeachDataBlob] = Field(
        default=None,
        description="Name of veggie species list file",
    )
    veggiemapfile: Optional[XBeachDataBlob] = Field(
        default=None,
        description="Name of veggie species map file",
    )
    vegcanflo: Optional[bool] = Field(
        default=None,
        description="Include incanopy flow (XBeach default: 0)",
    )
    vegnonlin: Optional[bool] = Field(
        default=None,
        description="Include non-linear wave effect (XBeach default: 0)",
    )
    veguntow: Optional[bool] = Field(
        default=None,
        description="Include undertow in phase-averaged vegetation (XBeach default: 1)",
    )

    def get(self, destdir: str | Path) -> dict:
        """Fetch external vegetation files if specified, and return the params dict.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for fetching files.

        Returns
        -------
        dict
            Parameters dictionary with file paths updated to workspace directory.

        """
        # Get base params (DataBlob fields are automatically excluded by serializer)
        params = super().get(destdir)

        # Fetch DataBlob files and add the fetched file paths
        if self.veggiefile:
            params["veggiefile"] = self.veggiefile.get(destdir).name
        if self.veggiemapfile:
            params["veggiemapfile"] = self.veggiemapfile.get(destdir).name

        return params
