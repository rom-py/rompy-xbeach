"""XBeach vegetation parameter configurations.

This module contains the Vegetation model used by the Physics.vegetation field.
"""

from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import Field

from rompy.core.data import DataBlob
from rompy_xbeach.types import XBeachBaseModel


class Vegetation(XBeachBaseModel):
    """Vegetation model configuration.

    When used in Physics.vegetation field, this enables vegetation modeling (vegetation=1)
    and allows specification of vegetation-specific parameters.
    """

    model_type: Literal[True] = Field(
        default=True,
        description="Model type discriminator - set to True to enable vegetation",
    )
    nveg: Optional[int] = Field(
        default=None,
        description="Number of vegetation species",
        ge=1,
    )
    veggiefile: Optional[Union[str, DataBlob]] = Field(
        default=None,
        description="Name of veggie species list file",
    )
    veggiemapfile: Optional[Union[str, DataBlob]] = Field(
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
        # Call parent get() to get base params with model_type included
        params = super().get(destdir)

        # Fetch DataBlob files and update params with the fetched file paths
        file_fields = ["veggiefile", "veggiemapfile"]
        for field in file_fields:
            if getattr(self, field) and isinstance(getattr(self, field), DataBlob):
                params[field] = getattr(self, field).get(destdir).name
        
        return params
