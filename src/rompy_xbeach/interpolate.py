from pydantic import Field
from typing import Literal, Optional
from abc import ABC, abstractmethod
import numpy as np
import logging
from pydantic_numpy.typing import Np1DArray, Np2DArray

from rompy.core.types import RompyBaseModel
logger = logging.getLogger(__name__)

class BaseInterpolator(ABC, RompyBaseModel):
    """Base interpolator class."""

    model_type: Literal["base"] = Field(
        default="base", description="Model type discriminator"
    )
    kwargs: dict = Field(
        default={}, description="Keyword arguments for the interpolator"
    )

    @abstractmethod
    def get(
        self,
        x: Np1DArray,
        y: Np1DArray,
        data: Np2DArray,
        xi: Np2DArray,
        yi: Np2DArray,
    ) -> Np2DArray:
        """Perform interpolation.

        New classes should implement this which must take the x, y, and data arrays
        and interpolate the data to the xi, yi coordinates.

        Parameters
        ----------
        x: Np1DArray
            The x-coordinates of the data.
        y: Np1DArray
            The y-coordinates of the data.
        data : Np2DArray
            The data to interpolate.
        xi: Np2DArray
            The x-coordinates of the interpolated data.
        yi: Np2DArray
            The y-coordinates of the interpolated data.

        Returns
        -------
        datai: Np2DArray
            The interpolated data.

        """
        pass


class RegularGridInterpolator(BaseInterpolator):
    """Regular grid interpolator based on scipy's RegularGridInterpolator."""

    model_type: Literal["scipy_regular_grid"] = Field(
        default="scipy_regular_grid", description="Model type discriminator"
    )

    def get(
        self,
        x: Np1DArray,
        y: Np1DArray,
        data: Np2DArray,
        xi: Np2DArray,
        yi: Np2DArray,
    ) -> Np2DArray:
        """Interpolate the data to the grid.

        Parameters
        ----------
        x: Np1DArray
            The x-coordinates of the data.
        y: Np1DArray
            The y-coordinates of the data.
        data : Np2DArray
            The data to interpolate.
        xi: Np2DArray
            The x-coordinates of the interpolated data.
        yi: Np2DArray
            The y-coordinates of the interpolated data.

        Returns
        -------
        datai: Np2DArray
            The interpolated data.

        """
        from scipy.interpolate import RegularGridInterpolator
        try:
            interp = RegularGridInterpolator(points=(y, x), values=data, **self.kwargs)
            return interp((yi, xi))
        except ValueError as e:
            interp = RegularGridInterpolator(points=(y, x), values=data, bounds_error=False, fill_value=None,  **self.kwargs)
            logging.warning(f"Interpolating grid failed {e}, allowing out of bounds data to be extrapolated")
            return interp((yi, xi))
