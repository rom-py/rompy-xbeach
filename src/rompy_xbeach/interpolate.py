from pydantic import BaseModel, Field
from typing import List, Tuple, Callable, Any, Literal
from abc import ABC, abstractmethod
import numpy as np

from rompy.core.types import RompyBaseModel


class BaseInterpolator(ABC, RompyBaseModel):
    """Base interpolator class."""

    model_type: Literal["base"] = Field(
        default="base",
        description="Model type discriminator"
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments for the interpolator"
    )

    @abstractmethod
    def interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
    ) -> np.ndarray:
        """Perform interpolation.

        New classes should implement this which must take the x, y, and data arrays
        and interpolate the data to the xi, yi coordinates.

        Parameters
        ----------
        x: np.ndarray
           The x-coordinates of the data.
        y: np.ndarray
              The y-coordinates of the data.
        data : np.ndarray
                The data to interpolate.
        xi: np.ndarray
            The x-coordinates of the interpolated data.
        yi: np.ndarray
            The y-coordinates of the interpolated data.

        Returns
        -------
        datai: np.ndarray
            The interpolated data.

        """
        pass


class RegularGridInterpolator(BaseInterpolator):
    """Regular grid interpolator based on scipy.interpolate.RegularGridInterpolator."""

    model_type: Literal["regular_grid"] = Field(
        default="regular_grid",
        description="Model type discriminator"
    )

    def interpolate(self,
            x: np.ndarray,
            y: np.ndarray,
            data: np.ndarray,
            xi: np.ndarray,
            yi: np.ndarray,
        ) -> np.ndarray:
        """Interpolate the data to the grid.

        Parameters
        ----------
        x: np.ndarray
           The x-coordinates of the data.
        y: np.ndarray
           The y-coordinates of the data.
        data : np.ndarray
           The data to interpolate.
        xi: np.ndarray
           The x-coordinates of the interpolated data.
        yi: np.ndarray
           The y-coordinates of the interpolated data.

        Returns
        -------
        datai: np.ndarray
            The interpolated data.

        """
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(points=(y, x), values=data, **self.kwargs)
        return interp((yi, xi))
