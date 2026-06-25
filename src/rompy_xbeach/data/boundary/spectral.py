"""Spectral wave boundary condition classes for XBeach.

This module contains boundary classes for spectral wave boundary types:
- JONS (JONSWAP parametric)
- JONSTABLE (time-varying JONSWAP)
- SWAN (2D spectral)
- VARDENS (placeholder)

Each class can either:
1. Generate boundary files from external data sources (spectra or parameters)
2. Use pre-existing boundary files via XBeachDataFetch source
"""

from abc import abstractmethod
from typing import Literal, Optional
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import Field, model_validator

from rompy.core.time import TimeRange

from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.boundary.writers import (
    JonsWriter,
    JonstableWriter,
    SwanWriter,
)
from rompy_xbeach.types import XBeachDataBlob
from rompy_xbeach.data.boundary.base import (
    SpectralWaveBoundaryParams,
    BoundaryBaseGrid,
    BoundaryBaseStation,
    BoundaryBasePoint,
    SpectraMixin,
    ParamMixin,
    FilelistMixin,
    SOURCE_TIMESERIES_TYPES,
    SOURCE_PARAM_TYPES,
)


logger = logging.getLogger(__name__)


# =====================================================================================
# JONS Mixin Class
# =====================================================================================
class BoundaryJonsBase(FilelistMixin):
    """Base class for JONS wave boundary from data sources.

    This class generates JONSWAP boundary files from wave data.
    """

    id: Literal["jons", "parametric"] = Field(
        default="jons",
        description="Boundary type identifier, used to define the wbctype",
    )
    fnyq: Optional[float] = Field(
        default=None,
        description=(
            "Highest frequency used to create JONSWAP spectrum [Hz] "
            "(XBeach default: 0.3)"
        ),
        ge=0.2,
        le=1.0,
    )
    dfj: Optional[float] = Field(
        default=None,
        description=(
            "Step size frequency used to create JONSWAP spectrum [Hz] within the "
            "range fnyq/1000 - fnyq/20 (XBeach default: fnyq/200)"
        ),
    )

    @abstractmethod
    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the Jonswap parameters from the data."""
        pass

    def _instantiate_boundary(self, data: xr.Dataset) -> JonsWriter:
        """Instantiate the boundary file writer object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        assert data.time.size == 1
        t = data.time.to_index().to_pydatetime()[0]
        logger.debug(f"Creating boundary for time {t}")
        kwargs = {}
        for param in ["hm0", "tp", "mainang", "gammajsp", "s"]:
            if param in data and not np.isnan(data[param]):
                kwargs[param] = float(data[param].squeeze())
            elif param in data and np.isnan(data[param]):
                raise ValueError(f"Parameter {param} is NaN for {data.time}")
        bcfile = f"{self.id}-{t:%Y%m%dT%H%M%S}.txt"
        return JonsWriter(bcfile=bcfile, fnyq=self.fnyq, dfj=self.dfj, **kwargs)

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the boundary files.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        dict
            XBeach parameters including wbctype, bcfile, and wave boundary settings.

        """
        ds = super().get(destdir, grid, time)
        if not self.filelist:
            # Write a single bcfile at the timerange start
            ds = ds.interp({self.coords.t: [time.start]})
            data = self._calculate_stats(ds)
            wb = self._instantiate_boundary(data)
            bcfile = wb.write(destdir)
        else:
            # Write a bcfile for each timestep in the timerange
            ds = self._adjust_time(ds, time)
            stats = self._calculate_stats(ds)
            times = stats.time.to_index().to_pydatetime()
            bcfiles = []
            durations = []
            for t0, t1 in zip(times[:-1], times[1:]):
                # Boundary data
                data = stats.sel(time=[t0])
                wb = self._instantiate_boundary(data)
                bcfiles.append(wb.write(destdir))
                # Boundary duration
                durations.append((t1 - t0).total_seconds())
            bcfile = self._write_filelist(destdir, bcfiles, durations)

        # Return XBeach parameters
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        # Add wave boundary parameters (mixin fields excluded, data interface fields
        # are excluded by the model_serializer in BoundaryBase classes)
        params.update(
            self.model_dump(
                exclude={"id", "model_type", "filelist", "fnyq", "dfj"},
                exclude_none=True,
            )
        )
        return params


# =====================================================================================
# JONSTABLE Mixin Class
# =====================================================================================
class BoundaryJonstableBase:
    """Base class for JONSTABLE wave boundary from data sources.

    This class generates JONSTABLE boundary files from wave data.
    """

    id: Literal["jonstable"] = Field(
        default="jonstable", description="Boundary type identifier"
    )

    @model_validator(mode="after")
    def default_params(self) -> "BoundaryJonstableBase":
        if hasattr(self, "gammajsp_var") and self.gammajsp_var is None:
            logger.debug("Setting default value for gammajsp_var of 3.3")
            self.gammajsp_var = 3.3
        if hasattr(self, "dspr_var") and self.dspr_var is None:
            logger.debug("Setting default value for dspr_var of 24.431 (s=10.0)")
            self.dspr_var = 24.43100247268452
        return self

    @abstractmethod
    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the Jonswap parameters from the data."""
        pass

    def _instantiate_boundary(self, data: xr.Dataset) -> JonstableWriter:
        """Instantiate the boundary file writer object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing the boundary spectral data for all times.

        """
        times = data.time.to_index().to_pydatetime()
        logger.debug(f"Creating jonstable boundary for times {times}")
        dts = [dt.total_seconds() for dt in np.diff(times)]
        bcfile = f"{self.id}-{times[0]:%Y%m%dT%H%M%S}-{times[-1]:%Y%m%dT%H%M%S}.txt"
        dtbc = self.dtbc or 1.0
        kwargs = dict(
            hm0=data.hm0.squeeze().values,
            tp=data.tp.squeeze().values,
            mainang=data.mainang.squeeze().values,
            gammajsp=data.gammajsp.squeeze().values,
            s=data.s.squeeze().values,
            duration=dts + [dts[-1]],
            dtbc=[dtbc] * len(times),
        )
        for key, val in kwargs.items():
            if any(np.isnan(val)):
                raise ValueError(
                    f"Parameter {key} has NaN for one or more times ({list(zip(times, val))})"
                )
        return JonstableWriter(bcfile=bcfile, **kwargs)

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the boundary files.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        dict
            XBeach parameters including wbctype, bcfile, and wave boundary settings.

        """
        ds = super().get(destdir, grid, time)
        ds = self._adjust_time(ds, time)
        data = self._calculate_stats(ds)
        wb = self._instantiate_boundary(data)
        bcfile = wb.write(destdir)

        # Return XBeach parameters
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        # Add wave boundary parameters (mixin fields excluded, data interface fields
        # are excluded by the model_serializer in BoundaryBase classes)
        params.update(
            self.model_dump(
                exclude={
                    "id",
                    "model_type",
                    "hm0_var",
                    "tp_var",
                    "mainang_var",
                    "gammajsp_var",
                    "dspr_var",
                },
                exclude_none=True,
            )
        )
        return params


# =====================================================================================
# File-based Spectral Mixin Class (pre-existing bcfiles)
# =====================================================================================
class BoundaryFileSpectralBase(SpectralWaveBoundaryParams):
    """Base class for file-based spectral boundaries.

    This class provides common functionality for boundary classes that fetch
    pre-existing bcfiles.
    """

    bcfile_source: XBeachDataBlob = Field(
        description="Source for the bcfile or FILELIST file",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Fetch bcfile(s) and return XBeach parameters.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for boundary files.
        grid : RegularGrid, optional
            Grid instance (not used).
        time : TimeRange, optional
            Time range (not used).

        Returns
        -------
        dict
            XBeach parameters including wbctype, bcfile, and wave boundary settings.

        """
        destdir = Path(destdir)

        # Fetch the main bcfile
        bcfile = self.bcfile_source.get(destdir)

        # If filelist, also fetch all referenced files
        if hasattr(self, "filelist") and self.filelist:
            self._fetch_filelist_files(destdir, bcfile)

        # Return XBeach parameters
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        exclude_fields = {"model_type", "id", "bcfile_source"}
        if hasattr(self, "filelist"):
            exclude_fields.add("filelist")
        params.update(
            self.model_dump(
                exclude=exclude_fields,
                exclude_none=True,
            )
        )
        return params


# =====================================================================================
# JONS Concrete Classes
# =====================================================================================
class BoundaryStationSpectraJons(
    SpectraMixin, BoundaryJonsBase, SpectralWaveBoundaryParams, BoundaryBaseStation
):
    """Wave boundary conditions from station type spectra dataset such as SMC."""

    model_type: Literal["station_spectra_jons"] = Field(
        default="station_spectra_jons",
        description="Model type discriminator",
    )


class BoundaryStationParamJons(
    ParamMixin, BoundaryJonsBase, SpectralWaveBoundaryParams, BoundaryBaseStation
):
    """Wave boundary conditions from station type parameters dataset such as SMC."""

    source: SOURCE_PARAM_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    model_type: Literal["station_param_jons"] = Field(
        default="station_param_jons",
        description="Model type discriminator",
    )


class BoundaryPointParamJons(
    ParamMixin, BoundaryJonsBase, SpectralWaveBoundaryParams, BoundaryBasePoint
):
    """Wave boundary conditions from point timeseries type parameters dataset."""

    model_type: Literal["point_param_jons"] = Field(
        default="point_param_jons",
        description="Model type discriminator",
    )
    source: SOURCE_TIMESERIES_TYPES = Field(
        description="Dataset source reader for point timeseries type data",
        discriminator="model_type",
    )


class BoundaryGridParamJons(
    ParamMixin, BoundaryJonsBase, SpectralWaveBoundaryParams, BoundaryBaseGrid
):
    """Wave boundary conditions from grid type parameters dataset."""

    source: SOURCE_PARAM_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    model_type: Literal["grid_param_jons"] = Field(
        default="grid_param_jons",
        description="Model type discriminator",
    )


class BoundaryFileJons(FilelistMixin, BoundaryFileSpectralBase):
    """JONSWAP boundary from pre-existing bcfile(s).

    Use this class when you have existing JONSWAP boundary files created outside
    of rompy (e.g., manually or from another tool).

    If `filelist=True`, the source bcfile is expected to be a FILELIST file, and
    all files referenced within it will also be fetched from the same directory.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> # Single bcfile
    >>> boundary = BoundaryFileJons(
    ...     bcfile_source=XBeachDataBlob(source="/path/to/bcfile"),
    ... )
    >>> # FILELIST with multiple bcfiles
    >>> boundary = BoundaryFileJons(
    ...     bcfile_source=XBeachDataBlob(source="/path/to/bcfile"),
    ...     filelist=True,
    ... )

    """

    id: Literal["jons"] = Field(default="jons", description="Boundary type identifier")
    model_type: Literal["file_jons"] = Field(
        default="file_jons",
        description="Model type discriminator",
    )


# =====================================================================================
# JONSTABLE Concrete Classes
# =====================================================================================
class BoundaryStationSpectraJonstable(
    SpectraMixin, BoundaryJonstableBase, SpectralWaveBoundaryParams, BoundaryBaseStation
):
    """Wave boundary conditions from station type spectra dataset such as SMC."""

    model_type: Literal["station_spectra_jonstable"] = Field(
        default="station_spectra_jonstable",
        description="Model type discriminator",
    )


class BoundaryStationParamJonstable(
    ParamMixin, BoundaryJonstableBase, SpectralWaveBoundaryParams, BoundaryBaseStation
):
    """Wave boundary conditions from station type parameters dataset such as SMC."""

    source: SOURCE_PARAM_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    model_type: Literal["station_param_jonstable"] = Field(
        default="station_param_jonstable",
        description="Model type discriminator",
    )


class BoundaryPointParamJonstable(
    ParamMixin, BoundaryJonstableBase, SpectralWaveBoundaryParams, BoundaryBasePoint
):
    """Wave boundary conditions from point timeseries type parameters dataset."""

    model_type: Literal["point_param_jonstable"] = Field(
        default="point_param_jonstable",
        description="Model type discriminator",
    )
    source: SOURCE_TIMESERIES_TYPES = Field(
        description="Dataset source reader for point timeseries type data",
        discriminator="model_type",
    )


class BoundaryGridParamJonstable(
    ParamMixin, BoundaryJonstableBase, SpectralWaveBoundaryParams, BoundaryBaseGrid
):
    """Generate XBeach JONSTABLE wave boundary conditions from gridded parameter data.

    This class reads wave parameters (Hm0, Tp, Dir, Spread, Gamma) from a gridded data
    source, selects/interpolates the data at the offshore boundary location of the
    XBeach grid, and writes the time-varying parameters to a JONSTABLE format file.

    """

    source: SOURCE_PARAM_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    model_type: Literal["grid_param_jonstable"] = Field(
        default="grid_param_jonstable",
        description="Model type discriminator",
    )


class BoundaryFileJonstable(BoundaryFileSpectralBase):
    """JONSTABLE boundary from pre-existing bcfile.

    Use this class when you have an existing JONSTABLE boundary file created
    outside of rompy (e.g., manually or from another tool).

    JONSTABLE files are always single files (no FILELIST support needed).

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> boundary = BoundaryFileJonstable(
    ...     bcfile_source=XBeachDataBlob(source="/path/to/jonstable.txt"),
    ... )

    """

    id: Literal["jonstable"] = Field(
        default="jonstable", description="Boundary type identifier"
    )
    model_type: Literal["file_jonstable"] = Field(
        default="file_jonstable",
        description="Model type discriminator",
    )


# =====================================================================================
# SWAN Concrete Class
# =====================================================================================
class BoundaryStationSpectraSwan(
    FilelistMixin, SpectraMixin, SpectralWaveBoundaryParams, BoundaryBaseStation
):
    """SWAN wave boundary from station type spectra dataset.

    XBeach assumes the directional information in the SWAN file is according to the
    nautical convention. If the file uses the Cartesian convention for directions, the
    user must specify the angle in degrees to rotate the x-axis in SWAN to the x-axis in
    XBeach (by the Cartesian convention). This value is specified in params.txt using
    the keyword dthetaS_XB.

    """

    id: Literal["swan"] = Field(default="swan", description="Boundary type identifier")
    model_type: Literal["station_spectra_swan"] = Field(
        default="station_spectra_swan",
        description="Model type discriminator",
    )

    def _instantiate_boundary(self, data: xr.Dataset) -> SwanWriter:
        """Instantiate the boundary file writer object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        assert data.time.size == 1
        t = data.time.to_index().to_pydatetime()[0]
        logger.debug(f"Creating boundary for time {t}")
        bcfile = f"{self.id}-{t:%Y%m%dT%H%M%S}.txt"
        if data.lon.size > 1 or data.lat.size > 1:
            raise ValueError("Data must be a single point")
        return SwanWriter(
            bcfile=bcfile,
            freq=data.freq.squeeze().values,
            dir=data.dir.squeeze().values,
            efth=data.efth.squeeze().values,
            lon=float(data.lon.squeeze().values),
            lat=float(data.lat.squeeze().values),
        )

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the boundary files.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        dict
            XBeach parameters including wbctype, bcfile, and wave boundary settings.

        """
        ds = super().get(destdir, grid, time)
        if not self.filelist:
            # Write a single bcfile at the timerange start
            ds = ds.interp({self.coords.t: [time.start]})
            wb = self._instantiate_boundary(ds)
            bcfile = wb.write(destdir)
        else:
            # Write a bcfile for each timestep in the timerange
            ds = self._adjust_time(ds, time)
            times = ds.time.to_index().to_pydatetime()
            bcfiles = []
            durations = []
            for t0, t1 in zip(times[:-1], times[1:]):
                # Boundary data
                data = ds.sel(time=[t0])
                wb = self._instantiate_boundary(data)
                bcfiles.append(wb.write(destdir))
                # Boundary duration
                durations.append((t1 - t0).total_seconds())
            bcfile = self._write_filelist(destdir, bcfiles, durations)

        # Return XBeach parameters
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        # Add wave boundary parameters (mixin fields excluded, data interface fields
        # are excluded by the model_serializer in BoundaryBase classes)
        params.update(
            self.model_dump(
                exclude={"id", "model_type", "filelist"},
                exclude_none=True,
            )
        )
        return params


class BoundaryFileSwan(FilelistMixin, BoundaryFileSpectralBase):
    """SWAN spectral boundary from pre-existing bcfile(s).

    Use this class when you have existing SWAN spectral boundary files created
    outside of rompy (e.g., manually or from another tool).

    If `filelist=True`, the source bcfile is expected to be a FILELIST file, and
    all files referenced within it will also be fetched from the same directory.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> # Single bcfile
    >>> boundary = BoundaryFileSwan(
    ...     bcfile_source=XBeachDataBlob(source="/path/to/swan_spectrum.txt"),
    ... )
    >>> # FILELIST with multiple bcfiles
    >>> boundary = BoundaryFileSwan(
    ...     bcfile_source=XBeachDataBlob(source="/path/to/filelist.txt"),
    ...     filelist=True,
    ... )

    """

    id: Literal["swan"] = Field(default="swan", description="Boundary type identifier")
    model_type: Literal["file_swan"] = Field(
        default="file_swan",
        description="Model type discriminator",
    )
