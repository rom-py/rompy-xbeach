"""Base classes and mixins for XBeach wave boundary conditions.

This module contains the foundational classes used by all wave boundary types:
- WaveBoundaryParams: Base wave boundary parameters (apply to all types)
- SpectralWaveBoundaryParams: Spectral-specific parameters
- NonSpectralWaveBoundaryParams: Non-spectral-specific parameters
- BoundaryBase, BoundaryBaseGrid, BoundaryBaseStation, BoundaryBasePoint: Data interface bases
- SpectraMixin, ParamMixin, FilelistMixin: Data processing mixins
"""

from typing import Literal, Union, Optional, Any
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import Field, field_validator, model_serializer

from rompy.utils import load_entry_points
from rompy.core.types import DatasetCoords, RompyBaseModel

from rompy_xbeach.data.base import BaseDataStation, BaseDataPoint, BaseDataGrid
from rompy_xbeach.source import (
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
    SourceCRSWavespectra,
)


logger = logging.getLogger(__name__)


SOURCE_TIMESERIES_TYPES = Union[load_entry_points("rompy.source", "timeseries")]

SOURCE_PARAM_TYPES = Union[
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
]

SOURCE_SPECTRA_TYPES = Union[
    SourceCRSWavespectra,
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
]


def dspr_to_s(dspr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the Jonswap spreading coefficient from the directional spread.

    Parameters
    ----------
    dspr: float | np.ndarray
        The directional spread in degrees.

    Returns
    -------
    s : float | np.ndarray
        The Jonswap spreading coefficient.

    """
    return (2 / np.radians(dspr) ** 2) - 1


def s_to_dspr(s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the directional spread from the Jonswap spreading coefficient.

    Parameters
    ----------
    s: float | np.ndarray
        The Jonswap spreading coefficient.

    Returns
    -------
    dspr : float | np.ndarray
        The directional spread in degrees.

    """
    return np.degrees(np.sqrt(2 / (s + 1)))


# =====================================================================================
# Wave Boundary Parameter Base Classes
# =====================================================================================
class WaveBoundaryParams(RompyBaseModel):
    """Base wave boundary condition parameters.

    These are general parameters that apply to ALL wave boundary condition types,
    whether spectral (jons, swan, vardens, jonstable) or non-spectral (stat, ts_1,
    ts_2, ts_nonh, bichrom).

    The boundary conditions affect wave generation, energy scaling, and the
    treatment of Stokes drift and wave group variance at the boundary.
    """

    nmax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum ratio of cg/c for computing long wave boundary conditions "
            "(XBeach default: 0.8)"
        ),
        ge=0.5,
        le=1.0,
    )
    wbcevarreduce: Optional[float] = Field(
        default=None,
        description=(
            "Reduction factor of short-wave group variance at the boundary "
            "(XBeach default: 1.0, no reduction)"
        ),
        ge=0.0,
        le=1.0,
    )
    bclwonly: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to run boundary conditions with long waves only (XBeach default: 0)"
        ),
    )
    swkhmin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum kh value to include in wave action balance. Waves with lower kh "
            "are included in NLSWE instead (XBeach default: -0.01)"
        ),
        ge=-0.01,
        le=0.35,
    )
    wbcRemoveStokes: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to remove long wave Stokes drift component at the offshore "
            "boundary (XBeach default: 1)"
        ),
    )
    wbcScaleEnergy: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to correct random time series of wave height to exactly "
            "match input Hm0 (XBeach default: 1)"
        ),
    )
    cyclicdiradjust: Optional[bool] = Field(
        default=None,
        description=(
            "Adjust alongshore wave length to fit inside domain with cyclic "
            "boundary conditions (XBeach default: 0)"
        ),
    )
    taper: Optional[float] = Field(
        default=None,
        description=(
            "Spin-up time of wave boundary conditions, in morphological time "
            "(XBeach default: 100.0)"
        ),
        ge=0.0,
        le=1000.0,
        examples=[100.0],
    )
    ARC: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for active reflection compensation at seaward boundary. "
            "Compensates for spurious long wave reflection (XBeach default: 1)"
        ),
    )
    freewave: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for free wave propagation at the boundary. When enabled, assumes "
            "incoming long waves propagate at sqrt(gh) instead of group velocity cg. "
            "Affects absorbing/radiating boundary calculations (XBeach default: 0)"
        ),
    )
    thetamin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum wave angle (degrees). When thetanaut=0, this is relative to the "
            "grid x-axis (shore-normal); when thetanaut=1, this is in nautical convention "
            "(N=0°, E=90°). Only used when swave=1 (XBeach default: -90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    thetamax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum wave angle (degrees). When thetanaut=0, this is relative to the "
            "grid x-axis (shore-normal); when thetanaut=1, this is in nautical convention "
            "(N=0°, E=90°). Only used when swave=1 (XBeach default: 90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    dtheta: Optional[float] = Field(
        default=None,
        description=(
            "Wave directional resolution (degrees). Automatically computed from "
            "thetamax-thetamin when single_dir=1. Only used when swave=1 "
            "(XBeach default: 10.0)"
        ),
        ge=0.1,
        le=180.0,
    )
    thetanaut: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for wave direction convention. When 0 (default), wave angles are "
            "relative to the grid x-axis and rotated by alfa internally. When 1, wave "
            "angles are in nautical convention (N=0°, E=90°) using real-world coordinates "
            "and alfa is ignored. Only used when swave=1 (XBeach default: 0)"
        ),
    )
    order: Optional[Literal[1, 2]] = Field(
        default=None,
        description=(
            "Order of wave steering at the boundary. 1 = first-order (short wave "
            "energy only), 2 = second-order (bound long wave corresponding to short "
            "wave forcing is added) (XBeach default: 2)"
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize_xbeach_params(self, handler) -> dict[str, Any]:
        """Convert booleans to integers for XBeach params.txt compatibility."""
        data = handler(self)
        return {k: int(v) if isinstance(v, bool) else v for k, v in data.items()}


class SpectralWaveBoundaryParams(WaveBoundaryParams):
    """Spectral wave boundary condition parameters.

    These parameters are specific to spectral boundary conditions (wbctype = jons,
    swan, vardens, jonstable). They control how wave spectra are generated and
    applied at the offshore boundary.

    Inherits all general wave boundary parameters from WaveBoundaryParams.
    """

    rt: Optional[float] = Field(
        default=None,
        description=(
            "Duration (s) of wave spectrum at offshore boundary, in morphological "
            "time (XBeach default: min(3600.d0, tstop))"
        ),
        ge=1200.0,
        le=7200.0,
        examples=[3600.0],
    )
    dtbc: Optional[float] = Field(
        default=1.0,
        description=(
            "Timestep (s) used to describe time series of wave energy and long wave "
            "flux at offshore boundary (not affected by morfac) (XBeach default: 1.0)"
        ),
        ge=0.1,
        le=2.0,
        examples=[1.0],
    )
    dthetas_xb: Optional[float] = Field(
        default=None,
        description=(
            "The (counter-clockwise) angle in the degrees needed to rotate from the "
            "x-axis in swan to the x-axis pointing east (XBeach default: 0.0)",
        ),
        ge=-360.0,
        le=360.0,
    )
    tm01switch: Optional[bool] = Field(
        default=None,
        description="Switch to enable tm01 rather than tm-10 (XBeach default: 0)",
    )
    correcthm0: Optional[bool] = Field(
        default=None,
        description="Switch to enable hm0 correction (XBeach default: 1)",
    )
    fcutoff: Optional[float] = Field(
        default=None,
        description=(
            "Low-freq cutoff frequency in Hz for jons, swan or vardens boundary "
            "conditions (XBeach default: 0.0)"
        ),
        ge=0.0,
        le=40.0,
    )
    nonhspectrum: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "Spectrum format for wave action balance of nonhydrostatic waves "
            "(XBeach default: 0)"
        ),
    )
    nspectrumloc: Optional[int] = Field(
        default=None,
        description=("Number of input spectrum locations (XBeach default: 1)"),
        ge=1,
    )
    nspr: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable long wave direction forced into centres of short wave "
            "bins (XBeach default: 0)",
        ),
    )
    random: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable random seed for jons, swan or vardens boundary "
            "conditions (XBeach default: 1)",
        ),
    )
    sprdthr: Optional[float] = Field(
        default=None,
        description=(
            "Threshold ratio to maximum value of s above which spectrum densities "
            "are read in (XBeach default: 0.08)"
        ),
        ge=0.0,
        le=1.0,
    )
    trepfac: Optional[float] = Field(
        default=None,
        description=(
            "Compute mean wave period over energy band: par%trepfac*maxval(sf) for "
            "jons, swan or vardens; converges to tm01 for trepfac = 0.0 "
            "(XBeach default: 0.01)",
        ),
        ge=0.0,
        le=1.0,
    )
    wbcversion: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description="Version of wave boundary conditions (XBeach default: 3)",
    )


# =====================================================================================
# Data Interface Base Classes
# =====================================================================================
class BoundaryBase:
    """Base class for wave boundary data interfaces.

    This class provides a custom serializer that returns an empty dict, ensuring
    that data interface fields from parent classes (BaseDataStation, BaseDataGrid,
    BaseDataPoint) are not serialized to params.txt. Only wave boundary parameters
    from WaveBoundaryParams/SpectralWaveBoundaryParams should be serialized.
    """

    location: Literal["offshore"] = Field(
        default="offshore",
        description="Location to extract the data from the source dataset",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler) -> dict[str, Any]:
        """Only keep fields defined on WaveBoundaryParams or SpectralWaveBoundaryParams.

        This ensures data-interface fields from parent classes (BaseDataStation,
        BaseDataGrid, etc.) are excluded from serialization without hardcoding
        field names — any changes to rompy core classes are handled automatically.
        """
        data = handler(self)
        wave_param_fields = set(WaveBoundaryParams.model_fields.keys()) | set(
            SpectralWaveBoundaryParams.model_fields.keys()
        )
        return {k: v for k, v in data.items() if k in wave_param_fields}


class BoundaryBaseGrid(BoundaryBase, BaseDataGrid):
    """Base class to construct XBeach wave boundary from gridded data."""


class BoundaryBaseStation(BoundaryBase, BaseDataStation):
    """Base class to construct XBeach wave boundary from stations type data."""


class BoundaryBasePoint(BoundaryBase, BaseDataPoint):
    """Base class to construct XBeach wave boundary from point timeseries type data."""


# =====================================================================================
# Data Processing Mixins
# =====================================================================================
class SpectraMixin:
    """Mixin class to calculate wave statistics from spectral data."""

    source: SOURCE_SPECTRA_TYPES = Field(
        description=(
            "Dataset source reader, must support CRS and have wavespectra accessor "
        ),
        discriminator="model_type",
    )
    coords: DatasetCoords = Field(
        default=DatasetCoords(x="lon", y="lat", t="time", s="site"),
        description="Names of the coordinates in the dataset",
    )

    @field_validator("source")
    def _validate_source_wavespectra(cls, source, values):
        if not hasattr(source.open(), "spec"):
            raise ValueError("source must have wavespectra accessor")
        return source

    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the wave statistics from the spectral data.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        stats = ds.chunk(freq=-1, dir=-1).spec.stats(
            ["hs", "tp", "dpm", "gamma", "dspr"]
        )
        stats["s"] = dspr_to_s(stats.dspr)
        return stats.rename(hs="hm0", dpm="mainang", gamma="gammajsp")


class ParamMixin:
    """Mixin class to get Jonswap statistics from parameter data."""

    hm0_var: Union[str, float] = Field(
        description=(
            "Variable name of the significant wave height Hm0 in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    tp_var: Union[str, float] = Field(
        description=(
            "Variable name of the peak period Tp in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    mainang_var: Union[str, float] = Field(
        description=(
            "Variable name of the main wave direction in the source data, "
            "or alternatively a constant  value to use for all times"
        ),
    )
    gammajsp_var: Optional[Union[str, float]] = Field(
        default=None,
        description=(
            "Variable name of the gamma parameter in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    dspr_var: Optional[Union[str, float]] = Field(
        default=None,
        description=(
            "Variable name of the directional spreading in the source data, used to "
            "calculate the Jonswap spreading coefficient, "
            "or alternatively a constant value to use for all times"
        ),
    )

    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the wave statistics from the spectral data.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        stats = xr.Dataset()
        for param, var_name in [
            ("hm0", "hm0_var"),
            ("tp", "tp_var"),
            ("mainang", "mainang_var"),
            ("gammajsp", "gammajsp_var"),
        ]:
            var_value = getattr(self, var_name)
            if var_value is None:
                continue
            if isinstance(var_value, str):
                stats[param] = ds[var_value]
            elif isinstance(var_value, (int, float)):
                stats[param] = xr.DataArray(var_value, coords=stats.coords)
        if self.dspr_var is not None:
            if isinstance(self.dspr_var, str):
                stats["s"] = dspr_to_s(ds[self.dspr_var])
            elif isinstance(self.dspr_var, (int, float)):
                stats["s"] = dspr_to_s([self.dspr_var] * ds.time.size)
        return stats


class FilelistMixin:
    """Mixin class for FILELIST functionality.

    This mixin provides support for FILELIST operations in two contexts:
    1. Writing FILELIST files when generating boundary data from external sources
    2. Fetching FILELIST files when using pre-existing boundary files

    For data-generating boundaries:
    - When filelist=True: Creates a FILELIST file that references individual bcfiles for
      each timestep, plus writes the individual bcfiles. The FILELIST file is specified
      as the bcfile parameter in params.txt.
    - When filelist=False: Creates a single bcfile with wave parameters interpolated
      at time.start.

    For file-based boundaries:
    - When filelist=True: The source is a FILELIST file and all referenced files will
      be fetched from the same directory.

    Example FILELIST format (as written to filelist.txt):
    ```
    FILELIST
    1800 0.2 jonswap1.inp
    1800 0.2 jonswap1.inp
    1350 0.2 jonswap2.inp
    1500 0.2 jonswap3.inp
    1200 0.2 jonswap2.inp
    3600 0.2 jonswap4.inp
    ```
    Each line contains: duration (seconds), timestep (seconds), and filename.
    """

    filelist: bool = Field(
        default=False,
        description=(
            "Controls FILELIST behavior. For data-generating boundaries: creates multiple "
            "bcfiles with a FILELIST index if True, single bcfile if False. "
            "For file-based boundaries: interprets source as FILELIST if True."
        ),
    )

    def _write_filelist(
        self, destdir: Path, bcfiles: list[str], durations: list[float]
    ) -> Path:
        """Write a filelist with the bcfiles.

        Parameters
        ----------
        destdir : Path
            Destination directory for the filelist.
        bcfiles : list[Path]
            List of bcfiles to include in the filelist.
        durations : list[float]
            List of durations for each bcfile.

        Returns
        -------
        filename : Path
            Path to the filelist file.

        """
        # Get dtbc from self if available, otherwise use default
        dtbc = getattr(self, "dtbc", None) or 1.0

        filename = Path(destdir) / f"{self.id}-filelist.txt"
        with open(filename, "w") as f:
            f.write("FILELIST\n")
            for bcfile, duration in zip(bcfiles, durations):
                f.write(f"{duration:g} {dtbc:g} {bcfile.name}\n")
        return filename

    def _fetch_filelist_files(self, destdir: Path, filelist_path: Path) -> None:
        """Fetch all files referenced in a FILELIST file.

        This method is used when reading pre-existing FILELIST files.
        It parses the FILELIST and fetches all referenced bcfiles from the same
        directory as the FILELIST file.

        Parameters
        ----------
        destdir : Path
            Destination directory where files will be copied.
        filelist_path : Path
            Path to the FILELIST file (already fetched to destdir).

        """
        from cloudpathlib import AnyPath

        # Get the source directory (where referenced files are located)
        # This assumes the FILELIST file was copied from its original location
        # and we need to find the original source directory
        # For file-based boundaries, this will be overridden in the specific class
        if hasattr(self, "bcfile_source"):
            source_dir = AnyPath(self.bcfile_source.source).parent
        else:
            # For data-generating classes, use the parent of the filelist
            source_dir = filelist_path.parent

        # Parse the FILELIST to get referenced files
        with open(filelist_path) as f:
            lines = f.readlines()

        # Skip the FILELIST header line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Format: <duration> <dtbc> <filename>
                filename = parts[2]
                source_file = source_dir / filename
                dest_file = destdir / filename
                if not dest_file.exists():
                    dest_file.write_bytes(source_file.read_bytes())
