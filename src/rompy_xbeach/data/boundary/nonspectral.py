"""Non-spectral wave boundary condition classes for XBeach.

This module contains boundary classes for non-spectral wave boundary types:
- stat: Stationary parametric waves (Hrms, Trep, dir0, m)
- bichrom: Bichromatic waves (Hrms, Trep, Tlong, dir0, m)
- stat_table: Time-varying parametric waves from file
- ts_1: Time series at single location from file
- ts_2: Time series at two locations from file
- ts_nonh: Non-hydrostatic time series from file
"""

from typing import Literal
from pathlib import Path
from pydantic import Field

from rompy.core.time import TimeRange

from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.types import XBeachDataBlob
from rompy_xbeach.data.boundary.base import WaveBoundaryParams


# =====================================================================================
# Stationary Parametric Waves (no file needed)
# =====================================================================================
class BoundaryStat(WaveBoundaryParams):
    """Stationary parametric wave boundary conditions.

    Defines wave conditions using bulk parameters (Hrms, Trep, dir0, m) without
    requiring any external files. XBeach generates the wave forcing internally.

    Examples
    --------
    >>> boundary = BoundaryStat(
    ...     Hrms=2.0,
    ...     Trep=12.0,
    ...     dir0=270.0,
    ...     m=10,
    ... )

    """

    id: Literal["stat"] = Field(default="stat", description="Boundary type identifier")
    model_type: Literal["stat"] = Field(
        default="stat",
        description="Model type discriminator",
    )
    Hrms: float = Field(
        description="Hrms wave height (m)",
        ge=0.0,
        le=10.0,
    )
    Trep: float = Field(
        description="Representative wave period (s)",
        ge=1.0,
        le=20.0,
    )
    dir0: float = Field(
        default=270.0,
        description="Mean wave direction, nautical convention (degrees)",
        ge=-360.0,
        le=360.0,
    )
    m: int = Field(
        default=10,
        description="Power in cos^m directional distribution",
        ge=2,
        le=128,
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for stationary wave boundary.

        Parameters
        ----------
        destdir : str | Path
            Destination directory (not used for stat, but required for interface).
        grid : RegularGrid, optional
            Grid instance (not used for stat).
        time : TimeRange, optional
            Time range (not used for stat).

        Returns
        -------
        dict
            XBeach parameters including wbctype and wave parameters.

        """
        params = {"wbctype": self.id}
        params.update(
            self.model_dump(
                exclude={"model_type", "id"},
                exclude_none=True,
            )
        )
        return params


# =====================================================================================
# Bichromatic Waves (no file needed)
# =====================================================================================
class BoundaryBichrom(WaveBoundaryParams):
    """Bichromatic wave boundary conditions.

    Defines bichromatic wave conditions using bulk parameters including the
    long wave period (Tlong). No external files required.

    Examples
    --------
    >>> boundary = BoundaryBichrom(
    ...     Hrms=1.5,
    ...     Trep=10.0,
    ...     Tlong=80.0,
    ...     dir0=270.0,
    ...     m=10,
    ... )

    """

    id: Literal["bichrom"] = Field(
        default="bichrom", description="Boundary type identifier"
    )
    model_type: Literal["bichrom"] = Field(
        default="bichrom",
        description="Model type discriminator",
    )
    Hrms: float = Field(
        description="Hrms wave height (m)",
        ge=0.0,
        le=10.0,
    )
    Trep: float = Field(
        description="Representative wave period (s)",
        ge=1.0,
        le=20.0,
    )
    Tlong: float = Field(
        description="Wave group period (s)",
        ge=20.0,
        le=300.0,
    )
    dir0: float = Field(
        default=270.0,
        description="Mean wave direction, nautical convention (degrees)",
        ge=-360.0,
        le=360.0,
    )
    m: int = Field(
        default=10,
        description="Power in cos^m directional distribution",
        ge=2,
        le=128,
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for bichromatic wave boundary.

        Parameters
        ----------
        destdir : str | Path
            Destination directory (not used for bichrom, but required for interface).
        grid : RegularGrid, optional
            Grid instance (not used for bichrom).
        time : TimeRange, optional
            Time range (not used for bichrom).

        Returns
        -------
        dict
            XBeach parameters including wbctype and wave parameters.

        """
        params = {"wbctype": self.id}
        params.update(
            self.model_dump(
                exclude={"model_type", "id"},
                exclude_none=True,
            )
        )
        return params


# =====================================================================================
# File-based Non-Spectral Boundaries
# =====================================================================================
class BoundaryStatTable(WaveBoundaryParams):
    """Time-varying parametric wave boundary from stat_table file.

    Requires a file with time-varying Hrms, Tp, direction, etc. in JONSWAP table format.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> boundary = BoundaryStatTable(
    ...     source=XBeachDataBlob(source="/path/to/stat_table.txt"),
    ... )

    """

    id: Literal["stat_table"] = Field(
        default="stat_table", description="Boundary type identifier"
    )
    model_type: Literal["file_stat_table"] = Field(
        default="file_stat_table",
        description="Model type discriminator",
    )
    source: XBeachDataBlob = Field(
        description="Source for stat_table boundary file",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for stat_table wave boundary.

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
            XBeach parameters including wbctype, bcfile, and wave parameters.

        """
        destdir = Path(destdir)
        bcfile = self.source.get(destdir)
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "source"},
                exclude_none=True,
            )
        )
        return params


class BoundaryTs1(WaveBoundaryParams):
    """Time series wave boundary at single location (ts_1).

    Requires a bc/gen.ezs file with columns: time, zs, E.
    The file will be fetched to destdir/bc/ subdirectory.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> boundary = BoundaryTs1(
    ...     source=XBeachDataBlob(source="/path/to/gen.ezs"),
    ... )

    """

    id: Literal["ts_1"] = Field(default="ts_1", description="Boundary type identifier")
    model_type: Literal["file_ts_1"] = Field(
        default="file_ts_1",
        description="Model type discriminator",
    )
    source: XBeachDataBlob = Field(
        description="Source for time series boundary file (bc/gen.ezs format)",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for ts_1 wave boundary.

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
            XBeach parameters including wbctype, bcfile, and wave parameters.

        """
        destdir = Path(destdir)
        bc_dir = destdir / "bc"
        bc_dir.mkdir(parents=True, exist_ok=True)
        bcfile = self.source.get(bc_dir)
        params = {"wbctype": self.id, "bcfile": f"bc/{bcfile.name}"}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "source"},
                exclude_none=True,
            )
        )
        return params


class BoundaryTs2(WaveBoundaryParams):
    """Time series wave boundary at two locations (ts_2).

    Requires a bc/gen.ezs file with columns: time, zs, E.
    The file will be fetched to destdir/bc/ subdirectory.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> boundary = BoundaryTs2(
    ...     source=XBeachDataBlob(source="/path/to/gen.ezs"),
    ... )

    """

    id: Literal["ts_2"] = Field(default="ts_2", description="Boundary type identifier")
    model_type: Literal["file_ts_2"] = Field(
        default="file_ts_2",
        description="Model type discriminator",
    )
    source: XBeachDataBlob = Field(
        description="Source for time series boundary file (bc/gen.ezs format)",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for ts_2 wave boundary.

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
            XBeach parameters including wbctype, bcfile, and wave parameters.

        """
        destdir = Path(destdir)
        bc_dir = destdir / "bc"
        bc_dir.mkdir(parents=True, exist_ok=True)
        bcfile = self.source.get(bc_dir)
        params = {"wbctype": self.id, "bcfile": f"bc/{bcfile.name}"}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "source"},
                exclude_none=True,
            )
        )
        return params


class BoundaryTsNonh(WaveBoundaryParams):
    """Non-hydrostatic time series wave boundary (ts_nonh).

    Requires a Boun_u.bcf file with columns: scalar/vector, t, U, Zs, W.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDataBlob
    >>> boundary = BoundaryTsNonh(
    ...     source=XBeachDataBlob(source="/path/to/Boun_u.bcf"),
    ... )

    """

    id: Literal["ts_nonh"] = Field(
        default="ts_nonh", description="Boundary type identifier"
    )
    model_type: Literal["file_ts_nonh"] = Field(
        default="file_ts_nonh",
        description="Model type discriminator",
    )
    source: XBeachDataBlob = Field(
        description="Source for non-hydrostatic time series boundary file (Boun_u.bcf format)",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for ts_nonh wave boundary.

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
            XBeach parameters including wbctype, bcfile, and wave parameters.

        """
        destdir = Path(destdir)
        bcfile = self.source.get(destdir)
        params = {"wbctype": self.id, "bcfile": bcfile.name}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "source"},
                exclude_none=True,
            )
        )
        return params
