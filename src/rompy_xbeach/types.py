from enum import Enum
from typing import Any, Union
from pathlib import Path

from cloudpathlib import AnyPath
from pydantic import ConfigDict, Field, field_validator, model_serializer

from rompy.core.config import BaseConfig
from rompy.core.types import RompyBaseModel
from rompy.core.data import DataBlob


class XBeachDataBlob(DataBlob):
    """Custom DataBlob for XBeach with special handling in get() method.

    XBeachDataBlob fields are excluded from .params serialization to prevent
    internal fields (id, source, link) from leaking into params.txt.
    In .get() they are replaced with the fetched file path.

    Usage:
        veggiefile: Optional[XBeachDataBlob] = Field(default=None, ...)

        def get(self, destdir: Path) -> dict:
            params = super().get(destdir)
            if self.veggiefile and destdir:
                params["veggiefile"] = self.veggiefile.get(destdir).name
            return params
    """

    @model_serializer(mode="wrap")
    def _serialize_skip_for_params(self, serializer: Any) -> dict:
        """Skip serialization to prevent field leakage in params."""
        # Return empty dict so the field gets excluded by exclude_none
        # The field will be added back in the component's get() method
        return {}


class XBeachHotstartBlob(XBeachDataBlob):
    """DataBlob for XBeach hotstart files with glob pattern support.

    Hotstart files follow the naming convention: hotstart_{varname}{fileno:06d}.dat
    where varname is the variable name (zs, zb, uu, vv, etc.) and fileno is the
    hotstart file number (0-999).

    The source should point to a directory containing hotstart files from a
    previous XBeach simulation.

    Usage:
        hotstart_source: Optional[XBeachHotstartBlob] = Field(default=None, ...)

        def get(self, destdir: Path, fileno: int = 0) -> list[Path]:
            # Returns list of copied hotstart files
            return self.hotstart_source.get(destdir, fileno)
    """

    source: AnyPath = Field(
        description=(
            "Directory containing hotstart files from a previous XBeach simulation. "
            "Can be a local path or remote URI (e.g., s3://bucket/path)."
        ),
    )

    @field_validator("source", mode="after")
    @classmethod
    def validate_source_is_directory(cls, v: AnyPath) -> AnyPath:
        """Validate that source is a directory."""
        if not v.is_dir():
            raise ValueError(
                f"Hotstart source must be a directory containing hotstart files, "
                f"got: {v}"
            )
        return v

    @field_validator("link", mode="after")
    @classmethod
    def validate_link_not_allowed(cls, v: bool) -> bool:
        """Validate that link is not enabled for hotstart directories."""
        if v:
            raise ValueError(
                "Symlinks are not supported for hotstart directories. "
                "Files must be copied."
            )
        return v

    def get(
        self, destdir: Union[str, Path], fileno: int = 0, *args, **kwargs
    ) -> list[Path]:
        """Copy hotstart files matching the pattern to destdir.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to copy hotstart files to.
        fileno : int, optional
            The hotstart file number to match (0-999). Default is 0.

        Returns
        -------
        list[Path]
            List of paths to the copied hotstart files.

        Raises
        ------
        FileNotFoundError
            If no hotstart files matching the pattern are found.
        """
        destdir = Path(destdir).resolve()
        destdir.mkdir(parents=True, exist_ok=True)

        source_path = AnyPath(self.source)
        pattern = f"hotstart_*{fileno:06d}.dat"
        copied_files = []

        for f in source_path.glob(pattern):
            dest_file = destdir / f.name
            # Use read_bytes/write_bytes for remote compatibility (same as DataBlob)
            dest_file.write_bytes(f.read_bytes())
            copied_files.append(dest_file)

        if not copied_files:
            raise FileNotFoundError(
                f"No hotstart files matching '{pattern}' found in {source_path}"
            )

        return copied_files


class XBeachBaseModel(RompyBaseModel):
    """Base model class for all XBeach parameter models.

    Provides automatic serialization with:
    - Recursive flattening of nested components (model_type discriminators)
    - Boolean to integer conversion for XBeach compatibility
    - Standard params property and get() method

    All XBeach parameter models should inherit from this class.

    """

    @model_serializer(mode="wrap")
    def _serialize_with_component_flattening(self, serializer: Any) -> dict:
        """Serialize model with recursive component flattening and bool to int conversion.

        This serializer:
        1. Recursively detects nested dictionaries (from parameter component serialization)
        2. Flattens them by setting outer key = inner model_type value
        3. Merges remaining inner key-values into the main dict
        4. Converts booleans to integers for XBeach compatibility

        Example:
            {'wavemodel': {'model_type': 'surfbeat', 'break': {'model_type': 'roelvink1', 'alpha': 1.0}}}
            becomes:
            {'wavemodel': 'surfbeat', 'break': 'roelvink1', 'alpha': 1.0}

        """
        data = serializer(self)

        def flatten_nested_dicts(d: dict) -> dict:
            """Recursively flatten nested dictionaries with model_type discriminators."""
            result = {}

            for key, value in d.items():
                if isinstance(value, dict) and "model_type" in value:
                    # This is a nested component - flatten it recursively
                    nested = value.copy()
                    model_type = nested.pop("model_type")

                    # Set the outer key to the model_type value
                    result[key] = model_type

                    # Recursively flatten any nested dicts within this component
                    flattened_nested = flatten_nested_dicts(nested)

                    # Merge the flattened nested values
                    result.update(flattened_nested)
                elif isinstance(value, dict):
                    # Dict without model_type - also flatten it recursively
                    # This handles nested components like wavfric inside break
                    flattened_nested = flatten_nested_dicts(value)
                    result.update(flattened_nested)
                else:
                    # Not a dict, keep as-is
                    result[key] = value

            return result

        # Flatten all nested dictionaries recursively
        data = flatten_nested_dicts(data)

        # Convert booleans to integers
        for key, value in list(data.items()):
            if isinstance(value, bool):
                data[key] = int(value)

        return data

    @property
    def params(self) -> dict:
        """Return the XBeach parameters as a flat dictionary."""
        # Identify XBeachDataBlob fields to exclude from params
        datablob_fields = []
        for field_name in self.model_fields_set:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, XBeachDataBlob):
                datablob_fields.append(field_name)

        return self.model_dump(
            exclude=["model_type"] + datablob_fields,
            exclude_none=True,
            by_alias=True,
        )

    def get(self, destdir: str | Path) -> dict:
        """Return the params dict with file fetching for nested components.

        This method handles two types of nested components:
        1. Components with explicit parameter fields: Need special handling to extract
           the field value and merge remaining params
        2. Discriminated union components: Already flattened by the serializer

        The default implementation:
        - For components with explicit fields: includes that field in return dict
        - For discriminated unions or leaf components: returns standard params
        - For parent components: processes XBeachBaseModel children recursively

        Override this method if you need custom file fetching logic (e.g., DataBlob).

        Parameters
        ----------
        destdir: str | Path
            Directory path for file operations.

        Returns
        -------
        Flat dictionary of XBeach parameters.

        """
        # Identify XBeachBaseModel child components
        # Three types:
        # 1. Discriminated unions (have model_type) - handled by serializer
        # 2. Components with explicit fields (e.g., Roller with 'roller') - need extraction
        # 3. Components without explicit fields (e.g., ShortWaveFriction) - just merge params
        components_with_explicit_fields = {}
        components_without_explicit_fields = {}

        for field_name in self.model_fields_set:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, XBeachBaseModel):
                # Check if this is a discriminated union (has model_type field)
                if hasattr(field_value, "model_type") and not isinstance(
                    getattr(field_value, "model_type", None), bool
                ):
                    # Discriminated union - will be handled by serializer
                    continue
                # Check if the component has a field matching the parent field name
                elif hasattr(field_value, field_name):
                    components_with_explicit_fields[field_name] = field_value
                else:
                    # Component without explicit field - just merge its params
                    components_without_explicit_fields[field_name] = field_value

        # If this component has child components, process them
        if components_with_explicit_fields or components_without_explicit_fields:
            # Serialize own fields, excluding child components
            # Discriminated unions are included and flattened by the serializer
            all_components = list(components_with_explicit_fields.keys()) + list(
                components_without_explicit_fields.keys()
            )
            params = self.model_dump(
                exclude=["model_type"] + all_components,
                exclude_none=True,
                exclude_unset=True,
                by_alias=True,
            )

            # Process components with explicit fields
            for field_name, field_value in components_with_explicit_fields.items():
                # Get the component's params
                component_params = field_value.get(destdir)

                # Extract the explicit field value and set it in parent
                if field_name in component_params:
                    field_param_value = component_params.pop(field_name)
                    # Convert boolean to integer for XBeach
                    if isinstance(field_param_value, bool):
                        params[field_name] = int(field_param_value)
                    else:
                        params[field_name] = field_param_value

                # Merge remaining component params
                params.update(component_params)

            # Process components without explicit fields - just merge their params
            for field_name, field_value in components_without_explicit_fields.items():
                component_params = field_value.get(destdir)
                params.update(component_params)

            return params

        # No child components - return all params including defaults
        # (don't use exclude_unset so explicit fields like 'roller' are included)
        return self.model_dump(
            exclude=["model_type"],
            exclude_none=True,
            by_alias=True,
        )


class XBeachBaseConfig(BaseConfig):
    """Base configuration class for all XBeach models."""

    model_config = ConfigDict(extra="forbid")


class WbcEnum(str, Enum):
    """Valid options for wbctype.

    Attributes
    ----------
    PARAMS: "params"
        Wave boundary conditions specified as a constant value.
    JONS: "jons"
        Wave boundary conditions specified as a single Jonswap spectrum.
    JONSTABLE: "jonstable"
        Wave boundary conditions specified as a time-series of wave parameters.
    SWAN: "swan"
        Wave boundary conditions specified as a SWAN 2D spectrum file.
    VARDENS: "vardens"
        Wave boundary conditions specified as a general spectrum file.
    TS_1: "ts_1"
        Wave boundary conditions specified as a variation in time of wave energy (first-order).
    TS_2: "ts_2"
        Wave boundary conditions specified as a variation in time of wave energy (second-order).
    TS_NONH: "ts_nonh"
        Wave boundary conditions specified as a variation in time of the horizontal
        velocity, vertical velocity and the free surface elevation.
    REUSE: "reuse"
        Wave boundary conditions specified from a previous run.
    OFF: "off"
        No wave boundary conditions.

    """

    PARAMS = "params"
    JONS = "jons"
    JONSTABLE = "jonstable"
    SWAN = "swan"
    VARDENS = "vardens"
    TS_1 = "ts_1"
    TS_2 = "ts_2"
    TS_NONH = "ts_nonh"
    REUSE = "reuse"
    OFF = "off"


class OutputVarsEnum(str, Enum):
    """XBeach output variables.

    Valid options for output variables that can be specified in meanvars,
    pointvars, globalvars, etc. Based on XBeach documentation:
    https://xbeach.readthedocs.io/en/latest/output_variables.html

    Attributes
    ----------
    AS: "As"
        Asymmetry of short waves.
    BR: "BR"
        Maximum wave surface slope used in roller dissipation formulation.
    CDRAG: "Cdrag"
        Vegetation drag coefficient.
    D: "D"
        Dissipation (W/m2).
    D15: "D15"
        D15 grain diameters for all sediment classes (m).
    D50: "D50"
        D50 grain diameters for all sediment classes (m).
    D50TOP: "D50top"
        Friction coefficient flow.
    D90: "D90"
        D90 grain diameters for all sediment classes (m).
    D90TOP: "D90top"
        Friction coefficient flow.
    DR: "DR"
        Roller energy dissipation (W/m2).
    DC: "Dc"
        Diffusion coefficient (m2/s).
    DF: "Df"
        Dissipation rate due to bed friction (W/m2).
    DP: "Dp"
        Dissipation rate in the swash due to transformation of kinetic wave
        energy to potential wave energy (W/m2).
    DVEG: "Dveg"
        Dissipation due to short wave attenuation by vegetation (W/m2).
    E: "E"
        Wave energy (Nm/m2).
    FVEGU: "Fvegu"
        X-forcing due to long wave attenuation by vegetation (N/m2).
    FVEGV: "Fvegv"
        Y-forcing due to long wave attenuation by vegetation (N/m2).
    FX: "Fx"
        Wave force, x-component (N/m2).
    FY: "Fy"
        Wave force, y-component (N/m2).
    H: "H"
        Hrms wave height based on instantaneous wave energy (m).
    HRUNUP: "Hrunup"
        Short wave height used in runup formulation (m).
    L1: "L1"
        Wave length used in dispersion relation (m).
    QB: "Qb"
        Fraction breaking waves.
    R: "R"
        Roller energy (Nm/m2).
    SK: "Sk"
        Skewness of short waves.
    SUBG: "Subg"
        Bed sediment transport for each sediment class (excluding pores),
        x-component (m2/s).
    SUSG: "Susg"
        Suspended sediment transport for each sediment class
        (excluding pores), x-component (m2/s).
    SUTOT: "Sutot"
        Sediment transport integrated over bed load and suspended and for
        all sediment grains, x-component (m2/s).
    SVBG: "Svbg"
        Bed sediment transport for each sediment class (excluding pores),
        y-component (m2/s).
    SVSG: "Svsg"
        Suspended sediment transport for each sediment class
        (excluding pores), y-component (m2/s).
    SVTOT: "Svtot"
        Sediment transport integrated over bed load and suspended and for
        all sediment grains, y-component (m2/s).
    SXX: "Sxx"
        Radiation stress, x-component (N/m).
    SXY: "Sxy"
        Radiation stress, xy-component (N/m).
    SYY: "Syy"
        Radiation stress, y-component (N/m).
    TBORE: "Tbore"
        Wave period interval associated with breaking induced
        turbulence (s).
    TSG: "Tsg"
        Sediment response time for each sediment class (s).
    ALFAU: "alfau"
        Grid orientation at u-point (rad).
    ALFAV: "alfav"
        Grid orientation at v-point (rad).
    ALFAZ: "alfaz"
        Grid orientation at z-point (rad).
    BEDFRICCOEF: "bedfriccoef"
        Dimensional/dimensionless input bed friction coefficient.
    BI: "bi"
        Incoming bound long wave (m).
    BREAKING: "breaking"
        Indicator whether cell has breaking nonh waves.
    BWALPHA: "bwalpha"
        Beachwizard weighting factor.
    C: "c"
        Wave celerity (m/s).
    CA: "ca"
        Reference concentration (m3/m3).
    CCG: "ccg"
        Depth-averaged suspended concentration for each sediment
        fraction (m3/m3).
    CCTOT: "cctot"
        Sediment concentration integrated over bed load and suspended and
        for all sediment grains (m3/m3).
    CCZ: "ccz"
        Concentration profile (m3/m3).
    CEQBG: "ceqbg"
        Depth-averaged bed equilibrium concentration for each sediment
        class (m3/m3).
    CEQSG: "ceqsg"
        Depth-averaged suspended equilibrium concentration for each
        sediment class (m3/m3).
    CF: "cf"
        Friction coefficient flow.
    CFU: "cfu"
        Friction coefficient flow in u-points.
    CFV: "cfv"
        Friction coefficient flow in v-points.
    CG: "cg"
        Group velocity (m/s).
    CGX: "cgx"
        Group velocity, x-component (m/s).
    CGX_S: "cgx_s"
        Group velocity, x-component (m/s).
    CGY: "cgy"
        Group velocity, y-component (m/s).
    CGY_S: "cgy_s"
        Group velocity, y-component (m/s).
    COBS: "cobs"
        Beachwizard observed wave celerity (m/s).
    COSTH: "costh"
        Cos of wave angles relative to grid direction.
    COSTH_S: "costh_s"
        Cos of wave angles relative to grid direction.
    CTHETA: "ctheta"
        Wave celerity theta-direction, refraction (rad/s).
    CTHETA_S: "ctheta_s"
        Wave celerity theta-direction, refraction (rad/s).
    CX: "cx"
        Wave celerity, x-component (m/s).
    CY: "cy"
        Wave celerity, y-component (m/s).
    DU: "dU"
        U-velocity difference between two vertical layers, reduced 2-layer
        non-hydrostatic model (m2/s2).
    DUI: "dUi"
        Velocity difference at boundary due to short waves (m/s).
    DV: "dV"
        V-velocity difference between two vertical layers, reduced 2-layer
        non-hydrostatic model (m2/s2).
    DASSIM: "dassim"
        Beachwizard depth change (m).
    DCBDX: "dcbdx"
        Bed concentration gradient x-direction (kg/m3/m).
    DCBDY: "dcbdy"
        Bed concentration gradient y-direction (kg/m3/m).
    DCMDO: "dcmdo"
        Beachwizard computed minus observed dissipation (W/m2).
    DCSDX: "dcsdx"
        Suspended concentration gradient x-direction (kg/m3/m).
    DCSDY: "dcsdy"
        Suspended concentration gradient y-direction (kg/m3/m).
    DEPO_EX: "depo_ex"
        Explicit bed deposition rate per fraction (m/s).
    DEPO_IM: "depo_im"
        Implicit bed deposition rate per fraction (m/s).
    DINFIL: "dinfil"
        Infiltration layer depth used in quasi-vertical flow model for
        groundwater (m).
    DNC: "dnc"
        Grid distance in n-direction, centered around c-point (m).
    DNU: "dnu"
        Grid distance in n-direction, centered around u-point (m).
    DNV: "dnv"
        Grid distance in n-direction, centered around v-point (m).
    DNZ: "dnz"
        Grid distance in n-direction, centered around z-point (m).
    DOBS: "dobs"
        Beachwizard observed dissipation (W/m2).
    DSC: "dsc"
        Grid distance in s-direction, centered around c-point (m).
    DSDNUI: "dsdnui"
        Inverse of grid cell surface, centered around u-point (1/m2).
    DSDNVI: "dsdnvi"
        Inverse of grid cell surface, centered around v-point (1/m2).
    DSDNZI: "dsdnzi"
        Inverse of grid cell surface, centered around z-point (1/m2).
    DSU: "dsu"
        Grid distance in s-direction, centered around u-point (m).
    DSV: "dsv"
        Grid distance in s-direction, centered around v-point (m).
    DSZ: "dsz"
        Grid distance in s-direction, centered around z-point (m).
    DZAV: "dzav"
        Total bed level change due to avalanching (m).
    DZBDT: "dzbdt"
        Rate of change bed level (m/s).
    DZBDX: "dzbdx"
        Bed level gradient in x-direction.
    DZBDY: "dzbdy"
        Bed level gradient in y-direction.
    DZBED: "dzbed"
        No description.
    DZBNOW: "dzbnow"
        Bed level change in current time step (m).
    DZS0DN: "dzs0dn"
        Alongshore water level gradient due to tide alone.
    DZSDT: "dzsdt"
        Rate of change water level (m/s).
    DZSDX: "dzsdx"
        Water surface gradient in x-direction (m/s).
    DZSDY: "dzsdy"
        Water surface gradient in y-direction (m/s).
    EE: "ee"
        Directionally distributed wave energy (J/m2/rad).
    EE_S: "ee_s"
        Directionally distributed wave energy (J/m2/rad).
    ERO: "ero"
        Bed erosion rate per fraction (m/s).
    FW: "fw"
        Wave friction coefficient.
    GW0BACK: "gw0back"
        Boundary condition back boundary for groundwater head (m).
    GWBOTTOM: "gwbottom"
        Level of the bottom of the aquifer (m).
    GWCURV: "gwcurv"
        Curvature coefficient of groundwater head function.
    GWHEAD: "gwhead"
        Groundwater head, differs from gwlevel (m).
    GWHEADB: "gwheadb"
        Groundwater head at bottom, differs from gwlevel (m).
    GWHEIGHT: "gwheight"
        Vertical size of aquifer through which groundwater can flow (m).
    GWLEVEL: "gwlevel"
        Groundwater table, min(zb,gwhead) (m).
    GWQX: "gwqx"
        Groundwater discharge in x-direction (m/s).
    GWQY: "gwqy"
        Groundwater discharge in y-direction (m/s).
    GWU: "gwu"
        Groundwater flow in x-direction (m/s).
    GWV: "gwv"
        Groundwater flow in y-direction (m/s).
    GWW: "gww"
        Groundwater flow in z-direction, interaction between surface and
        ground water (m/s).
    HH: "hh"
        Water depth (m).
    HHW: "hhw"
        Water depth used in all wave computations, includes
        h*par%delta (m).
    HHWCINS: "hhwcins"
        Water depth used in wave instationary computation in case of
        wci (m).
    HHWS: "hhws"
        Water depth used in wave stationary computation and single_dir
        wave directions (m).
    HOLD: "hold"
        Water depth previous time step (m).
    HU: "hu"
        Water depth in u-points (m).
    HUM: "hum"
        Water depth in u-points (m).
    HV: "hv"
        Water depth in v-points (m).
    HVM: "hvm"
        Water depth in v-points (m).
    IDRIFT: "idrift"
        Drifter x-coordinate in grid space.
    INFIL: "infil"
        Rate of exchange of water between surface and groundwater,
        positive from sea to groundwater (m/s).
    ISTRUCT: "istruct"
        Location of revetments toe.
    IWL: "iwl"
        Location of water line including long wave runup.
    JDRIFT: "jdrift"
        Drifter y-coordinate in grid space.
    K: "k"
        Wave number (rad/m).
    KB: "kb"
        Near bed turbulence intensity due to depth induced
        breaking (m2/s2).
    KTURB: "kturb"
        Depth averaged turbulence intensity due to long wave
        breaking (m2/s2).
    MAXZS: "maxzs"
        Maximum elevation in simulation (m).
    MINZS: "minzs"
        Minimum elevation in simulation (m).
    N: "n"
        Ratio group velocity/wave celerity.
    ND: "nd"
        Number of bed layers, can be different for each computational
        cell.
    NDIST: "ndist"
        Cumulative distance from right boundary along n-direction (m).
    NUH: "nuh"
        Horizontal viscosity coefficient (m2/s).
    NUTZ: "nutz"
        Turbulence viscosity.
    PBBED: "pbbed"
        No description.
    PDISCH: "pdisch"
        Discharge locations.
    PH: "ph"
        Pressure head due to ship (m).
    PNTDISCH: "pntdisch"
        Point discharge locations, no momentum.
    PRES: "pres"
        Normalized dynamic pressure (m2/s2).
    QDISCH: "qdisch"
        Discharges (m2/s).
    QX: "qx"
        Discharge in u-points, x-component (m2/s).
    QY: "qy"
        Discharge in u-points, y-component (m2/s).
    REFA: "refA"
        Reference level (m).
    ROLTHICK: "rolthick"
        Long wave roller thickness (m).
    RR: "rr"
        Directionally distributed roller energy (J/m2/rad).
    RUNUP: "runup"
        Short wave runup height (m).
    SDIST: "sdist"
        Cumulative distance from offshore boundary along s-direction (m).
    SEDCAL: "sedcal"
        Equilibrium sediment concentration factor for each sediment class.
    SEDERO: "sedero"
        Cumulative sedimentation/erosion (m).
    SETBATHY: "setbathy"
        Prescribed bed levels (m).
    SHIPFX: "shipFx"
        Force on ship in x-direction (N).
    SHIPFY: "shipFy"
        Force on ship in y-direction (N).
    SHIPFZ: "shipFz"
        Force on ship in z-direction (N).
    SHIPMX: "shipMx"
        Moment on ship around x-axis (Nm).
    SHIPMY: "shipMy"
        Moment on ship around y-axis (Nm).
    SHIPMZ: "shipMz"
        Moment on ship around z-axis (Nm).
    SHIPCHI: "shipchi"
        Turning angle around y-axis (deg).
    SHIPPHI: "shipphi"
        Turning angle around x-axis (deg).
    SHIPPSI: "shippsi"
        Turning angle around z-axis (deg).
    SHIPXCG: "shipxCG"
        X-coordinate of ship center of gravity (m).
    SHIPYCG: "shipyCG"
        Y-coordinate of ship center of gravity (m).
    SHIPZCG: "shipzCG"
        Z-coordinate of ship center of gravity (m).
    SHOBS: "shobs"
        Beachwizard observed shoreline (m).
    SIG2PRIOR: "sig2prior"
        Beachwizard prior std squared (m2).
    SIGM: "sigm"
        Mean frequency (rad/s).
    SIGT: "sigt"
        Relative frequency (rad/s).
    SIGZ: "sigz"
        Vertical distribution of sigma layers q3d.
    SINTH: "sinth"
        Sin of wave angles relative to grid direction.
    SINTH_S: "sinth_s"
        Sin of wave angles relative to grid direction.
    STRUCSLOPE: "strucslope"
        Slope of structure.
    STRUCTDEPTH: "structdepth"
        Depth of structure in relation to instantaneous bed level (m).
    TAUBX: "taubx"
        Bed shear stress, x-component (N/m2).
    TAUBX_ADD: "taubx_add"
        Additional bed shear stress due to boundary layer effects,
        x-component (N/m2).
    TAUBY: "tauby"
        Bed shear stress, y-component (N/m2).
    TAUBY_ADD: "tauby_add"
        Additional bed shear stress due to boundary layer effects,
        y-component (N/m2).
    TDISCH: "tdisch"
        Discharge time series.
    TDRIFTB: "tdriftb"
        Drifter release time (s).
    TDRIFTE: "tdrifte"
        Drifter retrieval time (s).
    THET: "thet"
        Wave angles (rad).
    THET_S: "thet_s"
        Wave angles (rad).
    THETA: "theta"
        Wave angles directional distribution w.r.t. computational
        x-axis (rad).
    THETA_S: "theta_s"
        Wave angles directional distribution w.r.t. computational
        x-axis (rad).
    THETAMEAN: "thetamean"
        Mean wave angle (rad).
    TIDEINPT: "tideinpt"
        Input time of input tidal signal (s).
    TIDEINPZ: "tideinpz"
        Input tidal signal (m).
    TSETBATHY: "tsetbathy"
        Points in time of prescribed bed levels (s).
    U: "u"
        GLM velocity in cell centre, x-component (m/s).
    UA: "ua"
        Time averaged flow velocity due to wave asymmetry (m/s).
    UCRCAL: "ucrcal"
        Calibration factor for u critical for each sediment class.
    UDUDX: "ududx"
        Advection (m2/s2).
    UDVDX: "udvdx"
        Advection (m2/s2).
    UE: "ue"
        Eulerian velocity in cell centre, x-component (m/s).
    UE_SED: "ue_sed"
        Advection velocity sediment in cell centre, x-component (m/s).
    UEU: "ueu"
        Eulerian velocity in u-points, x-component (m/s).
    UI: "ui"
        Incident bound wave velocity, x-component (m/s).
    UMEAN: "umean"
        Longterm mean velocity at boundaries in u-points,
        x-component (m/s).
    UMWCI: "umwci"
        Velocity time-averaged for wci, x-component (m/s).
    UR: "ur"
        Reflected velocity at boundaries in u-points (m/s).
    UREPB: "urepb"
        Representative flow velocity for sediment advection and diffusion,
        x-component (m/s).
    UREPS: "ureps"
        Representative flow velocity for sediment advection and diffusion,
        x-component (m/s).
    URMS: "urms"
        Orbital velocity (m/s).
    USD: "usd"
        Return flow due to roller after breaker delay (m/s).
    UST: "ust"
        Stokes drift (m/s).
    USTR: "ustr"
        Return flow due to roller (m/s).
    USTZ: "ustz"
        Stokes velocity q3d.
    UU: "uu"
        GLM velocity in u-points, x-component (m/s).
    UV: "uv"
        GLM velocity in v-points, x-component (m/s).
    UWCINS: "uwcins"
        U-velocity used in wave stationary computation in case of
        wci (m/s).
    UWF: "uwf"
        Stokes drift, x-component (m/s).
    UWS: "uws"
        U-velocity used in wave stationary computation and single_dir
        wave directions (m/s).
    UZ: "uz"
        Velocity q3d ksi-component.
    V: "v"
        GLM velocity in cell centre, y-component (m/s).
    VDUDY: "vdudy"
        Advection (m2/s2).
    VDVDY: "vdvdy"
        Advection (m2/s2).
    VE: "ve"
        Eulerian velocity in cell centre, y-component (m/s).
    VE_SED: "ve_sed"
        Advection velocity sediment in cell centre, y-component (m/s).
    VEGTYPE: "vegtype"
        Vegetation type index.
    VEV: "vev"
        Eulerian velocity in v-points, y-component (m/s).
    VI: "vi"
        Incident bound wave velocity, y-component (m/s).
    VISCU: "viscu"
        Viscosity (m2/s2).
    VISCV: "viscv"
        Viscosity (m2/s2).
    VMAG: "vmag"
        Velocity magnitude in cell centre (m/s).
    VMAGEU: "vmageu"
        Eulerian velocity magnitude u-points (m/s).
    VMAGEV: "vmagev"
        Eulerian velocity magnitude v-points (m/s).
    VMAGU: "vmagu"
        GLM velocity magnitude u-points (m/s).
    VMAGV: "vmagv"
        GLM velocity magnitude v-points (m/s).
    VMEAN: "vmean"
        Longterm mean velocity at boundaries in v-points,
        y-component (m/s).
    VMWCI: "vmwci"
        Velocity time-averaged for wci, y-component (m/s).
    VREPB: "vrepb"
        Representative flow velocity for sediment advection and diffusion,
        y-component (m/s).
    VREPS: "vreps"
        Representative flow velocity for sediment advection and diffusion,
        y-component (m/s).
    VU: "vu"
        GLM velocity in u-points, y-component (m/s).
    VV: "vv"
        GLM velocity in v-points, y-component (m/s).
    VWCINS: "vwcins"
        V-velocity used in wave stationary computation in case of
        wci (m/s).
    VWF: "vwf"
        Stokes drift, y-component (m/s).
    VWS: "vws"
        V-velocity used in wave stationary computation and single_dir
        wave directions (m/s).
    VZ: "vz"
        Velocity q3d eta-component.
    WB: "wb"
        Vertical velocity at the bottom (m/s).
    WETE: "wete"
        Mask wet/dry wave-points.
    WETU: "wetu"
        Mask wet/dry u-points.
    WETV: "wetv"
        Mask wet/dry v-points.
    WETZ: "wetz"
        Mask wet/dry eta-points.
    WI: "wi"
        Vertical velocity at boundary due to short waves (m/s).
    WINDDIRTS: "winddirts"
        Input wind direction (deg_nautical).
    WINDINPT: "windinpt"
        Input time of input wind signal (s).
    WINDNV: "windnv"
        Wind velocity in n direction in v point at current time
        step (m/s).
    WINDSU: "windsu"
        Wind velocity in s direction in u point at current time
        step (m/s).
    WINDVELTS: "windvelts"
        Input wind velocity (m/s).
    WINDXTS: "windxts"
        Time series of input wind velocity, not s direction,
        x-component (m/s).
    WINDYTS: "windyts"
        Time series of input wind velocity, not n direction,
        y-component (m/s).
    WM: "wm"
        Mean absolute frequency (rad/s).
    WS: "ws"
        Vertical velocity at the free surface (m/s).
    WSCRIT: "wscrit"
        Critical vertical velocity at the free surface for
        breaking (m/s).
    X: "x"
        X-coordinate original computational grid (m).
    XHRUNUP: "xhrunup"
        Location at which short wave height for runup is taken (m).
    XU: "xu"
        X-coordinate computational grid u-points (m).
    XV: "xv"
        X-coordinate computational grid v-points (m).
    XYZS01: "xyzs01"
        Global xy coordinates of corner (x=1,y=1).
    XYZS02: "xyzs02"
        Global xy coordinates of corner (x=1,y=n).
    XYZS03: "xyzs03"
        Global xy coordinates of corner (x=n,y=n).
    XYZS04: "xyzs04"
        Global xy coordinates of corner (x=n,y=1).
    XZ: "xz"
        X-coordinate computational grid, positive shoreward,
        perpendicular to coastline (m).
    Y: "y"
        Y-coordinate original computational grid (m).
    YU: "yu"
        Y-coordinate computational grid u-points (m).
    YV: "yv"
        Y-coordinate computational grid v-points (m).
    YZ: "yz"
        Y-coordinate computational grid (m).
    Z0BED: "z0bed"
        No description.
    ZB: "zb"
        Bed level (m).
    ZB0: "zb0"
        Initial bed level (m).
    ZBOBS: "zbobs"
        Beachwizard observed depth (m).
    ZI: "zi"
        Surface elevation at boundary due to short waves (m).
    ZS: "zs"
        Water level (m).
    ZS0: "zs0"
        Water level due to tide alone (m).
    ZS0FAC: "zs0fac"
        Relative weight of offshore boundary and bay boundary for each
        grid point.
    ZS1: "zs1"
        Water level minus tide (m).
    ZSWCI: "zswci"
        Waterlevel time-averaged for wci (m).

    """

    AS = "As"
    BR = "BR"
    CDRAG = "Cdrag"
    D = "D"
    D15 = "D15"
    D50 = "D50"
    D50TOP = "D50top"
    D90 = "D90"
    D90TOP = "D90top"
    DR = "DR"
    DC = "Dc"
    DF = "Df"
    DP = "Dp"
    DVEG = "Dveg"
    E = "E"
    FVEGU = "Fvegu"
    FVEGV = "Fvegv"
    FX = "Fx"
    FY = "Fy"
    H = "H"
    HRUNUP = "Hrunup"
    L1 = "L1"
    QB = "Qb"
    R = "R"
    SK = "Sk"
    SUBG = "Subg"
    SUSG = "Susg"
    SUTOT = "Sutot"
    SVBG = "Svbg"
    SVSG = "Svsg"
    SVTOT = "Svtot"
    SXX = "Sxx"
    SXY = "Sxy"
    SYY = "Syy"
    TBORE = "Tbore"
    TSG = "Tsg"
    ALFAU = "alfau"
    ALFAV = "alfav"
    ALFAZ = "alfaz"
    BEDFRICCOEF = "bedfriccoef"
    BI = "bi"
    BREAKING = "breaking"
    BWALPHA = "bwalpha"
    C = "c"
    CA = "ca"
    CCG = "ccg"
    CCTOT = "cctot"
    CCZ = "ccz"
    CEQBG = "ceqbg"
    CEQSG = "ceqsg"
    CF = "cf"
    CFU = "cfu"
    CFV = "cfv"
    CG = "cg"
    CGX = "cgx"
    CGX_S = "cgx_s"
    CGY = "cgy"
    CGY_S = "cgy_s"
    COBS = "cobs"
    COSTH = "costh"
    COSTH_S = "costh_s"
    CTHETA = "ctheta"
    CTHETA_S = "ctheta_s"
    CX = "cx"
    CY = "cy"
    DU = "dU"
    DUI = "dUi"
    DV = "dV"
    DASSIM = "dassim"
    DCBDX = "dcbdx"
    DCBDY = "dcbdy"
    DCMDO = "dcmdo"
    DCSDX = "dcsdx"
    DCSDY = "dcsdy"
    DEPO_EX = "depo_ex"
    DEPO_IM = "depo_im"
    DINFIL = "dinfil"
    DNC = "dnc"
    DNU = "dnu"
    DNV = "dnv"
    DNZ = "dnz"
    DOBS = "dobs"
    DSC = "dsc"
    DSDNUI = "dsdnui"
    DSDNVI = "dsdnvi"
    DSDNZI = "dsdnzi"
    DSU = "dsu"
    DSV = "dsv"
    DSZ = "dsz"
    DZAV = "dzav"
    DZBDT = "dzbdt"
    DZBDX = "dzbdx"
    DZBDY = "dzbdy"
    DZBED = "dzbed"
    DZBNOW = "dzbnow"
    DZS0DN = "dzs0dn"
    DZSDT = "dzsdt"
    DZSDX = "dzsdx"
    DZSDY = "dzsdy"
    EE = "ee"
    EE_S = "ee_s"
    ERO = "ero"
    FW = "fw"
    GW0BACK = "gw0back"
    GWBOTTOM = "gwbottom"
    GWCURV = "gwcurv"
    GWHEAD = "gwhead"
    GWHEADB = "gwheadb"
    GWHEIGHT = "gwheight"
    GWLEVEL = "gwlevel"
    GWQX = "gwqx"
    GWQY = "gwqy"
    GWU = "gwu"
    GWV = "gwv"
    GWW = "gww"
    HH = "hh"
    HHW = "hhw"
    HHWCINS = "hhwcins"
    HHWS = "hhws"
    HOLD = "hold"
    HU = "hu"
    HUM = "hum"
    HV = "hv"
    HVM = "hvm"
    IDRIFT = "idrift"
    INFIL = "infil"
    ISTRUCT = "istruct"
    IWL = "iwl"
    JDRIFT = "jdrift"
    K = "k"
    KB = "kb"
    KTURB = "kturb"
    MAXZS = "maxzs"
    MINZS = "minzs"
    N = "n"
    ND = "nd"
    NDIST = "ndist"
    NUH = "nuh"
    NUTZ = "nutz"
    PBBED = "pbbed"
    PDISCH = "pdisch"
    PH = "ph"
    PNTDISCH = "pntdisch"
    PRES = "pres"
    QDISCH = "qdisch"
    QX = "qx"
    QY = "qy"
    REFA = "refA"
    ROLTHICK = "rolthick"
    RR = "rr"
    RUNUP = "runup"
    SDIST = "sdist"
    SEDCAL = "sedcal"
    SEDERO = "sedero"
    SETBATHY = "setbathy"
    SHIPFX = "shipFx"
    SHIPFY = "shipFy"
    SHIPFZ = "shipFz"
    SHIPMX = "shipMx"
    SHIPMY = "shipMy"
    SHIPMZ = "shipMz"
    SHIPCHI = "shipchi"
    SHIPPHI = "shipphi"
    SHIPPSI = "shippsi"
    SHIPXCG = "shipxCG"
    SHIPYCG = "shipyCG"
    SHIPZCG = "shipzCG"
    SHOBS = "shobs"
    SIG2PRIOR = "sig2prior"
    SIGM = "sigm"
    SIGT = "sigt"
    SIGZ = "sigz"
    SINTH = "sinth"
    SINTH_S = "sinth_s"
    STRUCSLOPE = "strucslope"
    STRUCTDEPTH = "structdepth"
    TAUBX = "taubx"
    TAUBX_ADD = "taubx_add"
    TAUBY = "tauby"
    TAUBY_ADD = "tauby_add"
    TDISCH = "tdisch"
    TDRIFTB = "tdriftb"
    TDRIFTE = "tdrifte"
    THET = "thet"
    THET_S = "thet_s"
    THETA = "theta"
    THETA_S = "theta_s"
    THETAMEAN = "thetamean"
    TIDEINPT = "tideinpt"
    TIDEINPZ = "tideinpz"
    TSETBATHY = "tsetbathy"
    U = "u"
    UA = "ua"
    UCRCAL = "ucrcal"
    UDUDX = "ududx"
    UDVDX = "udvdx"
    UE = "ue"
    UE_SED = "ue_sed"
    UEU = "ueu"
    UI = "ui"
    UMEAN = "umean"
    UMWCI = "umwci"
    UR = "ur"
    UREPB = "urepb"
    UREPS = "ureps"
    URMS = "urms"
    USD = "usd"
    UST = "ust"
    USTR = "ustr"
    USTZ = "ustz"
    UU = "uu"
    UV = "uv"
    UWCINS = "uwcins"
    UWF = "uwf"
    UWS = "uws"
    UZ = "uz"
    V = "v"
    VDUDY = "vdudy"
    VDVDY = "vdvdy"
    VE = "ve"
    VE_SED = "ve_sed"
    VEGTYPE = "vegtype"
    VEV = "vev"
    VI = "vi"
    VISCU = "viscu"
    VISCV = "viscv"
    VMAG = "vmag"
    VMAGEU = "vmageu"
    VMAGEV = "vmagev"
    VMAGU = "vmagu"
    VMAGV = "vmagv"
    VMEAN = "vmean"
    VMWCI = "vmwci"
    VREPB = "vrepb"
    VREPS = "vreps"
    VU = "vu"
    VV = "vv"
    VWCINS = "vwcins"
    VWF = "vwf"
    VWS = "vws"
    VZ = "vz"
    WB = "wb"
    WETE = "wete"
    WETU = "wetu"
    WETV = "wetv"
    WETZ = "wetz"
    WI = "wi"
    WINDDIRTS = "winddirts"
    WINDINPT = "windinpt"
    WINDNV = "windnv"
    WINDSU = "windsu"
    WINDVELTS = "windvelts"
    WINDXTS = "windxts"
    WINDYTS = "windyts"
    WM = "wm"
    WS = "ws"
    WSCRIT = "wscrit"
    X = "x"
    XHRUNUP = "xhrunup"
    XU = "xu"
    XV = "xv"
    XYZS01 = "xyzs01"
    XYZS02 = "xyzs02"
    XYZS03 = "xyzs03"
    XYZS04 = "xyzs04"
    XZ = "xz"
    Y = "y"
    YU = "yu"
    YV = "yv"
    YZ = "yz"
    Z0BED = "z0bed"
    ZB = "zb"
    ZB0 = "zb0"
    ZBOBS = "zbobs"
    ZI = "zi"
    ZS = "zs"
    ZS0 = "zs0"
    ZS0FAC = "zs0fac"
    ZS1 = "zs1"
    ZSWCI = "zswci"
