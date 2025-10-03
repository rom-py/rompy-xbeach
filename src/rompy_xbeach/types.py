from enum import Enum
from pydantic import ConfigDict

from rompy.core.config import BaseConfig


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
    AS: "as"
        Asymmetry of short waves.
    BR: "br"
        Maximum wave surface slope used in roller dissipation formulation.
    CDRAG: "cdrag"
        Vegetation drag coefficient.
    D: "d"
        Dissipation (W/m2).
    D15: "d15"
        D15 grain diameters for all sediment classes (m).
    D50: "d50"
        D50 grain diameters for all sediment classes (m).
    D50TOP: "d50top"
        Friction coefficient flow.
    D90: "d90"
        D90 grain diameters for all sediment classes (m).
    D90TOP: "d90top"
        Friction coefficient flow.
    DR: "dr"
        Roller energy dissipation (W/m2).
    DC: "dc"
        Diffusion coefficient (m2/s).
    DF: "df"
        Dissipation rate due to bed friction (W/m2).
    DP: "dp"
        Dissipation rate in the swash due to transformation of kinetic wave
        energy to potential wave energy (W/m2).
    DVEG: "dveg"
        Dissipation due to short wave attenuation by vegetation (W/m2).
    E: "e"
        Wave energy (Nm/m2).
    FVEGU: "fvegu"
        X-forcing due to long wave attenuation by vegetation (N/m2).
    FVEGV: "fvegv"
        Y-forcing due to long wave attenuation by vegetation (N/m2).
    FX: "fx"
        Wave force, x-component (N/m2).
    FY: "fy"
        Wave force, y-component (N/m2).
    H: "h"
        Hrms wave height based on instantaneous wave energy (m).
    HRUNUP: "hrunup"
        Short wave height used in runup formulation (m).
    L1: "l1"
        Wave length used in dispersion relation (m).
    QB: "qb"
        Fraction breaking waves.
    R: "r"
        Roller energy (Nm/m2).
    SK: "sk"
        Skewness of short waves.
    SUBG: "subg"
        Bed sediment transport for each sediment class (excluding pores),
        x-component (m2/s).
    SUSG: "susg"
        Suspended sediment transport for each sediment class
        (excluding pores), x-component (m2/s).
    SUTOT: "sutot"
        Sediment transport integrated over bed load and suspended and for
        all sediment grains, x-component (m2/s).
    SVBG: "svbg"
        Bed sediment transport for each sediment class (excluding pores),
        y-component (m2/s).
    SVSG: "svsg"
        Suspended sediment transport for each sediment class
        (excluding pores), y-component (m2/s).
    SVTOT: "svtot"
        Sediment transport integrated over bed load and suspended and for
        all sediment grains, y-component (m2/s).
    SXX: "sxx"
        Radiation stress, x-component (N/m).
    SXY: "sxy"
        Radiation stress, xy-component (N/m).
    SYY: "syy"
        Radiation stress, y-component (N/m).
    TBORE: "tbore"
        Wave period interval associated with breaking induced
        turbulence (s).
    TSG: "tsg"
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
    DU: "du"
        U-velocity difference between two vertical layers, reduced 2-layer
        non-hydrostatic model (m2/s2).
    DUI: "dui"
        Velocity difference at boundary due to short waves (m/s).
    DV: "dv"
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
    REFA: "refa"
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
    SHIPFX: "shipfx"
        Force on ship in x-direction (N).
    SHIPFY: "shipfy"
        Force on ship in y-direction (N).
    SHIPFZ: "shipfz"
        Force on ship in z-direction (N).
    SHIPMX: "shipmx"
        Moment on ship around x-axis (Nm).
    SHIPMY: "shipmy"
        Moment on ship around y-axis (Nm).
    SHIPMZ: "shipmz"
        Moment on ship around z-axis (Nm).
    SHIPCHI: "shipchi"
        Turning angle around y-axis (deg).
    SHIPPHI: "shipphi"
        Turning angle around x-axis (deg).
    SHIPPSI: "shippsi"
        Turning angle around z-axis (deg).
    SHIPXCG: "shipxcg"
        X-coordinate of ship center of gravity (m).
    SHIPYCG: "shipycg"
        Y-coordinate of ship center of gravity (m).
    SHIPZCG: "shipzcg"
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
