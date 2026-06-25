# Parameter Reference

This document maps XBeach parameters to their location in rompy-xbeach components.

The structure follows the **rompy-xbeach component hierarchy**, with references to the corresponding XBeach manual tables. This ensures consistency between the documentation and the actual code structure.

!!! tip "Finding Parameters"
    - **By rompy-xbeach component**: Use the sections below (Physics, Sediment, Output, etc.)
    - **By XBeach table**: Look for the table reference in each section header (e.g., "Table 36")
    - **By XBeach parameter name**: Use your browser's search (Ctrl+F / Cmd+F)

---

## Physics Component

The [`Physics`](../api-reference/components.md#rompy_xbeach.components.physics.Physics) component controls wave models, flow computation, friction, viscosity, and related numerical settings.

### Process Switches

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wavemodel` | [`physics.wavemodel`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wavemodel) | Wave model type (stationary/surfbeat/nonh) |
| `swave` | [`physics.swave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.swave) | Enable short wave action balance |
| `lwave` | [`physics.lwave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.lwave) | Enable long wave propagation |
| `flow` | [`physics.flow`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.flow) | Enable flow computation |
| `avalanching` | [`physics.avalanching`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.avalanching) | Enable avalanching |
| `gwflow` | [`physics.gwflow`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.gwflow) | Enable groundwater flow |
| `wind` | [`physics.wind`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wind) | Enable wind forcing |
| `vegetation` | [`physics.vegetation`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.vegetation) | Enable vegetation effects |
| `ships` | [`physics.ships`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.ships) | Enable ship-induced waves |
| `roller` | [`physics.roller`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.roller) | Enable roller model |
| `wci` | [`physics.wci`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wci) | Wave-current interaction |

### Wave Model Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `break` | [`physics.wavemodel.breaktype`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Surfbeat.breaktype) | Breaker formulation |
| `gamma` | [`physics.wavemodel.breaktype.gamma`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.gamma) | Breaker parameter |
| `alpha` | [`physics.wavemodel.breaktype.alpha`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.alpha) | Wave dissipation coefficient |
| `n` | [`physics.wavemodel.breaktype.n`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.n) | Power in breaker formulation |
| `gammax` | [`physics.wavemodel.breaktype.gammax`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.gammax) | Maximum ratio Hb/hb |
| `single_dir` | [`physics.wavemodel.single_dir`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Surfbeat.single_dir) | Single directional bin (Surfbeat) |
| `nhbreaker` | [`physics.wavemodel.nhbreaker`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.nhbreaker) | Non-hydrostatic breaker (Nonh) |
| `solver` | [`physics.wavemodel.solver`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.solver) | Pressure solver (Nonh) |
| `Topt` | [`physics.wavemodel.Topt`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.Topt) | Optimal timestep (Nonh) |
| `kdmin` | [`physics.wavemodel.kdmin`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.kdmin) | Minimum kd for dispersion (Nonh) |

### Bed Friction Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `bedfriction` | [`physics.bedfriction`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.bedfriction) | Bed friction formulation (cf/chezy/manning/white-colebrook/white-colebrook-grainsize) |
| `bedfriccoef` | [`physics.bedfriction.bedfriccoef`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.bedfriccoef) | Friction coefficient |
| `bedfricfile` | [`physics.bedfriction.bedfricfile`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.bedfricfile) | Spatially varying friction file |
| `mincf` | [`physics.bedfriction.mincf`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Manning.mincf) | Minimum friction coefficient (Manning/WhiteColebrook) |
| `maxcf` | [`physics.bedfriction.maxcf`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Manning.maxcf) | Maximum friction coefficient (Manning/WhiteColebrook) |
| `friction_acceleration` | [`physics.bedfriction.friction_acceleration`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.friction_acceleration) | Acceleration effect on roughness |
| `friction_infiltration` | [`physics.bedfriction.friction_infiltration`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.friction_infiltration) | Infiltration effect on roughness |
| `friction_turbulence` | [`physics.bedfriction.friction_turbulence`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.friction_turbulence) | Turbulence effect on roughness |
| `gamma_turb` | [`physics.bedfriction.gamma_turb`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.gamma_turb) | Turbulence calibration factor |

### Viscosity Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `nuh` | [`physics.viscosity.nuh`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.nuh) | Horizontal viscosity coefficient |
| `nuhfac` | [`physics.viscosity.nuhfac`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.nuhfac) | Viscosity calibration factor |
| `nuhv` | [`physics.viscosity.nuhv`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.nuhv) | Longshore viscosity enhancement |
| `smag` | [`physics.viscosity.smag`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.smag) | Enable Smagorinsky model |

### Wave Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `scheme` | [`physics.wave_numerics.scheme`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.WaveNumerics.scheme) | Numerical scheme |
| `wavint` | [`physics.wave_numerics.wavint`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.WaveNumerics.wavint) | Wave integration interval |
| `maxerror` | [`physics.wave_numerics.maxerror`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.WaveNumerics.maxerror) | Maximum wave convergence error |
| `maxiter` | [`physics.wave_numerics.maxiter`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.WaveNumerics.maxiter) | Maximum wave iterations |

### Flow Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `cfl` | [`physics.flow_numerics.cfl`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.cfl) | CFL criterion |
| `eps` | [`physics.flow_numerics.eps`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.eps) | Threshold depth |
| `eps_sd` | [`physics.flow_numerics.eps_sd`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.eps_sd) | Threshold velocity difference |
| `epsi` | [`physics.flow_numerics.epsi`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.epsi) | Mean/varying current ratio |
| `hmin` | [`physics.flow_numerics.hmin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.hmin) | Minimum water depth |
| `deltahmin` | [`physics.flow_numerics.deltahmin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.deltahmin) | Dimensionless min depth coefficient |
| `oldhmin` | [`physics.flow_numerics.oldhmin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.oldhmin) | Use old hmin parameter |
| `umin` | [`physics.flow_numerics.umin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.umin) | Minimum velocity |
| `secorder` | [`physics.flow_numerics.secorder`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.secorder) | Second-order advection |
| `oldhu` | [`physics.flow_numerics.oldhu`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.oldhu) | Old hu/hv formulation |
| `defuse` | [`physics.flow_numerics.defuse`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.defuse) | Enable diffusion in flow solver |
| `dtset` | [`physics.flow_numerics.dtset`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.dtset) | Fixed timestep |
| `maxdtfac` | [`physics.flow_numerics.maxdtfac`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.maxdtfac) | Maximum timestep factor |

### Physical Constants

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `rho` | [`physics.constants.rho`](../api-reference/components.md#rompy_xbeach.components.physics.constants.PhysicalConstants.rho) | Water density |
| `rhoa` | [`physics.constants.rhoa`](../api-reference/components.md#rompy_xbeach.components.physics.constants.PhysicalConstants.rhoa) | Air density |
| `g` | [`physics.constants.g`](../api-reference/components.md#rompy_xbeach.components.physics.constants.PhysicalConstants.g) | Gravitational acceleration |
| `depthscale` | [`physics.constants.depthscale`](../api-reference/components.md#rompy_xbeach.components.physics.constants.PhysicalConstants.depthscale) | Depth scale for lab tests |
| `lat` | [`physics.coriolis.lat`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Coriolis.lat) | Latitude for Coriolis |
| `wearth` | [`physics.coriolis.wearth`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Coriolis.wearth) | Earth angular velocity |

### Vegetation Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `vegetation` | [`physics.vegetation`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.vegetation) | Enable vegetation |
| `nsec` | [`physics.vegetation.nsec`](../api-reference/components.md#rompy_xbeach.components.physics.vegetation.Vegetation.nsec) | Number of vegetation sections |
| `ah` | [`physics.vegetation.ah`](../api-reference/components.md#rompy_xbeach.components.physics.vegetation.Vegetation.ah) | Vegetation height |
| `bv` | [`physics.vegetation.bv`](../api-reference/components.md#rompy_xbeach.components.physics.vegetation.Vegetation.bv) | Stem diameter |
| `Nv` | [`physics.vegetation.Nv`](../api-reference/components.md#rompy_xbeach.components.physics.vegetation.Vegetation.Nv) | Stem density |
| `Cd` | [`physics.vegetation.Cd`](../api-reference/components.md#rompy_xbeach.components.physics.vegetation.Vegetation.Cd) | Drag coefficient |

---

## Sediment Component

The [`Sediment`](../api-reference/components.md#rompy_xbeach.components.sediment.Sediment) component controls sediment transport, morphology, bed composition, and groundwater flow. This corresponds to XBeach Tables 36-41.

!!! note "Process Switches"
    The main switches `sedtrans` and `morphology` are fields on the `Sediment` component, not `Physics`. Set `sediment.sedtrans=True` or provide a `SedimentTransport` object to enable sediment transport.

### Sediment Transport (XBeach Table 36)

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `form` | [`sediment.sedtrans.form`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.form) | Transport formulation |
| `waveform` | [`sediment.sedtrans.waveform`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.waveform) | Wave shape formulation |
| `turb` | [`sediment.sedtrans.turb`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.turb) | Turbulence formulation |
| `turbadv` | [`sediment.sedtrans.turbadv`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.turbadv) | Turbulence advection model |
| `sws` | [`sediment.sedtrans.sws`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.sws) | Short wave stirring |
| `lws` | [`sediment.sedtrans.lws`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.lws) | Long wave stirring |
| `lwt` | [`sediment.sedtrans.lwt`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.lwt) | Long wave turbulence |
| `BRfac` | [`sediment.sedtrans.BRfac`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.BRfac) | Bore runup factor |
| `Tbfac` | [`sediment.sedtrans.Tbfac`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.Tbfac) | Bore interval factor |
| `Tsmin` | [`sediment.sedtrans.Tsmin`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.Tsmin) | Minimum adaptation time |
| `facua` | [`sediment.sedtrans.facua`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.facua) | Onshore transport factor |
| `facAs` | [`sediment.sedtrans.facAs`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.facAs) | Asymmetry factor |
| `facSk` | [`sediment.sedtrans.facSk`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.facSk) | Skewness factor |
| `facsl` | [`sediment.sedtrans.facsl`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.facsl) | Bed slope factor |
| `facDc` | [`sediment.sedtrans.facDc`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.facDc) | Diffusion coefficient factor |
| `bed` | [`sediment.sedtrans.bed`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.bed) | Bed transport calibration |
| `sus` | [`sediment.sedtrans.sus`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.sus) | Suspended transport calibration |
| `bulk` | [`sediment.sedtrans.bulk`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.bulk) | Bulk transport switch |
| `ci` | [`sediment.sedtrans.ci`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.ci) | Mass coefficient (inertia) |
| `cm` | [`sediment.sedtrans.cm`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.cm) | Mass coefficient (shields) |
| `dilatancy` | [`sediment.sedtrans.dilatancy`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.dilatancy) | Dilatancy switch |
| `fallvelred` | [`sediment.sedtrans.fallvelred`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.fallvelred) | Fall velocity reduction |
| `bdslpeffmag` | [`sediment.sedtrans.bdslpeffmag`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.bdslpeffmag) | Bed slope magnitude effect |
| `bdslpeffdir` | [`sediment.sedtrans.bdslpeffdir`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.bdslpeffdir) | Bed slope direction effect |
| `bdslpeffini` | [`sediment.sedtrans.bdslpeffini`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.bdslpeffini) | Bed slope initiation effect |
| `reposeangle` | [`sediment.sedtrans.reposeangle`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.reposeangle) | Angle of internal friction |
| `tsfac` | [`sediment.sedtrans.tsfac`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.tsfac) | Sediment source term factor |
| `z0` | [`sediment.sedtrans.z0`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.SedimentTransport.z0) | Zero velocity level |

### Morphology (XBeach Table 39)

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `morfac` | [`sediment.morphology.morfac`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morfac) | Morphological acceleration |
| `morfacopt` | [`sediment.morphology.morfacopt`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morfacopt) | Adjust output times for morfac |
| `morstart` | [`sediment.morphology.morstart`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstart) | Morphology start time |
| `morstop` | [`sediment.morphology.morstop`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstop) | Morphology stop time |
| `wetslp` | [`sediment.morphology.wetslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.wetslp) | Critical wet slope |
| `dryslp` | [`sediment.morphology.dryslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.dryslp) | Critical dry slope |
| `dzmax` | [`sediment.morphology.dzmax`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.dzmax) | Maximum avalanching change |
| `hswitch` | [`sediment.morphology.hswitch`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.hswitch) | Wet/dry slope switch depth |
| `lsgrad` | [`sediment.morphology.lsgrad`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.lsgrad) | Longshore gradient factor |
| `struct` | [`sediment.morphology.struct`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.struct) | Enable structures |
| `ne_layer` | [`sediment.morphology.ne_layer`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.ne_layer) | Non-erodible layer file |

### Bed Update (XBeach Table 40)

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|--------------|
| `fwfile` | `sediment.bed_update.fwfile` | Wave friction file |
| `fwcutoff` | `sediment.bed_update.fwcutoff` | Wave friction cutoff |

### Bed Composition

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `D50` | [`sediment.bed_composition.D50`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.D50) | Median grain size |
| `D90` | [`sediment.bed_composition.D90`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.D90) | 90th percentile grain size |
| `D15` | [`sediment.bed_composition.D15`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.D15) | 15th percentile grain size |
| `ngd` | [`sediment.bed_composition.ngd`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.ngd) | Number of grain classes |
| `nd` | [`sediment.bed_composition.nd`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.nd) | Number of bed layers |
| `rhos` | [`sediment.bed_composition.rhos`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.rhos) | Sediment density |
| `por` | [`sediment.bed_composition.por`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.por) | Porosity |
| `dzg1` | [`sediment.bed_composition.dzg1`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.dzg1) | Layer 1 thickness |
| `dzg2` | [`sediment.bed_composition.dzg2`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.dzg2) | Layer 2 thickness |
| `dzg3` | [`sediment.bed_composition.dzg3`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.dzg3) | Layer 3 thickness |
| `sedcal` | [`sediment.bed_composition.sedcal`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.sedcal) | Sediment calibration factor |
| `ucrcal` | [`sediment.bed_composition.ucrcal`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.ucrcal) | Critical velocity calibration |
| `ws_nonh` | [`sediment.bed_composition.ws_nonh`](../api-reference/components.md#rompy_xbeach.components.sediment.composition.BedComposition.ws_nonh) | Fall velocity (nonh mode) |

### Transport Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `cmax` | [`sediment.transport_numerics.cmax`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.TransportNumerics.cmax) | Maximum concentration |
| `dtlimts` | [`sediment.transport_numerics.dtlimts`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.TransportNumerics.dtlimts) | Timestep limiter factor |
| `oldTsmin` | [`sediment.transport_numerics.oldTsmin`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.TransportNumerics.oldTsmin) | Use old Tsmin parameter |
| `sourcesink` | [`sediment.transport_numerics.sourcesink`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.TransportNumerics.sourcesink) | Source-sink bed update |
| `thetanum` | [`sediment.transport_numerics.thetanum`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.TransportNumerics.thetanum) | Upwind/central scheme |

### Quasi-3D Transport

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `q3d` | [`sediment.quasi3d.q3d`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.q3d) | Enable quasi-3D transport |
| `kmax` | [`sediment.quasi3d.kmax`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.kmax) | Number of sigma layers |
| `sigfac` | [`sediment.quasi3d.sigfac`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.sigfac) | Layer distribution factor |
| `vicmol` | [`sediment.quasi3d.vicmol`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.vicmol) | Molecular viscosity |
| `vonkar` | [`sediment.quasi3d.vonkar`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.vonkar) | Von Karman constant |
| `rwave` | [`sediment.quasi3d.rwave`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Quasi3D.rwave) | Wave roughness factor |

### Groundwater Flow (XBeach Table 41)

!!! note "Groundwater Switch"
    The main switch `gwflow` is on the `Physics` component (`physics.gwflow`), but the detailed groundwater parameters are on `Sediment.groundwater`.

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|--------------|
| `gwflow` | [`physics.gwflow`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.gwflow) | Enable groundwater flow (on Physics) |
| `gwnonh` | [`sediment.groundwater.gwnonh`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.GroundwaterFlow.gwnonh) | Non-hydrostatic groundwater |
| `kx` | [`sediment.groundwater.kx`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.GroundwaterFlow.kx) | Horizontal permeability |
| `ky` | [`sediment.groundwater.ky`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.GroundwaterFlow.ky) | Vertical permeability |
| `gwheadmodel` | [`sediment.groundwater.gwheadmodel`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.GroundwaterFlow.gwheadmodel) | Head boundary model |

---

## Boundary Conditions

Boundary conditions are configured at the `Config` level, not within Physics or Sediment components.

### Flow Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `front` | [`flow_boundary.front`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.front) | Front boundary type |
| `back` | [`flow_boundary.back`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.back) | Back boundary type |
| `left` | [`flow_boundary.left`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.left) | Left boundary type |
| `right` | [`flow_boundary.right`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.right) | Right boundary type |
| `lateralwave` | [`flow_boundary.lateralwave`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions.lateralwave) | Lateral wave boundary |

### Tide Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `tideloc` | [`tide_boundary.tideloc`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.tideloc) | Number of tide locations |
| `tidetype` | [`tide_boundary.tidetype`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.tidetype) | Tide boundary type |
| `zs0` | [`tide_boundary.zs0`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.zs0) | Initial water level |
| `paulrevere` | [`tide_boundary.paulrevere`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundaryConditions.paulrevere) | Sea/land boundary |

### Wave Boundary Parameters

Wave boundary parameters are specified directly on the wave boundary data classes via `Config.input.wave`. See [Wave Boundaries](../data-interfaces/boundaries.md) for full documentation.

**Common Parameters (All Boundaries):**

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wbctype` | `input.wave` (class determines type) | Wave boundary type |
| `bcfile` | `input.wave.get()` (generated) | Boundary condition file |
| `thetamin` | [`input.wave.thetamin`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Minimum wave direction |
| `thetamax` | [`input.wave.thetamax`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Maximum wave direction |
| `dtheta` | [`input.wave.dtheta`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Directional resolution |
| `thetanaut` | [`input.wave.thetanaut`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Nautical convention |
| `ARC` | [`input.wave.ARC`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Active reflection compensation |
| `freewave` | [`input.wave.freewave`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Free wave boundary |
| `nmax` | [`input.wave.nmax`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Maximum cg/c ratio |
| `taper` | [`input.wave.taper`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Spin-up time |

**Spectral Parameters (Spectral Boundaries Only):**

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `rt` | [`input.wave.rt`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Wave spectrum duration |
| `dtbc` | [`input.wave.dtbc`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Boundary update interval |
| `random` | [`input.wave.random`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Random seed switch |
| `fcutoff` | [`input.wave.fcutoff`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Low-frequency cutoff |
| `correcthm0` | [`input.wave.correcthm0`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Hm0 correction switch |
| `sprdthr` | [`input.wave.sprdthr`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Spreading threshold |
| `trepfac` | [`input.wave.trepfac`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Representative period factor |
| `nspr` | [`input.wave.nspr`](../api-reference/data.md#rompy_xbeach.data.boundary.SpectralWaveBoundaryParams) | Long wave direction switch |
| `wbcRemoveStokes` | [`input.wave.wbcRemoveStokes`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Remove Stokes drift |
| `wbcScaleEnergy` | [`input.wave.wbcScaleEnergy`](../api-reference/data.md#rompy_xbeach.data.boundary.WaveBoundaryParams) | Scale energy to match Hm0 |

---

## Hotstart Component

The [`Hotstart`](../api-reference/components.md#rompy_xbeach.components.hotstart.Hotstart) component enables initialization from a previous simulation state.

### Hotstart Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `hotstart` | [`hotstart`](../api-reference/config.md#rompy_xbeach.config.Config.hotstart) | Enable hotstart |
| `hotstartfileno` | [`hotstart.hotstartfileno`](../api-reference/components.md#rompy_xbeach.components.hotstart.Hotstart.hotstartfileno) | Hotstart file number |
| `writehotstart` | [`output.writehotstart`](../api-reference/components.md#rompy_xbeach.components.output.Output.writehotstart) | Write hotstart files |
| `tinth` | [`output.tinth`](../api-reference/components.md#rompy_xbeach.components.output.Output.tinth) | Hotstart output interval |

---

## Output Component

The [`Output`](../api-reference/components.md#rompy_xbeach.components.output.Output) component controls what variables are written and at what intervals.

### Output Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `outputformat` | [`output.outputformat`](../api-reference/components.md#rompy_xbeach.components.output.Output.outputformat) | Output file format |
| `tintg` | [`output.tintg`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintg) | Global output interval |
| `tintm` | [`output.tintm`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintm) | Mean output interval |
| `tintp` | [`output.tintp`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintp) | Point output interval |
| `tstart` | [`output.tstart`](../api-reference/components.md#rompy_xbeach.components.output.Output.tstart) | Output start time |
| `nglobalvar` | [`output.globalvars`](../api-reference/components.md#rompy_xbeach.components.output.Output.globalvars) | Computed from globalvars list length |
| `nmeanvar` | [`output.meanvars`](../api-reference/components.md#rompy_xbeach.components.output.Output.meanvars) | Computed from meanvars list length |
| `npointvar` | [`output.pointvars`](../api-reference/components.md#rompy_xbeach.components.output.Output.pointvars) | Computed from pointvars list length |
| `nrugauge` | [`output.rugauges`](../api-reference/components.md#rompy_xbeach.components.output.Output.rugauges) | Computed from rugauges list length |
| `nrugdepth` | [`output.nrugdepth`](../api-reference/components.md#rompy_xbeach.components.output.Output.nrugdepth) | Number of runup depths |
| `rugdepth` | [`output.rugdepth`](../api-reference/components.md#rompy_xbeach.components.output.Output.rugdepth) | Runup depth thresholds |

---

## MPI Component

The [`Mpi`](../api-reference/components.md#rompy_xbeach.components.mpi.Mpi) component controls domain decomposition for parallel execution.

### MPI Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `mpiboundary` | [`mpi.mpiboundary`](../api-reference/components.md#rompy_xbeach.components.mpi.Mpi.mpiboundary) | MPI boundary type |
| `mmpi` | [`mpi.mmpi`](../api-reference/components.md#rompy_xbeach.components.mpi.Mpi.mmpi) | MPI partitions in m |
| `nmpi` | [`mpi.nmpi`](../api-reference/components.md#rompy_xbeach.components.mpi.Mpi.nmpi) | MPI partitions in n |

---

## Data-Driven Parameters

These parameters are generated automatically by data interface classes. They should not be set manually - instead, configure the appropriate data source and the parameters will be computed during model generation.

### Grid Parameters

Generated by [`RegularGrid`](../api-reference/types.md#grid) from `grid.py`:

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `nx` | `grid.nx - 1` | Number of computational cells in x-direction |
| `ny` | `grid.ny - 1` | Number of computational cells in y-direction |
| `dx` | `grid.dx` | Grid spacing in x-direction |
| `dy` | `grid.dy` | Grid spacing in y-direction |
| `xori` | `grid.x0` | X-coordinate of grid origin |
| `yori` | `grid.y0` | Y-coordinate of grid origin |
| `alfa` | `grid.alfa` | Angle of x-axis from east (degrees) |
| `projection` | `grid.proj4` | PROJ4 projection string |

### Bathymetry Parameters

Generated by bathymetry data classes:

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `depfile` | `bathy.get()` | Bathymetry file name |
| `posdwn` | `bathy.get()` | Positive down convention (1 or -1) |
| `vardx` | `bathy.get()` | Variable grid spacing flag |

### Water Level / Tide Parameters

Generated by [`WaterLevelGrid`](../api-reference/data.md#tide-water-level), [`TideConsGrid`](../api-reference/data.md#tide-water-level), etc. from `data/waterlevel.py`:

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `zs0file` | `input.tide.get()` | Tide/water level time series file |
| `tideloc` | `input.tide.tideloc` | Number of tide boundary locations (1, 2, or 4) |
| `tidelen` | `input.tide.get()` | Number of time steps in tide series |

### Wind Parameters

Generated by [`WindGrid`](../api-reference/data.md#wind), [`WindStation`](../api-reference/data.md#wind), [`WindPoint`](../api-reference/data.md#wind) from `data/wind.py`:

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `windfile` | `input.wind.get()` | Wind forcing file (contains windv, windth time series) |

### Wave Boundary Parameters

Generated by boundary classes in `data/boundary.py`. These return a `SpectralWaveBoundary` specification:

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `wbctype` | `input.wave.id` | Wave boundary type (jons, jonstable, swan, etc.) |
| `bcfile` | `input.wave.get()` | Wave boundary condition file or FILELIST |

**JONS/Parametric boundaries** (`BoundaryStationParamJons`, `BoundaryGridParamJons`, etc.):

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `Hm0` | Data-derived | Significant wave height |
| `Tp` | Data-derived | Peak period |
| `mainang` | Data-derived | Main wave direction |
| `gammajsp` | Data-derived | JONSWAP peak enhancement factor |
| `s` | Data-derived | Directional spreading coefficient |
| `fnyq` | `input.wave.fnyq` | Nyquist frequency for spectrum |
| `dfj` | `input.wave.dfj` | Frequency step for spectrum |

**JONSTABLE boundaries** (`BoundaryStationParamJonstable`, `BoundaryGridParamJonstable`, etc.):

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `Hm0` | Data-derived (time series) | Significant wave height |
| `Tp` | Data-derived (time series) | Peak period |
| `mainang` | Data-derived (time series) | Main wave direction |
| `gammajsp` | Data-derived (time series) | JONSWAP gamma |
| `s` | Data-derived (time series) | Spreading coefficient |
| `duration` | Computed from times | Duration of each condition |
| `dtbc` | `input.wave.wbc.dtbc` | Boundary update timestep |

**SWAN spectral boundaries** (`BoundaryStationSpectraSwan`):

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `freq` | Data-derived | Frequency array |
| `dir` | Data-derived | Direction array |
| `efth` | Data-derived | 2D energy density spectrum |

### Wave Boundary Condition Parameters

These can be set via `input.wave.wbc` (a `SpectralWaveBoundaryConditions` object):

| XBeach Parameter | Source | Description |
|-----------------|--------|-------------|
| `dtbc` | `input.wave.wbc.dtbc` | Boundary condition timestep |
| `nmax` | `input.wave.wbc.nmax` | Maximum ratio cg/c for long waves |
| `rt` | `input.wave.wbc.rt` | Duration of wave spectrum |
| `taper` | `input.wave.wbc.taper` | Spin-up time for boundary |
| `fcutoff` | `input.wave.wbc.fcutoff` | Low-frequency cutoff |
| `sprdthr` | `input.wave.wbc.sprdthr` | Spreading threshold |
| `trepfac` | `input.wave.wbc.trepfac` | Representative period factor |
| `correcthm0` | `input.wave.wbc.correcthm0` | Correct Hm0 for directional spreading |
| `random` | `input.wave.wbc.random` | Random seed for boundary generation |
| `nspr` | `input.wave.wbc.nspr` | Long wave direction switch |
| `wbcversion` | `input.wave.wbc.wbcversion` | Wave boundary version |
| `wbcRemoveStokes` | `input.wave.wbc.wbcRemoveStokes` | Remove Stokes drift |
| `wbcScaleEnergy` | `input.wave.wbc.wbcScaleEnergy` | Scale energy to match Hm0 |

!!! note "Data Interface Pattern"
    The data interfaces follow a consistent pattern: configure the source data and variable mappings, then call `get(destdir, grid, time)` which writes the necessary files and returns a dict or object with the XBeach parameters to include in `params.txt`.
