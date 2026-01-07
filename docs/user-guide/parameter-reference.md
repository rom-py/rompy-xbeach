# Parameter Reference

This document provides a lookup table mapping XBeach parameters to their location in rompy-xbeach.

## Quick Reference

### Physics Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wavemodel` | [`physics.wavemodel`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wavemodel) | Wave model type (stationary/surfbeat/nonh) |
| `swave` | [`physics.swave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.swave) | Enable short wave action balance |
| `lwave` | [`physics.lwave`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.lwave) | Enable long wave propagation |
| `flow` | [`physics.flow`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.flow) | Enable flow computation |
| `sedtrans` | [`physics.sedtrans`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.sedtrans) | Enable sediment transport |
| `morphology` | [`physics.morphology`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.morphology) | Enable morphological updating |
| `avalanching` | [`physics.avalanching`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.avalanching) | Enable avalanching |
| `wind` | [`physics.wind`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.wind) | Enable wind forcing |
| `vegetation` | [`physics.vegetation`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.vegetation) | Enable vegetation effects |
| `ships` | [`physics.ships`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.ships) | Enable ship-induced waves |
| `wci` | [`physics.wavemodel.wci`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Surfbeat.wci) | Wave-current interaction (Surfbeat only) |

### Wave Model Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `break` | [`physics.wavemodel.break_type`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Surfbeat.break_type) | Breaker formulation |
| `gamma` | [`physics.wavemodel.break_type.gamma`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.gamma) | Breaker parameter |
| `alpha` | [`physics.wavemodel.break_type.alpha`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.alpha) | Wave dissipation coefficient |
| `n` | [`physics.wavemodel.break_type.n`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.n) | Power in breaker formulation |
| `gammax` | [`physics.wavemodel.break_type.gammax`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Roelvink1.gammax) | Maximum ratio Hb/hb |
| `single_dir` | [`physics.wavemodel.single_dir`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Surfbeat.single_dir) | Single directional bin (Surfbeat) |
| `nhbreaker` | [`physics.wavemodel.nhbreaker`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.nhbreaker) | Non-hydrostatic breaker (Nonh) |
| `solver` | [`physics.wavemodel.solver`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.solver) | Pressure solver (Nonh) |
| `Topt` | [`physics.wavemodel.Topt`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.Topt) | Optimal timestep (Nonh) |
| `kdmin` | [`physics.wavemodel.kdmin`](../api-reference/components.md#rompy_xbeach.components.physics.wavemodel.Nonh.kdmin) | Minimum kd for dispersion (Nonh) |

### Bed Friction Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `bedfriction` | [`physics.bedfriction`](../api-reference/components.md#rompy_xbeach.components.physics.Physics.bedfriction) | Bed friction formulation |
| `bedfriccoef` | [`physics.bedfriction.bedfriccoef`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.bedfriccoef) | Friction coefficient |
| `bedfricfile` | [`physics.bedfriction.bedfricfile`](../api-reference/components.md#rompy_xbeach.components.physics.friction.BedFriction.bedfricfile) | Spatially varying friction file |

### Viscosity Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `nuh` | [`physics.viscosity.nuh`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.nuh) | Horizontal viscosity coefficient |
| `nuhfac` | [`physics.viscosity.nuhfac`](../api-reference/components.md#rompy_xbeach.components.physics.friction.Viscosity.nuhfac) | Viscosity calibration factor |
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
| `hmin` | [`physics.flow_numerics.hmin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.hmin) | Minimum water depth |
| `umin` | [`physics.flow_numerics.umin`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.umin) | Minimum velocity |
| `secorder` | [`physics.flow_numerics.secorder`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.secorder) | Second-order advection |
| `oldhu` | [`physics.flow_numerics.oldhu`](../api-reference/components.md#rompy_xbeach.components.physics.numerics.FlowNumerics.oldhu) | Old hu/hv formulation |

### Physical Constants

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `rho` | [`physics.constants.rho`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Constants.rho) | Water density |
| `rhoa` | [`physics.constants.rhoa`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Constants.rhoa) | Air density |
| `g` | [`physics.constants.g`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Constants.g) | Gravitational acceleration |
| `lat` | [`physics.constants.lat`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Constants.lat) | Latitude for Coriolis |
| `wearth` | [`physics.constants.wearth`](../api-reference/components.md#rompy_xbeach.components.physics.constants.Constants.wearth) | Earth angular velocity |

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

### Sediment Transport Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `form` | [`sediment.transport.form`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.form) | Transport formulation |
| `waveform` | [`sediment.transport.waveform`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.waveform) | Wave shape formulation |
| `turb` | [`sediment.transport.turb`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.turb) | Turbulence formulation |
| `sws` | [`sediment.transport.sws`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.sws) | Short wave stirring |
| `lws` | [`sediment.transport.lws`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.lws) | Long wave stirring |
| `lwt` | [`sediment.transport.lwt`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.lwt) | Long wave turbulence |
| `BRfac` | [`sediment.transport.BRfac`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.BRfac) | Bore runup factor |
| `facua` | [`sediment.transport.facua`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.facua) | Onshore transport factor |
| `facAs` | [`sediment.transport.facAs`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.facAs) | Skewness factor |
| `facSk` | [`sediment.transport.facSk`](../api-reference/components.md#rompy_xbeach.components.sediment.transport.Transport.facSk) | Asymmetry factor |

### Morphology Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `morfac` | [`sediment.morphology.morfac`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morfac) | Morphological acceleration |
| `morstart` | [`sediment.morphology.morstart`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstart) | Morphology start time |
| `morstop` | [`sediment.morphology.morstop`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstop) | Morphology stop time |
| `wetslp` | [`sediment.morphology.wetslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.wetslp) | Critical wet slope |
| `dryslp` | [`sediment.morphology.dryslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.dryslp) | Critical dry slope |
| `struct` | [`sediment.morphology.struct`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.struct) | Enable structures |
| `ne_layer` | [`sediment.morphology.ne_layer`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.ne_layer) | Non-erodible layer file |

### Bed Update Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `fwfile` | [`sediment.bed_update.fwfile`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_update.BedUpdate.fwfile) | Wave friction file |
| `fwcutoff` | [`sediment.bed_update.fwcutoff`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_update.BedUpdate.fwcutoff) | Wave friction cutoff |

### Bed Composition Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `D50` | [`sediment.bed_composition.D50`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.D50) | Median grain size |
| `D90` | [`sediment.bed_composition.D90`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.D90) | 90th percentile grain size |
| `D15` | [`sediment.bed_composition.D15`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.D15) | 15th percentile grain size |
| `ngd` | [`sediment.bed_composition.ngd`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.ngd) | Number of grain classes |
| `nd` | [`sediment.bed_composition.nd`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.nd) | Number of bed layers |
| `rhos` | [`sediment.bed_composition.rhos`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.rhos) | Sediment density |
| `por` | [`sediment.bed_composition.por`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.por) | Porosity |
| `dzg1` | [`sediment.bed_composition.dzg1`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.dzg1) | Layer 1 thickness |
| `dzg2` | [`sediment.bed_composition.dzg2`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.dzg2) | Layer 2 thickness |
| `dzg3` | [`sediment.bed_composition.dzg3`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.dzg3) | Layer 3 thickness |
| `sedcal` | [`sediment.bed_composition.sedcal`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.sedcal) | Sediment calibration factor |
| `ucrcal` | [`sediment.bed_composition.ucrcal`](../api-reference/components.md#rompy_xbeach.components.sediment.bed_composition.BedComposition.ucrcal) | Critical velocity calibration |

### Groundwater Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `gwflow` | [`sediment.groundwater.gwflow`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.Groundwater.gwflow) | Enable groundwater flow |
| `gwnonh` | [`sediment.groundwater.gwnonh`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.Groundwater.gwnonh) | Non-hydrostatic groundwater |
| `kx` | [`sediment.groundwater.kx`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.Groundwater.kx) | Horizontal permeability |
| `ky` | [`sediment.groundwater.ky`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.Groundwater.ky) | Vertical permeability |
| `gwheadmodel` | [`sediment.groundwater.gwheadmodel`](../api-reference/components.md#rompy_xbeach.components.sediment.groundwater.Groundwater.gwheadmodel) | Head boundary model |

---

### Flow Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `front` | [`flow_boundary.front`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundary.front) | Front boundary type |
| `back` | [`flow_boundary.back`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundary.back) | Back boundary type |
| `left` | [`flow_boundary.left`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundary.left) | Left boundary type |
| `right` | [`flow_boundary.right`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundary.right) | Right boundary type |
| `lateralwave` | [`flow_boundary.lateralwave`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.FlowBoundary.lateralwave) | Lateral wave boundary |

### Tide Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `tideloc` | [`tide_boundary.tideloc`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundary.tideloc) | Number of tide locations |
| `tidetype` | [`tide_boundary.tidetype`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundary.tidetype) | Tide boundary type |
| `zs0` | [`tide_boundary.zs0`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundary.zs0) | Initial water level |
| `paulrevere` | [`tide_boundary.paulrevere`](../api-reference/components.md#rompy_xbeach.components.boundary.parameters.TideBoundary.paulrevere) | Sea/land boundary |

### Wave Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wbctype` | [`wave_boundary.wbctype`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.wbctype) | Wave boundary type |
| `bcfile` | [`wave_boundary.bcfile`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.bcfile) | Boundary condition file |
| `dtbc` | [`wave_boundary.dtbc`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.dtbc) | Boundary update interval |
| `thetamin` | [`wave_boundary.thetamin`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.thetamin) | Minimum wave direction |
| `thetamax` | [`wave_boundary.thetamax`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.thetamax) | Maximum wave direction |
| `dtheta` | [`wave_boundary.dtheta`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.dtheta) | Directional resolution |
| `thetanaut` | [`wave_boundary.thetanaut`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.thetanaut) | Nautical convention |
| `ARC` | [`wave_boundary.ARC`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.ARC) | Active reflection compensation |
| `freewave` | [`wave_boundary.freewave`](../api-reference/components.md#rompy_xbeach.components.wbc.WaveBoundarySpectral.freewave) | Free wave boundary |

---

### Hotstart Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `hotstart` | [`hotstart`](../api-reference/config.md#rompy_xbeach.config.Config.hotstart) | Enable hotstart |
| `hotstartfileno` | [`hotstart.hotstartfileno`](../api-reference/config.md#rompy_xbeach.components.hotstart.Hotstart.hotstartfileno) | Hotstart file number |
| `writehotstart` | [`output.writehotstart`](../api-reference/components.md#rompy_xbeach.components.output.Output.writehotstart) | Write hotstart files |
| `tinth` | [`output.tinth`](../api-reference/components.md#rompy_xbeach.components.output.Output.tinth) | Hotstart output interval |

---

### Output Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `outputformat` | [`output.outputformat`](../api-reference/components.md#rompy_xbeach.components.output.Output.outputformat) | Output file format |
| `tintg` | [`output.tintg`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintg) | Global output interval |
| `tintm` | [`output.tintm`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintm) | Mean output interval |
| `tintp` | [`output.tintp`](../api-reference/components.md#rompy_xbeach.components.output.Output.tintp) | Point output interval |
| `tstart` | [`output.tstart`](../api-reference/components.md#rompy_xbeach.components.output.Output.tstart) | Output start time |
| `nglobalvar` | [`output.nglobalvar`](../api-reference/components.md#rompy_xbeach.components.output.Output.nglobalvar) | Number of global variables |
| `nmeanvar` | [`output.nmeanvar`](../api-reference/components.md#rompy_xbeach.components.output.Output.nmeanvar) | Number of mean variables |
| `npointvar` | [`output.npointvar`](../api-reference/components.md#rompy_xbeach.components.output.Output.npointvar) | Number of point variables |
| `nrugauge` | [`output.nrugauge`](../api-reference/components.md#rompy_xbeach.components.output.Output.nrugauge) | Number of runup gauges |
| `nrugdepth` | [`output.nrugdepth`](../api-reference/components.md#rompy_xbeach.components.output.Output.nrugdepth) | Number of runup depths |
| `rugdepth` | [`output.rugdepth`](../api-reference/components.md#rompy_xbeach.components.output.Output.rugdepth) | Runup depth thresholds |

---

### MPI Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `mpiboundary` | [`mpi.mpiboundary`](../api-reference/config.md#rompy_xbeach.components.mpi.MPI.mpiboundary) | MPI boundary type |
| `mmpi` | [`mpi.mmpi`](../api-reference/config.md#rompy_xbeach.components.mpi.MPI.mmpi) | MPI partitions in m |
| `nmpi` | [`mpi.nmpi`](../api-reference/config.md#rompy_xbeach.components.mpi.MPI.nmpi) | MPI partitions in n |

---

## Data-Driven Parameters

Some parameters are generated automatically from data interfaces rather than set directly:

| XBeach Parameter | Generated By | Notes |
|-----------------|--------------|-------|
| `zs0file` | [`input.tide`](../api-reference/data.md#tide--water-level) | Tide time series file |
| `tidelen` | [`input.tide`](../api-reference/data.md#tide--water-level) | Length of tide series |
| `bcfile` | [`input.wave`](../api-reference/data.md#wave-boundaries) | Wave boundary file |
| `Hrms`, `Tp`, `dir` | [`input.wave`](../api-reference/data.md#wave-boundaries) | Wave parameters (for stat/bichrom) |
| `windfile` | [`input.wind`](../api-reference/data.md#wind) | Wind forcing file |
| `depfile` | [`bathy`](../api-reference/types.md#bathymetry) | Bathymetry file |
| `xfile`, `yfile` | [`grid`](../api-reference/types.md#grid) | Grid coordinate files |

These are set automatically when using the data interface classes and should not be set manually.
