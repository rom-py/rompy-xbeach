# Parameter Reference

This document provides a lookup table mapping XBeach parameters to their location in rompy-xbeach.

## Quick Reference

### Physics Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wavemodel` | [`physics.wavemodel`](../api-reference/components.md#physics) | Wave model type (stationary/surfbeat/nonh) |
| `swave` | [`physics.swave`](../api-reference/components.md#physics) | Enable short wave action balance |
| `lwave` | [`physics.lwave`](../api-reference/components.md#physics) | Enable long wave propagation |
| `flow` | [`physics.flow`](../api-reference/components.md#physics) | Enable flow computation |
| `sedtrans` | [`physics.sedtrans`](../api-reference/components.md#physics) | Enable sediment transport |
| `morphology` | [`physics.morphology`](../api-reference/components.md#physics) | Enable morphological updating |
| `avalanching` | [`physics.avalanching`](../api-reference/components.md#physics) | Enable avalanching |
| `wind` | [`physics.wind`](../api-reference/components.md#physics) | Enable wind forcing |
| `vegetation` | [`physics.vegetation`](../api-reference/components.md#physics) | Enable vegetation effects |
| `ships` | [`physics.ships`](../api-reference/components.md#physics) | Enable ship-induced waves |
| `wci` | [`physics.wavemodel.wci`](../api-reference/components.md#wave-models) | Wave-current interaction (Surfbeat only) |

### Wave Model Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `break` | [`physics.wavemodel.break_type`](../api-reference/components.md#wave-models) | Breaker formulation |
| `gamma` | [`physics.wavemodel.break_type.gamma`](../api-reference/components.md#breaker-formulations) | Breaker parameter |
| `alpha` | [`physics.wavemodel.break_type.alpha`](../api-reference/components.md#breaker-formulations) | Wave dissipation coefficient |
| `n` | [`physics.wavemodel.break_type.n`](../api-reference/components.md#breaker-formulations) | Power in breaker formulation |
| `gammax` | [`physics.wavemodel.break_type.gammax`](../api-reference/components.md#breaker-formulations) | Maximum ratio Hb/hb |
| `single_dir` | [`physics.wavemodel.single_dir`](../api-reference/components.md#wave-models) | Single directional bin (Surfbeat) |
| `nhbreaker` | [`physics.wavemodel.nhbreaker`](../api-reference/components.md#wave-models) | Non-hydrostatic breaker (Nonh) |
| `solver` | [`physics.wavemodel.solver`](../api-reference/components.md#wave-models) | Pressure solver (Nonh) |
| `Topt` | [`physics.wavemodel.Topt`](../api-reference/components.md#wave-models) | Optimal timestep (Nonh) |
| `kdmin` | [`physics.wavemodel.kdmin`](../api-reference/components.md#wave-models) | Minimum kd for dispersion (Nonh) |

### Bed Friction Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `bedfriction` | [`physics.bedfriction`](../api-reference/components.md#friction) | Bed friction formulation |
| `bedfriccoef` | [`physics.bedfriction.bedfriccoef`](../api-reference/components.md#friction) | Friction coefficient |
| `bedfricfile` | [`physics.bedfriction.bedfricfile`](../api-reference/components.md#friction) | Spatially varying friction file |

### Viscosity Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `nuh` | [`physics.viscosity.nuh`](../api-reference/components.md#viscosity) | Horizontal viscosity coefficient |
| `nuhfac` | [`physics.viscosity.nuhfac`](../api-reference/components.md#viscosity) | Viscosity calibration factor |
| `smag` | [`physics.viscosity.smag`](../api-reference/components.md#viscosity) | Enable Smagorinsky model |

### Wave Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `scheme` | [`physics.wave_numerics.scheme`](../api-reference/components.md#numerics) | Numerical scheme |
| `wavint` | [`physics.wave_numerics.wavint`](../api-reference/components.md#numerics) | Wave integration interval |
| `maxerror` | [`physics.wave_numerics.maxerror`](../api-reference/components.md#numerics) | Maximum wave convergence error |
| `maxiter` | [`physics.wave_numerics.maxiter`](../api-reference/components.md#numerics) | Maximum wave iterations |

### Flow Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `cfl` | [`physics.flow_numerics.cfl`](../api-reference/components.md#numerics) | CFL criterion |
| `eps` | [`physics.flow_numerics.eps`](../api-reference/components.md#numerics) | Threshold depth |
| `hmin` | [`physics.flow_numerics.hmin`](../api-reference/components.md#numerics) | Minimum water depth |
| `umin` | [`physics.flow_numerics.umin`](../api-reference/components.md#numerics) | Minimum velocity |
| `secorder` | [`physics.flow_numerics.secorder`](../api-reference/components.md#numerics) | Second-order advection |
| `oldhu` | [`physics.flow_numerics.oldhu`](../api-reference/components.md#numerics) | Old hu/hv formulation |

### Physical Constants

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `rho` | [`physics.constants.rho`](../api-reference/components.md#constants) | Water density |
| `rhoa` | [`physics.constants.rhoa`](../api-reference/components.md#constants) | Air density |
| `g` | [`physics.constants.g`](../api-reference/components.md#constants) | Gravitational acceleration |
| `lat` | [`physics.constants.lat`](../api-reference/components.md#constants) | Latitude for Coriolis |
| `wearth` | [`physics.constants.wearth`](../api-reference/components.md#constants) | Earth angular velocity |

### Vegetation Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `vegetation` | [`physics.vegetation`](../api-reference/components.md#vegetation) | Enable vegetation |
| `nsec` | [`physics.vegetation.nsec`](../api-reference/components.md#vegetation) | Number of vegetation sections |
| `ah` | [`physics.vegetation.ah`](../api-reference/components.md#vegetation) | Vegetation height |
| `bv` | [`physics.vegetation.bv`](../api-reference/components.md#vegetation) | Stem diameter |
| `Nv` | [`physics.vegetation.Nv`](../api-reference/components.md#vegetation) | Stem density |
| `Cd` | [`physics.vegetation.Cd`](../api-reference/components.md#vegetation) | Drag coefficient |

---

### Sediment Transport Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `form` | [`sediment.transport.form`](../api-reference/components.md#sediment) | Transport formulation |
| `waveform` | [`sediment.transport.waveform`](../api-reference/components.md#sediment) | Wave shape formulation |
| `turb` | [`sediment.transport.turb`](../api-reference/components.md#sediment) | Turbulence formulation |
| `sws` | [`sediment.transport.sws`](../api-reference/components.md#sediment) | Short wave stirring |
| `lws` | [`sediment.transport.lws`](../api-reference/components.md#sediment) | Long wave stirring |
| `lwt` | [`sediment.transport.lwt`](../api-reference/components.md#sediment) | Long wave turbulence |
| `BRfac` | [`sediment.transport.BRfac`](../api-reference/components.md#sediment) | Bore runup factor |
| `facua` | [`sediment.transport.facua`](../api-reference/components.md#sediment) | Onshore transport factor |
| `facAs` | [`sediment.transport.facAs`](../api-reference/components.md#sediment) | Skewness factor |
| `facSk` | [`sediment.transport.facSk`](../api-reference/components.md#sediment) | Asymmetry factor |

### Morphology Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `morfac` | [`sediment.morphology.morfac`](../api-reference/components.md#sediment) | Morphological acceleration |
| `morstart` | [`sediment.morphology.morstart`](../api-reference/components.md#sediment) | Morphology start time |
| `morstop` | [`sediment.morphology.morstop`](../api-reference/components.md#sediment) | Morphology stop time |
| `wetslp` | [`sediment.morphology.wetslp`](../api-reference/components.md#sediment) | Critical wet slope |
| `dryslp` | [`sediment.morphology.dryslp`](../api-reference/components.md#sediment) | Critical dry slope |
| `struct` | [`sediment.morphology.struct`](../api-reference/components.md#sediment) | Enable structures |
| `ne_layer` | [`sediment.morphology.ne_layer`](../api-reference/components.md#sediment) | Non-erodible layer file |

### Bed Update Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `fwfile` | [`sediment.bed_update.fwfile`](../api-reference/components.md#sediment) | Wave friction file |
| `fwcutoff` | [`sediment.bed_update.fwcutoff`](../api-reference/components.md#sediment) | Wave friction cutoff |

### Bed Composition Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `D50` | [`sediment.bed_composition.D50`](../api-reference/components.md#sediment) | Median grain size |
| `D90` | [`sediment.bed_composition.D90`](../api-reference/components.md#sediment) | 90th percentile grain size |
| `D15` | [`sediment.bed_composition.D15`](../api-reference/components.md#sediment) | 15th percentile grain size |
| `ngd` | [`sediment.bed_composition.ngd`](../api-reference/components.md#sediment) | Number of grain classes |
| `nd` | [`sediment.bed_composition.nd`](../api-reference/components.md#sediment) | Number of bed layers |
| `rhos` | [`sediment.bed_composition.rhos`](../api-reference/components.md#sediment) | Sediment density |
| `por` | [`sediment.bed_composition.por`](../api-reference/components.md#sediment) | Porosity |
| `dzg1` | [`sediment.bed_composition.dzg1`](../api-reference/components.md#sediment) | Layer 1 thickness |
| `dzg2` | [`sediment.bed_composition.dzg2`](../api-reference/components.md#sediment) | Layer 2 thickness |
| `dzg3` | [`sediment.bed_composition.dzg3`](../api-reference/components.md#sediment) | Layer 3 thickness |
| `sedcal` | [`sediment.bed_composition.sedcal`](../api-reference/components.md#sediment) | Sediment calibration factor |
| `ucrcal` | [`sediment.bed_composition.ucrcal`](../api-reference/components.md#sediment) | Critical velocity calibration |

### Groundwater Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `gwflow` | [`sediment.groundwater.gwflow`](../api-reference/components.md#sediment) | Enable groundwater flow |
| `gwnonh` | [`sediment.groundwater.gwnonh`](../api-reference/components.md#sediment) | Non-hydrostatic groundwater |
| `kx` | [`sediment.groundwater.kx`](../api-reference/components.md#sediment) | Horizontal permeability |
| `ky` | [`sediment.groundwater.ky`](../api-reference/components.md#sediment) | Vertical permeability |
| `gwheadmodel` | [`sediment.groundwater.gwheadmodel`](../api-reference/components.md#sediment) | Head boundary model |

---

### Flow Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `front` | [`flow_boundary.front`](../api-reference/components.md#boundaries) | Front boundary type |
| `back` | [`flow_boundary.back`](../api-reference/components.md#boundaries) | Back boundary type |
| `left` | [`flow_boundary.left`](../api-reference/components.md#boundaries) | Left boundary type |
| `right` | [`flow_boundary.right`](../api-reference/components.md#boundaries) | Right boundary type |
| `lateralwave` | [`flow_boundary.lateralwave`](../api-reference/components.md#boundaries) | Lateral wave boundary |

### Tide Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `tideloc` | [`tide_boundary.tideloc`](../api-reference/components.md#boundaries) | Number of tide locations |
| `tidetype` | [`tide_boundary.tidetype`](../api-reference/components.md#boundaries) | Tide boundary type |
| `zs0` | [`tide_boundary.zs0`](../api-reference/components.md#boundaries) | Initial water level |
| `paulrevere` | [`tide_boundary.paulrevere`](../api-reference/components.md#boundaries) | Sea/land boundary |

### Wave Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wbctype` | [`wave_boundary.wbctype`](../api-reference/components.md#boundaries) | Wave boundary type |
| `bcfile` | [`wave_boundary.bcfile`](../api-reference/components.md#boundaries) | Boundary condition file |
| `dtbc` | [`wave_boundary.dtbc`](../api-reference/components.md#boundaries) | Boundary update interval |
| `thetamin` | [`wave_boundary.thetamin`](../api-reference/components.md#boundaries) | Minimum wave direction |
| `thetamax` | [`wave_boundary.thetamax`](../api-reference/components.md#boundaries) | Maximum wave direction |
| `dtheta` | [`wave_boundary.dtheta`](../api-reference/components.md#boundaries) | Directional resolution |
| `thetanaut` | [`wave_boundary.thetanaut`](../api-reference/components.md#boundaries) | Nautical convention |
| `ARC` | [`wave_boundary.ARC`](../api-reference/components.md#boundaries) | Active reflection compensation |
| `freewave` | [`wave_boundary.freewave`](../api-reference/components.md#boundaries) | Free wave boundary |

---

### Hotstart Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `hotstart` | [`hotstart`](../api-reference/config.md) | Enable hotstart |
| `hotstartfileno` | [`hotstart.hotstartfileno`](../api-reference/config.md) | Hotstart file number |
| `writehotstart` | [`output.writehotstart`](../api-reference/components.md#output) | Write hotstart files |
| `tinth` | [`output.tinth`](../api-reference/components.md#output) | Hotstart output interval |

---

### Output Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `outputformat` | [`output.outputformat`](../api-reference/components.md#output) | Output file format |
| `tintg` | [`output.tintg`](../api-reference/components.md#output) | Global output interval |
| `tintm` | [`output.tintm`](../api-reference/components.md#output) | Mean output interval |
| `tintp` | [`output.tintp`](../api-reference/components.md#output) | Point output interval |
| `tstart` | [`output.tstart`](../api-reference/components.md#output) | Output start time |
| `nglobalvar` | [`output.nglobalvar`](../api-reference/components.md#output) | Number of global variables |
| `nmeanvar` | [`output.nmeanvar`](../api-reference/components.md#output) | Number of mean variables |
| `npointvar` | [`output.npointvar`](../api-reference/components.md#output) | Number of point variables |
| `nrugauge` | [`output.nrugauge`](../api-reference/components.md#output) | Number of runup gauges |
| `nrugdepth` | [`output.nrugdepth`](../api-reference/components.md#output) | Number of runup depths |
| `rugdepth` | [`output.rugdepth`](../api-reference/components.md#output) | Runup depth thresholds |

---

### MPI Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `mpiboundary` | [`mpi.mpiboundary`](../api-reference/config.md) | MPI boundary type |
| `mmpi` | [`mpi.mmpi`](../api-reference/config.md) | MPI partitions in m |
| `nmpi` | [`mpi.nmpi`](../api-reference/config.md) | MPI partitions in n |

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
