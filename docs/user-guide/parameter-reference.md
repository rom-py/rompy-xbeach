# Parameter Reference

This document provides a lookup table mapping XBeach parameters to their location in rompy-xbeach.

## Quick Reference

### Physics Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wavemodel` | `physics.wavemodel` | Wave model type (stationary/surfbeat/nonh) |
| `swave` | `physics.swave` | Enable short wave action balance |
| `lwave` | `physics.lwave` | Enable long wave propagation |
| `flow` | `physics.flow` | Enable flow computation |
| `sedtrans` | `physics.sedtrans` | Enable sediment transport |
| `morphology` | `physics.morphology` | Enable morphological updating |
| `avalanching` | `physics.avalanching` | Enable avalanching |
| `wind` | `physics.wind` | Enable wind forcing |
| `vegetation` | `physics.vegetation` | Enable vegetation effects |
| `ships` | `physics.ships` | Enable ship-induced waves |
| `wci` | `physics.wavemodel.wci` | Wave-current interaction (Surfbeat only) |

### Wave Model Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `break` | `physics.wavemodel.break_type` | Breaker formulation |
| `gamma` | `physics.wavemodel.break_type.gamma` | Breaker parameter |
| `alpha` | `physics.wavemodel.break_type.alpha` | Wave dissipation coefficient |
| `n` | `physics.wavemodel.break_type.n` | Power in breaker formulation |
| `gammax` | `physics.wavemodel.break_type.gammax` | Maximum ratio Hb/hb |
| `single_dir` | `physics.wavemodel.single_dir` | Single directional bin (Surfbeat) |
| `nhbreaker` | `physics.wavemodel.nhbreaker` | Non-hydrostatic breaker (Nonh) |
| `solver` | `physics.wavemodel.solver` | Pressure solver (Nonh) |
| `Topt` | `physics.wavemodel.Topt` | Optimal timestep (Nonh) |
| `kdmin` | `physics.wavemodel.kdmin` | Minimum kd for dispersion (Nonh) |

### Bed Friction Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `bedfriction` | `physics.bedfriction` | Bed friction formulation |
| `bedfriccoef` | `physics.bedfriction.bedfriccoef` | Friction coefficient |
| `bedfricfile` | `physics.bedfriction.bedfricfile` | Spatially varying friction file |

### Viscosity Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `nuh` | `physics.viscosity.nuh` | Horizontal viscosity coefficient |
| `nuhfac` | `physics.viscosity.nuhfac` | Viscosity calibration factor |
| `smag` | `physics.viscosity.smag` | Enable Smagorinsky model |

### Wave Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `scheme` | `physics.wave_numerics.scheme` | Numerical scheme |
| `wavint` | `physics.wave_numerics.wavint` | Wave integration interval |
| `maxerror` | `physics.wave_numerics.maxerror` | Maximum wave convergence error |
| `maxiter` | `physics.wave_numerics.maxiter` | Maximum wave iterations |

### Flow Numerics

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `cfl` | `physics.flow_numerics.cfl` | CFL criterion |
| `eps` | `physics.flow_numerics.eps` | Threshold depth |
| `hmin` | `physics.flow_numerics.hmin` | Minimum water depth |
| `umin` | `physics.flow_numerics.umin` | Minimum velocity |
| `secorder` | `physics.flow_numerics.secorder` | Second-order advection |
| `oldhu` | `physics.flow_numerics.oldhu` | Old hu/hv formulation |

### Physical Constants

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `rho` | `physics.constants.rho` | Water density |
| `rhoa` | `physics.constants.rhoa` | Air density |
| `g` | `physics.constants.g` | Gravitational acceleration |
| `lat` | `physics.constants.lat` | Latitude for Coriolis |
| `wearth` | `physics.constants.wearth` | Earth angular velocity |

### Vegetation Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `vegetation` | `physics.vegetation` | Enable vegetation |
| `nsec` | `physics.vegetation.nsec` | Number of vegetation sections |
| `ah` | `physics.vegetation.ah` | Vegetation height |
| `bv` | `physics.vegetation.bv` | Stem diameter |
| `Nv` | `physics.vegetation.Nv` | Stem density |
| `Cd` | `physics.vegetation.Cd` | Drag coefficient |

---

### Sediment Transport Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `form` | `sediment.transport.form` | Transport formulation |
| `waveform` | `sediment.transport.waveform` | Wave shape formulation |
| `turb` | `sediment.transport.turb` | Turbulence formulation |
| `sws` | `sediment.transport.sws` | Short wave stirring |
| `lws` | `sediment.transport.lws` | Long wave stirring |
| `lwt` | `sediment.transport.lwt` | Long wave turbulence |
| `BRfac` | `sediment.transport.BRfac` | Bore runup factor |
| `facua` | `sediment.transport.facua` | Onshore transport factor |
| `facAs` | `sediment.transport.facAs` | Skewness factor |
| `facSk` | `sediment.transport.facSk` | Asymmetry factor |

### Morphology Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `morfac` | `sediment.morphology.morfac` | Morphological acceleration |
| `morstart` | `sediment.morphology.morstart` | Morphology start time |
| `morstop` | `sediment.morphology.morstop` | Morphology stop time |
| `wetslp` | `sediment.morphology.wetslp` | Critical wet slope |
| `dryslp` | `sediment.morphology.dryslp` | Critical dry slope |
| `struct` | `sediment.morphology.struct` | Enable structures |
| `ne_layer` | `sediment.morphology.ne_layer` | Non-erodible layer file |

### Bed Update Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `fwfile` | `sediment.bed_update.fwfile` | Wave friction file |
| `fwcutoff` | `sediment.bed_update.fwcutoff` | Wave friction cutoff |

### Bed Composition Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `D50` | `sediment.bed_composition.D50` | Median grain size |
| `D90` | `sediment.bed_composition.D90` | 90th percentile grain size |
| `D15` | `sediment.bed_composition.D15` | 15th percentile grain size |
| `ngd` | `sediment.bed_composition.ngd` | Number of grain classes |
| `nd` | `sediment.bed_composition.nd` | Number of bed layers |
| `rhos` | `sediment.bed_composition.rhos` | Sediment density |
| `por` | `sediment.bed_composition.por` | Porosity |
| `dzg1` | `sediment.bed_composition.dzg1` | Layer 1 thickness |
| `dzg2` | `sediment.bed_composition.dzg2` | Layer 2 thickness |
| `dzg3` | `sediment.bed_composition.dzg3` | Layer 3 thickness |
| `sedcal` | `sediment.bed_composition.sedcal` | Sediment calibration factor |
| `ucrcal` | `sediment.bed_composition.ucrcal` | Critical velocity calibration |

### Groundwater Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `gwflow` | `sediment.groundwater.gwflow` | Enable groundwater flow |
| `gwnonh` | `sediment.groundwater.gwnonh` | Non-hydrostatic groundwater |
| `kx` | `sediment.groundwater.kx` | Horizontal permeability |
| `ky` | `sediment.groundwater.ky` | Vertical permeability |
| `gwheadmodel` | `sediment.groundwater.gwheadmodel` | Head boundary model |

---

### Flow Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `front` | `flow_boundary.front` | Front boundary type |
| `back` | `flow_boundary.back` | Back boundary type |
| `left` | `flow_boundary.left` | Left boundary type |
| `right` | `flow_boundary.right` | Right boundary type |
| `lateralwave` | `flow_boundary.lateralwave` | Lateral wave boundary |

### Tide Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `tideloc` | `tide_boundary.tideloc` | Number of tide locations |
| `tidetype` | `tide_boundary.tidetype` | Tide boundary type |
| `zs0` | `tide_boundary.zs0` | Initial water level |
| `paulrevere` | `tide_boundary.paulrevere` | Sea/land boundary |

### Wave Boundary Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `wbctype` | `wave_boundary.wbctype` | Wave boundary type |
| `bcfile` | `wave_boundary.bcfile` | Boundary condition file |
| `dtbc` | `wave_boundary.dtbc` | Boundary update interval |
| `thetamin` | `wave_boundary.thetamin` | Minimum wave direction |
| `thetamax` | `wave_boundary.thetamax` | Maximum wave direction |
| `dtheta` | `wave_boundary.dtheta` | Directional resolution |
| `thetanaut` | `wave_boundary.thetanaut` | Nautical convention |
| `ARC` | `wave_boundary.ARC` | Active reflection compensation |
| `freewave` | `wave_boundary.freewave` | Free wave boundary |

---

### Hotstart Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `hotstart` | `hotstart` or `hotstart.hotstart` | Enable hotstart |
| `hotstartfileno` | `hotstart.hotstartfileno` | Hotstart file number |
| `writehotstart` | `output.writehotstart` | Write hotstart files |
| `tinth` | `output.tinth` | Hotstart output interval |

---

### Output Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `outputformat` | `output.outputformat` | Output file format |
| `tintg` | `output.tintg` | Global output interval |
| `tintm` | `output.tintm` | Mean output interval |
| `tintp` | `output.tintp` | Point output interval |
| `tstart` | `output.tstart` | Output start time |
| `nglobalvar` | `output.nglobalvar` | Number of global variables |
| `nmeanvar` | `output.nmeanvar` | Number of mean variables |
| `npointvar` | `output.npointvar` | Number of point variables |
| `nrugauge` | `output.nrugauge` | Number of runup gauges |
| `nrugdepth` | `output.nrugdepth` | Number of runup depths |
| `rugdepth` | `output.rugdepth` | Runup depth thresholds |

---

### MPI Parameters

| XBeach Parameter | Rompy Location | Description |
|-----------------|----------------|-------------|
| `mpiboundary` | `mpi.mpiboundary` | MPI boundary type |
| `mmpi` | `mpi.mmpi` | MPI partitions in m |
| `nmpi` | `mpi.nmpi` | MPI partitions in n |

---

## Data-Driven Parameters

Some parameters are generated automatically from data interfaces rather than set directly:

| XBeach Parameter | Generated By | Notes |
|-----------------|--------------|-------|
| `zs0file` | `input.tide` | Tide time series file |
| `tidelen` | `input.tide` | Length of tide series |
| `bcfile` | `input.wave` | Wave boundary file |
| `Hrms`, `Tp`, `dir` | `input.wave` | Wave parameters (for stat/bichrom) |
| `windfile` | `input.wind` | Wind forcing file |
| `depfile` | `bathy` | Bathymetry file |
| `xfile`, `yfile` | `grid` | Grid coordinate files |

These are set automatically when using the data interface classes and should not be set manually.
