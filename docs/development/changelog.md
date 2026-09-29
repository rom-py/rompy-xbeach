# Changelog

## Unreleased

### Breaking changes

- `Config.physics` is required, and `Physics.wavemodel` is required: XBeach does not run without a wave model.
- `BoundaryStat` and `BoundaryBichrom` are replaced by [`BoundaryParams`][rompy_xbeach.data.boundary.nonspectral.BoundaryParams] (`model_type: params`). Set `Tlong` for bichromatic waves.
- The JONSWAP boundaries (`Boundary*Jons`) use `id: parametric`, the XBeach `wbctype`, so their files are named `parametric-*.txt` and `parametric-filelist.txt`. Every wave boundary `id` is now its `wbctype`. YAML with `id: jons` must change to `id: parametric`. Class names and `model_type` values are unchanged.
- `WbcEnum.JONS` is renamed `WbcEnum.PARAMETRIC`.
- Three fields wrote parameter names XBeach does not read, so their settings had no effect. They now use XBeach's names: `ShortWaveFriction.wavfriccoef` and `wavfricfile` are renamed `fw` and `fwfile`, `Output.tspoint` is renamed `tspoints`, and `Nonh.nonhq3d` is written as `nonhq3d` (it was written as `nhq3d`).

### New features

- `Config` rejects wave boundary types that XBeach refuses with the chosen wave model: bichromatic waves need surfbeat; parametric, swan, vardens and reuse boundaries cannot be stationary; `ts_1` and `ts_2` are surfbeat only; `ts_nonh` is nonh only.
- `sediment=None` and `mpi=None` leave out those components.
- [`CombinedWaterLevel`][rompy_xbeach.data.waterlevel.CombinedWaterLevel] (`model_type: combined_water_level`) combines tide constituents and a water level series as `input.tide`.
- The wave boundaries have a `dtheta_s` field.
- Public Docker images `ghcr.io/rom-py/xbeach` with serial and MPI builds (see [Running XBeach](../user-guide/running.md)).

### Bug fixes

- `RegularGrid.params` writes `vardx = 0`.
- Field descriptions give XBeach's actual defaults (for example `wind`, `morphology`, `wetslp`, `tintg`, `tintp`, `maxcf` and the breaker `gamma`), checked against the XBeach source.
- Field descriptions of `dthetas_xb`, `nspr`, `random` and `trepfac` were tuples and did not render.
- `BoundaryOff` writes its wave parameters, and `BoundaryReuse` copies the boundary list and series files.
- The left lateral extension of the bathymetry is fixed.
- `bedfricfile` and `wavfricfile` set in `Physics` are copied to the workspace. File fields are handled in one place, `XBeachBaseModel.get()`, for every component.

## 0.1.0 (2024-09-11)

- First release on PyPI.
