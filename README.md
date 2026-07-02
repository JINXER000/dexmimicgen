# installation of sdp_dmg

> **DR-LfD customized fork** — branch `playback_3d`, based on upstream DexMimicGen
> `0.1.0`. The bimanual environment definitions are unchanged from upstream; the
> fork's value-add lives entirely in `scripts/`. The original upstream README
> follows after the divider.

## DR-LfD fork — what's customized

DR-LfD uses this fork to (a) play back human demonstrations in 3D and (b) roll out
the learned TAMP plan inside the robomimic evaluation loop. Two new scripts provide
this:

1. **robomimic-compatible env wrapper** — `scripts/robomimic_dmg_wrapper.py`.
   Class `DMG_env_switchable` subclasses robomimic's `EnvRobosuite` and adds
   switchable controllers (e.g. `OSC_POSE`, absolute/relative action) plus the
   TAMP execution hooks DR-LfD calls: `replay_tamp_step`, `rollout_from_biop`,
   `inference`, `reset_ts` / `step_ts`, `get_cur_eef_xyz_robosuite` /
   `get_cur_jpose_robosuite`, and `save_mj_observation` (point-cloud recording).
   It reads the scene-graph from the demonstration HDF5 via `networkx`.

2. **Depth / 3D point-cloud demonstration playback** — `scripts/playback_depth.py`
   (the branch's namesake). `get_pcd_dict_fn` builds per-object point clouds from
   depth cameras (`agentview` / `birdview` / `frontview`) and optionally writes
   `.ply` assets. It imports `get_individual_pcd` / `get_name2id` from the
   **customized robosuite** (`robosuite.demos.vis_depth_seg`).

### Tasks
Provides the two DR-LfD bimanual tasks — `two_arm_threading` and
`two_arm_three_piece_assembly` (env classes under `dexmimicgen/environments/`,
registered automatically on import). All upstream tasks remain available.

### Dependencies & integration
- **Hard dependency on the customized robosuite fork** (`switchable` branch): the
  wrapper and playback scripts use `robosuite.demos.vis_depth_seg` and
  `refactor_composite_controller_config`, which exist only in that fork.
- `requirements.txt` is intentionally de-pinned so it coexists with the DR-LfD
  conda environment rather than re-pinning it.
- DR-LfD locates this repo through the `${DEXMIMICGEN_ROOT}` macro (its task
  configs reference it via `env_dir`); set that macro to wherever you clone it.

### Install
```bash
git clone https://github.com/Dr-LfD/dexmimicgen -b playback_3d "${DEXMIMICGEN_ROOT}"
pip install -e "${DEXMIMICGEN_ROOT}"
```

---

<p align="center">
  <img width="95.0%" src="images/dexmimicgen.gif">
</p>

This repository contains the official release of simulation environments and datasets for the [ICRA 2025](https://2025.ieee-icra.org) paper "DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning".

Website: https://dexmimicgen.github.io

Paper: https://arxiv.org/abs/2410.24185

For business inquiries, please submit this form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

-------

## Getting Started

To use this repository, you need to first install the latest robosuite. For more
information, please refer to [robosuite](https://github.com/ARISE-Initiative/robosuite).

```bash
git clone https://github.com/ARISE-Initiative/robosuite
pip install -e robosuite
```
