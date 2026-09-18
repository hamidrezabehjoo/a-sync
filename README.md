# Mpipi-GG α-Synuclein Simulation Ladder

COSMO/OpenMM pipeline for running **Mpipi-GG** coarse-grained simulations of
human α-synuclein (140 aa) as a staged, convergence-labeled ladder. Built to
audit whether the force field behind generative IDP models (STARLING) is
sampled long enough to capture the rare compact states that PRE experiments
detect.

## What this does

- Registers a validated **Mpipi-GG** model in [COSMO](https://github.com/vuqv/cosmo)
  by splicing parameter tables from [FINCHES](https://github.com/idptools/finches)
  into COSMO's native `mpipi` model (with full self-validation — see below).
- Generates per-replica random-coil starting structures (self-avoiding walk)
  and simulation inputs matching the Lotthammer 2024 / STARLING training
  protocol: NVT 300 K, 150 mM implicit salt, dt = 20 fs, 500 Å box,
  10 ns equilibration (discarded), 2 ns frame spacing.
- Runs a staged ladder: `pilot` (100 ns × 1) → `E2s` (6 μs × 6,
  protocol-matched) → `E2m` (20 μs × 3) → `E2l` (100 μs × 3, as ten chained
  10 μs segments with cumulative `md_steps` and shared checkpoints).
- Screens convergence with a monitor: Rg / R_e, per-label long-range
  ⟨r⁻⁶⟩ contact scores at the five Dedmon 2005 MTSL sites (Q24, S42, Q62,
  S87, N103), block drift, cross-replica spread.

## Requirements

| Package  | Source | Role |
|---|---|---|
| Python ≥ 3.10 | conda | |
| OpenMM ≥ 8.x | conda-forge | MD engine (CUDA build for GPU) |
| COSMO | GitHub (`vuqv/cosmo`) | CG IDP simulation framework |
| FINCHES | GitHub (`idptools/finches`) | Mpipi / Mpipi-GG parameter tables |
| mdtraj | conda-forge / pip | trajectory analysis (monitor) |
| sparrow | pip (optional) | ALBATROSS Rg prediction for the sanity band |

## Installation

```bash
# 1. environment
conda create -n mpipigg python=3.11 -y
conda activate mpipigg
conda install -c conda-forge openmm cudatoolkit mdtraj -y

# 2. COSMO (GitHub only)
git clone https://github.com/vuqv/cosmo.git
cd cosmo && pip install -e . && cd ..

# 3. FINCHES (GitHub only; note: requires numpy >= 2)
pip install git+https://github.com/idptools/finches.git

# 4. optional: ALBATROSS Rg prediction for the sanity band
pip install sparrow
```

Verify GPU support once: `python -c "from openmm import Platform; \
print(Platform.getPlatformByName('CUDA'))"` — should not raise.

## Usage

```bash
# stage 1: pilot (also regenerates the validated parameter pickle)
python setup_asyn_mpipigg.py --only pilot
sbatch --array=1-1 submit_pilot.slurm

# stage 2: E2s (6 replicas, one GPU each)
python setup_asyn_mpipigg.py --only E2s
sbatch --array=1-6 submit_E2s.slurm

# monitor anytime (safe on running trajectories)
python monitor.py 'E2s/rep*/*'
MONITOR_STRIDE=5 python monitor.py 'E2l/rep*/*'    # lower memory for long arms

# later arms, in order, only after the gates below pass
python setup_asyn_mpipigg.py --only E2m && sbatch --array=1-3 submit_E2m.slurm
python setup_asyn_mpipigg.py --only E2l           # then, sequentially:
sbatch --array=1-3 submit_E2l_seg1.slurm          #   seg N after seg N-1
sbatch --array=1-3 submit_E2l_seg2.slurm
# ... through seg10
```

**Adjust the conda environment name** in the generated `submit_*.slurm`
templates if yours is not `myenv`.

### Gates between arms

Proceed to the next arm only when the previous one shows, via
`monitor_summary.csv` and the per-replica `.monitor.json` files:

1. small cross-replica `range/mean`, `CV`, and `|drift|` for the five
   LR(⟨r⁻⁶⟩) scores — *stationarity of the tail observable, not just Rg*;
2. Rg inside the ALBATROSS Mpipi-GG band (set `RG_REFERENCE_AA` in
   `monitor.py` from sparrow; this is a sanity check, not a pass/fail gate);
3. (E2l only) the two standing verifications below.

## Force-field self-validation

`setup_asyn_mpipigg.py` refuses to run unless, at setup time:

- σ, ν, μ, and rc reproduce COSMO's built-in `mpipi` (< 10⁻³ relative);
- the **full 20×20 ε block** of FINCHES `Mpipi_original` reproduces COSMO's
  `mpipi` after unit scaling (mode A). Unit scales are auto-detected
  (ε × 4.184, σ × 0.1 on the tested FINCHES 0.1.3) — never assumed;
- FINCHES `CHARGE_ALL` matches COSMO's per-residue charges (GG must not
  alter the charge set);
- the `debye_length` registry entry is present and copied (otherwise the
  salt concentration would silently default).

The GG-vs-base difference matrices are written to `mpipi_gg_delta.npz`
(suitable for a supplementary information table), and the validated model is
serialized to `mpipi_gg_params.pkl`, which compute nodes load in ~10 ms
without re-running validation.

## Outputs (per replica, self-contained)

```
E2s/rep1/
  asyn_E2s_rep1.pdb        # random-coil start (topology for the trajectory)
  asyn_E2s_rep1.dcd        # trajectory, 2 ns/frame, first 5 frames = discarded equilibration
  asyn_E2s_rep1.monitor.json
monitor_summary.csv        # cross-replica screening table
versions.txt               # pinned software stack (recorded at setup)
mpipi_gg_delta.npz         # GG vs base parameter differences
```

## Standing verifications (read before E2l)

- **rc = 3σ convention for GG**: asserted exact for COSMO's built-in Mpipi,
  *assumed* for GG — verify against the Lotthammer 2024 supplementary
  information before submitting E2l.
- **COSMO restart semantics**: E2l relies on `md_steps` being a cumulative
  target on restart. Confirm on your COSMO build with the two-leg probe
  (`probe/` pattern: leg 1 = 100k steps, leg 2 = 150k target → 15 frames
  means cumulative, 25 means per-segment).
- **`nstcomm`** (COM removal): recommended by COSMO for single chains; add
  to the INI template once the key name is confirmed against your COSMO
  version's docs.

## Protocol reference

Simulation parameters follow Lotthammer et al., *Nat. Methods* 21, 465–476
(2024) (Mpipi-GG / ALBATROSS; also the training physics of STARLING,
Novak et al., *Nature* 652, 240–250, 2026): NVT 300 K, 150 mM implicit
salt, dt = 20 fs, 500 Å cubic box, 10 ns equilibration discarded,
6 μs production for sequences < 250 residues, 2 ns coordinate saves.
Langevin friction τ = 0.01 ps⁻¹ (weak coupling ≈ 100 ps).
PRE label sites per Dedmon et al., *JACS* 127, 476–477 (2005).

## Notes

- The ladder is a **convergence design, not the STARLING training protocol**
  (which ran ~78k sequences at 6–10 μs each). Arm lengths exist to answer
  "how long must Mpipi-GG run before ⟨r⁻⁶⟩ at the Dedmon sites converges?"
- The monitor's ⟨r⁻⁶⟩ scores are Cα–Cα CG proxies for onset/convergence
  screening. Physical PRE back-calculation (backbone reconstruction + MTSL
  rotamer averaging, e.g. DEER-PREdict) is a downstream step.
