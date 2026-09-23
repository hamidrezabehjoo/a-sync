# Mpipi-GG α-Synuclein Simulation Ladder

COSMO/OpenMM pipeline for running **Mpipi-GG** coarse-grained simulations of
human α-synuclein (140 aa) as a staged, convergence-labeled ladder. Built to
audit whether the force field behind generative IDP models (STARLING) is
sampled long enough to capture the rare compact states that PRE experiments
detect.

## What this does

- Builds a validated **Mpipi-GG** model from [COSMO](https://github.com/vuqv/cosmo)
  + [FINCHES](https://github.com/idptools/finches) parameter tables (with full
  self-validation — see below).
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

## Repository layout

```
setup_asyn_mpipigg.py   # setup + force-field self-validation + INI/PDB generation
run_cosmo_mpipigg.py    # static runner: splices the validated GG tables, calls mdrun
monitor.py              # convergence screening (safe on running trajectories)
probe_restart.py        # two-leg probe of COSMO restart semantics (run once per build)
```

Generated at setup time (not tracked): `md_<arm>_<rep>_seg<seg>.ini`,
`<arm>/rep*/asyn_*.pdb`, `mpipi_gg_params.pkl`, `mpipi_gg_delta.npz`,
`versions.txt`.

## Requirements

| Package  | Source | Role |
|---|---|---|
| Python ≥ 3.10 | conda | |
| OpenMM ≥ 8.x | conda-forge | MD engine (CUDA build for GPU) |
| COSMO | GitHub (`vuqv/cosmo`) | CG IDP simulation framework |
| FINCHES | GitHub (`idptools/finches`) | Mpipi / Mpipi-GG parameter tables |
| mdtraj | conda-forge / pip | trajectory analysis (monitor) |
| sparrow | pip (optional) | ALBATROSS Rg prediction for the sanity band |

```bash
conda create -n mpipigg python=3.11 -y
conda activate mpipigg
conda install -c conda-forge openmm cudatoolkit mdtraj -y
git clone https://github.com/vuqv/cosmo.git && pip install -e ./cosmo
pip install git+https://github.com/idptools/finches.git   # tested: 0.1.3 and current master (1.0.0); pin a commit for production
pip install sparrow   # optional
```

**Version compatibility.** Verified end-to-end on COSMO `2026.2.dev2` +
FINCHES `1.0.0` + OpenMM `8.6.1`. The runner handles both COSMO entry
points (`main` on older builds, `mdrun` on current). FINCHES pulls in
`metapredict` (hence torch) at import time — install the full dependency
chain; only the parameter tables are used here, but the import is eager.

## Usage

```bash
# stage 1: pilot (also regenerates the validated parameter pickle)
python setup_asyn_mpipigg.py --only pilot
python run_cosmo_mpipigg.py -f md_pilot_1_seg1.ini

# stage 2: E2s (6 independent replicas; run them in parallel however
# your site prefers — one GPU each)
python setup_asyn_mpipigg.py --only E2s
for r in $(seq 1 6); do
    python run_cosmo_mpipigg.py -f md_E2s_${r}_seg1.ini &
done; wait

# monitor anytime (safe on running trajectories)
python monitor.py 'E2s/rep*/*'
MONITOR_STRIDE=5 python monitor.py 'E2l/rep*/*'    # lower memory for long arms

# later arms, in order, only after the gates below pass
python setup_asyn_mpipigg.py --only E2m
# E2l: segments are SEQUENTIAL per replica (seg N resumes seg N-1's checkpoint)
for r in 1 2 3; do
    for s in $(seq 1 10); do
        python run_cosmo_mpipigg.py -f md_E2l_${r}_seg${s}.ini
    done
done
```

The INIs request `device = GPU`. For a CPU run (e.g. a local smoke test),
set `device = CPU` and add `ppn = <cores>` (COSMO defaults to 1 thread).
There is no GPU→CPU automatic fallback: a GPU INI on a GPU-less node fails
at platform selection.

### Gates between arms

Proceed to the next arm only when the previous one shows, via
`monitor_summary.csv` and the per-replica `.monitor.json` files:

1. small cross-replica `range/mean`, `CV`, and `|drift|` for the five
   LR(⟨r⁻⁶⟩) scores — *stationarity of the tail observable, not just Rg*;
2. Rg inside the ALBATROSS Mpipi-GG band (set `RG_REFERENCE_AA` in
   `monitor.py` from sparrow; this is a sanity check, not a pass/fail gate).

## Force-field construction and self-validation

`setup_asyn_mpipigg.py` refuses to run unless, at setup time:

- σ, ν, μ, and rc reproduce COSMO's built-in `mpipi` (< 10⁻³ relative);
- the **full 20×20 ε block** of FINCHES `Mpipi_original` reproduces COSMO's
  `mpipi` after unit scaling (mode A). Unit scales are auto-detected
  (ε × 4.184, σ × 0.1 on tested FINCHES 0.1.3 and 1.0.0) — never assumed;
  a neutral-only match with charged-pair mismatch (mode B, DH folded into
  ε) aborts with a distinct message;
- FINCHES `CHARGE_ALL` matches COSMO's per-residue charges (GG must not
  alter the charge set);
- the `debye_length` registry entry for `mpipi` is present (0.795 nm at
  150 mM) and is re-checked by the runner at run time.

**Why the GG tables are spliced in place.** COSMO dispatches force terms
and per-residue atom types by the model *name*, hard-coding `'mpipi'` in
three places (`models.py` per-residue and nonbonded dispatch,
`addWangFrenkelForces`' assert, `setCAIDPerResidueType`'s table lookup).
Registering a new `mpipi_gg` model name therefore builds no Wang-Frenkel
force on unmodified COSMO (the run dies with `UnboundLocalError: nb_name`).
Instead, `run_cosmo_mpipigg.py` **replaces** `parameters['mpipi']` with
the validated GG entry before calling `mdrun`; the INI keeps
`model = mpipi`, and provenance lives in `mpipi_gg_delta.npz`
(GG-vs-base difference matrices, suitable for an SI table), the
`*_runinfo.log` files, and `versions.txt`.

## Outputs (per replica, self-contained)

```
E2s/rep1/
  asyn_E2s_rep1.pdb        # random-coil start (topology for the trajectory)
  asyn_E2s_rep1.dcd        # trajectory, 2 ns/frame, first 5 frames = discarded equilibration
  asyn_E2s_rep1.chk/.log/_runinfo.log/_init.pdb/_final.pdb/.psf
  asyn_E2s_rep1.monitor.json
monitor_summary.csv        # cross-replica screening table (needs ≥2 finished replicas)
versions.txt               # pinned software stack (recorded at setup)
mpipi_gg_delta.npz         # GG vs base parameter differences
```

## Verifications

- **PDB output is mdtraj-parseable — verified.** `write_random_coil_pdb`
  writes strict fixed-column PDB (atom name cols 13–16, resName 18–20,
  chainID 22, resSeq 23–26, x from col 31); `mdtraj.load` parses the
  generated topology with no `ValueError`. (The pre-rewrite script was one
  space short in the atom-name field and every PDB it wrote failed to load.)
- **Per-replica seeds are collision-free — verified.** Seeds are
  `10_000 * ARM_OFFSET[arm] + r` with distinct offsets per arm
  (pilot/E2s/E2m/E2l = 0/1/2/3): all 13 per-replica seeds across the
  ladder are unique (checked exhaustively for replica indices 1–6 in
  every arm). (The old `1000*len(arm)+r` formula collided across all
  three-character arm names.)
- **MONITOR_STRIDE 5–10 — verified.** Any positive stride runs: the discard
  window is recomputed as `ceil(EQ_NS / (2 ns * stride))` with a printed
  note on slight over-discard, instead of the old assertion that rejected
  every documented value except 5. Swept 5–10 through `analyze()` on a real
  DCD: all pass.
- **Restart semantics — verified.** COSMO treats `md_steps` as a
  *cumulative* target on restart (`nsteps_remain = md_steps − done_steps`),
  confirmed on `2026.2.dev2` both by code inspection and empirically by
  `probe_restart.py` (60k + restart@120k → 12 frames, one appended DCD).
  COSMO is a moving target: re-run `python probe_restart.py md_pilot_1_seg1.ini`
  once on your build before launching E2l.
- **rc = 3σ convention for GG**: asserted exact for COSMO's built-in Mpipi
  and corroborated by FINCHES' own Wang-Frenkel reference energy
  (`R_ij = 3·sigma_ij` hard-coded for the same tables). Still verify against
  the Lotthammer 2024 supplementary information before E2l.
- **`nstcomm`** (COM removal): supported by current COSMO (default off);
  the INI sets `nstcomm = 100`. Delete the key if your build predates it.

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
- The ladder table (`ARMS` in `setup_asyn_mpipigg.py`) is the single source
  of truth for arm lengths/replica counts — note the pre-cov design doc
  (Sept 2026) lists E2l as 6 replicas and drops E2m; reconcile before the
  campaign.
