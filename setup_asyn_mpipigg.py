#!/usr/bin/env python
"""
setup_asyn_mpipigg.py -- Mpipi-GG alpha-synuclein ladder for COSMO/OpenMM.

PROTOCOL PARAMETERS (Lotthammer 2024 Methods, which STARLING's training
data used): Mpipi-GG, one bead/residue, NVT 300 K, 150 mM implicit
salt, dt 20 fs, 500 A cubic box, minimization + 500,000 equilibration
steps (10 ns; first 5 frames at 2-ns spacing DISCARDED in analysis --
a continuous run with discard is the protocol's equilibration), then
6 us production for sequences < 250 residues.

THE LADDER IS OUR CONVERGENCE DESIGN, NOT THE STARLING TRAINING
PROTOCOL. The Lotthammer protocol used 6 us (<250 aa) / 10 us (>=250
aa) per sequence; STARLING's training set ultimately comprised nearly
12 million distance maps from ~78,000 such simulations. We run, for
alpha-syn only: pilot (100 ns x1), E2s (6 us x6), E2m (20 us x3),
E2l (100 us x3, CHAINED SEGMENTS via checkpoint/restart). E2l is ~76%
of production compute -- do NOT run it before E2s gates pass.

Force-field construction: FINCHES tables spliced into the protein block
of cosmo's (24,24) arrays by residue 'id'; unit scales auto-detected by
requiring reproduction of cosmo's built-in mpipi (sigma x0.1, eps x4.184
on tested FINCHES 0.1.3 and 1.0.0). eps transferred as cosmo-bare +
(GG - original) from FINCHES -- exact at equal salt whether or not
FINCHES folds any electrostatics into eps, since the common part cancels
in the difference. rc: cosmo's rc/sigma ratio is EXACTLY 3.0 for mpipi
(asserted); the same convention is used for GG, corroborated by FINCHES'
own Wang-Frenkel reference energy (R_ij = 3*sigma_ij hard-coded there);
still verify against Lotthammer 2024 SI before E2l.

RUNTIME SPLICE (IMPORTANT): COSMO dispatches force terms and per-residue
atom types by the model NAME and hard-codes 'mpipi' in three places
(models.py per-residue + nonbonded dispatch, addWangFrenkelForces'
assert, setCAIDPerResidueType's table lookup). Registering a new model
name therefore cannot run on unmodified COSMO. Instead, the validated GG
model entry is serialized to mpipi_gg_params.pkl, and
run_cosmo_mpipigg.py replaces parameters['mpipi'] with it at run time;
the INI keeps model = mpipi. Provenance lives in mpipi_gg_delta.npz,
the runinfo logs, and versions.txt.

RUN ORDER:
  1. python setup_asyn_mpipigg.py --only pilot ; run pilot; sanity-check
  2. python setup_asyn_mpipigg.py --only E2s  ; run E2s
  3. gates: monitor.py cross-replica spread + block drift (ALBATROSS Rg
     is a SANITY band, not a pass/fail criterion)
  4. then --only E2m, then E2l (chained segments, restart=yes)

All runs are plain invocations of `python run_cosmo_mpipigg.py -f <ini>`;
scheduler integration (arrays, job dependencies) is deliberately left to
the user's site and is not part of this repo.
"""
import os, sys, json, platform, subprocess

SALT_M, DIELECTRIC = 0.150, 80.0
DT_PS = 0.02
NSTXOUT = 100000                     # 2 ns/frame, per protocol
EQ_DISCARD_NS = 10
EQ_STEPS = int(EQ_DISCARD_NS * 1000 / DT_PS)
EQ_FRAMES = EQ_STEPS // NSTXOUT
assert EQ_STEPS == 500_000 and EQ_FRAMES == 5
NSTCHK = 500_000                     # checkpoint every 10 ns (E2l chaining)
NSTCOMM = 100                        # COM removal every 100 steps (2 ps);
                                     # our choice -- COSMO default is off
BOND_A, EV_MIN_A = 3.8, 3.5          # our SAW choice; not from the paper
ARM_OFFSET = {'pilot': 0, 'E2s': 1, 'E2m': 2, 'E2l': 3}  # distinct seeds per arm

ASYN = ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNV"
        "GGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEE"
        "GYQDYEPEA")
assert len(ASYN) == 140
THREE = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
         'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
         'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
AAS = "ACDEFGHIKLMNPQRSTVWY"

def _saw(rng):
    import numpy as np
    n = len(ASYN)
    coords = np.zeros((n, 3))
    coords[0] = rng.uniform(-5, 5, 3)
    for i in range(1, n):
        for _ in range(20000):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            trial = coords[i - 1] + BOND_A * v
            if i == 1 or (np.linalg.norm(trial - coords[:i-1], axis=1) > EV_MIN_A).all():
                coords[i] = trial
                break
        else:
            raise RuntimeError(f"SAW trapped at residue {i+1}")
    return coords - coords.mean(axis=0)

def write_random_coil_pdb(path, seed):
    import numpy as np
    for attempt in range(100):
        try:
            coords = _saw(np.random.default_rng(seed + attempt))
            break
        except RuntimeError:
            continue
    else:
        sys.exit("could not generate SAW in 100 attempts")
    with open(path, 'w') as f:
        for i, aa in enumerate(ASYN, 1):
            x, y, z = coords[i - 1]
            f.write(f"ATOM  {i:5d}  CA  {THREE[aa]} A{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("TER\nEND\n")

def load_finches_tables():
    from finches.forcefields.mpipi import Mpipi_model
    gg = Mpipi_model("Mpipi_GGv1", salt=SALT_M, dielectric=DIELECTRIC)
    orig = Mpipi_model("Mpipi_original", salt=SALT_M, dielectric=DIELECTRIC)
    grab = lambda m: {"eps": m.EPSILON_ALL, "sigma": m.SIGMA_ALL,
                      "nu": m.NU_ALL, "mu": m.MU_ALL}
    gg_d, orig_d = grab(gg), grab(orig)
    charges = dict(gg.CHARGE_ALL)   # GG must not change the charge set
    for tag, d in (("GG", gg_d), ("original", orig_d)):
        for k, tab in d.items():
            for a in AAS:
                for b in AAS:
                    v = float(tab[a][b])
                    if v != v or abs(v) == float("inf"):
                        sys.exit(f"non-finite FINCHES {tag}.{k}[{a}][{b}] = {v}")
    return gg_d, orig_d, charges

def build_tables(base, finches_d, sigma_scale, eps_scale):
    """Splice FINCHES pair tables into cosmo's (24,24) protein block.
    FINCHES table units are version-dependent; the (sigma, eps) scale pair
    is auto-detected by requiring reproduction of cosmo's built-in mpipi
    (tested installs: sigma x0.1 [Ang->nm], eps x4.184 [kcal->kJ]).
    eps returned as scaled FINCHES values -- caller applies the delta
    correction against Mpipi_original (exact whether or not FINCHES folds
    the Debye-Huckel term into eps; see mode A/B logic in build_gg_model)."""
    import numpy as np
    eps = np.array(base['eps_ij'], dtype=float)
    sig = np.array(base['sigma_ij'], dtype=float)
    nu  = np.array(base['nu_ij'], dtype=float)
    mu  = np.array(base['mu_ij'], dtype=float)
    rc_b = np.asarray(base['rc_ij'], dtype=float)
    assert np.allclose(rc_b / sig, 3.0, rtol=0, atol=1e-12), \
        "base mpipi rc/sigma is not exactly 3.0; re-examine rc convention"
    ratio = rc_b / sig
    id_of = {t: base[t]['id'] for t in THREE.values()}
    for a1 in AAS:
        i = id_of[THREE[a1]]
        for a2 in AAS:
            j = id_of[THREE[a2]]
            eps[i, j] = float(finches_d['eps'][a1][a2]) * eps_scale
            sig[i, j] = float(finches_d['sigma'][a1][a2]) * sigma_scale
            nu[i, j]  = float(finches_d['nu'][a1][a2])
            mu[i, j]  = float(finches_d['mu'][a1][a2])
    return {'eps_ij': eps, 'sigma_ij': sig, 'nu_ij': nu, 'mu_ij': mu,
            'rc_ij': ratio * sig}

def build_gg_model():
    import numpy as np, copy, importlib
    import cosmo.parameters.model_parameters as mp
    assert mp.parameters is importlib.import_module(
        'cosmo.parameters.model_parameters').parameters

    base = mp.parameters['mpipi']
    gg_d, orig_d, charges = load_finches_tables()
    for a in AAS:
        c_cos = float(base[THREE[a]]['charge'])
        c_fin = float(charges[a])
        if abs(c_cos - c_fin) > 1e-6:
            sys.exit(f"charge mismatch {a}: cosmo={c_cos} vs FINCHES={c_fin}; "
                     "GG must not alter the charge set")
    print("[2] charge set matches FINCHES (GG unchanged).")
    id_of = {t: base[t]['id'] for t in THREE.values()}
    charged_ids = {id_of[t] for t in THREE.values() if base[t]['charge'] != 0}
    prot = [id_of[THREE[a]] for a in AAS]
    unch = [(i, j) for i in prot for j in prot
            if i not in charged_ids and j not in charged_ids]

    def relmax(new, ref, pairs=None):
        if pairs is None:
            return float(np.nanmax(np.abs(new - ref))) / (float(np.nanmax(np.abs(ref))) or 1.0)
        if not pairs:
            return 0.0                     # no neutral pairs: nothing to check
        d = max(abs(new[i, j] - ref[i, j]) for i, j in pairs)
        s = max(abs(ref[i, j]) for i, j in pairs) or 1.0
        return d / s

    best = None
    mode_b_seen = False
    for ss in (1.0, 0.1):
        for es in (1.0, 4.184):
            o = build_tables(base, orig_d, ss, es)
            base_eps_full = np.asarray(base['eps_ij'], float)
            r_eps_full = relmax(o['eps_ij'], base_eps_full)            # all pairs
            r_eps_unch = relmax(o['eps_ij'], base_eps_full, unch)      # neutral only
            r = {k: relmax(o[k], np.asarray(base[k], float),
                          unch if k == 'eps_ij' else None)
                 for k in ['eps_ij', 'sigma_ij', 'nu_ij', 'mu_ij', 'rc_ij']}
            print(f"[2] validate sig={ss} eps={es}: " +
                  "  ".join(f"{k} rel={v:.2e}" for k, v in r.items()) +
                  f"  | eps full={r_eps_full:.2e}")
            other = max(r[k] for k in ['sigma_ij', 'nu_ij', 'mu_ij', 'rc_ij'])
            if other >= 1e-3:
                continue                       # wrong unit scales; keep looking
            if r_eps_full < 1e-3:
                best = ((ss, es), o)
                print(f"[2] self-validation passed (sigma x{ss}, eps x{es}); "
                      f"FINCHES eps mode: A (bare eps, full-matrix match).")
                break
            if r_eps_unch < 1e-3:
                mode_b_seen = True             # neutral matches, charged does not
        if best is not None:
            break
    if best is None:
        if mode_b_seen:
            sys.exit("mode B detected: charged-pair eps does not match cosmo "
                     "after unit scaling (DH folded into FINCHES eps?). "
                     "Expected mode A on cosmo. Inspect the charged-pair "
                     "delta matrix before trusting any splice.")
        sys.exit("FINCHES Mpipi_original does not reproduce cosmo mpipi. Aborting.")
    (ss, es), o = best
    base_eps = np.asarray(base['eps_ij'], float)

    cn = [(i, j) for i in charged_ids for j in prot if j not in charged_ids]
    d_cn = max(abs(o['eps_ij'][i, j] - base_eps[i, j]) for i, j in cn) if cn else 0.0
    print(f"[2] charged-neutral eps max|delta| = {d_cn:.4f} kJ/mol (expect ~0).")

    g = build_tables(base, gg_d, ss, es)
    g['eps_ij'] = base_eps + (g['eps_ij'] - o['eps_ij'])
    np.savez('mpipi_gg_delta.npz',
             **{f'delta_{k}': g[k] - np.asarray(base[k], float)
                for k in ['eps_ij', 'sigma_ij', 'nu_ij', 'mu_ij', 'rc_ij']})

    entry = copy.deepcopy(base)
    for k in ['eps_ij', 'sigma_ij', 'nu_ij', 'mu_ij', 'rc_ij']:
        entry[k] = g[k]
    dl = getattr(mp, 'debye_length', None)
    if not (isinstance(dl, dict) and 'mpipi' in dl):
        sys.exit("cosmo debye_length registry not found -- cannot guarantee "
                 "150 mM; aborting rather than simulating the wrong salt.")
    print(f"[2] debye_length['mpipi'] = {dl['mpipi']} nm (150 mM); recorded "
          f"in pkl for a runtime sanity check (no registry edit needed: the "
          f"INI keeps model = mpipi).")

    d = np.abs(g['eps_ij'] - base_eps)
    rows = sorted(range(24), key=lambda k: -float(np.nanmax(d[k, :])))
    inv = {v: k for k, v in id_of.items()}
    print("[2] built Mpipi-GG entry. epsilon rows most changed by GG:")
    for k in rows[:5]:
        print(f"      {inv.get(k, f'idx{k}'):4s}: max|deps| = {np.nanmax(d[k, :]):.4f} kJ/mol")
    print("[2] NOTE before E2l: rc = 3*sigma for GG is corroborated by "
          "FINCHES' own Wang-Frenkel energy (R_ij = 3*sigma_ij hard-coded); "
          "still verify against Lotthammer 2024 SI.")
    import pickle
    with open('mpipi_gg_params.pkl', 'wb') as fh:
        pickle.dump({'gg_entry': entry,
                     'debye_length_mpipi': dl['mpipi'],
                     'unit_scales': {'sigma': ss, 'eps': es},
                     'spliced_into': 'mpipi'}, fh)
    print("[2] saved validated GG entry to mpipi_gg_params.pkl "
          "(run_cosmo_mpipigg.py splices it into parameters['mpipi'] "
          "at run time; no re-validation on compute nodes)")
    return entry

# Ladder design lives here only. NOTE: the pre-cov proposal (Sept 2026)
# lists E2l as 6 replicas and drops E2m (20 us is a nested prefix of E2l);
# reconcile this table with the design doc before the campaign.
ARMS = {'pilot': (0.1, 1), 'E2s': (6, 6), 'E2m': (20, 3), 'E2l': (100, 3)}
E2L_SEG_US = 10        # E2l runs as explicit 10-us chained segments
E2L_NSEG = ARMS['E2l'][0] // E2L_SEG_US
assert ARMS['E2l'][0] % E2L_SEG_US == 0, 'E2l length must divide segment size'
INI = """[OPTIONS]
# {arm} replica {rep}: Mpipi-GG, 300 K, 150 mM, dt 20 fs, 500 A box.
# Equilibration = first {eq} ns (frames 1-{eqf}), discarded in analysis.
# tau_t = Langevin friction coefficient in ps^-1; 0.01 matches the
# Lotthammer weak-coupling (~100 ps) protocol and COSMO's example.
# nstcomm = COM removal every 100 steps (2 ps); our choice (COSMO default
# is off). Delete the key if your COSMO build predates nstcomm support.
# model stays 'mpipi' because COSMO dispatches force terms by that name;
# run_cosmo_mpipigg.py splices the validated Mpipi-GG tables into
# parameters['mpipi'] at run time (see README).
# E2l: chain segments -- run seg N after seg N-1 (restart=yes resumes
# from the shared checkpoint; md_steps is a CUMULATIVE target, verified
# on COSMO 2026.2.dev2 by probe_restart.py).
md_steps      = {steps}
dt            = 0.02
nstxout       = {nstxout}
nstchk        = {nstchk}
nstlog        = 50000
nstcomm       = {nstcomm}
model         = mpipi
tcoupl        = yes
ref_t         = 300
tau_t         = 0.01
pcoupl        = no
pbc           = yes
box_dimension = 50
pdb_file      = {pdb}
output_dir    = {outdir}
outname       = {outname}
device        = GPU
restart       = {restart}
minimize      = yes
"""

def record_versions():
    import numpy as np
    info = {"python": platform.python_version(), "numpy": np.__version__}
    for pkg in ("cosmo", "openmm", "finches"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
            info[f"{pkg}_path"] = getattr(mod, "__file__", "n/a") or "n/a"
        except ImportError:
            info[pkg] = "NOT IMPORTABLE"
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
                              '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
        info['gpu_submit_node'] = out or 'n/a'
    except Exception:
        pass
    with open('versions.txt', 'w') as f:
        json.dump(info, f, indent=2)
    print("[0] versions ->", json.dumps(info))

def main():
    only = None
    if '--only' in sys.argv:
        try:
            only = sys.argv[sys.argv.index('--only') + 1]
        except IndexError:
            sys.exit("usage: python setup_asyn_mpipigg.py [--only ARM] "
                     f"with ARM in {list(ARMS)}")
        if only not in ARMS:
            sys.exit(f"--only must be one of {list(ARMS)}")
    record_versions()
    build_gg_model()
    arms = {only: ARMS[only]} if only else ARMS
    for arm, (us, nrep) in arms.items():
        prod = int(us * 1e6 / DT_PS)      # production only; equilibration
        segs = range(1, E2L_NSEG + 1) if arm == 'E2l' else range(1, 2)
        steps_per_seg = int(E2L_SEG_US * 1e6 / DT_PS) if arm == 'E2l' else prod
        for r in range(1, nrep + 1):
            outdir = f"{arm}/rep{r}"; os.makedirs(outdir, exist_ok=True)
            pdb = f"{outdir}/asyn_{arm}_rep{r}.pdb"
            write_random_coil_pdb(pdb, seed=10_000 * ARM_OFFSET[arm] + r)
            for seg in segs:
                # segment 1 carries minimization + equilibration; later
                # segments restart from the .chk and add pure production
                eq_ns = EQ_DISCARD_NS if seg == 1 else 0
                eqf = EQ_FRAMES if seg == 1 else 0
                # COSMO restart treats md_steps as a CUMULATIVE target:
                # it runs (md_steps - steps_already_done), verified on
                # COSMO 2026.2.dev2 both by code inspection and by
                # probe_restart.py. So segment N's md_steps must be the
                # total through segment N, not the length of segment N.
                seg_steps = seg * steps_per_seg + EQ_STEPS
                restart = 'no' if seg == 1 else 'yes'
                ini = INI.format(arm=arm, rep=r, eq=eq_ns, eqf=eqf,
                                 steps=seg_steps, nstxout=NSTXOUT,
                                 nstchk=NSTCHK, nstcomm=NSTCOMM,
                                 pdb=pdb, outdir=outdir,
                                 outname=f"asyn_{arm}_rep{r}",
                                 restart=restart)
                with open(f"md_{arm}_{r}_seg{seg}.ini", 'w') as fh:
                    fh.write(ini)
    print(f"[3] wrote {sum(n for _, n in arms.values())} replicas "
          f"across {len(arms)} arm(s): per-replica PDBs + per-segment INIs.")
    print("[3] run with the static repo runner, e.g.:")
    for arm in arms:
        nrep = ARMS[arm][1]
        if arm == 'E2l':
            print(f"      # {arm}: segments sequential per replica")
            print(f"      for r in $(seq 1 {nrep}); do for s in $(seq 1 {E2L_NSEG}); do "
                  f"python run_cosmo_mpipigg.py -f md_{arm}_${{r}}_seg${{s}}.ini; done; done")
        else:
            print(f"      for r in $(seq 1 {nrep}); do "
                  f"python run_cosmo_mpipigg.py -f md_{arm}_${{r}}_seg1.ini; done")
    print("[3] gates before next arm: monitor.py (cross-replica spread, "
          "block drift; ALBATROSS Rg = sanity band only).")

if __name__ == '__main__':
    main()
