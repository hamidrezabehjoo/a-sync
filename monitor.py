#!/usr/bin/env python
"""
monitor.py -- convergence screening for the Mpipi-GG alpha-syn ladder.

Production statistics only: the first EQ_FRAMES frames (10 ns
equilibration at 2 ns/frame) are discarded -- that discard IS the
equilibration. All frame arithmetic is DERIVED from the simulation
parameters (asserts catch setup/monitor drift).

Observables (production region):
  * Rg, Re (+ percentiles, block means)
  * per-label long-range r^-6 contact score:
        LR(s) = < sum_{j: |j-s| > 10} r_{s,j}^{-6} >
    for the five DEDMON 2005 MTSL sites (JACS 127:476-477, verified
    against the primary source: Q24, S42, Q62, S87, N103). Label-
    referenced C-alpha CG proxy for PRE's long-range contact
    sensitivity -- NOT the experimental quantity (which needs backbone
    reconstruction + MTSL rotamer averaging via DEER-PREdict).
    NORMALIZATION: unnormalized sum; all simulations share the same
    140-residue sequence, so a per-residue mean would only apply a
    common constant.
  * distribution stats (p50/p95) and 10-block means per observable.

Convergence claims: CAN establish stationarity of these observables
within replicas (block trends) and consistency across replicas
(range/mean, CV, drift). CANNOT prove full state-space exploration --
that is the Phase-0 notebook's job (contact maps, clustering, PCA/tICA,
state populations). ALBATROSS Rg is a SANITY BAND, not a gate.

Writes: stdout, <traj>.monitor.json per replica, monitor_summary.csv.
MONITOR_STRIDE=5..10 for long arms (discard window is recomputed and
asserted to align with EQ_NS exactly).

Usage:
  python monitor.py 'E2s/rep*/*'
  python monitor.py 'E2s/rep*/*.dcd'
  python monitor.py E2s/rep1
  MONITOR_STRIDE=5 python monitor.py 'E2l/rep*/*'
"""
import sys, os, json, glob, csv
import numpy as np

try:
    import mdtraj as md
except ImportError:
    sys.exit("pip install mdtraj")

DT_PS, NSTXOUT, EQ_NS = 0.02, 100000, 10.0
FRAME_INTERVAL_NS = DT_PS * NSTXOUT / 1000.0
EQ_FRAMES = int(round(EQ_NS / FRAME_INTERVAL_NS))
assert FRAME_INTERVAL_NS == 2.0 and EQ_FRAMES == 5, \
    "frame arithmetic out of sync with the simulation INIs"
STRIDE = int(os.environ.get('MONITOR_STRIDE', 1))
N_BLOCKS = 10
RG_REFERENCE_AA = None        # <-- sparrow Mpipi-GG aSyn Rg; None disables
RG_BAND_AA = 3.0
LOCAL_EXCL = 10

SITES = {'Q24': 24, 'S42': 42, 'Q62': 62, 'S87': 87, 'N103': 103}
# Dedmon et al., JACS 127:476-477 (2005), primary-source verified.
# (Bertoncini et al. PNAS 2005 used A18C/A90C/A140C -- different paper.)

def find_trajectories(pattern):
    if pattern.endswith(('.dcd', '.xtc')):
        return sorted(glob.glob(pattern))
    if os.path.isdir(pattern):
        pats = [os.path.join(pattern, '*.dcd'),
                os.path.join(pattern, '*.xtc')]
    else:
        pats = [pattern + '.dcd', pattern + '.xtc']
    hits = []
    for p in pats:
        hits.extend(glob.glob(p))
    return sorted(set(hits))

def find_topology(traj):
    d = os.path.dirname(traj) or '.'
    stem = os.path.splitext(traj)[0]
    for cand in (stem + '.pdb', stem + '_init.pdb',
                 *sorted(glob.glob(d + '/*.pdb')),
                 *sorted(glob.glob('asyn_*.pdb'))):
        if os.path.exists(cand):
            return cand
    return None

def block_means(series, n_blocks=N_BLOCKS):
    # variable length (< n_blocks) for short trajectories -- downstream
    # consumers must handle len < n_blocks; no NaN padding (not strict JSON)
    n = len(series)
    if n < n_blocks:
        return series.reshape(1, -1).mean(axis=1)
    b = n // n_blocks
    return series[:b * n_blocks].reshape(n_blocks, b).mean(axis=1)

def analyze(traj, top_pdb):
    t = md.load(traj, top=top_pdb, stride=STRIDE)
    xyz = t.xyz * 10.0          # mdtraj returns nm; convert to Angstrom
    n = len(xyz)
    eff_interval_ns = FRAME_INTERVAL_NS * STRIDE
    eq_frames_eff = int(np.ceil(EQ_NS / eff_interval_ns))
    if abs(eq_frames_eff * eff_interval_ns - EQ_NS) > 1e-9:
        print(f"note: STRIDE={STRIDE} does not exactly tile EQ_NS={EQ_NS} "
              f"-- discarding {eq_frames_eff * eff_interval_ns:.1f} ns "
              f"(slight over-discard, harmless)")
    if n <= eq_frames_eff:
        return {'frames_total': n * STRIDE, 'frames_prod': 0}
    xyz_p = xyz[eq_frames_eff:]
    np_ = len(xyz_p)
    cm = xyz_p.mean(axis=1, keepdims=True)
    rg = np.sqrt(((xyz_p - cm) ** 2).sum(-1).mean(-1))
    re = np.linalg.norm(xyz_p[:, 0, :] - xyz_p[:, -1, :], axis=-1)
    out = {'frames_total': n * STRIDE, 'frames_prod': np_ * STRIDE,
           'stride': STRIDE, 'ok': True,
           'Rg_mean': float(rg.mean()),
           'Rg_p16': float(np.percentile(rg, 16)),
           'Rg_p84': float(np.percentile(rg, 84)),
           'Rg_blocks': block_means(rg).round(4).tolist(),
           'Re_mean': float(re.mean()),
           'Re_p16': float(np.percentile(re, 16)),
           'Re_p84': float(np.percentile(re, 84))}
    nres = xyz_p.shape[1]
    for name, s in SITES.items():
        d = np.linalg.norm(xyz_p - xyz_p[:, s - 1:s, :], axis=-1)
        mask = np.abs(np.arange(1, nres + 1) - s) <= LOCAL_EXCL
        d[:, mask] = np.inf
        with np.errstate(divide='ignore'):
            r6 = np.nan_to_num(d ** -6, nan=0.0, posinf=0.0).sum(axis=1)
        half = np_ // 2
        out[f'LR_{name}'] = float(r6.mean())
        out[f'LR_{name}_p50'] = float(np.percentile(r6, 50))
        out[f'LR_{name}_p95'] = float(np.percentile(r6, 95))
        out[f'LR_{name}_drift'] = float((r6[:half].mean() - r6[half:].mean())
                                        / r6.mean())
        out[f'LR_{name}_blocks'] = block_means(r6).round(6).tolist()
    return out

def fmt(o):
    if o.get('frames_prod', 0) == 0:
        return f"n={o['frames_total']} (still in equilibration window)"
    flag = ''
    if RG_REFERENCE_AA is not None and \
            abs(o['Rg_mean'] - RG_REFERENCE_AA) > RG_BAND_AA:
        flag = '  <-- outside ALBATROSS sanity band (investigate, not fatal)'
    s = (f"n_prod={o['frames_prod']:6d}  "
         f"Rg={o['Rg_mean']:5.1f}[{o['Rg_p16']:4.1f},{o['Rg_p84']:4.1f}]  "
         f"Re={o['Re_mean']:5.1f}[{o['Re_p16']:4.1f},{o['Re_p84']:4.1f}]{flag}")
    for name in SITES:
        s += (f"\n    {name}: LR={o['LR_' + name]:.3e} "
              f"[p50 {o['LR_' + name + '_p50']:.2e}, "
              f"p95 {o['LR_' + name + '_p95']:.2e}] "
              f"drift={o['LR_' + name + '_drift']:+.1%}")
    return s

if __name__ == '__main__':
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'E2*/rep*/*'
    files = find_trajectories(pattern)
    if not files:
        print(f"no trajectories match '{pattern}' (.dcd/.xtc) yet")
        sys.exit(0)
    summary = []
    for traj in files:
        top = find_topology(traj)
        if top is None:
            print(f"{traj}: no topology; skipping"); continue
        try:
            o = analyze(traj, top)
        except Exception as e:
            print(f"{traj}: not readable yet ({e})"); continue
        print(f"{traj}:\n    {fmt(o)}")
        if o.get('ok'):
            with open(os.path.splitext(traj)[0] + '.monitor.json', 'w') as fh:
                json.dump(o, fh, indent=2)
        summary.append((traj, o))

    done = [(x, o) for x, o in summary if o.get('ok')]
    if len(done) >= 2:
        print("\ncross-replica summary (label-referenced LR scores):")
        print(f"  {'site':6s} {'mean':>10s} {'range/mean':>11s} {'CV':>7s} "
              f"{'mean drift':>11s} {'mean |drift|':>12s}")
        for name in SITES:
            vals = np.array([o[f'LR_{name}'] for _, o in done])
            dr = np.array([o[f'LR_{name}_drift'] for _, o in done])
            print(f"  {name:6s} {vals.mean():10.3e} "
                  f"{(vals.max() - vals.min()) / vals.mean():11.1%} "
                  f"{vals.std() / vals.mean():7.1%} "
                  f"{dr.mean():+11.1%} {np.abs(dr).mean():12.1%}")
        print("\nreading: small range/mean, CV, and |drift| are evidence of "
              "stationarity and cross-replica consistency FOR THESE "
              "OBSERVABLES. Large range/mean -> written justification for "
              "E2m/E2l. Large |drift| -> not yet stationary. None of this "
              "proves full state-space exploration (Phase-0 notebook).")
        with open('monitor_summary.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            hdr = ['traj', 'frames_prod', 'Rg_mean', 'Rg_p16', 'Rg_p84',
                   'Re_mean'] + [f'LR_{s}' for s in SITES] + \
                  [f'LR_{s}_drift' for s in SITES]
            w.writerow(hdr)
            for x, o in done:
                w.writerow([x, o['frames_prod'], o['Rg_mean'], o['Rg_p16'],
                            o['Rg_p84'], o['Re_mean']] +
                           [o[f'LR_{s}'] for s in SITES] +
                           [o[f'LR_{s}_drift'] for s in SITES])
        print("wrote monitor_summary.csv and per-replica .monitor.json files.")
