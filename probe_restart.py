#!/usr/bin/env python
"""
probe_restart.py -- two-leg probe of COSMO's restart semantics.

E2l relies on `md_steps` being a CUMULATIVE target on restart (segment N
specifies the total steps through segment N, and COSMO runs the
difference). This is verified on COSMO 2026.2.dev2 -- both by code
inspection (`nsteps_remain = cfg.md_steps - done_steps`) and by this
probe -- but COSMO is a moving target, so re-run this probe once on
your own build before launching E2l.

Design: leg 1 runs 60k steps from scratch; leg 2 restarts with
md_steps = 120k. If md_steps is cumulative the pair yields 12 frames
(10k-step spacing); if it is per-segment it yields 18.

Usage:
    python probe_restart.py md_pilot_1_seg1.ini

Takes the PDB/device from the given INI; writes to ./probe/ and leaves
the arm outputs untouched.
"""
import os, re, subprocess, sys

LEG1_STEPS, LEG2_STEPS, NSTXOUT = 60_000, 120_000, 10_000
EXPECTED = {'cumulative': (LEG2_STEPS) // NSTXOUT,
            'per-segment': (LEG1_STEPS + LEG2_STEPS) // NSTXOUT}

def _set(ini, key, value):
    if not re.search(rf'^{key}\s*=', ini, flags=re.M):
        sys.exit(f'[probe] key "{key}" not found in base INI -- '
                 f'template drift; refusing to continue')
    return re.sub(rf'^{key}\s*=.*$', f'{key} = {value}', ini,
                  count=1, flags=re.M)

def main(base_ini):
    base = open(base_ini).read()
    os.makedirs('probe', exist_ok=True)
    legs = []
    for tag, steps, restart in (('leg1', LEG1_STEPS, 'no'),
                                ('leg2', LEG2_STEPS, 'yes')):
        ini = _set(base, 'md_steps', steps)
        ini = _set(ini, 'nstxout', NSTXOUT)
        ini = _set(ini, 'nstchk', 30_000)
        ini = _set(ini, 'output_dir', 'probe')
        ini = _set(ini, 'outname', 'probe')
        ini = _set(ini, 'restart', restart)
        path = f'probe/md_probe_{tag}.ini'
        open(path, 'w').write(ini)
        legs.append(path)
    for path in legs:
        print(f'[probe] running {path} ...', flush=True)
        r = subprocess.run([sys.executable, 'run_cosmo_mpipigg.py',
                            '-f', path])
        if r.returncode != 0:
            sys.exit(f'[probe] {path} failed (exit {r.returncode})')
    import glob
    pdb = glob.glob('pilot/rep*/*.pdb') or glob.glob('*/rep*/*.pdb')
    if not pdb:
        sys.exit('[probe] no topology PDB found (looked in pilot/rep*/ '
                 'and */rep*/); run an arm setup first')
    import mdtraj as md
    t = md.load('probe/probe.dcd', top=pdb[0])
    n = t.n_frames
    print(f'[probe] total frames after two legs: {n}')
    if n == EXPECTED['cumulative']:
        print('[probe] VERDICT: md_steps is a CUMULATIVE target -- '
              'E2l segment arithmetic in setup_asyn_mpipigg.py is correct.')
    elif n == EXPECTED['per-segment']:
        print('[probe] VERDICT: md_steps is PER-SEGMENT on this COSMO '
              'build -- do NOT launch E2l; the segment arithmetic in '
              'setup_asyn_mpipigg.py must be changed to per-segment steps.')
    else:
        sys.exit(f'[probe] unexpected frame count {n} (expected '
                 f'{EXPECTED["cumulative"]} or {EXPECTED["per-segment"]}); '
                 f'investigate before launching E2l.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python probe_restart.py <base.ini> '
                 '(e.g. md_pilot_1_seg1.ini)')
    main(sys.argv[1])
