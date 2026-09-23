#!/usr/bin/env python
"""
run_cosmo_mpipigg.py -- runner for the Mpipi-GG alpha-syn ladder.

Loads the validated Mpipi-GG model entry written by
setup_asyn_mpipigg.py (mpipi_gg_params.pkl) and splices it INTO
parameters['mpipi'] before calling COSMO's mdrun.

Why an in-place splice: COSMO dispatches force terms and per-residue
atom types by the model NAME and hard-codes 'mpipi' in three places
(cosmo/core/models.py per-residue and nonbonded dispatch,
cosmo/core/system.py addWangFrenkelForces' assert and
setCAIDPerResidueType's table lookup). A newly registered model name
('mpipi_gg') therefore builds no Wang-Frenkel force on unmodified
COSMO. Replacing the 'mpipi' entry keeps every code path intact; the
INI says model = mpipi, and the GG provenance lives in
mpipi_gg_delta.npz, the runinfo logs, and versions.txt.

Usage:  python run_cosmo_mpipigg.py -f md_E2s_1_seg1.ini
"""
import sys, os, pickle

# --- pre-flight: fail loudly on missing/malformed INI ------------------
# configparser.read() silently accepts a nonexistent file, so COSMO's
# config['OPTIONS'] then dies with a cryptic KeyError. Catch it here.
_ini = None
for i, a in enumerate(sys.argv[1:], 1):
    if a in ('-f', '--file') and i + 1 < len(sys.argv):
        _ini = sys.argv[i + 1]
if _ini is not None:
    if not os.path.isfile(_ini):
        sys.exit(f"[wrapper] INI not found: {_ini}\n"
                 f"  (did setup_asyn_mpipigg.py --only <that arm> run on "
                 f"this machine? ls md_*_seg*.ini)")
    with open(_ini) as _fh:
        if '[OPTIONS]' not in _fh.read():
            sys.exit(f"[wrapper] {_ini} has no [OPTIONS] section -- stale "
                     f"or foreign INI; regenerate with setup_asyn_mpipigg.py")

sys.path.insert(0, '.')
import cosmo.parameters.model_parameters as mp

with open('mpipi_gg_params.pkl', 'rb') as fh:
    state = pickle.load(fh)

# --- in-place splice: parameters['mpipi'] now carries Mpipi-GG --------
mp.parameters['mpipi'] = state['gg_entry']
print('[wrapper] parameters["mpipi"] = validated Mpipi-GG tables '
      f"(in-place splice; unit scales {state.get('unit_scales')})")

# --- sanity: the simulation salt comes from debye_length['mpipi'] -----
dl = getattr(mp, 'debye_length', None)
if isinstance(dl, dict) and 'mpipi' in dl:
    if abs(float(dl['mpipi']) - float(state['debye_length_mpipi'])) > 1e-6:
        print(f"[wrapper] WARNING: this COSMO's debye_length['mpipi'] = "
              f"{dl['mpipi']} nm, expected {state['debye_length_mpipi']} nm "
              f"(150 mM). Check your COSMO version before trusting energies.")

# --- entry point: name changed across COSMO versions ------------------
try:
    from cosmo.mdrun import main as cosmo_main     # older COSMO builds
except ImportError:
    from cosmo.mdrun import mdrun as cosmo_main    # COSMO >= 2026.x

sys.argv = ['mdrun'] + sys.argv[1:]
cosmo_main()
