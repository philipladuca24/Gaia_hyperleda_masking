""" Utils. """

from lsst.daf.butler import Butler
import threading
import os

os.environ['DAF_BUTLER_REPOSITORY_INDEX'] = '/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml'

QUANTITIES = ["gT", "gX", "R", "flux", "size", "snr", 'nn', 'nr']
NULL_TESTS = ["gT", "gX", "R", "flux", "size", "snr", 'dens']
NPATCH = 100
COLLECTIONS = ["dp2","refcats/dm-39298/gaia_dr3_20230707","u/xiangchl/anacal-v0/merge","refcats/DM-49042/the_monster_20250219"]
AB_ZP_NJY = 31.4

_local = threading.local()
_butler_lock = threading.Lock()
def get_butler():
    if not hasattr(_local, "butler"):
        with _butler_lock:
            _local.butler = Butler("dp2", collections=COLLECTIONS)
    return _local.butler