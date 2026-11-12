import sys
import numpy as np
from lsst.daf.butler import Butler
import treecorr
from astropy.table import Table, vstack
from tqdm.notebook import tqdm
from lsst.geom import degrees
from dataregistry import DataRegistry
from matplotlib.path import Path
import healsparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import threading
import traceback
import os
import healpy as hp
from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter, NullFormatter,MaxNLocator
import rustfits
from numpy.lib import recfunctions as rfn
from pathlib import Path as filePath

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

### load catalogs:
def get_anacal_tracts():
    butler = Butler("dp2",collections=["dp2","refcats/dm-39298/gaia_dr3_20230707","u/xiangchl/anacal-v0/merge","refcats/DM-49042/the_monster_20250219"])
    tract_list_refs = list(butler.registry.queryDatasets('deep_coadd_cell_anacal_merged', skymap='lsst_cells_v2'))
    tract_list = np.array([trf.dataId['tract'] for trf in tract_list_refs])
    return tract_list

def get_mdet_tracts():
    butler = Butler("dp2",collections=["dp2","refcats/dm-39298/gaia_dr3_20230707","u/xiangchl/anacal-v0/merge","refcats/DM-49042/the_monster_20250219"])
    tract_list_refs = list(butler.registry.queryDatasets('object_shear_all', skymap='lsst_cells_v2'))
    tract_list = np.array([trf.dataId['tract'] for trf in tract_list_refs])
    return tract_list

def get_erin_tracts():
    source = filePath("/global/cfs/projectdirs/lsst/groups/PO/users/esheldon/desc-mdet/lsst-mdet-runs/run-dp2-v01.1")
    tract_list = sorted(int(p.name) for p in source.glob("[0-9]" * 5) if p.is_dir())
    return tract_list
    
def load_baseline(tract_list, n_jobs=32):
    """ Returns: tract_ids_cut, monsters, star_randoms, tract_mask, footprint"""
    def get_footprint():
        feet =[]
        datareg = DataRegistry()
        for band in [90,91,92]:
            temp_foot = datareg.query.get_dataset_absolute_path(band)
            feet.append(healsparse.HealSparseMap.read(temp_foot))
        footprint = feet[0]
        for foot in feet[1:]:
            footprint = footprint & foot
        del(feet)
        return footprint

    footprint = get_footprint()
    print('footprinted')

    butler = Butler("dp2",collections=["dp2","refcats/dm-39298/gaia_dr3_20230707","u/xiangchl/anacal-v0/merge","refcats/DM-49042/the_monster_20250219"])
    skyMap = butler.get("skyMap", skymap="lsst_cells_v2")

    ra = []
    dec = []
    tract_ids = []
    for tract_id in tract_list:
        tract_info = skyMap[tract_id]
        center = tract_info.getCtrCoord()
        ra.append(center.getRa().asDegrees())
        dec.append(center.getDec().asDegrees())
        tract_ids.append(tract_id)
    print(f"{len(tract_ids)} tracts selected")
    
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    # mask = (ra >= 315) & (ra <= 325) & (dec >= -20) & (dec <= -10)
    polygon = Path([
        (360,15),
        (300,-5),
        (300,-35),
        (360,-15)])
    mask = polygon.contains_points(np.column_stack((ra, dec)))
    tract_ids_cut = np.asarray(tract_ids)[mask]
    tract_ids_cut_stars = tract_ids_cut
    print(f"{len(tract_ids_cut)} tracts filtered")
    print(f"{len(tract_ids_cut_stars)} tracts stared")

    polys = []
    for tid in tract_ids_cut_stars:
        isr = skyMap[tid].inner_sky_region
        ra_1, ra_2 = isr.getLon().getA().asDegrees(), isr.getLon().getB().asDegrees()
        dec_1, dec_2 = isr.getLat().getA().asDegrees(), isr.getLat().getB().asDegrees()
        polys.append(healsparse.geom.Polygon(
            ra=np.array([ra_1, ra_1, ra_2, ra_2]),
            dec=np.array([dec_1, dec_2, dec_2, dec_1]),
            value=True,))
    tract_mask = healsparse.HealSparseMap.make_empty(32, 32768, bool, bit_packed=True)
    for p in polys:
        tract_mask |= p

    star_columns = ['id','coord_ra', 'coord_dec', 'phot_g_mean_flux']
    def get_monsters_for_tract(tid):
        try:
            butler = get_butler()
            where_star = f"skymap='lsst_cells_v2' AND tract = {tid}"
            t_ref = butler.query_datasets("the_monster_20250219", where=where_star)
            tract = skyMap[tid]
            tract_stack = []
            for ref in t_ref:
                t = butler.get(ref).asAstropy()[star_columns]
                t = t[t['phot_g_mean_flux'] > 0]
                t['coord_ra'] = np.rad2deg(t['coord_ra'])
                t['coord_dec'] = np.rad2deg(t['coord_dec'])
                t['g_mag'] = -2.5 * np.log10(t['phot_g_mean_flux']) + AB_ZP_NJY
                inside = np.array(tract.inner_sky_region.contains(t['coord_ra'] *degrees, t['coord_dec'] *degrees))
                tract_stack.append(t[inside])
            return vstack(tract_stack)
        except Exception:
            print(f"tract {tid} failed:\n{traceback.format_exc()}", flush=True)
            return None

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(get_monsters_for_tract)(tid)
        for tid in tqdm(tract_ids_cut_stars, mininterval=60, file=sys.stdout))
    results = [t for t in results if t is not None]
    monsters = vstack(results)

    to_gal = hp.Rotator(coord=["C", "G"])
    valid = footprint.get_values_pos(monsters['coord_ra'], monsters['coord_dec'], lonlat=True, valid_mask=True)
    _, b_star = to_gal(np.asarray(monsters['coord_ra']), np.asarray(monsters['coord_dec']), lonlat=True)
    keep = valid & (b_star < -25)
    monsters = monsters[keep]
    print('len monster', len(monsters), flush=True)

    # collect random points
    star_area = tract_mask.get_valid_area(degrees=True) * 3600
    ra_rand, dec_rand = healsparse.make_uniform_randoms_fast(tract_mask, int(10 * star_area))
    rand_valid = footprint.get_values_pos(ra_rand, dec_rand, lonlat=True, valid_mask=True)
    _, b_rand = to_gal(ra_rand, dec_rand, lonlat=True)
    keep = rand_valid & (b_rand < -25)
    star_randoms = Table([ra_rand[keep], dec_rand[keep]], names=['ra', 'dec'])
    print('len star randoms', len(star_randoms))

    return tract_ids_cut, monsters, star_randoms, tract_mask, footprint

def plot_tracts(tract_list):
    butler = Butler("dp2",collections=["dp2","refcats/dm-39298/gaia_dr3_20230707","u/xiangchl/anacal-v0/merge","refcats/DM-49042/the_monster_20250219"])
    skyMap = butler.get("skyMap", skymap="lsst_cells_v2")
    
    ra = []
    dec = []
    for tract_id in tract_list:
        tract_info = skyMap[tract_id]
        center = tract_info.getCtrCoord()
        ra.append(center.getRa().asDegrees())
        dec.append(center.getDec().asDegrees())
    print(f"{len(ra)} tracts selected")
    
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    # mask = (ra >= 315) & (ra <= 325) & (dec >= -20) & (dec <= -10)
    polygon = Path([
        (360,15),
        (300,-5),
        (300,-35),
        (360,-15)])
    mask = polygon.contains_points(np.column_stack((ra, dec)))
    ra_cut = ra[mask]
    dec_cut = dec[mask]
    
    ra_plot = -np.deg2rad(((ra + 180) % 360) - 180)
    dec_plot = np.deg2rad(dec)
    ra_plot_cut = -np.deg2rad(((ra_cut + 180) % 360) - 180)
    dec_plot_cut = np.deg2rad(dec_cut)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(ra_plot,dec_plot,s=8,alpha=0.7,edgecolors="none")
    ax.scatter(ra_plot_cut,dec_plot_cut,s=8,alpha=0.7,edgecolors="none")
    ra_ticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    tick_x = -np.deg2rad(((ra_ticks + 180) % 360) - 180)
    ax.set_xticks(tick_x)
    ax.set_xticklabels([f"{x}°" for x in ra_ticks])
    ax.grid(True, alpha=0.3)
    ax.set_title(f"LSST DP2 tracts ({len(tract_list)} tracts)",pad=15)
    plt.tight_layout()
    plt.show()

def load_anacal(tract_ids_cut, tract_mask, footprint, n_jobs=32):
    # collect anacal catalogs
    def cut_minimal(data):
        minimal_cuts = np.logical_and.reduce((
            data["n_mask_base"] < 0.035,
            data["wsel"] > 1e-5,
            data["esq"] < 0.4,
            data["lsst_i_s2n_fpfs1"] > 9 , ## CHANGE THIS TO 9 for SNR meas then go back to 10 lim
            # data["lsst_i_mag_gauss2"] < 24,
        ))
        trace = (data['fpfs1_m00'] + data['fpfs1_m20'])/data['fpfs1_m00']
        t_filt = (trace > 0.15)
        return data[minimal_cuts & t_filt]

    cut_columns = [
        'n_mask_base',
        'esq',
    ]
    meas_columns = [
        'ra',
        'dec',
        'wsel',
        'dwsel_dg1',
        'dwsel_dg2',
        'fpfs1_e1',
        'fpfs1_e2',
        'fpfs1_de1_dg1',
        'fpfs1_de1_dg2',
        'fpfs1_de2_dg1',
        'fpfs1_de2_dg2',
        'fpfs1_m00',
        'fpfs1_m20',
        'lsst_i_mag_gauss2',
        'lsst_i_flux_gauss2',
        'lsst_i_s2n_fpfs1',
        'lsst_i_ext_shapeHSM_HsmPsfMoments_xx',
        'lsst_i_ext_shapeHSM_HsmPsfMoments_yy',
        'lsst_i_ext_shapeHSM_HsmPsfMoments_xy',
    ]
    where = f"skymap='lsst_cells_v2' AND tract IN {tuple(set(int(t) for t in tract_ids_cut))}"
    meta_refs = get_butler().query_datasets("deep_coadd_cell_anacal_merged", where=where)

    def get_anacal_for_tract(ref):
        try:
            butler = get_butler()
            t = butler.get(ref, parameters={"columns": cut_columns + meas_columns})
            t = cut_minimal(t)[meas_columns]
            return t
        except Exception:
            print(f"tract {ref} failed:\n{traceback.format_exc()}", flush=True)
            return None

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(get_anacal_for_tract)(m_ref)
        for m_ref in tqdm(meta_refs, mininterval=60, file=sys.stdout))
    results = [t for t in results if t is not None]
    metas = vstack(results)

    to_gal = hp.Rotator(coord=["C", "G"])
    valid_gal = footprint.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    valid_gal &= tract_mask.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    _, b = to_gal(np.asarray(metas['ra']), np.asarray(metas['dec']), lonlat=True)
    keep = valid_gal & (b < -25)
    metas = metas[keep]
    print('len anacal', len(metas), flush=True)
    return metas

def load_mdet(tract_ids_cut, tract_mask, footprint, n_jobs=32):
    def cut_minimal(data):
        minimal_cuts = (
            (data["bmask_flags"] == 0) &
            (data["gauss_flags"] == 0) &
            (data["gauss_object_flags"] == 0) &
            (data["gauss_psfReconvolved_flags"] == 0) &
            (data["gauss_shape_flags"] == 0) &
            (data["image_flags"] == 0) &
            (data["pgauss_flags"] == 0) &
            (data["pgauss_object_flags"] == 0) &
            (data["pgauss_shape_flags"] == 0) &
            (data["psfOriginal_flags"] == 0) &
            (data['r_gaussFlux_flags'] == 0) &
            (data['i_gaussFlux_flags'] == 0) &
            (data['z_gaussFlux_flags'] == 0) &
            (data['r_pgaussFlux_flags'] == 0) &
            (data['i_pgaussFlux_flags'] == 0) &
            (data['z_pgaussFlux_flags'] == 0) &
            (data["is_primary"] == True) &
            (data['mfrac'] < 0.1) &
            (data['gauss_snr'] > 9) & ## used to measure depth then cut to >10 later
            (data['gauss_T']/data['gauss_psfReconvolved_T'] > 0.5))
        return data[minimal_cuts]
    cut_columns = [
        'bmask_flags',
        'gauss_flags',
        'gauss_object_flags',
        'gauss_psfReconvolved_flags',
        'gauss_shape_flags',
        'image_flags',
        'pgauss_flags',
        'pgauss_object_flags',
        'pgauss_shape_flags',
        'psfOriginal_flags',
        'i_gaussFlux_flags',
        'i_pgaussFlux_flags',
        'r_gaussFlux_flags',
        'r_pgaussFlux_flags',
        'z_gaussFlux_flags',
        'z_pgaussFlux_flags',
        'is_primary',
        'mfrac',
    ]
    meas_columns = [
        'metaStep',
        'ra',
        'dec',
        'gauss_g1',
        'gauss_g2',
        'gauss_snr',
        'gauss_T',
        'gauss_TErr',
        'pgauss_snr',
        'pgauss_T',
        'pgauss_TErr',
        'i_pgaussFlux',
        'r_pgaussFlux',
        'z_pgaussFlux',
        'gauss_psfReconvolved_T',
        'psfOriginal_g1',
        'psfOriginal_g2',
        'psfOriginal_T',
    ]
    where = f"skymap='lsst_cells_v2' AND tract IN {tuple(set(int(t) for t in tract_ids_cut))}"
    meta_refs = get_butler().query_datasets("object_shear_all", where=where)

    def get_mdet_for_tract(ref):
        try:
            butler = get_butler()
            t = butler.get(ref, parameters={"columns": cut_columns + meas_columns})
            t = cut_minimal(t)[meas_columns]
            return t
        except Exception:
            print(f"tract {ref} failed:\n{traceback.format_exc()}", flush=True)
            return None

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(get_mdet_for_tract)(m_ref)
        for m_ref in tqdm(meta_refs, mininterval=60, file=sys.stdout))
    results = [t for t in results if t is not None]
    metas = vstack(results)

    to_gal = hp.Rotator(coord=["C", "G"])
    valid_gal = footprint.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    valid_gal &= tract_mask.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    _, b = to_gal(np.asarray(metas['ra']), np.asarray(metas['dec']), lonlat=True)
    keep = valid_gal & (b < -25)
    metas = metas[keep]
    metas['i_pgaussMag'] = -2.5 * np.log10(metas['i_pgaussFlux']) + 31.4
    print('len mdet', len(metas), flush=True)
    return metas

def load_erin(tract_ids_cut, tract_mask, footprint, n_jobs=32):
    source = filePath("/global/cfs/projectdirs/lsst/groups/PO/users/esheldon/desc-mdet/lsst-mdet-runs/run-dp2-v01.1")
    def cut_minimal(data):
        minimal_cuts = (
            (data["is_primary"] == True) &
            (data["flags"] == 0) &
            (data["g_flags"] == 0) &
            (data['mfrac'] < 0.1) &
            (data['s2n'] > 9) &
            (data['T']/data['psf_T'] > 0.5))
        return data[minimal_cuts]
    meas_columns = [
        'mcal_step',
        'ra',
        'dec',
        's2n',
        'g1',
        'g2',
        'T',
        'flux_i',
        'psf_T_i',
    ]
    PATCH_GLOB = "[0-9]" * 5 + "-" + "[0-9]" * 5 + "-mdet.fits"
    def get_mdet_for_patch(patch_path):
        try:
            with rustfits.FITS(patch_path) as fits:
                data = fits[1][:]                      # structured numpy array
            data = cut_minimal(data)
            data = rfn.repack_fields(data[meas_columns])  # compact copy of just these columns
            return data
        except Exception:
            print(f"{patch_path} failed:\n{traceback.format_exc()}", flush=True)
        return None

    patch_paths = [p for tid in tract_ids_cut
                   for p in sorted((source / f"{tid:05d}").glob(PATCH_GLOB))]
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(get_mdet_for_patch)(p)
        for p in tqdm(patch_paths, mininterval=64, file=sys.stdout))
    metas = np.concatenate([t for t in results if t is not None])

    to_gal = hp.Rotator(coord=["C", "G"])
    valid_gal = footprint.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    valid_gal &= tract_mask.get_values_pos(metas['ra'], metas['dec'], lonlat=True, valid_mask=True)
    _, b = to_gal(np.asarray(metas['ra']), np.asarray(metas['dec']), lonlat=True)
    keep = valid_gal & (b < -25)
    metas = metas[keep]
    metas['mag_i'] = -2.5 * np.log10(metas['flux_i']) + 31.4
    print('len erin', len(metas), flush=True)
    return metas

### galaxy randoms
def plot_density_maps(gal_randoms, info, metas, wcol="w", nside_plot=128,
                      min_frac=0.5, rot_ra=330, reso=5.0, xsize=900, ysize=700,
                      delta_lim=0.2):
    nside    = info["nside"]
    npix     = hp.nside2npix(nside)
    npix_p   = hp.nside2npix(nside_plot)
    shift    = 2 * int(np.log2(nside // nside_plot))       # NEST: coarse pix = pix >> shift
    pix_area = hp.nside2pixarea(nside, degrees=True) * 3600.0
    full     = hp.nside2pixarea(nside_plot, degrees=True) * 3600.0

    # galaxies: counts and effective area summed into coarse pixels
    coarse = np.arange(npix) >> shift
    G = np.bincount(coarse, weights=info["n_gal"],           minlength=npix_p)
    A = np.bincount(coarse, weights=info["frac"] * pix_area, minlength=npix_p)

    # weighted randoms, scaled so total weighted randoms = total galaxies
    w    = np.asarray(gal_randoms[wcol])
    rpix = hp.ang2pix(nside_plot, gal_randoms["ra"], gal_randoms["dec"], lonlat=True, nest=True)
    R    = np.bincount(rpix, weights=w, minlength=npix_p) * G.sum() / w.sum()

    ok = (A > min_frac * full) & (R > 0)                    # drop mostly-empty edge pixels
    dens_g = np.full(npix_p, hp.UNSEEN); dens_g[ok] = G[ok] / A[ok]
    dens_r = np.full(npix_p, hp.UNSEEN); dens_r[ok] = R[ok] / A[ok]
    delta  = np.full(npix_p, hp.UNSEEN); delta[ok]  = G[ok] / R[ok] - 1

    vmin, vmax = np.percentile(dens_g[ok], [2, 98])
    rot  = (rot_ra, np.median(metas["dec"]))
    fig  = plt.figure(figsize=(21, 5))
    base = dict(nest=True, rot=rot, reso=reso, xsize=xsize, ysize=ysize, fig=fig.number)
    dens_kw = dict(base, min=vmin, max=vmax, unit="gal / arcmin$^2$")

    hp.gnomview(dens_g, sub=(1, 3, 1), title="galaxies", **dens_kw)
    hp.gnomview(dens_r, sub=(1, 3, 2), title=f"weighted randoms ({wcol})", **dens_kw)
    hp.gnomview(delta,  sub=(1, 3, 3), title="gal / rand − 1",
                min=-delta_lim, max=delta_lim, cmap="RdBu_r", unit="δ", **base)
    plt.show()

    return dict(dens_gal=dens_g, dens_rand=dens_r, delta=delta, ok=ok, nside=nside_plot)

def plot_weight_check(gal_randoms, info, wcol="w", n_bins=12, extra_maps=None):
    """
    Galaxies / randoms in equal-count bins of each systematic, normalised to 1,
    for unweighted and weighted randoms. Flat = systematic is corrected.
    extra_maps: optional dict {label: map at info['nside']} for more panels.
    """
    nside = info["nside"]
    npix  = hp.nside2npix(nside)
    p     = info["fit_pix"]
    G     = info["n_gal"][p]
    rpix  = hp.ang2pix(nside, gal_randoms["ra"], gal_randoms["dec"], lonlat=True, nest=True)

    weights = {"unweighted": None, wcol: np.asarray(gal_randoms[wcol])}
    R = {k: np.bincount(rpix, weights=w, minlength=npix)[p] for k, w in weights.items()}

    xs = {"PSF T [pix]": info["psf_map"][p],
          f"{info['threshold']:g}σ limiting mag [i]": info["depth_map"][p]}
    for lab, m in (extra_maps or {}).items():
        xs[lab] = m[p]

    fig, ax = plt.subplots(1, len(xs), figsize=(6 * len(xs), 4), sharey=True, squeeze=False)
    for a, (xlabel, x) in zip(ax[0], xs.items()):
        ok    = np.isfinite(x)
        edges = np.quantile(x[ok], np.linspace(0, 1, n_bins + 1))
        idx   = np.clip(np.digitize(x[ok], edges) - 1, 0, n_bins - 1)
        xb    = np.bincount(idx, x[ok], n_bins) / np.bincount(idx, minlength=n_bins)
        gb    = np.bincount(idx, G[ok], n_bins)
        gb2 = np.bincount(idx, info["n_gal2"][p][ok], n_bins)
        for k, Rk in R.items():
            rb   = np.bincount(idx, Rk[ok], n_bins)
            norm = G[ok].sum() / Rk[ok].sum()
            a.errorbar(xb, gb / rb / norm, np.sqrt(gb2) / rb / norm,
                       fmt="o", ms=4, capsize=2, label=k,
                       ls="--" if k == "unweighted" else "-")
        a.axhline(1, c="k", lw=0.5)
        a.set_xlabel(xlabel)
    ax[0, 0].set_ylabel("normalized galaxies / randoms")
    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()

def make_gal_randoms(metas, footprint, tract_mask, sys_maps=None,
                     nside=256, threshold=10.0, snr_window=0.2, min_count=3, snr_cut=10.0,
                     b_max=-25.0, min_frac=0.5, n_depth_bins=20,
                     rand_per_arcmin2=100, model="logquad", weight_col="wsel", seed=42,
                     snr_col="lsst_i_s2n_fpfs1", mag_col="lsst_i_mag_gauss2",
                     psf_prefix="lsst_i_ext_shapeHSM_HsmPsfMoments_", is_trace=False, make_plots=True):
    """
    make sure to use metas[metas["metaStep"] == "ns"] and weight_col=None for mdet and
    use metas[metas["mcal_step"] == "ns"] and weight_col=None for erin
    
    Build randoms weighted by a joint fit of galaxy density vs depth, PSF size
    and any extra systematics maps.

    1. Depth map: mean magnitude of objects with |S/N - threshold| < snr_window per
       pixel; pixels with < min_count such objects borrow depth from a coarser parent,
       so no area is masked. Uses the full input catalog.
    2. PSF size map: mean trace [pix] per pixel,
       from the full input catalog.
    3. Randoms: uniform over every pixel containing objects, then cut with the same
       footprint / tract / galactic-latitude masks as the galaxies.
    4. Joint fit (quasi-Poisson likelihood per pixel) on the S/N > snr_cut sample,
       using the sum of `weight_col` per pixel (plain counts if weight_col=None):
           log10 n = lg0 + s*x + c*x^2 + sum_i b_i * z_i
       x = depth - m_med, z_i = standardised PSF size and extra maps.
       Errors are inflated by the dispersion phi = sum(w^2) / sum(w).
    5. Weights: model density at each random's pixel, clipped to the fitted ranges,
       normalised to mean 1.

    sys_maps: optional dict {name: map at `nside`, NEST, NaN where missing}.
    Returns (randoms Table with ra, dec, w;  info dict).
    """
    rng      = np.random.RandomState(seed)
    npix     = hp.nside2npix(nside)
    pix_area = hp.nside2pixarea(nside, degrees=True) * 3600.0   # arcmin^2
    to_gal   = hp.Rotator(coord=["C", "G"])
    allpix   = np.arange(npix)

    def in_masks(ra, dec):
        ok  = footprint.get_values_pos(ra, dec, lonlat=True, valid_mask=True)
        ok &= tract_mask.get_values_pos(ra, dec, lonlat=True, valid_mask=True)
        _, b = to_gal(ra, dec, lonlat=True)
        return ok & (b < b_max)

    snr = np.asarray(metas[snr_col], dtype=float)
    mag = np.asarray(metas[mag_col], dtype=float)
    pix = hp.ang2pix(nside, metas["ra"], metas["dec"], lonlat=True, nest=True)

    # ---------- 1. depth map ----------
    sel   = (np.abs(snr - threshold) < snr_window) & np.isfinite(mag) & np.isfinite(snr)
    psel, m_lim = pix[sel], mag[sel]

    good     = np.bincount(pix, minlength=npix) > 0          # footprint: any object
    good_pix = np.flatnonzero(good)

    depth_map   = np.full(npix, hp.UNSEEN)
    depth_level = np.full(npix, -1)
    todo = good.copy()
    for k in range(int(np.log2(nside)) + 1):                 # NEST: parent = pix >> 2k
        shift  = 2 * k
        c      = np.bincount(psel >> shift, minlength=npix >> shift)
        s      = np.bincount(psel >> shift, weights=m_lim, minlength=npix >> shift)
        parent = allpix >> shift
        ok     = todo & (c[parent] >= min_count)
        depth_map[ok]   = s[parent[ok]] / c[parent[ok]]
        depth_level[ok] = k
        todo &= ~ok
        if not todo.any():
            break
    depth_map[todo] = m_lim.mean()
    print(f"{good_pix.size:,} pixels; native depth in "
          f"{np.mean(depth_level[good] == 0):.1%}, max coarsening level "
          f"{depth_level[good].max()}")

    # ---------- 2. PSF size map ----------
    if is_trace:
        psf_size = np.asarray(metas[psf_prefix])
    else:
        xx, yy, xy = (np.asarray(metas[psf_prefix + c], dtype=float) for c in ("xx", "yy", "xy"))
        psf_size = xx + yy
        # with np.errstate(invalid="ignore"):
        #     psf_size = (xx * yy - xy**2) ** 0.25

    okp   = np.isfinite(psf_size)
    c_psf = np.bincount(pix[okp], minlength=npix)
    s_psf = np.bincount(pix[okp], weights=psf_size[okp], minlength=npix)
    psf_map = np.full(npix, np.nan)
    psf_map[c_psf > 0] = s_psf[c_psf > 0] / c_psf[c_psf > 0]

    # weighted galaxy counts for the fit: final science sample, sum of selection weights
    wgal   = (np.asarray(metas[weight_col], dtype=float) if weight_col
              else np.ones(len(snr)))
    cut    = np.isfinite(snr) & (snr > snr_cut) & np.isfinite(wgal)
    n_gal  = np.bincount(pix[cut], weights=wgal[cut],      minlength=npix)
    n_gal2 = np.bincount(pix[cut], weights=wgal[cut] ** 2, minlength=npix)   # for errors

    # ---------- 3. randoms ----------
    cov_map = healsparse.HealSparseMap.make_empty(32, nside, bool, bit_packed=True)
    cov_map.update_values_pix(good_pix, True)
    n_rand = int(rand_per_arcmin2 * pix_area * good_pix.size)
    rra_all, rdec_all = healsparse.make_uniform_randoms_fast(cov_map, n_rand, rng=rng)
    rpix_all = hp.ang2pix(nside, rra_all, rdec_all, lonlat=True, nest=True)

    keep = in_masks(rra_all, rdec_all)
    rra, rdec, rpix = rra_all[keep], rdec_all[keep], rpix_all[keep]

    n_drawn = np.bincount(rpix_all, minlength=npix)
    n_kept  = np.bincount(rpix,     minlength=npix)
    frac = np.divide(n_kept, n_drawn, out=np.zeros(npix), where=n_drawn > 0)

    # ---------- 4. joint fit ----------
    maps  = {"psf": psf_map, **(sys_maps or {})}
    names = list(maps)

    base = good & (frac > min_frac)
    lo, hi = np.quantile(depth_map[base], [0.01, 0.99])
    fp = base & (depth_map > lo) & (depth_map < hi)
    for nm in names:
        fp &= np.isfinite(maps[nm])
    fit_pix = np.flatnonzero(fp)
    m_med   = np.median(depth_map[fit_pix])

    # standardise each systematics map over the fit pixels (clip 1-99%)
    sys_stats = {}
    for nm in names:
        v = maps[nm][fit_pix]
        vlo, vhi = np.percentile(v, [1, 99])
        sys_stats[nm] = dict(median=np.median(v), std=np.std(np.clip(v, vlo, vhi)),lo=vlo, hi=vhi)

    def zmap(nm, v):
        st = sys_stats[nm]
        v  = np.where(np.isfinite(v), v, st["median"])        # missing -> neutral
        return (np.clip(v, st["lo"], st["hi"]) - st["median"]) / st["std"]

    nq = 3
    
    def design(p):
        x    = np.clip(depth_map[p], lo, hi) - m_med
        cols = [np.ones_like(x), x, x**2]
        cols += [zmap(nm, maps[nm][p]) for nm in names]
        return np.column_stack(cols)

    X    = design(fit_pix)
    N    = n_gal[fit_pix].astype(float)
    expo = frac[fit_pix] * pix_area
    ln10 = np.log(10.0)

    def nll(th):
        eta = ln10 * (X @ th)
        return np.sum(expo * np.exp(eta) - N * eta)

    def grad(th):
        mu = expo * np.exp(ln10 * (X @ th))
        return ln10 * X.T @ (mu - N)

    th0 = np.zeros(X.shape[1])
    th0[0] = np.log10(N.sum() / expo.sum())
    th0[1] = 0.3
    fit = minimize(nll, th0, jac=grad, method="BFGS")

    phi = n_gal2[fit_pix].sum() / n_gal[fit_pix].sum()
    theta, theta_err = fit.x, np.sqrt(phi * np.diag(fit.hess_inv))

    # depth-only part (at median systematics), in the form the plotting code expects
    f = lambda m, n0, s, c: n0 * 10**(s * (m - m_med) + c * (m - m_med)**2)
    params = np.array([10**theta[0], theta[1], theta[2]])

    sys_coef = {}
    for i, nm in enumerate(names):
        b, e = theta[nq + i], theta_err[nq + i]
        sys_coef[nm] = dict(dex_per_sigma=b, err=e,
                            dex_per_unit=b / sys_stats[nm]["std"],
                            pct_per_sigma=100 * (10**b - 1))

    unit = f"sum({weight_col})" if weight_col else "gal"
    print(f"{fit_pix.size:,} pixels in joint fit ({model}), dispersion phi = {phi:.3f}")
    print(f"  depth: n(m_med={m_med:.2f}) = {params[0]:.2f} {unit}/arcmin^2, "
          f"s = {theta[1]:.3f}, c = {theta[2]:.3f}")
    for nm, cf in sys_coef.items():
        r = np.corrcoef(X[:, nq + names.index(nm)], X[:, 1])[0, 1]
        print(f"  {nm:>10s}: {cf['dex_per_sigma']:+.4f} ± {cf['err']:.4f} dex/σ "
              f"({cf['pct_per_sigma']:+.2f}%/σ, {cf['dex_per_unit']:+.4f} dex/unit), "
              f"r(depth) = {r:+.2f}")

    # binned density vs depth (diagnostics / plotting only)
    d        = depth_map[fit_pix]
    dens_pix = N / expo
    edges = np.linspace(lo, hi, n_depth_bins + 1)
    idx   = np.digitize(d, edges) - 1
    inb   = (idx >= 0) & (idx < n_depth_bins)
    nb    = np.bincount(idx[inb], minlength=n_depth_bins)

    def bin_mean(v):
        return np.bincount(idx[inb], weights=v[inb], minlength=n_depth_bins) / nb

    with np.errstate(invalid="ignore", divide="ignore"):
        dmean    = bin_mean(d)
        dens     = bin_mean(dens_pix)
        dens_err = np.sqrt((bin_mean(dens_pix**2) - dens**2) / nb)
    use = (nb > 1) & (dens_err > 0)

    # ---------- 5. weights (computed per pixel, then looked up) ----------
    w_map = np.zeros(npix)
    w_map[good_pix] = 10 ** (design(good_pix) @ theta)
    w = w_map[rpix]
    w /= w.mean()
    w_map /= w_map[rpix].mean()
    print(f"{rra.size:,} randoms, weights in [{w.min():.3f}, {w.max():.3f}]")

    randoms = Table([rra, rdec, w], names=["ra", "dec", "w"])
    info = dict(depth_map=depth_map, depth_level=depth_level, psf_map=psf_map,
                frac=frac, n_gal=n_gal, n_gal2=n_gal2, weight_col=weight_col, phi=phi,
                w_map=w_map,
                params=params, model=f, model_name=f"joint {model}", m_med=m_med,
                theta=theta, theta_err=theta_err, sys_names=names,
                sys_stats=sys_stats, sys_coef=sys_coef,
                dmean=dmean, dens=dens, dens_err=dens_err, use=use,
                edges=edges, fit_pix=fit_pix, nside=nside, threshold=threshold,
                snr_cut=snr_cut)

    if make_plots:
        plot_density_maps(randoms, info, metas)
        plot_weight_check(randoms, info)
    return randoms

### star maps
NEFF      = 25      # stars per smoothing kernel -> sets sigma per bin (weight noise ~ 1/sqrt(NEFF))
ALPHA     = 5      # shrinkage toward the joint-fit prior, in stars per kernel (0 = pure data)
SIGMA_FIX = None    # deg; set to use one sigma for every bin (else per-bin from NEFF)
NSIDE_W   = 256     # healpix grid for the kernel smoothing (pixels << sigma)
DEG, MAG_DEG = 2, 1 # joint prior: poly degree in (l, b), and in magnitude
ROT_RA, RESO, XSIZE, YSIZE = 330, 15.0, 400, 300   # gnomview settings

def _design(l, b, l0, b0, ls, bs, deg):
    x, y = (l - l0) / ls, (b - b0) / bs
    return np.column_stack([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)])

def _to_gal(ra, dec, l_c=None):
    l, b = map(np.asarray, hp.Rotator(coord=["C", "G"])(ra, dec, lonlat=True))
    if l_c is None:
        l_c = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(l)))))
    return (l - l_c + 180) % 360 - 180 + l_c, b, l_c

def joint_star_weights(star_ra, star_dec, star_mag, bins, rand_ra, rand_dec,
                       deg=2, mag_deg=1, nside_fit=32, lam=0.0):
    """log mu_kp = log R_p + a_k + sum_j X_pj c_j(m_k), c_j polynomial in bin-centre mag."""
    lr, br, _ = _to_gal(rand_ra, rand_dec)
    npix = hp.nside2npix(nside_fit)
    rpix = hp.ang2pix(nside_fit, rand_ra, rand_dec, lonlat=True, nest=True)
    spix = hp.ang2pix(nside_fit, star_ra, star_dec, lonlat=True, nest=True)
    R = np.bincount(rpix, minlength=npix).astype(float)
    p = np.flatnonzero(R > 0)
    lp = np.bincount(rpix, lr, npix)[p] / R[p]
    bp = np.bincount(rpix, br, npix)[p] / R[p]
    l0, b0, lsc, bsc = lp.mean(), bp.mean(), lp.std(), bp.std()

    X   = _design(lp, bp, l0, b0, lsc, bsc, deg)[:, 1:]
    off = np.log(R[p])
    N = np.array([np.bincount(spix[(star_mag > lo) & (star_mag < hi)], minlength=npix)[p]
                  for lo, hi in bins], float)
    mc = np.array([0.5 * (lo + hi) for lo, hi in bins])
    T  = np.vander((mc - mc.mean()) / (mc.std() or 1), mag_deg + 1, increasing=True)
    K, J, D = len(bins), X.shape[1], T.shape[1]

    def fg(th):
        a, B = th[:K], th[K:].reshape(J, D)
        eta = off[None] + a[:, None] + (T @ B.T) @ X.T
        mu  = np.exp(eta)
        r   = mu - N
        f   = np.sum(mu - N * eta) + 0.5 * lam * np.sum(B[:, 1:]**2)
        gB  = X.T @ r.T @ T
        gB[:, 1:] += lam * B[:, 1:]
        return f, np.concatenate([r.sum(1), gB.ravel()])

    th0 = np.concatenate([np.log(np.maximum(N.sum(1), 1) / R[p].sum()), np.zeros(J * D)])
    fit = minimize(fg, th0, jac=True, method="L-BFGS-B")
    B = fit.x[K:].reshape(J, D)
    Xr = _design(lr, br, l0, b0, lsc, bsc, deg)[:, 1:]
    ws = [np.exp(Xr @ (B @ T[k])) for k in range(K)]
    return [w / w.mean() for w in ws], fit

def sigma_for_neff(n_star, rand_pix_w, nside=NSIDE_W, neff=NEFF):
    area = len(np.unique(rand_pix_w)) * hp.nside2pixarea(nside, degrees=True)
    return np.sqrt(neff * area / (4 * np.pi * max(n_star, 1)))

def kernel_ratio_weights(star_ra, star_dec, rand_pix_w, sigma_deg,
                         nside=NSIDE_W, prior_w=None, alpha=0.0):
    """Random weights following the star density on scales >~ sigma_deg (RING pixels)."""
    npix = hp.nside2npix(nside)
    spix = hp.ang2pix(nside, star_ra, star_dec, lonlat=True)
    pw = np.ones(len(rand_pix_w)) if prior_w is None else np.asarray(prior_w, float)
    N = np.bincount(spix, minlength=npix).astype(float)
    M = np.bincount(rand_pix_w, weights=pw, minlength=npix)
    M *= N.sum() / M.sum()
    sig = np.radians(sigma_deg)
    SN = hp.smoothing(N, sigma=sig, use_pixel_weights=False)
    SM = hp.smoothing(M, sigma=sig, use_pixel_weights=False)
    a = alpha * hp.nside2pixarea(nside) / (4 * np.pi * sig**2)
    c = (np.clip(SN, 0, None) + a) / (np.clip(SM, 1e-12, None) + a)
    w = pw * c[rand_pix_w]
    return w / w.mean()

def _nside_for_scale(sigma_deg, lo=4, hi=128):
    """Coarsest-adequate plotting nside: pixel size ~ sigma."""
    ns = [n for n in (4, 8, 16, 32, 64, 128) if lo <= n <= hi]
    res = np.array([hp.nside2resol(n, arcmin=True) / 60 for n in ns])
    return ns[int(np.argmin(np.abs(np.log(res / sigma_deg))))]

def _profile(ax, xs, xr, w, label, nbins=8):
    edges = np.quantile(xr, np.linspace(0, 1, nbins + 1))
    N  = np.histogram(xs, edges)[0].astype(float)
    out = {}
    for name, ww, mk in (("prior", w[0], "s"), ("kernel", w[1], "o")):
        Rw = np.histogram(xr, edges, weights=ww)[0]
        mu = Rw * N.sum() / Rw.sum()
        chi2 = np.sum((N - mu)**2 / mu)
        bc = 0.5 * (edges[1:] + edges[:-1])
        ax.errorbar(bc + (0.02 if mk == "o" else -0.02) * np.ptp(edges), N / mu,
                    np.sqrt(np.maximum(N, 1)) / mu, fmt=mk, ms=4, capsize=2,
                    label=f"{name}")
        out[name] = chi2
    ax.axhline(1, c="k", lw=0.8)
    ax.set_xlabel(f"{label} [deg]"); ax.set_ylabel("stars / model")
    ax.legend(fontsize=8, frameon=False)
    return out

def plot_bin(star_ra, star_dec, rand_ra, rand_dec, w_prior, w_kern, sigma_deg, title):
    nside = _nside_for_scale(sigma_deg)
    npix  = hp.nside2npix(nside)
    spix  = hp.ang2pix(nside, star_ra, star_dec, lonlat=True, nest=True)
    rpix  = hp.ang2pix(nside, rand_ra, rand_dec, lonlat=True, nest=True)
    N  = np.bincount(spix, minlength=npix).astype(float)
    Ru = np.bincount(rpix, minlength=npix).astype(float)
    ok = Ru > 0.5 * np.median(Ru[Ru > 0])
    area = Ru / Ru[ok].mean()

    def rel(c):
        m = np.full(npix, hp.UNSEEN); m[ok] = c[ok] / area[ok] / (c[ok].sum() / area[ok].sum()); return m
    def delta(w):
        Rw = np.bincount(rpix, weights=w, minlength=npix)
        mu = Rw * N[ok].sum() / Rw[ok].sum()
        d = np.full(npix, hp.UNSEEN); d[ok] = N[ok] / mu[ok] - 1
        chi2 = np.sum((N[ok] - mu[ok])**2 / mu[ok]) / ok.sum()
        return d, Rw, chi2, np.sqrt(np.mean(1 / mu[ok]))

    d_p, _,  chi_p, pois = delta(w_prior)
    d_k, Rk, chi_k, _    = delta(w_kern)
    rs, rk = rel(N), rel(Rk)
    vmin, vmax = np.percentile(rs[ok], [2, 98])
    dl = max(3 * pois, 0.05)

    fig = plt.figure(figsize=(24, 10), dpi=60)
    base = dict(nest=True, rot=(ROT_RA, np.median(star_dec)), reso=RESO,
                xsize=XSIZE, ysize=YSIZE, fig=fig.number)
    dk  = dict(base, min=vmin, max=vmax, unit="relative density")
    ddk = dict(base, min=-dl, max=dl, cmap="RdBu_r", unit="δ")
    # top row: 4 maps
    hp.gnomview(rs,  sub=(2, 4, 1), title=f"stars {title}", **dk)
    hp.gnomview(rk,  sub=(2, 4, 2), title="kernel-weighted randoms", **dk)
    hp.gnomview(d_p, sub=(2, 4, 3), title=f"δ vs parametric  χ²/pix={chi_p:.2f}", **ddk)
    hp.gnomview(d_k, sub=(2, 4, 4), title=f"δ vs kernel  χ²/pix={chi_k:.2f}", **ddk)

    # bottom row: 2 profiles, each spanning two map columns
    lr, br, l_c = _to_gal(rand_ra, rand_dec)
    ls, bs, _   = _to_gal(star_ra, star_dec, l_c)
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, bottom=0.07, top=0.95,
                          wspace=0.15, hspace=0.25)
    _profile(fig.add_subplot(gs[1, 0]), bs, br, (w_prior, w_kern), "b")
    _profile(fig.add_subplot(gs[1, 1]), ls, lr, (w_prior, w_kern), "l")

    fig.suptitle(f"{title}:  σ = {sigma_deg * 60:.0f}′,  map nside = {nside} "
                 f"({hp.nside2resol(nside, arcmin=True):.0f}′ pix),  Poisson δ rms ≈ {pois:.3f}",
                 y=1.04, fontsize=13)
    plt.show()

def make_star_randoms(monsters, star_randoms, mag_min_max, mag_cuts, mag_bin_args, make_plots=True):
    s_ra_all  = np.asarray(monsters["coord_ra"], float)
    s_dec_all = np.asarray(monsters["coord_dec"], float)
    s_mag_all = np.asarray(monsters["g_mag"], float)
    r_ra  = np.asarray(star_randoms["ra"], float)
    r_dec = np.asarray(star_randoms["dec"], float)
    r_pix_w = hp.ang2pix(NSIDE_W, r_ra, r_dec, lonlat=True)

    prior_weights, jfit = joint_star_weights(s_ra_all, s_dec_all, s_mag_all, mag_min_max,
                                            r_ra, r_dec, deg=DEG, mag_deg=MAG_DEG)
    print(f"joint prior fit: converged={jfit.success}, nll={jfit.fun:.1f}")

    sigma_floor = 5
    re_weights = []
    for (b, c, mag, dif), (lo, hi) in zip(mag_cuts, mag_min_max):
        m = (s_mag_all > lo) & (s_mag_all < hi)
        theta_max = mag_bin_args[b][1] / 3600
        sig = max(sigma_for_neff(m.sum(), r_pix_w), sigma_floor * theta_max)
        area = len(np.unique(r_pix_w)) * hp.nside2pixarea(NSIDE_W, degrees=True)
        neff = m.sum() / area * 4 * np.pi * sig**2
        w = kernel_ratio_weights(s_ra_all[m], s_dec_all[m], r_pix_w, sig, prior_w=prior_weights[c], alpha=ALPHA)
        re_weights.append(w)
        print(f"g ∈ ({lo}, {hi}]: {m.sum():,} stars, σ = {sig*60:.0f}′ "
            f"(θ_max = {theta_max*60:.1f}′), N_eff = {neff:,.0f}, "
            f"leak ≲ {1/neff:.1e}")
        if make_plots:
            plot_bin(s_ra_all[m], s_dec_all[m], r_ra, r_dec, prior_weights[c], w, sig, f"g ∈ ({lo}, {hi}]")
    return re_weights

### make correlations
def get_patch_centers(monst, npatch, path):
    """k-means patches from the lens monsters
    Saved to `path` so reruns use identical patches."""
    if os.path.exists(path):
        return np.load(path)
    cat = treecorr.Catalog(ra=monst["coord_ra"], dec=monst["coord_dec"],
                           ra_units="deg", dec_units="deg", npatch=npatch)
    np.save(path, cat.patch_centers)
    return cat.patch_centers

def lens_catalog(ra, dec, centers, w=None):
    return treecorr.Catalog(ra=ra, dec=dec, w=w, ra_units="deg", dec_units="deg",
                            patch_centers=centers)

def measure(lens_cat, gal_cats, gal_random, sep_min_as, sep_max_as, bin_size, out_path,
            extra_config=None, dvec=None, bin_type="Log"):
    """Run all correlations for one lens catalog and save everything."""
    if dvec is None:
        raise ValueError('Derived vector function is None.')
    cfg = dict(min_sep=sep_min_as / 60, max_sep=sep_max_as / 60, bin_size=bin_size,
               bin_type=bin_type, sep_units="arcmin",
               var_method="jackknife", cross_patch_weight="match")
    corrs = {}
    for k in gal_cats.keys():
        if k == "ng":
            corrs[k] = treecorr.NGCorrelation(**cfg)
        elif k == 'nn':
            corrs[k] = treecorr.NNCorrelation(**cfg)
        else:
            corrs[k] = treecorr.NKCorrelation(**cfg)
            
    for k, c in corrs.items():
        c.process(lens_cat, gal_cats[k])
        print(f"  done {k}", flush=True)

    corrs['nr'] = treecorr.NNCorrelation(**cfg)
    corrs['nr'].process(lens_cat, gal_random)
    print('done nr')

    keys = list(corrs)
    func = dvec(keys)
    d = func([corrs[k] for k in keys])
    A, w = treecorr.build_multi_cov_design_matrix([corrs[k] for k in keys], "jackknife", func=func)
    Np = len(A)
    D = A - A.mean(0)
    cov = (Np - 1) / Np * D.T @ D             # identical to estimate_multi_cov

    nb = corrs["ng"].nbins
    out = dict(
        quantities=np.array(QUANTITIES),
        nbins=nb,
        r_as=np.asarray(corrs["ng"].meanr) * 60,
        edges_as=np.append(corrs["ng"].left_edges, corrs["ng"].right_edges[-1]) * 60,
        npairs=np.asarray(corrs["ng"].npairs),
        data=d,
        cov=cov,
        design=A,
        design_w=w,
        patch_centers=lens_cat.patch_centers,
        n_lens=lens_cat.nobj,
        sumw_lens=lens_cat.sumw,
        config=dict(sep_min_as=sep_min_as, sep_max_as=sep_max_as,
                    bin_size=bin_size, bin_type=bin_type, **(extra_config or {})),
    )
    var = np.diag(cov)
    for i, q in enumerate(QUANTITIES):
        sl = slice(i * nb, (i + 1) * nb)
        out[q] = d[sl]
        out[q + "_var"] = var[sl]
    np.savez(out_path + ".npz", **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v)
                                   for k, v in out.items()})
    return out

def make_rand_corr(rand_lens, gal_cats, gal_random, out_path,
                   sep_min, sep_max, bin_sep, bin_type="Log", dvec=None, extra_config=None):
    print(f"meas randoms -> {out_path}.npz")
    return measure(rand_lens, gal_cats, gal_random, sep_min, sep_max, bin_sep, out_path,
                   bin_type=bin_type, dvec=dvec, extra_config=extra_config)

def make_star_corr(monster, star_mag, mag_dif, gal_cats, gal_random, centers, out_path,
                   sep_min, sep_max, bin_sep, bin_type="Log", dvec=None):
    sel = (monster["g_mag"] > star_mag - mag_dif) & (monster["g_mag"] < star_mag + mag_dif)
    m = monster[sel]
    lens = lens_catalog(m["coord_ra"], m["coord_dec"], centers)
    print(f"meas star {star_mag}")
    return measure(lens, gal_cats, gal_random, sep_min, sep_max, bin_sep, out_path,
                   bin_type=bin_type, dvec=dvec,
                   extra_config=dict(star_mag=star_mag, frac_dif=mag_dif,
                                     mean_mag=float(np.mean(m["g_mag"]))))

## catalog specific response
def build_gal_cats_anacal(meta, centers):
    kw  = dict(ra=meta["ra"], dec=meta["dec"], ra_units="deg", dec_units="deg",
               patch_centers=centers)
    wkw = dict(kw, w=meta["wsel"])                 # selection-weighted averages / counts
    r1 = meta["wsel"] * meta["fpfs1_de1_dg1"] + meta["dwsel_dg1"] * meta["fpfs1_e1"]
    r2 = meta["wsel"] * meta["fpfs1_de2_dg2"] + meta["dwsel_dg2"] * meta["fpfs1_e2"]
    return {
        "ng":   treecorr.Catalog(g1=meta["fpfs1_e1"] * meta["wsel"],
                                 g2=meta["fpfs1_e2"] * meta["wsel"], **kw),
        "R":    treecorr.Catalog(k=0.5 * (r1 + r2), **kw),
        "flux": treecorr.Catalog(k=meta["lsst_i_flux_gauss2"], **wkw),
        "size": treecorr.Catalog(k=1 + meta["fpfs1_m20"] / meta["fpfs1_m00"], **wkw),
        "snr":  treecorr.Catalog(k=meta["lsst_i_s2n_fpfs1"], **wkw),
        "nn":   treecorr.Catalog(**wkw),
    }

def derived_vector_anacal(keys):
    """Function mapping the list of correlations -> concatenated data vector
    in QUANTITIES order. treecorr calls it once per jackknife realization."""
    def func(cl):
        c = dict(zip(keys, cl))
        R = c["R"].xi
        norm = lambda nn: nn.weight / nn.tot
        return np.concatenate([c["ng"].xi / R, c["ng"].xi_im / R, R,
                               c["flux"].xi, c["size"].xi, c["snr"].xi,
                               norm(c["nn"]), norm(c["nr"])])
    return func

def run_anacal_corrs(metas, monsters, gal_randoms, star_randoms, mag_cuts, mag_bin_args, re_weights, save_path):
    metas_cut = metas[metas['lsst_i_s2n_fpfs1'] > 10]
    centers    = get_patch_centers(monsters, NPATCH, f"{save_path}/patch_centers.npy")
    gal_cats   = build_gal_cats_anacal(metas_cut, centers)
    gal_random = lens_catalog(gal_randoms["ra"], gal_randoms["dec"], centers, w=gal_randoms['w'])
    print("built gal")

    rand_ra, rand_dec = star_randoms["ra"].data, star_randoms["dec"].data

    for (b, c, mag, dif) in tqdm(mag_cuts):
        smin, smax, bsz, btype = mag_bin_args[b]
        rand_lens = lens_catalog(rand_ra, rand_dec, centers, w=re_weights[c])
        make_rand_corr(rand_lens, gal_cats, gal_random, f"{save_path}/rand_g{mag}",
                    smin, smax, bsz, btype, dvec=derived_vector_anacal, 
                    extra_config=dict(star_mag=mag))

    for b, c, mag, dif in tqdm(mag_cuts):
        make_star_corr(monsters, mag, dif, gal_cats, gal_random, centers,
                    f"{save_path}/star_mag{mag}", *mag_bin_args[b], dvec=derived_vector_anacal)                

def build_gal_cats_mdet(meta, centers):
    def cat(step, **cols):
        s = np.asarray(meta["metaStep"] == step)
        return treecorr.Catalog(ra=np.asarray(meta["ra"])[s], dec=np.asarray(meta["dec"])[s],
                                ra_units="deg", dec_units="deg", patch_centers=centers,
                                **{k: np.asarray(v)[s] for k, v in cols.items()})
    return {
        "ng":   cat("ns", g1=meta["gauss_g1"], g2=meta["gauss_g2"]),
        "g1p":  cat("1p", k=meta["gauss_g1"]),
        "g1m":  cat("1m", k=meta["gauss_g1"]),
        "g2p":  cat("2p", k=meta["gauss_g2"]),
        "g2m":  cat("2m", k=meta["gauss_g2"]),
        "flux": cat("ns", k=meta['i_pgaussFlux']),
        "size": cat("ns", k=meta['gauss_T']),
        "snr":  cat("ns", k=meta['gauss_snr']),
        "nn":   cat("ns"),
    }

def derived_vector_mdet(keys):
    DG = 0.01 # shear applied in the 1p/1m/2p/2m images
    def func(cl):
        c = dict(zip(keys, cl))
        R11 = (c["g1p"].xi - c["g1m"].xi) / (2 * DG)
        R22 = (c["g2p"].xi - c["g2m"].xi) / (2 * DG)
        R = 0.5 * (R11 + R22)
        norm = lambda nn: nn.weight / nn.tot
        return np.concatenate([c["ng"].xi / R, c["ng"].xi_im / R, R,
                               c["flux"].xi, c["size"].xi, c["snr"].xi,
                               norm(c["nn"]), norm(c["nr"])])
    return func

def run_mdet_corrs(metas, monsters, gal_randoms, star_randoms, mag_cuts, mag_bin_args, re_weights, save_path):
    metas_cut = metas[metas['gauss_snr'] > 10]
    centers    = get_patch_centers(monsters, NPATCH, f"{save_path}/patch_centers.npy")
    gal_cats   = build_gal_cats_mdet(metas_cut, centers)
    gal_random = lens_catalog(gal_randoms["ra"], gal_randoms["dec"], centers, w=gal_randoms['w'])
    print("built gal")

    rand_ra, rand_dec = star_randoms["ra"].data, star_randoms["dec"].data

    for (b, c, mag, dif) in tqdm(mag_cuts):
        smin, smax, bsz, btype = mag_bin_args[b]
        rand_lens = lens_catalog(rand_ra, rand_dec, centers, w=re_weights[c])
        make_rand_corr(rand_lens, gal_cats, gal_random, f"{save_path}/rand_g{mag}",
                    smin, smax, bsz, btype, dvec=derived_vector_mdet, 
                    extra_config=dict(star_mag=mag))

    for b, c, mag, dif in tqdm(mag_cuts):
        make_star_corr(monsters, mag, dif, gal_cats, gal_random, centers,
                    f"{save_path}/star_mag{mag}", *mag_bin_args[b], dvec=derived_vector_mdet)

def build_gal_cats_erin(meta, centers):
    def cat(step, **cols):
        s = np.asarray(meta["mcal_step"] == step)
        return treecorr.Catalog(ra=np.asarray(meta["ra"])[s], dec=np.asarray(meta["dec"])[s],
                                ra_units="deg", dec_units="deg", patch_centers=centers,
                                **{k: np.asarray(v)[s] for k, v in cols.items()})
    return {
        "ng":   cat("ns", g1=meta["g1"], g2=meta["g2"]),
        "g1p":  cat("1p", k=meta["g1"]),
        "g1m":  cat("1m", k=meta["g1"]),
        "flux": cat("ns", k=meta['flux_i']),
        "size": cat("ns", k=meta['T']),
        "snr":  cat("ns", k=meta['s2n']),
        "nn":   cat("ns"),
    }

def derived_vector_erin(keys):
    DG = 0.01 # shear applied in the 1p/1m images
    def func(cl):
        c = dict(zip(keys, cl))
        R11 = (c["g1p"].xi - c["g1m"].xi) / (2 * DG)
        R = R11
        norm = lambda nn: nn.weight / nn.tot
        return np.concatenate([c["ng"].xi / R, c["ng"].xi_im / R, R,
                               c["flux"].xi, c["size"].xi, c["snr"].xi,
                               norm(c["nn"]), norm(c["nr"])])
    return func

def run_erin_corrs(metas, monsters, gal_randoms, star_randoms, mag_cuts, mag_bin_args, re_weights, save_path):
    metas_cut = metas[metas['s2n'] > 10]
    centers    = get_patch_centers(monsters, NPATCH, f"{save_path}/patch_centers.npy")
    gal_cats   = build_gal_cats_erin(metas_cut, centers)
    gal_random = lens_catalog(gal_randoms["ra"], gal_randoms["dec"], centers, w=gal_randoms['w'])
    print("built gal")

    rand_ra, rand_dec = star_randoms["ra"].data, star_randoms["dec"].data

    for (b, c, mag, dif) in tqdm(mag_cuts):
        smin, smax, bsz, btype = mag_bin_args[b]
        rand_lens = lens_catalog(rand_ra, rand_dec, centers, w=re_weights[c])
        make_rand_corr(rand_lens, gal_cats, gal_random, f"{save_path}/rand_g{mag}",
                    smin, smax, bsz, btype, dvec=derived_vector_erin, 
                    extra_config=dict(star_mag=mag))

    for b, c, mag, dif in tqdm(mag_cuts):
        make_star_corr(monsters, mag, dif, gal_cats, gal_random, centers,
                    f"{save_path}/star_mag{mag}", *mag_bin_args[b], dvec=derived_vector_erin)

### plot outputs
LABELS = {"gT": r"\gamma_{\rm T}", "gX": r"\gamma_\times", "R": r"R",
          "flux": r"{\rm flux}", "size": r"{\rm size}", "snr": r"{\rm S/N}",
          "dens": r"\xi_{\rm LS}"}
C_DATA, C_SIM, C_FAIL, C_DENS = "#222222", "#4C72B0", "#C44E52", "#55A868"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "axes.linewidth": 0.7, "lines.linewidth": 0.9,
    "legend.frameon": False, "legend.handlelength": 1.5,
})

def load_null(star_file, rand_file):
    """star - random for every test, for the data and for every jackknife row.
    Returns d (ntest, nb), A (Np, ntest, nb), bin edges and centers [arcsec]."""
    S = np.load(star_file, allow_pickle=True)
    Q = np.load(rand_file, allow_pickle=True)
    assert np.allclose(S["patch_centers"], Q["patch_centers"])
    nq, nb = len(QUANTITIES), int(S["nbins"])
    dS, dQ = S["data"].reshape(nq, nb), Q["data"].reshape(nq, nb)
    AS, AQ = S["design"].reshape(-1, nq, nb), Q["design"].reshape(-1, nq, nb)
    ig, ir = QUANTITIES.index("nn"), QUANTITIES.index("nr")

    def null(s, q, name):                    # works on (nq, nb) and (Np, nq, nb)
        if name == "dens":                   # Landy-Szalay
            DD, DR = s[..., ig, :], s[..., ir, :]
            RD, RR = q[..., ig, :], q[..., ir, :]
            return (DD - DR - RD + RR) / RR
        j = QUANTITIES.index(name)
        return s[..., j, :] - q[..., j, :]

    d = np.stack([null(dS, dQ, n) for n in NULL_TESTS])          # (ntest, nb)
    A = np.stack([null(AS, AQ, n) for n in NULL_TESTS], axis=1)  # (Np, ntest, nb)

    # drop inner bins that are empty in some jackknife row (RR = 0 -> NaN)
    bad = ~np.isfinite(d).all(0) | ~np.isfinite(A).all((0, 1))
    k0 = np.flatnonzero(bad).max() + 1 if bad.any() else 0
    return d[:, k0:], A[:, :, k0:], S["edges_as"][k0:], S["r_as"][k0:]

def chi2_terms(v, A, sel, width, hartlap=True):
    """chi^2 = x^T C^-1 x vs zero for every window of `width` bins, using the tests
    `sel` jointly (full jackknife covariance between tests and bins), split into
    one additive piece per test:  chi2 = sum_t x_t . (C^-1 x)_t.
    A piece can be negative when tests are correlated.
    v: (..., ntest, nb); A: (Np, ntest, nb). Returns (..., len(sel), nwin);
    sum over axis -2 to get the chi^2 itself."""
    Np, _, nb = A.shape
    nwin, k = nb - width + 1, len(sel) * width
    h = (Np - k - 2) / (Np - 1) if hartlap else 1.0              # Hartlap correction
    lead = v.shape[:-2]
    out = np.empty(lead + (len(sel), nwin))
    for i in range(nwin):
        X = A[:, sel, i:i + width].reshape(Np, k)
        D = X - X.mean(0)
        Cinv = h * np.linalg.inv((Np - 1) / Np * D.T @ D)        # k x k jackknife cov
        x = v[..., sel, i:i + width].reshape(lead + (k,))
        out[..., i] = (x * (x @ Cinv)).reshape(lead + (len(sel), width)).sum(-1)
    return out

def window_test(star_file, rand_file, joint_tests,
                width=1, alpha=1e-4, n_sims=5000, dens_cut=0.15, spot=0.5, seed=0):
    d, A, edges, r = load_null(star_file, rand_file)
    Np, nt = len(A), len(NULL_TESTS)
    sel = [NULL_TESTS.index(q) for q in joint_tests]
    sigma = np.sqrt((Np - 1) / Np * ((A - A.mean(0)) ** 2).sum(0))   # per-bin error

    # 1. joint chi2 of every window, its per-test pieces, and per-test chi2s (diagnosis)
    terms = chi2_terms(d, A, sel, width)
    jchi = terms.sum(0)
    contrib = np.full((nt, jchi.size), np.nan)
    contrib[sel] = terms
    share = contrib / jchi
    chi = np.stack([chi2_terms(d, A, [t], width)[0] for t in range(nt)])

    # 2. threshold: pure-noise data vectors with the full covariance, run through the
    #    same scan (with the covariance they came from); largest joint chi2 per sim
    D = A.reshape(Np, -1) - A.reshape(Np, -1).mean(0)
    z = np.random.default_rng(seed).standard_normal((n_sims, Np))
    sims = (np.sqrt((Np - 1) / Np) * z @ D).reshape(n_sims, *d.shape)
    jchi_sims = chi2_terms(sims, A, sel, width, hartlap=False).sum(-2)
    worst = jchi_sims.max(-1)
    thresh = np.quantile(worst, 1 - alpha)

    # 3. window labels: center data point (inner of the two middle bins for even width)
    c = np.arange(jchi.size) + int(spot)
    r_ext = np.append(r, edges[-1])
    r_win = r_ext[c] if spot == int(spot) else 0.5 * (r_ext[c] + r_ext[c + 1])

    # 4. a window fails if its joint chi2 is above the threshold;
    #    the radius is that of the outermost failing window
    failing = np.flatnonzero(jchi > thresh)
    big = np.flatnonzero(np.abs(d[NULL_TESTS.index("dens")]) >= dens_cut)
    res = dict(d=d, sigma=sigma, r=r, edges=edges, chi=chi, jchi=jchi, thresh=thresh,
               x_win=r[c], r_win=r_win, failing=failing, no_fail=failing.size == 0,
               sims=sims, jchi_sims=jchi_sims, worst_sims=worst, worst_data=jchi.max(),
               contrib=contrib, share=share,
               dens_radius=r[big[-1]] if big.size else np.nan,
               joint_tests=joint_tests, width=width, alpha=alpha, dens_cut=dens_cut)

    if failing.size:
        i = failing[-1]                      # tests driving it: largest chi2 shares
        top = sorted(sel, key=lambda t: -share[t, i])[:3]
        res.update(win=i, radius=r_win[i], win_lo=edges[i], win_hi=edges[i + width],
                   tests=[f"{NULL_TESTS[t]} ({share[t, i]:.0%})" for t in top])
    else:                                    # no failure: innermost non-empty bin
        has = np.flatnonzero(np.any(d != 0, axis=0))
        k = has[0] if has.size else 0
        res.update(win=int(np.argmax(jchi)), radius=edges[k],
                   win_lo=edges[k], win_hi=edges[k + 1], tests=[])
    return res

## plotting
def _log_x(ax, edges):
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())

def _fail_spans(failing, edges, width):
    """Merge overlapping failing windows into contiguous [lo, hi] spans."""
    spans = []
    for i in failing:
        lo, hi = edges[i], edges[i + width]
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return spans

def _markers(ax, res, spans):
    for lo, hi in spans:
        ax.axvspan(lo, hi, color=C_FAIL, alpha=0.08, lw=0, zorder=0)
    if not res["no_fail"]:
        ax.axvline(res["radius"], color=C_FAIL, lw=0.8, ls="--")
    if np.isfinite(res["dens_radius"]):
        ax.axvline(res["dens_radius"], color=C_DENS, lw=0.9, ls="--")

def _robust_ylim(y, noise, clip=5.0, pad=0.15):
    """y-limits from the bulk of the points. A point is an outlier if it is more than
    `clip` x scale from the median, where scale = max(typical 1-sigma error, robust
    spread of the points), so a panel with real signal in many bins keeps it.
    Always shows at least +-2 typical sigma. clip=None: use every point (old behaviour)."""
    ok = np.isfinite(y)
    noise_med = None
    if clip is not None and ok.sum() > 2:
        med = np.median(y[ok])
        mad = 1.4826 * np.median(np.abs(y[ok] - med))
        nz = noise[np.isfinite(noise) & (noise > 0)]
        noise_med = np.median(nz) if nz.size else 0.0
        scale = max(noise_med, mad)
        if scale > 0:
            ok &= np.abs(y - med) <= clip * scale
    yk = y[ok]
    lo, hi = min(yk.min(), 0.0), max(yk.max(), 0.0)
    if noise_med:
        lo, hi = min(lo, -2 * noise_med), max(hi, 2 * noise_med)
    span = hi - lo
    p = pad * span if span > 0 else 1.0
    return lo - p, hi + p

def _mark_clipped(ax, x, y, lo, hi):
    """Open triangles at the panel edge for points outside the y-range."""
    inset = 0.04 * (hi - lo)
    for sel, yy, mk in ((y > hi, hi - inset, "^"), (y < lo, lo + inset, "v")):
        if sel.any():
            ax.plot(x[sel], np.full(sel.sum(), yy), mk, ms=4, color=C_DATA,
                    mfc="white", mew=0.8, zorder=6)

def plot_bin_rad(res, title="", show_sims=True, n_show=0,
             theta_pow=2.5, relative=False, y_pad=0.15,clip=5.0, save_plot=False):
    """One figure per star bin.
    Left:   theta-weighted (star - random) per test, with the noise-sim 68/95% band.
    Right:  each test's share of the joint chi^2 per window.
    Bottom: joint chi^2 per window, and the test shares in the deciding window."""
    joint_tests = res["joint_tests"]
    nt, nj = len(NULL_TESTS), len(joint_tests)
    r, edges, xw, share = res["r"], res["edges"], res["x_win"], res["share"]
    spans = _fail_spans(res["failing"], edges, res["width"])
    fail_win = res["jchi"] > res["thresh"]
    fin = share[np.isfinite(share)]
    share_lim = (min(-0.05, fin.min() * 1.1), max(1.0, fin.max() * 1.1))
    wlab = ("" if theta_pow == 0 else r"\theta\," if theta_pow == 1
            else rf"\theta*{{{theta_pow:g}}}\,")

    fig = plt.figure(figsize=(7.2, 1.25 * nt + 2.4), layout="constrained")
    fig.get_layout_engine().set(hspace=0, h_pad=0.01)
    gs = fig.add_gridspec(nt + 1, 2, height_ratios=[1] * nt + [1.6], width_ratios=[1.7, 1])
    axj, axh = fig.add_subplot(gs[nt, 0]), fig.add_subplot(gs[nt, 1])

    for t, q in enumerate(NULL_TESTS):
        a = fig.add_subplot(gs[t, 0], sharex=axj)
        b = fig.add_subplot(gs[t, 1], sharex=axj)
        s = res["sigma"][t]
        # NOTE: r * theta_pow, not r ** theta_pow (kept as in the original)
        w = 1 / s if relative else r * theta_pow

        # left: data with noise band
        if show_sims:
            sims_t = res["sims"][:, t] * w
            for (lo_p, hi_p), al in (((2.5, 97.5), 0.12), ((16, 84), 0.20)):
                lo, hi = np.percentile(sims_t, [lo_p, hi_p], axis=0)
                a.fill_between(r, lo, hi, color=C_SIM, alpha=al, lw=0, zorder=1)
            if n_show:
                a.plot(r, sims_t[:n_show].T, color=C_SIM, alpha=0.15, lw=0.4, zorder=1)
        a.axhline(0, color="0.4", lw=0.5, zorder=2)
        y = res["d"][t] * w
        a.errorbar(r, y, yerr=s * w, fmt="o", ms=2.5, color=C_DATA,
                   elinewidth=0.7, capsize=0, zorder=5)

        lab = LABELS.get(q, q)
        if relative:
            a.set_ylabel(rf"$\Delta {{{lab}}}/\sigma$")
        elif q == "dens":
            a.set_ylabel(rf"${wlab}{lab}$")
            for sgn in (-1, 1):
                a.plot(r, sgn * res["dens_cut"] * w, color=C_DENS, lw=0.7, ls=":")
        else:
            a.set_ylabel(rf"${wlab}\Delta {{{lab}}}$")

        # lo, hi = min(y.min(), 0.0), max(y.max(), 0.0)
        # pad = y_pad * (hi - lo) if hi > lo else 1.0
        # a.set_ylim(lo - pad, hi + pad)
        lo, hi = _robust_ylim(y, s * np.abs(w), clip=clip, pad=y_pad)
        a.set_ylim(lo, hi)
        _mark_clipped(a, r, y, lo, hi)
        a.yaxis.set_major_locator(MaxNLocator(4, symmetric=True, prune="both"))
        a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

        # right: share of the joint chi2 per window
        b.set_ylim(*share_lim)
        if q in joint_tests:
            b.axhline(0, color="0.6", lw=0.5)
            b.axhline(1 / nj, color="0.6", lw=0.5, ls=":")
            b.plot(xw, share[t], color="k", lw=0.7)
            b.plot(xw[~fail_win], share[t][~fail_win], "o", ms=2.5,
                   mfc="white", mec=C_SIM, mew=0.7)
            b.plot(xw[fail_win], share[t][fail_win], "o", ms=3, color=C_FAIL)
        else:
            b.text(0.5, 0.5, "not in joint", transform=b.transAxes,
                   ha="center", va="center", color="0.5")
            b.set_yticklabels([])
        b.yaxis.set_major_locator(MaxNLocator(3))
        if t == 0:
            b.set_title(r"fraction of joint $\chi^2$", fontsize=9)

        for x in (a, b):
            _markers(x, res, spans)
            x.tick_params(labelbottom=False)

    # bottom left: joint chi2 per window
    floor = 1e-12
    if show_sims:
        lo, hi = np.percentile(np.maximum(res["jchi_sims"], floor), [2.5, 97.5], axis=0)
        axj.fill_between(xw, lo, hi, color=C_SIM, alpha=0.15, lw=0, zorder=1)
    jp = np.maximum(res["jchi"], floor)
    axj.plot(xw, jp, "-", color="0.4", lw=0.7)
    axj.plot(xw[~fail_win], jp[~fail_win], "o", ms=3, color=C_DATA)
    axj.plot(xw[fail_win], jp[fail_win], "o", ms=3.5, color=C_FAIL)
    axj.axhline(res["thresh"], color=C_FAIL, ls="--", lw=0.8,
                label=rf"threshold ($\alpha={res['alpha']:g}$)")
    axj.set_yscale("log")
    _markers(axj, res, spans)
    _log_x(axj, edges)
    axj.set_ylim(max(res["jchi"].min() / 2, 1e-2), max(res["thresh"], res["jchi"].max()) * 3)
    axj.set_ylabel(r"joint $\chi^2$")
    axj.set_xlabel(r"$\theta$ [arcsec]")
    axj.legend(loc="upper right")

    # bottom right: shares in the deciding window
    # (outermost failing window; if nothing fails, the window with the largest chi2)
    i = res["win"]
    sh = share[[NULL_TESTS.index(q) for q in joint_tests], i]
    order = np.argsort(sh)                       # largest at the top
    y = np.arange(nj)
    cols = [C_SIM] * nj
    cols[-1] = C_DATA if res["no_fail"] else C_FAIL
    axh.barh(y, sh[order], height=0.65, color=cols, lw=0)
    axh.axvline(0, color="0.4", lw=0.5)
    axh.axvline(1 / nj, color="0.6", lw=0.5, ls=":")
    axh.set_yticks(y)
    axh.set_yticklabels([rf"${LABELS.get(joint_tests[k], joint_tests[k])}$" for k in order])
    axh.tick_params(axis="y", which="both", length=0, right=False)
    axh.minorticks_off()
    for yy, v in zip(y, sh[order]):
        axh.text(max(v, 0) + 0.02, yy, f"{v:.0%}", va="center", fontsize=7, color="0.3")
    axh.set_xlim(min(0, sh.min()) - 0.05, max(sh.max(), 1 / nj) + 0.18)
    axh.set_xlabel(r"fraction of joint $\chi^2$")
    kind = "max-$\\chi^2$ window " if res["no_fail"] else "radius window "
    axh.set_title(f"{kind}({res['r_win'][i]:.1f}'')", fontsize=9)

    rad = "none" if res["no_fail"] else f"{res['radius']:.1f}''"
    dens = f"{res['dens_radius']:.1f}''" if np.isfinite(res["dens_radius"]) else "none"
    fig.suptitle(f"{title}   (joint radius {rad}, density cut {dens})", fontsize=10)
    if save_plot:
        plt.savefig(f'test{title}.png')
    plt.show()
    return fig

def _log_fit(m, r, mag_range=None):
    """Straight-line fit log10(r) = slope * m + intercept over finite, positive r
    (optionally only for mag_range[0] <= m <= mag_range[1]). None if < 2 points."""
    use = np.isfinite(r) & (r > 0)
    if mag_range is not None:
        use &= (m >= mag_range[0]) & (m <= mag_range[1])
    return np.polyfit(m[use], np.log10(r[use]), 1) if use.sum() >= 2 else None
 
 
def plot_radius_fit(results, dens_cut=0.15, pixel_scale=0.2,
                    fit_range=None, dens_fit_range=None, x_line=None, include_old=True):
    """Radius vs star magnitude, with a straight-line fit of log10(radius) vs mag
    for the joint radius and for the density-cut radius.
    results:        array from determine_radius
                    (mag, win_lo, win_hi, radius, dens_radius[, no_fail]) in arcsec.
    fit_range:      (mag_lo, mag_hi) used for the joint-radius fit (None = all).
    dens_fit_range: same for the density-cut fit.
    x_line:         magnitudes to draw the fit lines over (default: data range ± 0.5).
    Returns {"joint": (slope, intercept), "dens": (slope, intercept)} for log10(R / arcsec)."""
    results = np.asarray(results, float)
    m, w_lo, w_hi, rad, rad_dens = results.T[:5]
    no_fail = (results[:, 5].astype(bool) if results.shape[1] > 5
               else np.zeros(len(m), bool))
    ok = np.isfinite(rad) & ~no_fail             # bins with a real failing window
    if x_line is None:
        x_line = np.linspace(m.min() - 0.5, m.max() + 0.5, 50)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m[ok], rad[ok], "o", color="C0", label="joint radius (window center)")
    ax.plot(m, rad_dens, "s", mfc="none", color="C2",
            label=rf"$|\xi_{{\rm LS}}| \geq {dens_cut:g}$")
    if (~ok).any():
        ax.plot(m[~ok], rad[~ok], "v", color="0.5", label="no failure (smallest bin)")
 
    # straight-line fits of log10(radius) vs magnitude (no-failure bins excluded)
    fits = {}
    for key, r, col, rng in (("joint", np.where(ok, rad, np.nan), "C0", fit_range),
                             ("dens", rad_dens, "C2", dens_fit_range)):
        p = _log_fit(m, r, rng)
        if p is None:
            continue
        fits[key] = tuple(p)
        ax.plot(x_line, 10 ** np.polyval(p, x_line), color=col, lw=1,
                label=rf"$\log_{{10}}(R/'') = {p[0]:.4f}\,G {p[1]:+.4f}$")
        print(f"{key:>5s}: log10(R/arcsec) = {p[0]:.5f} * G {p[1]:+.5f}   "
              f"(log10(R/pix) = {p[0]:.5f} * G {p[1] - np.log10(pixel_scale):+.5f})")

    if include_old:                              # old fits are in pixels -> arcsec
        old = {"anacal": ("C3", [(-0.12, 3.839), (-0.077, 2.772)]),
               "mdet":   ("C4", [(-0.14712, 4.06523), (-0.08855, 3.19652)])}
        for name, (col, lines) in old.items():
            for j, (slope, icpt) in enumerate(lines):
                ax.plot(x_line, pixel_scale * 10 ** (slope * x_line + icpt), ls=":",
                        color=col, label=name if j == 0 else None)
        r_erin = np.clip(45.0 * 10 ** (0.115 * (15.0 - x_line)), 20.0, 450.0)
        ax.plot(x_line, pixel_scale * r_erin, ls="--", color="C5", label="erin")
            
        circ = np.clip(45.0 * 10 ** (0.115 * (15.0 - x_line)), 20.0, 450.0)   # px
        circ = np.where(x_line < 6.0, np.maximum(circ, 900.0), circ)           # ghost disk
        r_excl = np.where(x_line < 19.0, (circ + 12.0) * pixel_scale + 4.0,    # mask+taper+4''
                          np.where(x_line < 20.0, 6.0, np.nan))                # faint stars
        ax.plot(x_line, r_excl, ls="--", color="C5", label="erin (exclusion)")
        ax.plot(x_line, pixel_scale * circ, ls="-.", color="C5", lw=0.7,
                label="erin (image mask)")
        
    ax.set_yscale("log")
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel("radius [arcsec]")
    ax.set_ylim(None, 1000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
 
    # right axis: same radius in pixels
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin / pixel_scale, ymax / pixel_scale)
    ax2.set_ylabel("radius [pixel]")
 
    fig.tight_layout()
    plt.show()
    return fits

## driver
def determine_radius(mag_cuts, save_path,
                     joint_tests=("gT", "gX", "R", "snr"),
                     width=1,           # radial bins per window
                     alpha=1e-4,        # chance that pure noise fails anywhere in the scan
                     n_sims=5000,       # noise simulations used to set the threshold
                     dens_cut=0.15,     # outermost bin with |xi_LS| >= dens_cut
                     theta_pow=2.5,     # left panels show theta-weighted Delta (0 = none)
                     relative=False,    # left panels show Delta / sigma instead
                     y_pad=0.15,
                     show_sims=True,    # draw the noise simulations in each figure
                     pixel_scale=0.2,
                     make_plots=True, save_plot=False):  # arcsec per pixel, for the printout
    results = []
    for _, _, mag, _ in mag_cuts:
        star_file = f"{save_path}/star_mag{mag}.npz"
        rand_file = f"{save_path}/rand_g{mag}.npz"
        res = window_test(star_file, rand_file, joint_tests,
                          width=width, alpha=alpha, n_sims=n_sims, dens_cut=dens_cut)
        mean_mag = np.load(star_file, allow_pickle=True)["config"].item()["mean_mag"]
        results.append((mean_mag, res["win_lo"], res["win_hi"],
                        res["radius"], res["dens_radius"]))
        tag = ("no failure, smallest bin" if res["no_fail"]
               else f"strongest tests: {', '.join(res['tests'])}")
        print(f"mag {mean_mag:.2f}, window radius {res['radius']:.1f} arcsec "
              f"({res['radius'] / pixel_scale:.0f} pix), {tag}, "
              f"dens_cut: {res['dens_radius']:.1f} arcsec "
              f"({res['dens_radius'] / pixel_scale:.0f} pix)")
        if make_plots:
            plot_bin_rad(res, title=f"mag {mean_mag:.2f}", show_sims=show_sims,
                     theta_pow=theta_pow, relative=relative, y_pad=y_pad, save_plot=save_plot)
    return np.array(results)
