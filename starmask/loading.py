""" Loading star catalogs and galaxy catalogs used in defining star masks.
    Also loads footprint and generates baseline randoms catalogs. """

import sys
import numpy as np
from astropy.table import Table, vstack
from tqdm.notebook import tqdm
from lsst.geom import degrees
from dataregistry import DataRegistry
from matplotlib.path import Path
import healsparse
from joblib import Parallel, delayed
import traceback
import healpy as hp
import rustfits
from numpy.lib import recfunctions as rfn
from pathlib import Path as filePath

from utils import get_butler, AB_ZP_NJY

### Load catalog tracts
def get_anacal_tracts():
    butler = get_butler()
    tract_list_refs = list(butler.registry.queryDatasets('deep_coadd_cell_anacal_merged', skymap='lsst_cells_v2'))
    tract_list = np.array([trf.dataId['tract'] for trf in tract_list_refs])
    return tract_list

def get_mdet_tracts():
    butler = get_butler()
    tract_list_refs = list(butler.registry.queryDatasets('object_shear_all', skymap='lsst_cells_v2'))
    tract_list = np.array([trf.dataId['tract'] for trf in tract_list_refs])
    return tract_list

def get_erin_tracts():
    source = filePath("/global/cfs/projectdirs/lsst/groups/PO/users/esheldon/desc-mdet/lsst-mdet-runs/run-dp2-v01.1")
    tract_list = sorted(int(p.name) for p in source.glob("[0-9]" * 5) if p.is_dir())
    return tract_list

### Load footprint, star catalog, randoms catalog, and tract mask for selected region
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

    butler = get_butler()
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

### Load individual object catalogs
def load_anacal(tract_ids_cut, tract_mask, footprint, n_jobs=32):
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
