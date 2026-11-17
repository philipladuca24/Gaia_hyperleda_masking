import os
import healsparse
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, unique
from matplotlib.path import Path
from tqdm import tqdm
import astropy.units as u
from joblib import Parallel, delayed
from lsst.daf.butler import Butler
from lsst.geom import (Box2I, Point2D, Point2I, Extent2I,SpherePoint, degrees)
from lsst.ip.diffim import GetTemplateTask
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.sphgeom import (Box as SphBox, ConvexPolygon, LonLat)
from dataregistry import DataRegistry
import threading
import traceback
import pickle
import argparse

from starmask.loading import get_mdet_tracts, load_baseline
os.environ['DAF_BUTLER_REPOSITORY_INDEX'] = '/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml'

## load all stars
mag_cuts =[ # bin_int, cut_int, mag, mag_cut
    # (0,5,1.58489),
    # (0,6,1.58489),
    # (0,7,1.58489),
    # (0,0, 8,1.58489),
    (0,0, 8,1.5),
    (0,1, 10,0.5),
    (0,2, 11,0.5),

    (1,3, 12,0.5),
    (1,4, 13,0.5),
    (1,5, 14,0.5),

    (2,6, 15,0.5),
    (2,7, 16,0.5),
    (2,8, 17,0.5),
    (2,9, 18,0.5),
    (2,10, 19,0.4),
    (2,11, 20,0.3)]
mag_min_max = [(m - d, m + d) for _,_,m,d in mag_cuts]

n_jobs = 32
pixel_radius = 1000

## use the mdet tracts
def load_stars():
    tract_list = get_mdet_tracts()
    tract_ids_cut, monsters, star_randoms, tract_mask, footprint = load_baseline(tract_list, n_jobs)
    return tract_ids_cut, monsters, star_randoms, tract_mask, footprint


def get_tract_wcs(tract_id, tract_wcs):
    if tract_id not in tract_wcs:
        tract_wcs[tract_id] = skyMap[tract_id].getWcs()
    return tract_wcs[tract_id]

def paste_span(span, origin_yx, src_tract, out_mask, ref_tract_id, img_bbox, tract_wcs, ref_wcs):
    yy, xx = np.nonzero(span > 0)
    if len(xx) == 0:
        return
    src_x = xx.astype(float) + origin_yx[1]
    src_y = yy.astype(float) + origin_yx[0]
    if src_tract == ref_tract_id:
        dst_x, dst_y = src_x, src_y    
    else:
        sra, sdec = get_tract_wcs(src_tract, tract_wcs).pixelToSkyArray(src_x, src_y, degrees=True)
        dst_x, dst_y = ref_wcs.skyToPixelArray(sra, sdec, degrees=True)

    ix = np.round(dst_x).astype(int) - img_bbox.getMinX()
    iy = np.round(dst_y).astype(int) - img_bbox.getMinY()
    ok = (ix >= 0) & (ix < out_mask.shape[1]) & (iy >= 0) & (iy < out_mask.shape[0])
    out_mask[iy[ok], ix[ok]] = True

def get_scarlet(tract_id, patch_id, scarlet_cache, butler):
    key = (tract_id, patch_id)
    if key not in scarlet_cache:
        try:
            scarlet_refs = list(butler.query_datasets("object_scarlet_models",
                where=f"skymap='lsst_cells_v2' AND "
                      f"tract={tract_id} AND "
                      f"patch={patch_id}"))
            scarlet_cache[key] = butler.get(scarlet_refs[0])
        except Exception:
            scarlet_cache[key] = None
    return scarlet_cache[key]

def bbox_to_cutout(bbox, wcs, ref_wcs, img_bbox):
    corners = bbox.getCorners()
    sky_points = [
        wcs.pixelToSky(Point2D(p))
        for p in corners]
    sky_points.append(sky_points[0])
    ra = np.array([p.getRa().asDegrees() for p in sky_points])
    dec = np.array([p.getDec().asDegrees() for p in sky_points])
    x, y = ref_wcs.skyToPixelArray(
        ra,
        dec,
        degrees=True,)
    x -= img_bbox.getMinX()
    y -= img_bbox.getMinY()
    return x, y

def get_cutout(ra, dec, radius_pix, bin_width, band, sep_cut, mad_sigma, butler, skyMap, do_plot=False, do_prints=False):
    skypos = SpherePoint(ra, dec, degrees)
    ref_tract_info = skyMap.findTract(skypos)
    ref_wcs = ref_tract_info.getWcs()
    ref_tract_id = ref_tract_info.getId()
    center = ref_wcs.skyToPixel(skypos)
    cut_bbox = Box2I(
        Point2I(int(round(center.x)) - radius_pix, int(round(center.y)) - radius_pix),
        Extent2I(2 * radius_pix + 1, 2 * radius_pix + 1),
    )
    corners = [ref_wcs.pixelToSky(Point2D(c)) for c in cut_bbox.getCorners()]
    region = ConvexPolygon([c.getVector() for c in corners])    
    coadd_refs = list(butler.query_datasets(
        "deep_coadd",
        where=f"patch.region OVERLAPS(region) and band in ('{band}')",
        bind={"region": region},
    ))
    if do_prints:
        print('coadds_num', len(coadd_refs), flush=True)
    mapping = {}
    for ref in set(coadd_refs):
        mapping.setdefault(ref.dataId["tract"], []).append(ref)
    getTemplateTask = GetTemplateTask()
    result = getTemplateTask.run(
        coaddExposureHandles={
            t: [InMemoryDatasetHandle(butler.get(r, storageClass="Exposure")) for r in refs]
            for t, refs in mapping.items()},
        bbox=cut_bbox, wcs=ref_wcs, physical_filter=band,
        dataIds={t: [r.dataId for r in refs] for t, refs in mapping.items()},
    )
    template = result.template
    img_bbox = template.getBBox()
    cutout = template.image.array.copy()
    col_names = ['coord_ra', 'coord_dec', 'objectId', 'patch', 'detect_isIsolated',
                 'detect_fromBlend', 'parentObjectId', 'deblend_blendId']
    obj_refs = list(butler.query_datasets(
        "object", where="tract.region OVERLAPS(region)", bind={"region": region}))
    if do_prints:
        print('num obj', len(obj_refs), flush=True)
    tabs = []
    for ref in obj_refs:
        t = butler.get(ref, parameters={"columns": col_names})
        t["tract"] = ref.dataId["tract"]
        tabs.append(t)
    obj = vstack(tabs)
    obj_x, obj_y = ref_wcs.skyToPixelArray(obj['coord_ra'], obj['coord_dec'], degrees=True)
    in_cut = ((obj_x >= img_bbox.getMinX()) & (obj_x <= img_bbox.getMaxX()) &
              (obj_y >= img_bbox.getMinY()) & (obj_y <= img_bbox.getMaxY()))
    obj_cut = obj[in_cut]
    
    # drop the blend containing the target star so it isn't masked
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    coords = SkyCoord(ra=obj_cut['coord_ra']*u.deg, dec=obj_cut['coord_dec']*u.deg)
    sep = target.separation(coords)
    idx = sep.argmin()
    obj_near = obj_cut[sep <= sep_cut*u.arcsec]
    obj_cut = obj_cut[
        (obj_cut['parentObjectId'] != obj_cut[idx]['parentObjectId']) &
        (obj_cut['deblend_blendId'] != obj_cut[idx]['deblend_blendId']) & 
        (obj_cut['objectId'] != obj_cut[idx]['objectId'])]
    for ob in obj_near:
        obj_cut = obj_cut[
            (obj_cut['parentObjectId'] != ob['parentObjectId']) &
            (obj_cut['deblend_blendId'] != ob['deblend_blendId']) & 
            (obj_cut['objectId'] != ob['objectId'])]
    
    tract_wcs = {} 
    scarlet_cache = {}
    im_mask = np.zeros(cutout.shape, dtype=bool)
    im_mask_iso = np.zeros(cutout.shape, dtype=bool)
    if do_prints:
        print('len obj cut', len(obj_cut), flush=True)
    
    for ob in obj_cut:
        sm = get_scarlet(int(ob['tract']), int(ob['patch']),scarlet_cache, butler)
        if sm is None:
            print('scarlet fail', flush=True)
            continue
        src_tract = int(ob['tract'])
        if ob['detect_isIsolated']:
            fp = sm.isolated[ob['objectId']]
            paste_span(fp.span_array, fp.origin, src_tract, im_mask_iso, ref_tract_id, img_bbox, tract_wcs, ref_wcs)
        else:
            blend = sm.blends[ob['parentObjectId']].children[ob['deblend_blendId']]
            for comp in blend.sources[str(ob['objectId'])].components:
                paste_span(comp.morph, comp.origin, src_tract, im_mask, ref_tract_id, img_bbox, tract_wcs, ref_wcs)
    cutout = np.where(im_mask | im_mask_iso, np.nan, cutout)
    cy = (cutout.shape[0] - 1) / 2
    cx = (cutout.shape[1] - 1) / 2
    radial_bins = np.arange(0, min(cx,cy)+1, bin_width)
    yy, xx = np.indices(cutout.shape)
    radius = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    clipped_image = cutout.copy()
    clipped_image[radius >= radial_bins[-1]] = np.nan
    
    radial_mean = []
    radial_sigma = []
    radial_radii = []
    for i, (r0, r1) in enumerate(zip(radial_bins[:-1], radial_bins[1:])): 
        mask = (radius >= r0) & (radius < r1)
        values = cutout[mask]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        sigma = 1.4826 * mad
        lower = median - mad_sigma * sigma
        upper = median + mad_sigma * sigma
        values_clipped = values[(values >= lower) & (values <= upper)]
        mean_clipped = np.mean(values_clipped)
        bin_pixels = mask & np.isfinite(cutout)
        clipped_image[bin_pixels & ((cutout < lower) | (cutout > upper))] = np.nan
        else:
            mean_clipped = np.mean(values)
            sigma = np.std(values)
        radial_radii.append((r0 + r1) / 2)
        radial_mean.append(mean_clipped)
        radial_sigma.append(sigma)

    count_img = ~np.isnan(clipped_image)
    
    return clipped_image, count_img

def _star_cutout(ra, dec, pixel_radius, bin_width, band, sep_cut, mad_sigma, skyMap):
    try:
        star_im, star_imp = get_cutout(ra, dec, pixel_radius, bin_width, band, sep_cut, mad_sigma, get_butler(), skyMap)
        return [star_im, star_imp]
    except Exception:
        print(f"--- star ra={ra:.5f} dec={dec:.5f} failed ---\n"
              f"{traceback.format_exc()}", flush=True)
        return None

def get_star_cutout(star_positions,njobs=32):
    stars_stack, input_count, n_ok = None, None, 0
    
    for res in Parallel(n_jobs=n_jobs, backend="threading",return_as="generator_unordered")(
        delayed(_star_cutout)(star['ra'],star['dec'], pixel_radius,10,'i',5,5,skyMap)
        for star in tqdm(star_positions)):
    if res is None:
        continue
    im, cnt = res
    if stars_stack is None:
        stars_stack = np.zeros(im.shape)
        input_count = np.zeros(cnt.shape, dtype=np.int32)
    im[np.isnan(im)] = 0
    stars_stack += im
    input_count += cnt
    n_ok += 1

print(n_ok, 'star_results', flush=True)
if n_ok == 0:
    raise SystemExit("No successful cutouts, nothing to stack.")
    return



# with open(f'/global/homes/p/pladuca/desc-pladuca/dp2_masks/star_stack_outputs/randoms_{n_ok}sum_{start_ind}region.pkl', 'wb') as f:
#     pickle.dump({'sum':stars_stack,'count':input_count}, f)
# print('done', flush=True)

# stack = np.divide(stars_stack,input_count,out=np.full_like(stars_stack, np.nan),where=input_count > 0)
# with open(f'/global/homes/p/pladuca/desc-pladuca/dp2_masks/star_stack_outputs/mag_{m}_{n_ok}stack.pkl', 'wb') as f:
#     pickle.dump(stack, f)
# print('done', flush=True)