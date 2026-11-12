""" Run correlations for each catalog. """

import numpy as np
import treecorr
from tqdm.notebook import tqdm
import os
from pathlib import Path


from utils import QUANTITIES, NULL_TESTS, NPATCH

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

### catalog specific functions

### Anacal
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
    Path(save_path).mkdir(parents=True, exist_ok=True)

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


### mdet
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
    Path(save_path).mkdir(parents=True, exist_ok=True)

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


### erin
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
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
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