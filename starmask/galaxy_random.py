""" Creating galaxy randoms catalog. """

import numpy as np
from astropy.table import Table
import healpy as hp
from scipy.optimize import minimize

from plotting import plot_density_maps, plot_weight_check

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