""" Create star randoms catalogs. """

import numpy as np
import healpy as hp
from scipy.optimize import minimize

from plotting import plot_bin

### star maps
NEFF      = 25      # stars per smoothing kernel -> sets sigma per bin (weight noise ~ 1/sqrt(NEFF))
ALPHA     = 10      # shrinkage toward the joint-fit prior, in stars per kernel (0 = pure data)
SIGMA_FIX = None    # deg; set to use one sigma for every bin (else per-bin from NEFF)
NSIDE_W   = 256     # healpix grid for the kernel smoothing (pixels << sigma)
DEG, MAG_DEG = 2, 1 # joint prior: poly degree in (l, b), and in magnitude
SIGMA_FLOOR = 5     # how many times larger the blurring kernel is to max correlation range

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

    re_weights = []
    for (b, c, mag, dif), (lo, hi) in zip(mag_cuts, mag_min_max):
        m = (s_mag_all > lo) & (s_mag_all < hi)
        theta_max = mag_bin_args[b][1] / 3600
        sig = max(sigma_for_neff(m.sum(), r_pix_w), SIGMA_FLOOR * theta_max)
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