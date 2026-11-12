""" Determine mask radius. """

import numpy as np

from utils import QUANTITIES, NULL_TESTS
from plotting import plot_radius_fit, plot_bin_rad

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
    fits = plot_radius_fit(results)
    return np.array(results), fits