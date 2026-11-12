""" Plotting. """

import numpy as np
from lsst.daf.butler import Butler
from matplotlib.path import Path
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.ticker import FuncFormatter, NullFormatter,MaxNLocator

from utils import NULL_TESTS
from star_random import _to_gal

ROT_RA, RESO, XSIZE, YSIZE = 330, 15.0, 400, 300   # gnomview settings for star plots

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

### Plot catalog tracts and selection
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


### star randoms plotting

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

## plotting star radius
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