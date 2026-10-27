"""
Orientation- and scale-adaptive fringe ridge detector for laser-plasma interferograms.

Pipeline (all heavy work runs once on the native grid and is cached):

  A. Features (native grid)
     1. despike           remove hot pixels / cosmic hits (median outlier test)
     2. fan filter        one FFT: removes the slowly varying background AND any
                          structure whose wave-vector lies outside +-max_angle of
                          vertical (the vertical streaks next to the target, edge
                          glints).  Fringes up to max_angle from horizontal pass.
     3. orientation       structure tensor of the filtered image -> fringe
                          direction theta(x, y), clamped to +-max_angle
     4. tangent smoothing average ALONG the fringes by following the orientation
                          field (curved streamlines), never across them.  This kills
                          speckle without blurring dense fringes together.
     5. amplitude norm.   divide by the local fringe amplitude, so faint and bright
                          fringes look the same to the ridge detector
     6. ridge strength    second derivative across the fringe at 6 scales; the
                          scale is chosen per pixel (dense fringes near the target
                          get a fine scale, wide fringes a coarse one)
  B. Ridge center lines   non-maximum suppression across the fringe, on the output
                          grid (the image's own size, 1x, or twice that, 2x).  The
                          response is smooth, so it is upsampled before NMS and the
                          2x lines sit between the image pixel centers.
  C. Clean-up (cheap)     hysteresis threshold, fringe-region mask, junction
                          breaking (so no two fringes are ever 8-connected), short
                          piece removal, orientation-guided gap bridging.
"""
import math
import time
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

F32 = np.float32

DEFAULTS = dict(
    max_angle=70.0,      # deg; largest fringe tilt from horizontal that is kept
    smooth_len=6.0,      # native px; sigma of the averaging along the fringes
    smooth_ratio='auto', # per-pixel cap: sigma <= ratio * local period ('auto' = smooth_len / 12)
    sensitivity=0.5,     # 0..1; higher = keeps fainter ridges
    min_length=30,       # native px; shorter pieces are deleted
    gap_bridge=12,       # native px; largest gap that is bridged
    mask_level=0.25,     # 0..1; how hard short pieces are rejected where fringes are weak (Noise Rejection)
    keep_main_region=False,
    orient_scale=8.0,    # native px; structure tensor integration scale
    fine_bias=-1.0,      # scale-selection bias toward fine scales
    scale_prior=0.0,     # >0: favor ridge scales near 0.225*local period (band factor)
    bg_period=48.0,      # native px; structure coarser than this is background
    despike=True,
    refine_period=False,
    link_rmin=0.0,
    link_rmean=0.1,
    max_dev=40.0,
)

SIGMAS = (0.6, 0.85, 1.2, 1.7, 2.4, 3.4)


class Cancelled(Exception):
    """Raised inside a detection when its should_stop() callback returns True."""


def _check(stop):
    if stop is not None and stop():
        raise Cancelled()


# --------------------------------------------------------------------------- #
#  Stage A helpers
# --------------------------------------------------------------------------- #
def despike(I, k=6.0):
    med = cv2.medianBlur(np.ascontiguousarray(I, dtype=F32), 3)
    d = I - med
    mad = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 8)) + F32(1e-6)
    out = I.copy()
    m = np.abs(d) > F32(k) * mad
    out[m] = med[m]
    return out


def fan_filter(I, max_ang=70.0, taper=10.0, pmax=48.0, want_period=False):
    H, W = I.shape
    F = np.fft.rfft2(I - I.mean())
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    kr = np.sqrt(kx * kx + ky * ky)
    wr = 1.0 - np.exp(-0.5 * (kr * pmax) ** 2)
    if max_ang >= 89.5:
        # every direction up to vertical is kept
        Ff = F * wr
    else:
        alpha = np.degrees(np.arctan2(np.abs(kx), np.abs(ky)))
        a0, a1 = max_ang - taper / 2.0, max_ang + taper / 2.0
        wa = np.clip((a1 - alpha) / (a1 - a0), 0, 1)
        wa = 0.5 - 0.5 * np.cos(np.pi * wa)
        Ff = F * (wa * wr)
    out = np.fft.irfft2(Ff, s=(H, W)).astype(F32)
    if not want_period:
        return out
    # exact (spectral) gradient -> local wavenumber k^2 = <|grad f|^2> / <f^2>
    fx = np.fft.irfft2(Ff * (2j * np.pi * kx), s=(H, W)).astype(F32)
    fy = np.fft.irfft2(Ff * (2j * np.pi * ky), s=(H, W)).astype(F32)
    g2 = cv2.GaussianBlur(fx * fx + fy * fy, (0, 0), 8)
    f2 = cv2.GaussianBlur(out * out, (0, 0), 8)
    k = np.sqrt(np.maximum(g2 / (f2 + F32(1e-12)), F32(1e-6)))
    P = (F32(2 * np.pi) / k).astype(F32)
    return out, P


def structure_orientation(f, rho=8.0, pre=0.6, max_ang=70.0, rho_map=None, rhos=(3.0, 5.0, 8.0)):
    g = cv2.GaussianBlur(f, (0, 0), pre) if pre > 0 else f
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    if rho_map is None:
        J11 = cv2.GaussianBlur(gx * gx, (0, 0), rho)
        J22 = cv2.GaussianBlur(gy * gy, (0, 0), rho)
        J12 = cv2.GaussianBlur(gx * gy, (0, 0), rho)
    else:
        # integration scale varies per pixel: interpolate between a few scales
        rhos = [r for r in rhos if r < rho] + [rho]
        gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
        J11 = np.zeros_like(f); J22 = np.zeros_like(f); J12 = np.zeros_like(f)
        rm = np.clip(rho_map, rhos[0], rhos[-1])
        for i, r in enumerate(rhos):
            if i == 0:
                w = np.clip((rhos[1] - rm) / (rhos[1] - rhos[0]), 0, 1) if len(rhos) > 1 else np.ones_like(rm)
            elif i == len(rhos) - 1:
                w = np.clip((rm - rhos[i - 1]) / (rhos[i] - rhos[i - 1]), 0, 1)
            else:
                w = np.clip(np.minimum((rm - rhos[i - 1]) / (rhos[i] - rhos[i - 1]),
                                       (rhos[i + 1] - rm) / (rhos[i + 1] - rhos[i])), 0, 1)
            w = w.astype(F32)
            J11 += w * cv2.GaussianBlur(gxx, (0, 0), r)
            J22 += w * cv2.GaussianBlur(gyy, (0, 0), r)
            J12 += w * cv2.GaussianBlur(gxy, (0, 0), r)
    phi = 0.5 * np.arctan2(2 * J12, J11 - J22)          # gradient direction
    theta = phi + F32(np.pi / 2)                          # fringe direction
    theta = (theta + F32(np.pi / 2)) % F32(np.pi) - F32(np.pi / 2)
    m = F32(np.deg2rad(max_ang))
    theta = np.clip(theta, -m, m).astype(F32)
    coh = (np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2) / (J11 + J22 + F32(1e-9))).astype(F32)
    return theta, coh


def streamline_smooth(f, theta, sigma, step=1.0, sigma_map=None, stop=None):
    """Gaussian average of f along the orientation field (follows curvature).
    sigma_map (optional) gives a per-pixel sigma <= sigma."""
    if sigma <= 0:
        return f
    H, W = f.shape
    tx = np.cos(theta).astype(F32)
    ty = np.sin(theta).astype(F32)
    xx, yy = np.meshgrid(np.arange(W, dtype=F32), np.arange(H, dtype=F32))
    J = int(math.ceil(2.5 * sigma / step))
    acc = f.copy()
    wsum = np.ones_like(f) if sigma_map is not None else 1.0
    inv2 = None if sigma_map is None else (F32(-0.5) / (sigma_map * sigma_map)).astype(F32)
    for sgn in (1.0, -1.0):
        X = xx.copy(); Y = yy.copy()
        dx = F32(sgn) * tx; dy = F32(sgn) * ty
        for j in range(1, J + 1):
            _check(stop)
            # midpoint (RK2) step along the field, keeping a consistent sense
            mx = X + F32(0.5 * step) * dx; my = Y + F32(0.5 * step) * dy
            ntx = cv2.remap(tx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            nty = cv2.remap(ty, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            flip = (ntx * dx + nty * dy) < 0
            ntx[flip] = -ntx[flip]; nty[flip] = -nty[flip]
            X = X + F32(step) * ntx; Y = Y + F32(step) * nty
            dx, dy = ntx, nty
            if inv2 is None:
                w = F32(math.exp(-0.5 * (j * step / sigma) ** 2))
            else:
                w = np.exp(inv2 * F32((j * step) ** 2))
            acc += w * cv2.remap(f, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum = wsum + w
    return (acc / wsum).astype(F32)


def carrier_period(I, tile=512, nbins=60, pmin=2.2, pmax=170.0, w=6):
    """Fringe period of the image in pixels and how clearly it stands out:
    (period, prominence) of the most prominent peak of the radially averaged
    power spectrum (averaged over overlapping tiles), plus whether there is
    also a clear peak at a period below 16 px. Returns (None, 0, False) when
    the image is too small."""
    I = np.asarray(I, F32)
    H, W = I.shape
    t = int(min(tile, H, W))
    t -= t % 2
    if t < 32:
        return None, 0.0, False
    pmax = min(pmax, t / 3.0)
    win = np.outer(np.hanning(t), np.hanning(t)).astype(F32)
    ny = max(1, int(round(2 * H / t)) - 1)
    nx = max(1, int(round(2 * W / t)) - 1)
    ys = np.linspace(0, H - t, ny).astype(int)
    xs = np.linspace(0, W - t, nx).astype(int)
    acc = np.zeros((t, t // 2 + 1), np.float64)
    for y in ys:
        for x in xs:
            a = I[y:y + t, x:x + t]
            acc += np.abs(np.fft.rfft2((a - a.mean()) * win)) ** 2
    ky = np.fft.fftfreq(t)[:, None]
    kx = np.fft.rfftfreq(t)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    edges = np.exp(np.linspace(np.log(1.0 / pmax), np.log(1.0 / pmin), nbins + 1))
    idx = np.digitize(k, edges) - 1
    sel = (idx >= 0) & (idx < nbins)
    cnt = np.bincount(idx[sel], minlength=nbins)
    S = np.bincount(idx[sel], weights=acc[sel], minlength=nbins) / np.maximum(1, cnt)
    kc = np.sqrt(edges[:-1] * edges[1:])
    s = np.convolve(S * kc, np.ones(3) / 3.0, mode='same')
    prom = np.zeros(nbins)
    for i in range(nbins):
        L = s[max(0, i - w):max(0, i - 1)]
        R = s[i + 2:i + w + 1]
        if len(L) == 0 or len(R) == 0 or cnt[i] == 0:
            continue
        prom[i] = s[i] / math.sqrt(max(L.min(), 1e-30) * max(R.min(), 1e-30))
    if not prom.any():
        return None, 0.0, False
    bi = int(np.argmax(prom))
    fine = bool(np.any((1.0 / kc < 16.0) & (prom >= 4.0)))
    return float(1.0 / kc[bi]), float(prom[bi]), fine


def work_scale(I):
    """1 for ordinary interferograms. Wide fringes (a clear carrier period of
    20 px or more and no fine fringes anywhere) are analyzed on a reduced copy
    of the image so that they look like ordinary 10 px fringes; this returns
    the reduction factor."""
    P, prom, fine = carrier_period(I)
    if P is None or P < 20.0 or prom < 4.0 or fine:
        return 1
    return int(max(2, min(16, round(P / 10.0))))


def spectral_period(f, win=8.0):
    H, W = f.shape
    F = np.fft.rfft2(f)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    fx = np.fft.irfft2(F * (2j * np.pi * kx), s=(H, W)).astype(F32)
    fy = np.fft.irfft2(F * (2j * np.pi * ky), s=(H, W)).astype(F32)
    g2 = cv2.GaussianBlur(fx * fx + fy * fy, (0, 0), win)
    f2 = cv2.GaussianBlur(f * f, (0, 0), win)
    k = np.sqrt(np.maximum(g2 / (f2 + F32(1e-12)), F32(1e-6)))
    return (F32(2 * np.pi) / k).astype(F32)


def hess_nn(n, s, c, sig):
    g = cv2.GaussianBlur(n, (0, 0), sig)
    gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * F32(0.25)
    gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * F32(0.25)
    gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * F32(0.25)
    nn = s * s * gxx - F32(2) * s * c * gxy + c * c * gyy
    return (-nn * F32(sig * sig)).astype(F32)


def ridge_scale_select(n, theta, sigmas=SIGMAS, win=6.0, bias=-1.0, sig_hint=None, hint_band=1.6):
    s, c = np.sin(theta).astype(F32), np.cos(theta).astype(F32)
    S = len(sigmas)
    Rs = np.empty((S,) + n.shape, F32)
    Es = np.empty((S,) + n.shape, F32)
    for i, sg in enumerate(sigmas):
        r = hess_nn(n, s, c, sg)
        Rs[i] = r
        Es[i] = cv2.GaussianBlur(r * r, (0, 0), win) * F32(sg ** bias)
        if sig_hint is not None:
            # soft prior: penalize scales far from the period-based hint
            lr = np.log(F32(sg) / sig_hint) / F32(np.log(hint_band))
            Es[i] *= np.exp(-0.5 * lr * lr)
    i = np.argmax(Es, 0)
    im = np.clip(i - 1, 0, S - 1); ip = np.clip(i + 1, 0, S - 1)
    e0 = np.take_along_axis(Es, i[None], 0)[0]
    em = np.take_along_axis(Es, im[None], 0)[0]
    ep = np.take_along_axis(Es, ip[None], 0)[0]
    den = em - 2 * e0 + ep
    ok = (i > 0) & (i < S - 1) & (den < 0)
    off = np.zeros_like(e0)
    off[ok] = 0.5 * (em[ok] - ep[ok]) / den[ok]
    fi = i + np.clip(off, -0.5, 0.5)
    lo = np.clip(np.floor(fi).astype(np.int64), 0, S - 2)
    w = (fi - lo).astype(F32)
    R = (1 - w) * np.take_along_axis(Rs, lo[None], 0)[0] + w * np.take_along_axis(Rs, (lo + 1)[None], 0)[0]
    sig = np.exp(np.interp(fi.ravel(), np.arange(S), np.log(sigmas))).reshape(fi.shape).astype(F32)
    return R.astype(F32), sig


# --------------------------------------------------------------------------- #
#  Stage B: ridge center lines on the output grid
# --------------------------------------------------------------------------- #
def nms_on_grid(R, theta, up=1, strip=512):
    H, W = R.shape
    if up > 1:
        Ru = cv2.resize(R, (W * up, H * up), interpolation=cv2.INTER_CUBIC)
        c2 = cv2.resize(np.cos(2 * theta), (W * up, H * up), interpolation=cv2.INTER_LINEAR)
        s2 = cv2.resize(np.sin(2 * theta), (W * up, H * up), interpolation=cv2.INTER_LINEAR)
        th = (0.5 * np.arctan2(s2, c2)).astype(F32)
        del c2, s2
    else:
        Ru, th = R, theta
    Hu, Wu = Ru.shape
    nx = (-np.sin(th)).astype(F32)
    ny = np.cos(th).astype(F32)
    keep = np.zeros((Hu, Wu), bool)
    xs = np.arange(Wu, dtype=F32)[None, :]
    for y0 in range(0, Hu, strip):
        y1 = min(Hu, y0 + strip)
        ys = np.arange(y0, y1, dtype=F32)[:, None]
        X = np.broadcast_to(xs, (y1 - y0, Wu))
        Y = np.broadcast_to(ys, (y1 - y0, Wu))
        sx, sy = nx[y0:y1], ny[y0:y1]
        p = cv2.remap(Ru, X + sx, Y + sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.remap(Ru, X - sx, Y - sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r = Ru[y0:y1]
        keep[y0:y1] = (r > p) & (r >= m) & (r > 0)
    return keep, Ru, th


# --------------------------------------------------------------------------- #
#  Stage C helpers
# --------------------------------------------------------------------------- #
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def neighbors(b):
    return cv2.filter2D(b.astype(np.uint8), cv2.CV_8U, _K8, borderType=cv2.BORDER_CONSTANT)


def thin(b):
    """Skeletonize only where the line map is thicker than 1 px (2x2 blocks).
    The ridge map is already 1 px almost everywhere, so this is much faster
    than thinning the whole frame at 2x."""
    from skimage.morphology import skeletonize
    b = b.astype(bool)
    # 2x2 windows holding 3 or 4 line pixels: blobs and 4-connected staircase corners
    s4 = cv2.filter2D(b.astype(np.uint8), cv2.CV_8U, np.ones((2, 2), np.uint8), anchor=(0, 0),
                      borderType=cv2.BORDER_CONSTANT)
    thick = s4 >= 3
    if not thick.any():
        return b
    zone = cv2.dilate(thick.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8), connectivity=8)
    out = b.copy()
    H, W = b.shape
    for i in range(1, n):
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        x0, y0 = max(0, x - 3), max(0, y - 3)
        x1, y1 = min(W, x + w + 3), min(H, y + h + 3)
        sk = skeletonize(np.ascontiguousarray(b[y0:y1, x0:x1]))
        reg = lab[y0:y1, x0:x1] == i
        sub = out[y0:y1, x0:x1]
        sub[reg] = sk[reg]
    return out


def break_junctions(b):
    b = b.copy()
    for _ in range(3):
        nb = neighbors(b)
        j = b & (nb >= 3)
        if not j.any():
            break
        b[j] = False
    return b


def break_off_axis(b, theta, max_dev_deg=50.0):
    """Cut the line wherever it runs across the local fringe direction (a jump
    from one fringe to the next).  Uses the direction between each pixel's two
    neighbors, so single-pixel staircase steps are not affected."""
    nb = neighbors(b)
    m = b & (nb == 2)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return b
    pad = np.pad(b, 1)
    vx = np.zeros(len(ys), np.int32); vy = np.zeros(len(ys), np.int32)
    first = np.ones(len(ys), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            hit = pad[ys + 1 + dy, xs + 1 + dx]
            # first neighbor found is subtracted, second added
            sub = hit & first
            add = hit & ~first
            vx[sub] -= dx; vy[sub] -= dy
            vx[add] += dx; vy[add] += dy
            first[sub] = False
    ang = np.arctan2(vy, vx)
    d = np.abs(((ang - theta[ys, xs]) + np.pi / 2) % np.pi - np.pi / 2)
    bad = d > np.deg2rad(max_dev_deg)
    out = b.copy()
    out[ys[bad], xs[bad]] = False
    return out


def remove_short(b, min_len):
    if min_len <= 1:
        return b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    good = stats[:, cv2.CC_STAT_AREA] >= min_len
    good[0] = False
    return good[lab]


def hysteresis(cand, R, lo, hi):
    weak = cand & (R > lo)
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    strong_lab = np.unique(lab[weak & (R > hi)])
    keep = np.zeros(n, bool); keep[strong_lab] = True; keep[0] = False
    return keep[lab]


def local_period(b, win):
    """Local fringe period from the detected lines: mean distance to the nearest
    line is P/4 for evenly spaced lines."""
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.minimum(d, F32(1e6))        # no line at all: the transform returns a huge value
    return (4.0 * cv2.blur(d, (int(win) | 1, int(win) | 1))).astype(F32)


def link_gaps(b, theta, period, max_gap, lat_frac=0.35, max_turn_deg=30.0, R=None, r_min=-0.1, r_mean=0.0):
    """Join endpoint pairs that continue each other along the orientation field."""
    H, W = b.shape
    nb = neighbors(b)
    ep = b & (nb == 1)
    ey, ex = np.nonzero(ep)
    if len(ey) < 2:
        return b, 0
    n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    elab = lab[ey, ex]
    # inward neighbor of each endpoint
    pad = np.pad(b, 1)
    inx = np.zeros(len(ey), F32); iny = np.zeros(len(ey), F32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            hit = pad[ey + 1 + dy, ex + 1 + dx]
            inx[hit] = dx; iny[hit] = dy
    t = theta[ey, ex]
    tx, ty = np.cos(t), np.sin(t)
    sgn = np.where(tx * (-inx) + ty * (-iny) >= 0, 1.0, -1.0)
    ox, oy = tx * sgn, ty * sgn                      # outward tangent
    tree = cKDTree(np.c_[ex, ey])
    pairs = tree.query_pairs(r=max_gap, output_type='ndarray')
    if len(pairs) == 0:
        return b, 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = elab[i] != elab[j]
    i, j = i[keep], j[keep]
    vx = (ex[j] - ex[i]).astype(F32); vy = (ey[j] - ey[i]).astype(F32)
    dist = np.sqrt(vx * vx + vy * vy) + F32(1e-6)
    # mean tangent of the two ends, oriented along i's outward direction
    mtx = ox[i] - ox[j]; mty = oy[i] - oy[j]
    mn = np.sqrt(mtx * mtx + mty * mty) + 1e-6
    mtx /= mn; mty /= mn
    along = vx * mtx + vy * mty
    lat = np.abs(-vx * mty + vy * mtx)
    P = 0.5 * (period[ey[i], ex[i]] + period[ey[j], ex[j]])
    cos_turn = math.cos(math.radians(max_turn_deg))
    ok = ((ox[i] * vx + oy[i] * vy) > 0) & ((-(ox[j] * vx + oy[j] * vy)) > 0)
    ok &= (-(ox[i] * ox[j] + oy[i] * oy[j])) > cos_turn
    ok &= lat <= np.maximum(1.0, lat_frac * P)
    ok &= along > 0
    if not ok.any():
        return b, 0
    i, j, dist, lat = i[ok], j[ok], dist[ok], lat[ok]
    cost = dist + 4.0 * lat
    order = np.argsort(cost)
    used = np.zeros(len(ey), bool)
    out = b.copy()
    occ = lab  # occupancy by label (original)
    links = 0
    for k in order:
        a, c = i[k], j[k]
        if used[a] or used[c]:
            continue
        x0, y0, x1, y1 = int(ex[a]), int(ey[a]), int(ex[c]), int(ey[c])
        L = int(max(abs(x1 - x0), abs(y1 - y0)))
        if L < 2:
            xs = np.array([x0, x1]); ys = np.array([y0, y1])
        else:
            tt = np.linspace(0, 1, L + 1)
            xs = np.rint(x0 + (x1 - x0) * tt).astype(int); ys = np.rint(y0 + (y1 - y0) * tt).astype(int)
        # reject if the bridge passes next to a third fringe
        la, lc = elab[a], elab[c]
        bad = False
        for xx, yy in zip(xs[1:-1], ys[1:-1]):
            win = occ[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]
            if np.any((win != 0) & (win != la) & (win != lc)):
                bad = True
                break
        if bad:
            continue
        if R is not None and len(xs) > 2:
            rv = R[ys[1:-1], xs[1:-1]]
            if rv.min() < r_min or rv.mean() < r_mean:
                continue   # the bridge would cross a dark fringe -> different fringe
        out[ys, xs] = True
        used[a] = used[c] = True
        links += 1
    return out, links


# --------------------------------------------------------------------------- #
#  Detector object with stage caching
# --------------------------------------------------------------------------- #
class RidgeDetector:
    def __init__(self, raw, params=None):
        raw = np.asarray(raw, dtype=F32)
        bad = ~np.isfinite(raw)
        if bad.any():
            # NaN would spread through the FFT and stall the smoothing step
            good = raw[~bad]
            raw = raw.copy()
            raw[bad] = float(np.median(good)) if good.size else 0.0
        self.raw = raw
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self._A = None; self._A_key = None
        self._B = {}; self._B_key = None
        self.timings = {}

    # --- stage A ---------------------------------------------------------- #
    def _a_key(self):
        p = self.p
        return (round(p['max_angle'], 3), round(p['smooth_len'], 3), str(p.get('smooth_ratio', 0)), round(p['orient_scale'], 3), int(p.get('orient_iter', 0)), round(p.get('orient_scale2', 5.0), 3), round(p.get('orient_ratio', 0), 3),
                round(p['fine_bias'], 3), round(p.get('scale_prior', 0), 3), round(p['bg_period'], 3), bool(p['despike']))

    def work_image(self):
        """(image the analysis runs on, reduction factor). The image with hot
        pixels removed; for wide fringes a reduced copy (see work_scale)."""
        flag = bool(self.p['despike'])
        hit = getattr(self, '_work', None)
        if hit is not None and hit[0] == flag:
            return hit[1], hit[2]
        I = despike(self.raw) if flag else self.raw
        s = work_scale(I)
        if s > 1:
            H, W = I.shape
            I = cv2.resize(I, (max(16, int(round(W / float(s)))), max(16, int(round(H / float(s))))),
                           interpolation=cv2.INTER_AREA)
        self._work = (flag, I, s)
        return I, s

    def _stage_a(self, I, p, progress, stop):
        """Stage A maps of image I (see the module docstring). Returns (maps, local period, timings)."""
        T = {}
        t = time.time()
        if progress: progress('Filtering background and streaks')
        f, Pest = fan_filter(I, p['max_angle'], pmax=p['bg_period'], want_period=True)
        T['fan'] = time.time() - t; t = time.time()
        _check(stop)
        if progress: progress('Estimating fringe orientation')
        rmap = None
        if p.get('orient_ratio', 0) > 0:
            rmap = (F32(p['orient_ratio']) * Pest).astype(F32)
        theta, coh = structure_orientation(f, rho=p['orient_scale'], max_ang=p['max_angle'], rho_map=rmap)
        T['orient'] = time.time() - t; t = time.time()
        if progress: progress('Smoothing along fringes')
        smap = None
        ratio = p.get('smooth_ratio', 0)
        if ratio == 'auto' or ratio is None:
            ratio = p['smooth_len'] / 12.0
        if ratio > 0:
            smap = np.clip(F32(ratio) * Pest, 0.75, max(0.75, p['smooth_len'])).astype(F32)
        _check(stop)
        # long smoothing follows the fringes in 2 px steps (half the work, same result to the eye)
        step = 2.0 if p['smooth_len'] > 15.0 else 1.0
        ft = streamline_smooth(f, theta, p['smooth_len'], step=step, sigma_map=smap, stop=stop)
        if ratio > 0 and p.get('refine_period', False):
            # speckle makes the first period estimate too short in noisy, wide-fringe
            # areas; re-measure on the along-fringe smoothed image and redo
            Pest = np.maximum(Pest, spectral_period(ft))
            smap = np.clip(F32(ratio) * Pest, 0.75, max(0.75, p['smooth_len'])).astype(F32)
            ft = streamline_smooth(f, theta, p['smooth_len'], sigma_map=smap, stop=stop)
        for _ in range(int(p.get('orient_iter', 0))):
            # re-estimate orientation on the along-fringe smoothed image (higher SNR),
            # then smooth again along the improved field
            theta, coh = structure_orientation(ft, rho=p.get('orient_scale2', 5.0), pre=0.0,
                                               max_ang=p['max_angle'])
            ft = streamline_smooth(f, theta, p['smooth_len'], sigma_map=smap, stop=stop)
        T['smooth'] = time.time() - t; t = time.time()
        A = np.sqrt(cv2.GaussianBlur(ft * ft, (0, 0), 6.0)) + F32(1e-6)
        n = (ft / A).astype(F32)
        _check(stop)
        if progress: progress('Measuring ridge strength')
        hint = None
        if p.get('scale_prior', 0) > 0:
            hint = np.clip(F32(0.225) * Pest, SIGMAS[0], SIGMAS[-1]).astype(F32)
        R, sig = ridge_scale_select(n, theta, bias=p['fine_bias'], sig_hint=hint,
                                    hint_band=max(1.05, p.get('scale_prior', 1.6)))
        T['ridge'] = time.time() - t; t = time.time()
        # fringe-region quality: local ridge energy, coherence, illumination
        ped = float(np.percentile(I, 0.5))
        M = cv2.GaussianBlur(I, (0, 0), 8) - F32(ped)
        upper = M[M > np.percentile(M, 50)]
        # (no pixel above the median: a blank image, or one that is mostly saturated)
        Mref = float(np.median(upper)) if upper.size else float(M.max())
        illum = (M / F32(max(Mref, 1e-6))).astype(F32)
        energy = np.sqrt(cv2.GaussianBlur(R * R, (0, 0), 12)).astype(F32)
        cohs = cv2.GaussianBlur(coh, (0, 0), 12)
        T['quality'] = time.time() - t
        return dict(theta=theta, R=R, sig=sig, illum=illum, energy=energy, coh=cohs, flat=n), Pest, T

    def features(self, progress=None, should_stop=None):
        key = self._a_key()
        if self._A is not None and self._A_key == key:
            return self._A
        cache = getattr(self, '_A_cache', {})
        if key in cache:
            # recently computed (e.g. auto-optimize or a slider moved back)
            self._A, self._B, self._Pest = cache[key]
            self._A_key = key
            return self._A
        p = self.p
        stop = should_stop
        _check(stop)
        I, s = self.work_image()
        if s == 1:
            A, Pest, T = self._stage_a(I, p, progress, stop)
        else:
            # wide fringes: analyze the reduced image (where they look like ordinary
            # 10 px fringes), then bring every map back to the image's own size
            ps = dict(p, smooth_len=p['smooth_len'] / float(s))
            A, Pest, T = self._stage_a(I, ps, progress, stop)
            H, W = self.raw.shape
            sx = W / float(I.shape[1])
            up = lambda a, interp=cv2.INTER_LINEAR: cv2.resize(a, (W, H), interpolation=interp)
            th = A['theta']
            c2 = up(np.cos(2 * th)); s2 = up(np.sin(2 * th))
            A = dict(theta=(0.5 * np.arctan2(s2, c2)).astype(F32), R=up(A['R'], cv2.INTER_CUBIC),
                     sig=(up(A['sig']) * F32(sx)).astype(F32), illum=up(A['illum']), energy=up(A['energy']),
                     coh=up(A['coh']), flat=up(A['flat'], cv2.INTER_CUBIC))
            Pest = (up(Pest) * F32(sx)).astype(F32)
        self._Pest = Pest
        self._A = A
        self._A_key = key
        self._B = {}
        self.timings.update(T)
        # keep the previous result too, so toggling between two settings is instant
        keep = max(1, int(getattr(self, 'A_keep', 2)) - 1)
        old = {k: v for k, v in getattr(self, '_A_cache', {}).items() if k != key}
        old = dict(list(old.items())[-keep:])
        old[key] = (self._A, self._B, self._Pest)
        self._A_cache = old
        return self._A

    # --- stage B ---------------------------------------------------------- #
    def candidates(self, up=1, progress=None, should_stop=None):
        A = self.features(progress, should_stop)
        if up in self._B:
            return self._B[up]
        _check(should_stop)
        if progress: progress('Locating fringe center lines')
        t = time.time()
        keep, Ru, th = nms_on_grid(A['R'], A['theta'], up=up)
        self.timings['nms'] = time.time() - t
        self._B[up] = (keep, Ru, th)
        return self._B[up]

    # --- region mask ------------------------------------------------------ #
    def region_mask(self, up=1):
        """Hard mask: only blanks unlit areas (target shadow, outside the probe beam)."""
        A = self.features()
        key = (self._A_key, up, bool(self.p.get('keep_main_region')), float(self.p['mask_level']) if self.p.get('keep_main_region') else None)
        hit = getattr(self, '_RM', None)
        if hit is not None and hit[0] == key:
            return hit[1]
        m = A['illum'] > 0.08
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        if self.p.get('keep_main_region'):
            q = self.quality_map() > 0.5
            n, lab, stats, _ = cv2.connectedComponentsWithStats((m & q).astype(np.uint8), connectivity=4)
            if n > 1:
                areas = stats[:, cv2.CC_STAT_AREA].copy(); areas[0] = 0
                main = lab == int(np.argmax(areas))
                main = cv2.morphologyEx(main.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
                main = ndi.binary_fill_holes(main)
                m &= main
        if up > 1:
            m = cv2.resize(m.astype(np.uint8), (m.shape[1] * up, m.shape[0] * up), interpolation=cv2.INTER_NEAREST) > 0
        self._RM = (key, m)
        return m

    def quality_map(self):
        """0..1 fringe quality from orientation coherence (smoothed)."""
        A = self.features()
        lvl = float(self.p['mask_level'])
        lo = 0.2 + 0.3 * lvl
        return np.clip((A['coh'] - lo) / 0.35, 0, 1).astype(F32)

    def quality_length_filter(self, b, up):
        """Short pieces are only trusted where the fringes are coherent: the
        required length grows as the local fringe quality drops."""
        lvl = float(self.p['mask_level'])
        if lvl <= 0:
            return b
        q = self.quality_map()
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
        if n <= 1:
            return b
        ys, xs = np.nonzero(lab)
        L = lab[ys, xs]
        qs = q[np.minimum(ys // up, q.shape[0] - 1), np.minimum(xs // up, q.shape[1] - 1)]
        qmean = np.bincount(L, weights=qs, minlength=n) / np.maximum(1, np.bincount(L, minlength=n))
        length = stats[:, cv2.CC_STAT_AREA]
        need = self.p['min_length'] * up / np.maximum(qmean, 1e-3)
        good = (length >= need) & (qmean > 0.12)
        good[0] = False
        return good[lab]

    # --- stage C ---------------------------------------------------------- #
    def _cached(self, level, key, fn, keep=3):
        """Small per-level cache of clean-up results, so moving one slider only
        redoes the steps after the one it affects."""
        store = self.__dict__.setdefault('_C', {}).setdefault(level, [])
        for k, v in store:
            if k == key:
                return v
        v = fn()
        store.append((key, v))
        if len(store) > keep:
            store.pop(0)
        return v

    def detect(self, up=1, progress=None, return_debug=False, should_stop=None):
        p = self.p
        cand, Ru, th = self.candidates(up, progress, should_stop)
        _check(should_stop)
        t = time.time()
        if progress: progress('Cleaning up')
        akey = (self._A_key, up)
        sens = float(np.clip(p['sensitivity'], 0, 1))
        ml = int(p['min_length'])
        gap = float(p['gap_bridge'])
        k1 = (akey, round(sens, 6), float(p.get('max_dev', 0)))

        def level1():
            lo = 0.55 - 0.6 * sens          # sens 0.55 -> 0.22
            hi = lo + 0.18
            mask = self.region_mask(up)
            b = hysteresis(cand & mask, Ru, lo, hi)
            b = thin(b)
            b = break_junctions(b)
            if p.get('max_dev', 0) > 0:
                b = break_off_axis(b, th, p['max_dev'])
            return b
        b1 = self._cached(1, k1, level1)
        k2 = (k1, ml)
        b2 = self._cached(2, k2, lambda: remove_short(b1, max(3, int(0.3 * ml * up))))
        k3 = (k2, round(gap, 6), float(p.get('link_rmin', -0.1)), float(p.get('link_rmean', 0.0)))

        def level3():
            b = b2
            nlinks = 0
            if gap > 0:
                per = local_period(b, 31 * up)
                pest = getattr(self, '_Pest', None)
                if pest is not None:
                    pe = pest if up == 1 else cv2.resize(pest, (pest.shape[1] * up, pest.shape[0] * up),
                                                         interpolation=cv2.INTER_LINEAR) * F32(up)
                    per = np.minimum(per, pe)
                per = np.clip(per, 3 * up, 40 * up)
                for frac in (0.5, 1.0):
                    b, nl = link_gaps(b, th, per, max_gap=max(2, frac * gap * up), R=Ru,
                                      r_min=p.get('link_rmin', -0.1), r_mean=p.get('link_rmean', 0.0))
                    nlinks += nl
                b = thin(b)
                b = break_junctions(b)
            return b, nlinks
        b3, nlinks = self._cached(3, k3, level3)
        _check(should_stop)
        b = remove_short(b3, int(ml * up))
        b = self.quality_length_filter(b, up)
        self.timings['cleanup'] = time.time() - t
        self.last_links = nlinks
        if return_debug:
            return b, dict(mask=self.region_mask(up), R=Ru)
        return b


# --------------------------------------------------------------------------- #
#  Auto-Optimize
# --------------------------------------------------------------------------- #
def crowding(det, b, up, frac=0.6):
    """Line pixels that have another line closer than frac * local period along
    the fringe normal: spurious ridges between fringes and jumps across them."""
    ys, xs = np.nonzero(b)
    if len(ys) == 0:
        return 0
    A = det.features()
    th = A['theta'][ys // up, xs // up]
    P = det._Pest[ys // up, xs // up] * up
    nx, ny = -np.sin(th), np.cos(th)
    H, W = b.shape
    lim = np.maximum(2.0, frac * P)
    crowded = np.zeros(len(ys), bool)
    tmax = int(np.ceil(lim.max()))
    for t in range(2, tmax + 1):
        act = (lim >= t) & ~crowded
        if not act.any():
            break
        for sg in (1, -1):
            yy = np.rint(ys[act] + sg * t * ny[act]).astype(np.int64)
            xx = np.rint(xs[act] + sg * t * nx[act]).astype(np.int64)
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            hit = np.zeros(ok.shape, bool)
            hit[ok] = b[yy[ok], xx[ok]]
            idx = np.nonzero(act)[0][hit]
            crowded[idx] = True
    return int(crowded.sum())


# Coherence of pure camera noise after the fan filter, against Max Fringe Angle
# (median of the smoothed coherence map). Narrow angle ranges make noise look
# orderly, so fringe coherence is always judged relative to this floor.
_NOISE_COH = ((0, 0.999), (5, 0.995), (10, 0.983), (15, 0.962), (20, 0.931), (25, 0.894), (30, 0.847),
              (35, 0.793), (40, 0.733), (45, 0.667), (50, 0.597), (55, 0.525), (60, 0.454), (65, 0.383),
              (70, 0.312), (75, 0.247), (80, 0.188), (85, 0.146), (90, 0.126))


def noise_coherence(max_angle):
    a, c = zip(*_NOISE_COH)
    return float(np.interp(float(max_angle), a, c))


# Weights of the Auto-Optimize score. The score estimates how much hand editing
# a trace saves: every pixel of real fringe counts, a broken end costs about as
# much as linking it, and lines squeezed between two fringes (where fringes
# merge) cost extra. Whether a piece is a real fringe is judged from how orderly
# the fringe pattern is around it (quality_reference with fringe tilts up to ref
# degrees, above the level pure noise reaches). More smoothing gives longer,
# cleaner looking lines but also lines that drift across dense fringes, which
# the score cannot see directly; ksm weighs against it (per px of smoothing, per
# 1000 px of line). The free numbers were fitted so that the settings it picks
# need the least hand editing on hand-edited real shots.
# Stray pieces (spur, doubt) then had their weight halved: they are quick to
# delete, so finding more fringe is worth a few more of them. Ends are only
# counted away from the edge of the lit area, so a stray near the edge costs
# less than one in the middle. A weight below half made the search pick too low
# a Max Fringe Angle on closed fringe loops and join fringes there.
SCORE_W = dict(ref=70.0, a=-0.1178, b=0.5027, w=0.598, gain=1 / 25.0, spur=0.5 * 1.521, end=0.666,
               doubt=0.5 * 0.954, crowd=0.0321, ksm=0.2628)


def quality_reference(det, angle=90.0, should_stop=None):
    """Coherence map the Auto-Optimize score uses to judge whether a traced piece
    is a real fringe: how orderly the fringe pattern is around each pixel, from
    the structure tensor of the image with the background removed and every
    fringe tilt up to angle kept. It does not depend on the sliders, so traces
    made with different Max Fringe Angle or Smoothing are judged alike. Made
    once per image (on the image's own grid)."""
    cache = det.__dict__.setdefault('_q_ref', {})
    if angle in cache:
        return cache[angle]
    I, s = det.work_image()
    f = fan_filter(I, angle, pmax=DEFAULTS['bg_period'])
    _check(should_stop)
    _th, coh = structure_orientation(f, rho=DEFAULTS['orient_scale'], max_ang=angle)
    if s > 1:
        H, W = det.raw.shape
        coh = cv2.resize(coh, (W, H), interpolation=cv2.INTER_LINEAR)
    cache[angle] = coh.astype(F32)
    return cache[angle]


def auto_score(det, b, up, W=None):
    """Unsupervised quality of a trace; higher is better (units: hand-editing
    effort saved, roughly one unit per 25 px of fringe)."""
    W = dict(SCORE_W, **(W or {}))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    info = dict(pieces=int(n - 1))
    if n <= 1:
        return -1e9, dict(info, length=0.0, trusted=0.0, ends=0, crowded=0.0)
    ys, xs = np.nonzero(b)
    L = lab[ys, xs]
    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    # how orderly the fringe pattern is around each piece, on a reference map that
    # is the same for every setting, above the level pure noise reaches
    coh = quality_reference(det, W['ref'])
    cn = noise_coherence(W['ref'])
    cm = np.bincount(L, weights=coh[np.minimum(ys // up, coh.shape[0] - 1), np.minimum(xs // up, coh.shape[1] - 1)]
                     .astype(np.float64), minlength=n) / cnt
    cnorm = (cm - cn) / max(1e-6, 1.0 - cn)
    q = np.clip((cnorm - W['a']) / W['b'], 0, 1) ** W['w']
    lens = st[:, cv2.CC_STAT_AREA] / float(up)
    q[0] = 0.0
    lens[0] = 0.0
    real = np.ones(n, bool); real[0] = False
    trusted = float((q * lens).sum())
    gain = W['gain'] * trusted
    stray = W['spur'] * float((1.0 - q)[real].sum()) + W['doubt'] * W['gain'] * float(((1.0 - q) * lens).sum())
    nb = neighbors(b)
    ep = b & (nb == 1)
    mg = int(40 * up)
    lit = det.region_mask(up)
    inner = cv2.erode(lit.astype(np.uint8), np.ones((2 * mg + 1, 2 * mg + 1), np.uint8)) > 0
    inner[:mg, :] = False; inner[-mg:, :] = False; inner[:, :mg] = False; inner[:, -mg:] = False
    n_end = int((ep & inner).sum())
    crowd = crowding(det, b, up) / float(up)
    score = gain - stray - W['end'] * n_end - W['crowd'] * crowd \
        - W.get('ksm', 0.0) * float(det.p['smooth_len']) * float(lens.sum()) / 1000.0
    info.update(length=float(lens.sum()), trusted=trusted, ends=n_end, crowded=crowd)
    return float(score), info


def estimate_max_angle(det, should_stop=None):
    """Max Fringe Angle the search starts from: 90 when most of the fringe pattern
    is steeper than 60 degrees (vertical or steep carrier fringes), otherwise 60,
    which traced the hand-edited shots best. The search then steps the angle up
    or down while the trace improves."""
    I, _s = det.work_image()
    _check(should_stop)
    f = fan_filter(I, 90.0, pmax=det.p['bg_period'])
    _check(should_stop)
    th, coh = structure_orientation(f, rho=det.p['orient_scale'], max_ang=90.0)
    E = cv2.GaussianBlur(f * f, (0, 0), 8)
    w = ((coh > 0.6) * E).ravel().astype(np.float64)
    if w.sum() <= 0:
        return 60.0
    a = np.degrees(np.abs(th)).ravel()
    order = np.argsort(a)
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    median = float(a[order][min(len(a) - 1, int(np.searchsorted(cw, 0.5)))])
    return 90.0 if median >= 60.0 else 60.0


def _fmt_eta(sec):
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f'{sec} s'
    return f'{sec // 60} min {sec % 60:02d} s'


# Values Auto-Optimize tries for the clean-up sliders: their whole useful range.
AUTO_C = dict(sensitivity=(0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
              mask_level=(0.0, 0.25, 0.5, 0.7, 0.85, 1.0),
              min_length=(10, 20, 30, 45, 60, 100, 150),
              gap_bridge=(0, 4, 8, 12, 16, 24, 32, 40))
# Max Fringe Angle and Smoothing need a full analysis per value, so they are
# stepped from the start value (60 degrees, 9 px) to the next value up while the
# trace improves, or down when the first step up did not help.
AUTO_ANGLES = (55.0, 60.0, 70.0, 85.0, 90.0)
AUTO_SMOOTH = (3.0, 6.0, 9.0, 12.0, 15.0)


class _AutoSearch:
    """The Auto-Optimize search (see auto_optimize)."""
    KEYS = ('smooth_len', 'max_angle', 'sensitivity', 'mask_level', 'gap_bridge', 'min_length')
    ORDER = ('sensitivity', 'mask_level', 'min_length', 'gap_bridge')

    def __init__(self, det, up, progress, should_stop):
        self.det, self.up, self.progress, self.stop = det, up, progress, should_stop
        self.t_start = time.time()
        # the plan, for the countdown: clean-up evaluations (about a second each),
        # full analyses (one step up and one down for the angle and for the
        # smoothing; a step that helps plans the next one in its direction) and
        # the comparisons on the 2x grid
        self.plan_clean = sum(len(AUTO_C[k]) - 1 for k in self.ORDER) + 4 * len(self.ORDER)
        self.plan_now = [9.0, 9.0]          # smoothing of each analysis the current step still needs
        self.plan_later = [12.0, 6.0]       # and of the later steps
        self.plan_2x = 3 if up > 1 else 0
        self.cur_sm = 9.0
        self.done = 0
        self.n_clean = 0
        self.n_2x = 0
        self.tA = []            # (smoothing, seconds) of evaluations that needed a full analysis
        self.tC = []            # seconds of clean-up only evaluations
        self.t2x = []
        self.seen = {}
        self.a_keys = set()

    def _t_analysis(self, sm, tc):
        """Seconds for a full analysis at smoothing sm, scaled from the ones done so far
        (the along-fringe smoothing dominates and grows with the smoothing length)."""
        units = lambda v: 3.0 + (v / 2.0 if v > 15 else v)
        if not self.tA:
            return 8.0 * units(sm) / units(9.0)
        k = float(np.mean([max(0.0, t - tc) / units(v) for v, t in self.tA]))
        return tc + k * units(sm)

    def remaining(self):
        tc = float(np.median(self.tC)) if self.tC else 0.6
        left = max(0, self.plan_clean - self.n_clean) * tc
        planned = self.plan_now + self.plan_later
        left += sum(self._t_analysis(sm, tc) for sm in planned)
        if self.plan_2x > self.n_2x:
            t2 = float(np.median(self.t2x)) if self.t2x else 4 * tc + 0.3 * self._t_analysis(self.cur_sm, tc)
            left += (self.plan_2x - self.n_2x) * t2
        n_left = max(0, self.plan_clean - self.n_clean) + len(planned) + max(0, self.plan_2x - self.n_2x)
        return left + tc, n_left

    def tick(self):
        if self.progress is None:
            return
        left, n_left = self.remaining()
        total = self.done + n_left
        msg = 'Auto-Optimize: step %d of about %d' % (self.done, total)
        if self.tA:
            msg += ', about %s left' % _fmt_eta(left)
        elapsed = time.time() - self.t_start
        self.progress(msg, elapsed / max(1e-6, elapsed + left))

    def evaluate(self, p, eval_up=1):
        key = (tuple(p[k] for k in self.KEYS), eval_up)
        if key in self.seen:
            return self.seen[key][0]
        _check(self.stop)
        self.det.p.update(p)
        akey = (p['smooth_len'], p['max_angle'], eval_up)
        t0 = time.time()
        b = self.det.detect(up=eval_up, should_stop=self.stop)
        s, info = auto_score(self.det, b, eval_up)
        dt = time.time() - t0
        if eval_up > 1:
            self.t2x.append(dt)
            self.n_2x += 1
        elif akey not in self.a_keys:
            self.tA.append((p['smooth_len'], dt))
        else:
            self.tC.append(dt)
            self.n_clean += 1
        self.a_keys.add(akey)
        self.seen[key] = (s, info)
        self.done += 1
        self.tick()
        return s

    def scan(self, cur, best, keys, near=False):
        """One pass over the clean-up sliders. near=False tries every value,
        near=True only the values next to the current one."""
        improved = False
        for k in keys:
            vals = AUTO_C[k]
            if near:
                i = min(range(len(vals)), key=lambda j: abs(vals[j] - cur[k]))
                vals = [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]
            for v in vals:
                if v == cur[k]:
                    continue
                p = dict(cur)
                p[k] = v
                s = self.evaluate(p)
                if s > best:
                    best, cur, improved = s, p, True
        return cur, best, improved

    def polish(self, cur, best, passes=3):
        for i in range(passes):
            cur, best, improved = self.scan(cur, best, self.ORDER, near=True)
            if not improved:
                break
            if i + 1 < passes:
                self.plan_clean += 2 * len(self.ORDER)      # another pass is coming
        return cur, best

    def step(self, cur, best, key, vals):
        """Move one setting (Max Fringe Angle or Smoothing) to the next value up
        while the score improves, or down when the first step up did not help."""
        vals = list(vals)
        i = min(range(len(vals)), key=lambda j: abs(vals[j] - cur[key]))
        sm0 = cur['smooth_len']

        def plan(j, d):
            # the analysis the step needs next, if there is a value there
            if not 0 <= j + d < len(vals):
                return []
            return [vals[j + d] if key == 'smooth_len' else sm0]
        moved = False
        for d in (1, -1):
            j = i
            self.plan_now = plan(i, d) + (plan(i, -1) if d == 1 else [])
            while 0 <= j + d < len(vals):
                p = dict(cur, **{key: vals[j + d]})
                if self.plan_now:
                    self.plan_now.pop(0)
                s = self.evaluate(p)
                if s > best:
                    best, cur, j, moved = s, p, j + d, True
                    self.plan_now = plan(j, d)
                else:
                    break
            if moved:
                break
        self.plan_now = []
        return cur, best, moved

    def run(self):
        det, up = self.det, self.up
        if self.progress:
            self.progress('Auto-Optimize: measuring the fringe directions in this image', 0.0)
        base = dict(DEFAULTS)
        a0 = estimate_max_angle(det, self.stop)
        cur = dict(base, smooth_len=9.0, max_angle=a0, sensitivity=0.6, mask_level=0.85, gap_bridge=8,
                   min_length=60)
        best = self.evaluate(cur)
        # 1. every value of each clean-up slider
        cur, best, _ = self.scan(cur, best, self.ORDER)
        # 2. Max Fringe Angle, stepped up or down from the start while it helps
        angles = AUTO_ANGLES if a0 >= 90 else tuple(a for a in AUTO_ANGLES if a < 90)
        cur, best, moved = self.step(cur, best, 'max_angle', angles)
        if moved:
            cur, best = self.polish(cur, best)
        # 3. smoothing, the same way
        self.plan_later = []
        cur, best, _ = self.step(cur, best, 'smooth_len', AUTO_SMOOTH)
        self.cur_sm = cur['smooth_len']
        cur, best = self.polish(cur, best)
        self.plan_clean = self.n_clean          # the clean-up searches are over
        if up > 1:
            # the best few with the winning smoothing and angle, compared again on the 2x grid
            same = [(v[0], k[0]) for k, v in self.seen.items()
                    if k[1] == 1 and (k[0][0], k[0][1]) == (cur['smooth_len'], cur['max_angle'])]
            best2 = None
            for _s, vals in sorted(same, reverse=True)[:3]:
                p = dict(base, **dict(zip(self.KEYS, vals)))
                s2 = self.evaluate(p, eval_up=up)
                if best2 is None or s2 > best2[0]:
                    best2 = (s2, p)
            cur = best2[1]
        _check(self.stop)
        det.p.update(cur)
        if self.progress:
            self.progress('Auto-Optimize: applying the best setting', self.done / (self.done + 1.0))
        b = det.detect(up=up, should_stop=self.stop)
        p = {k: cur[k] for k in cur if k in DEFAULTS}
        for k in ('gap_bridge', 'min_length'):
            p[k] = int(round(p[k]))
        report = ('Auto-Optimize done in %s (%d settings tried): sensitivity %.2f, smoothing %.1f, max angle %.0f, '
                  'gap %d, min length %d, noise rejection %.2f'
                  % (_fmt_eta(time.time() - self.t_start), len(self.seen), p['sensitivity'], p['smooth_len'],
                     p['max_angle'], p['gap_bridge'], p['min_length'], p['mask_level']))
        return p, b, report


def auto_optimize(det, up=1, progress=None, should_stop=None):
    """Search all six detection settings for the trace with the best auto_score.

    1. Start at Smoothing 9 and Max Fringe Angle 60 (90 when most fringes are
       steeper than 60 degrees, see estimate_max_angle).
    2. Every value of Sensitivity, Noise Rejection, Minimum Length and Gap
       Bridging (AUTO_C, their whole useful range), one slider at a time.
    3. Max Fringe Angle (AUTO_ANGLES) stepped to the next value up while the
       score improves, or down when the first step up did not help; the
       clean-up sliders are then polished (the values next to the current ones,
       until nothing improves).
    4. Smoothing (AUTO_SMOOTH, 3 to 15 px) stepped the same way, then polished.
    5. On the 2x grid the search runs on the image's own grid (fast) and the
       three best candidates with the winning smoothing and angle are compared
       again on the 2x grid.

    The search never looks at the current sliders, so the answer is the same
    whatever they were. progress(message, fraction) reports the countdown.
    Returns (best_params, binary, report); raises Cancelled when
    should_stop() becomes true."""
    keep = getattr(det, 'A_keep', 2)
    det.A_keep = 4          # coming back to an earlier smoothing must not redo its analysis
    try:
        return _AutoSearch(det, up, progress, should_stop).run()
    finally:
        det.A_keep = keep
