"""
Fringe Editor.

A control panel on the left, the image on the right.

Layers (all the same size: the working grid, which is the shot image's own
size at 1x or twice that at 2x):
  shot binary       self.mask          uint8, 0 = fringe, 255 = background (drawn in red)
  gray mask         self.gray_mask     uint8, 255 = masked (saved as 128, in the binary and gray mask files)
  reference binary  self.overlay_mask  uint8, 0 = fringe, drawn in cyan
  shot image        self._shot_img     uint8 photo plate (from self._shot_src, the image at 1x)
  reference image   self._ref_img      uint8 photo plate (from self._ref_src)

Every tool acts on the layer chosen under "Editing" (shot or reference binary),
except Mask Pen, which always edits the gray mask of the shot. Masking a
pixel removes any fringe under it, exactly as the saved PNG will have it.

Nudge (arrow keys) moves the reference image and reference binary together
by whole working-grid pixels, so you can see where shot and reference line
up. It is stored as an offset (self._nudge); the reference layers themselves
never move, so tools that edit the reference binary work at the offset and a
saved reference binary is unchanged. Merge Binaries uses the offset.

Undo and redo cover everything that changes a layer, including opening a
reference binary, a gray mask or an image, merging, nudging and switching the
grid. Pixel edits store only the rectangle that changed; whole-layer changes
store the old layer itself (no copy), so undo stays quick.

Binaries are read with one rule everywhere: below 128 is fringe, exactly 128 is
gray mask, anything else is background (SpellBook writes 0, 128 and 255).
"""
import hashlib
import math
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from fringe_detection.ui_helpers import (MOD, help_icon, attach_tooltip, bold_font, fixed_font, bind_wheel,
                                         bind_pan, scrollable_panel, make_inline_slider, imread_any, imwrite_png,
                                         image_filetypes, sanitize_image, mod_pressed, shift_pressed, ui_scale,
                                         numeric_validator, overlay_button, CanvasPrompt, position_text, image_px)

ORANGE = (255, 128, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)          # shot binary, as on the Detection tab
BLACK = (0, 0, 0)          # fringes not touched by a Connection Check
BLUE = (40, 120, 255)      # mirrored view
YELLOW = (255, 230, 0)     # the piece Delete Fringe would remove
OUTSIDE = (51, 51, 51)     # canvas color outside the working grid

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EDITED_DIR = os.path.join(APP_DIR, 'EditedImages')
INPUT_DIR = os.path.join(APP_DIR, 'InputImages')
ZOOM_STEP = 1.15
ZOOM_MIN, ZOOM_MAX = 0.02, 64.0
NUDGE_GROUP_S = 0.7        # arrow presses closer together than this are one undo step

TOOLS = [
    # key, label, shortcut, tooltip
    ('pen', 'Pen', 'P', f'Draw fringe pixels on the layer selected under Editing. Size is set below '
                        f'({MOD} + mouse wheel also changes it). Shortcut: P'),
    ('erase', 'Eraser', 'E', 'Remove fringe pixels from the layer selected under Editing. Shortcut: E'),
    ('line', 'Line', 'L', 'Straight 1 px line: click and drag, or click the start then the end. With '
                          f'Polyline on, every click adds a corner; Enter or {MOD}+click finishes. With Snap to '
                          'ends on, clicks jump to the nearest fringe end. Shortcut: L'),
    ('spline', 'Spline', 'S', f'Smooth 1 px curve through your clicks. Click to add points; {MOD}+click '
                              f'adds the last point and finishes (Enter also finishes); Esc cancels. Shortcut: S'),
    ('delete', 'Delete Fringe', 'D', 'Click a fringe line to delete that whole connected piece. The piece under '
                                     'the cursor is highlighted in yellow first. Only the exact pixel you click '
                                     'counts, so zoom in to pick a line. Shortcut: D'),
    ('mask', 'Mask Pen', 'M', 'Mark out areas without fringes to isolate the proper region. The mask is drawn '
                              'in gray and saved as gray (128) in the binary. Masking removes the fringes '
                              'underneath, as in the saved file. Hold Shift while painting to remove mask. '
                              'Shortcut: M'),
    ('ruler', 'Ruler', 'R', 'Drag to measure a distance in pixels of the image. Drag either end to adjust; '
                            'Lock keeps it horizontal or vertical. Shortcut: R'),
    ('check', 'Connection Check', 'K', 'Drag a line across fringes: every fringe it touches turns orange. '
                                       'Fringes that stay black are not connected to them. The orange stays '
                                       'while you use other tools and follows your edits; Esc clears it. '
                                       'Shortcut: K'),
]
SIZED_TOOLS = ('pen', 'erase', 'mask')
DEFAULT_SIZES = {'pen': 0.5, 'erase': 1.5, 'mask': 10.0}
SIZE_SLIDER_MAX = 60.0     # end of the Size slider
SIZE_MAX = 500.0           # Ctrl + wheel and a typed size can go past the slider up to this
TEXT_WIDGETS = ('Entry', 'TEntry', 'Spinbox', 'TSpinbox', 'TCombobox', 'Text')


# =========================================================== pure helpers ==
_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))   # N, NE, E, SE, S, SW, W, NW


def fringe_ends(ink):
    """(ys, xs) of fringe ends: pixels whose neighbors form one run of at most 3
    around them. That is the end of a 1 px line, and also the end of a pen
    stroke that starts with a small clump. Isolated pixels count too."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return ys, xs
    pad = np.pad(ink, 1)
    ring = np.stack([pad[ys + 1 + dy, xs + 1 + dx] for dy, dx in _RING], axis=1)
    cnt = ring.sum(1)
    runs = (ring & ~np.roll(ring, -1, axis=1)).sum(1)
    end = (cnt == 0) | ((runs == 1) & (cnt <= 3))
    return ys[end], xs[end]


def upsample_lines_2x(ink):
    """Fringe lines on a grid twice the size, still 1 px wide and with exactly
    the same connections (no fringes touch that did not touch before)."""
    H, W = ink.shape
    out = np.zeros((2 * H, 2 * W), bool)
    out[::2, ::2] = ink
    out[::2, 1:2 * W - 1:2] = ink[:, :-1] & ink[:, 1:]
    out[1:2 * H - 1:2, ::2] = ink[:-1, :] & ink[1:, :]
    a, b, c, d = ink[:-1, :-1], ink[:-1, 1:], ink[1:, :-1], ink[1:, 1:]
    diag = (a & d & ~b & ~c) | (b & c & ~a & ~d) | (a & b & c & d)
    out[1:2 * H - 1:2, 1:2 * W - 1:2] = diag
    return out


def classify_binary(img):
    """(fringe layer, gray mask) from a binary image, with the same rule everywhere:
    below 128 is fringe, exactly 128 is gray mask, anything else is background."""
    gray = np.where(img == 128, 255, 0).astype(np.uint8)
    mask = np.where(img < 128, 0, 255).astype(np.uint8)
    return mask, gray


def reduce_any_2x(m):
    H, W = m.shape
    h, w = H // 2, W // 2
    return m[:2 * h, :2 * w].reshape(h, 2, w, 2).any(axis=(1, 3))


def to_display(img):
    """Photo plate -> uint8 with an automatic contrast stretch (0.5 / 99.5
    percentile), so a few hot pixels cannot make the whole image dark."""
    a, _n = sanitize_image(img)
    lo, hi = np.percentile(a, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def plate_on_grid(src, factor):
    if src is None:
        return None
    if factor == 1:
        return src
    h, w = src.shape[:2]
    return cv2.resize(src, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def gray_mask_from_file(img):
    """Masked pixels of a gray mask file. Saved gray masks use the shot binary's
    convention (128 = masked, 255 = not masked), so a shot binary also works as a
    gray mask. Files from SpellBook 1 used 255 = masked on 0, which is still read."""
    if (img == 128).any():
        return img == 128
    if (img < 128).any():
        return img >= 128          # SpellBook 1 file: 0 = not masked, 255 = masked
    return np.zeros(img.shape, bool)


def link_endpoints(ink, tol, angle_deg, blocked=None):
    """Pairs of fringe ends to join with a straight 1 px stroke.

    A pair is joined when the ends belong to different pieces, are within tol
    pixels, the stroke is within angle_deg of horizontal (0 = horizontal only,
    90 = any angle up to vertical), each end points toward the other, the
    stroke runs along the fringes rather than across to the next fringe
    (sideways offset small next to the local fringe spacing, and no longer
    than six fringe spacings), and it does not touch any other piece or the
    gray mask.
    Each end is used once, closest and straightest pairs first.
    Returns a list of ((x0, y0), (x1, y1))."""
    H, W = ink.shape
    ys, xs = fringe_ends(ink)
    n = len(ys)
    if n < 2:
        return []
    _nl, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    el = lab[ys, xs]
    tree = cKDTree(np.c_[xs, ys])
    pairs = tree.query_pairs(r=float(tol), output_type='ndarray')
    if len(pairs) == 0:
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    keep = el[i] != el[j]
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return []
    vx = (xs[j] - xs[i]).astype(np.float64)
    vy = (ys[j] - ys[i]).astype(np.float64)
    dist = np.hypot(vx, vy)
    ok = dist > 0
    if angle_deg < 90:
        # joining line within angle_deg of horizontal; 0 allows only horizontal joins
        ok &= np.abs(vy) <= math.tan(math.radians(max(0.0, angle_deg))) * np.abs(vx) + 1e-9
        ok &= vx != 0
    i, j, vx, vy, dist = i[ok], j[ok], vx[ok], vy[ok], dist[ok]
    if len(i) == 0:
        return []
    # outward direction of each end involved: from the centroid of its own
    # piece within 5 px, to the end pixel
    used_ends = np.unique(np.r_[i, j])
    dirs = np.zeros((n, 2))
    R = 5
    for k in used_ends:
        y, x = int(ys[k]), int(xs[k])
        y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
        py, px = np.nonzero(lab[y0:y1, x0:x1] == el[k])
        d = np.array([x - (px.mean() + x0), y - (py.mean() + y0)])
        nn = math.hypot(d[0], d[1])
        if nn > 0.3:
            dirs[k] = d / nn
    ux, uy = vx / dist, vy / dist
    di, dj = dirs[i], dirs[j]
    face_i = di[:, 0] * ux + di[:, 1] * uy
    face_j = -(dj[:, 0] * ux + dj[:, 1] * uy)
    has_i = np.abs(di).sum(1) > 0
    has_j = np.abs(dj).sum(1) > 0
    ok = (~has_i | (face_i > 0.2)) & (~has_j | (face_j > 0.2))
    # sideways offset relative to the mean fringe direction at the two ends
    tx = di[:, 0] - dj[:, 0]; ty = di[:, 1] - dj[:, 1]
    tn = np.hypot(tx, ty)
    tx = np.where(tn > 1e-6, tx / np.maximum(tn, 1e-6), ux); ty = np.where(tn > 1e-6, ty / np.maximum(tn, 1e-6), uy)
    lat = np.abs(-vx * ty + vy * tx)
    # local fringe spacing at each end: distance to the nearest other piece straight
    # across the fringe (both sides), and the binary-wide estimate (mean distance to
    # the nearest line is a quarter period); the smaller one is used
    dt = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    per_dt = 4.0 * cv2.blur(dt, (31, 31))
    ends_used = np.unique(np.r_[i, j])
    spacing = np.full(n, np.inf)
    for k in ends_used:
        spacing[k] = per_dt[ys[k], xs[k]]
    nx_, ny_ = -dirs[ends_used, 1], dirs[ends_used, 0]
    has = np.abs(dirs[ends_used]).sum(1) > 0
    T = int(max(8, 2 * tol))
    best = np.full(len(ends_used), np.inf)
    for sgn in (1.0, -1.0):
        found = ~has
        for t in range(1, T + 1):
            if found.all():
                break
            yy = np.rint(ys[ends_used] + sgn * t * ny_).astype(int)
            xx = np.rint(xs[ends_used] + sgn * t * nx_).astype(int)
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            lv = np.zeros(len(ends_used), np.int32)
            lv[inside] = lab[yy[inside], xx[inside]]
            hit = ~found & (lv != 0) & (lv != el[ends_used])
            best[hit] = np.minimum(best[hit], t)
            found |= hit
    spacing[ends_used] = np.minimum(spacing[ends_used], best)
    P = np.minimum(spacing[i], spacing[j])
    ok &= lat <= np.maximum(0.75, 0.35 * P)
    # where fringes are only a few pixels apart a long bridge is a guess: keep it short
    ok &= dist <= np.maximum(6.0, 6.0 * P)
    i, j, dist, lat = i[ok], j[ok], dist[ok], lat[ok]
    if len(i) == 0:
        return []
    order = np.argsort(dist + 4.0 * lat, kind='stable')
    used = np.zeros(n, bool)
    links = []
    for k in order:
        a, c = int(i[k]), int(j[k])
        if used[a] or used[c]:
            continue
        x0, y0, x1, y1 = int(xs[a]), int(ys[a]), int(xs[c]), int(ys[c])
        L = max(abs(x1 - x0), abs(y1 - y0))
        tt = np.linspace(0.0, 1.0, L + 1)
        px = np.rint(x0 + (x1 - x0) * tt).astype(int); py = np.rint(y0 + (y1 - y0) * tt).astype(int)
        bad = False
        la, lc = el[a], el[c]
        for xx, yy in zip(px[1:-1], py[1:-1]):
            win = lab[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]
            if np.any((win != 0) & (win != la) & (win != lc)):
                bad = True
                break
            if blocked is not None and blocked[yy, xx]:
                bad = True
                break
        if bad:
            continue
        used[a] = used[c] = True
        links.append(((x0, y0), (x1, y1)))
    return links


# =============================================================== editor ===
class FringeEditorFrame(ttk.Frame):
    """Embeddable fringe editor."""

    MAX_UNDO = 100
    MAX_UNDO_BYTES = 1_500_000_000

    def __init__(self, master=None, file_callback=None, grid_callback=None, cursor_callback=None,
                 status_callback=None):
        super().__init__(master)
        self._file_cb = file_callback or (lambda t: None)
        self._status_cb = status_callback or (lambda t: None)
        self._grid_cb = grid_callback or (lambda f: None)
        self._cursor_cb = cursor_callback or (lambda t: None)

        # ---- data -------------------------------------------------------- #
        self.grid_factor = 1
        self.gh, self.gw = 2048, 2048
        self.native_hw = None               # (H, W) of the shot image at 1x, when known
        self.mask = np.full((self.gh, self.gw), 255, dtype=np.uint8)
        self.gray_mask = np.zeros((self.gh, self.gw), dtype=np.uint8)
        self.overlay_mask = None
        self._shot_src = None
        self._ref_src = None
        self._shot_img = None
        self._ref_img = None
        self._shot_raw = None               # the shot image's own values (for the cursor readout)
        self._nudge = (0, 0)                # offset of the reference image and binary, working-grid px
        self._nudge_t = 0.0
        self._names = {'binary': None, 'shot': None, 'ovl': None, 'ref': None}
        self._loaded = False                # a binary or image has been loaded
        self._saved_hash = self._doc_hash()
        self._saved_ref_hash = None
        self._plate_dir = INPUT_DIR if os.path.isdir(INPUT_DIR) else APP_DIR
        self._bin_dir = EDITED_DIR if os.path.isdir(EDITED_DIR) else APP_DIR

        # ---- undo / redo -------------------------------------------------- #
        self._undo = []
        self._redo = []
        self._snap_before = None

        # ---- caches ------------------------------------------------------- #
        self._labels_cache = None
        self._labels_dirty = True
        self._labels_gen = 0
        self._ends_cache = {}
        self._ovl_labels = None
        self._hover = None                  # (layer, label, bbox, mask) under the cursor for Delete Fringe
        self._ver = {}
        self._live = {}
        self._pyr = {}
        self._samples = {}
        self.magic2_labels = set()          # labels of the fringes the Connection Check touched
        self._check_line = None             # that check's line (grid coords) while its orange is shown
        self._check_shape = None            # grid size it was made on
        self._check_px = None               # bool: pixels of the fringes it touched (kept with every tool)
        self._check_key = None              # which version of the shot binary _check_px was made from

        # ---- view state --------------------------------------------------- #
        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._fitted = False
        self._frame_rgb = None
        self._photo = None
        self._img_item = None
        self._render_id = None

        # ---- interaction state --------------------------------------------- #
        self._painting = False
        self._paint_layer = None
        self._paint_rect = None
        self._last_paint_xy = None
        self._tool_points = []
        self._preview_items = []
        self._drag_start = None
        self._ruler_points = []
        self._ruler_items = []
        self._ruler_drag_idx = None
        self._cursor_item = None
        self._snap_item = None
        self._last_mouse = None
        self._pan_start = None
        self._sizes = dict(DEFAULT_SIZES)

        self._build_ui()
        self._bind_events()
        self._set_tool('pen')
        self.set_status('Open a binary to edit, or use Send to Editor on the Detection tab')

    # ================================================================== UI ==
    def _section(self, parent, title, help_text):
        row = ttk.Frame(parent)
        row.pack(anchor='w', fill='x', pady=(8, 2))
        ttk.Label(row, text=title, font=bold_font(self)).pack(side='left')
        help_icon(row, help_text)
        return row

    def _slider(self, parent, label, var, frm, to, command, fmt='{:.2f}', resolution=0.01, help_text=None,
                type_max=None):
        row = ttk.Frame(parent)
        row.pack(anchor='w', fill='x')
        ttk.Label(row, text=label, width=11).pack(side='left')
        sc = make_inline_slider(row, var, frm, to, resolution=resolution, fmt=fmt, command=lambda _=None: command(),
                                type_max=type_max)
        if help_text:
            help_icon(row, help_text)
        return sc

    def _build_ui(self):
        container, ctrl = scrollable_panel(self, 300)
        container.pack(side='left', fill='y', padx=8, pady=8)
        self._panel = ctrl
        s = ui_scale(self)

        title = ttk.Frame(ctrl)
        title.pack(anchor='w', fill='x')
        ttk.Label(title, text='Editor', font=bold_font(self, 1)).pack(side='left')
        help_icon(title, 'Right or middle drag: pan.  Mouse wheel or two finger scroll: zoom.\n'
                         'Fit (top right of the image) shows the whole image.\n'
                         f'{MOD} + wheel: pen size.\n'
                         f'{MOD}+Z: undo.  {MOD}+Y or {MOD}+Shift+Z: redo.\n'
                         f'Arrow keys or {MOD} + right drag: nudge the reference image and binary.\n'
                         'Tool shortcuts are listed on each tool button.')

        # --- Files ---
        self._section(ctrl, 'Files', 'Open loads a binary or an image. Images and binaries must match the '
                                     'working grid: a shot image is the binary\'s size at 1x or half of it at '
                                     '2x. Binaries keep their gray mask (value 128). Save always writes a '
                                     'PNG: 0 = fringe, 255 = background, 128 = gray mask.')
        frow = ttk.Frame(ctrl)
        frow.pack(anchor='w', fill='x', pady=2)
        mb_open = ttk.Menubutton(frow, text='Open')
        m_open = tk.Menu(mb_open, tearoff=False)
        m_open.add_command(label='Shot Binary...', command=self.open_binary)
        m_open.add_command(label='Reference Binary...', command=self.open_overlay_binary)
        m_open.add_separator()
        m_open.add_command(label='Shot Image...', command=self._load_shot)
        m_open.add_command(label='Reference Image...', command=self._load_reference)
        m_open.add_separator()
        m_open.add_command(label='Gray Mask...', command=self.open_gray_mask)
        mb_open['menu'] = m_open
        mb_open.pack(side='left')
        mb_save = ttk.Menubutton(frow, text='Save')
        m_save = tk.Menu(mb_save, tearoff=False)
        m_save.add_command(label='Shot Binary...', command=self.save_binary)
        m_save.add_command(label='Reference Binary...', command=self.save_reference_binary)
        m_save.add_command(label='Gray Mask...', command=self.save_gray_mask)
        mb_save['menu'] = m_save
        mb_save.pack(side='left', padx=(6, 0))

        # --- Editing ---
        self._section(ctrl, 'Editing', 'Which binary the tools change: the shot binary (red) or '
                                       'the reference binary (cyan). Mask Pen always edits the gray '
                                       'mask of the shot.')
        self.edit_layer_var = tk.StringVar(value='shot')
        erow = ttk.Frame(ctrl)
        erow.pack(anchor='w', fill='x')
        ttk.Radiobutton(erow, text='Shot binary', value='shot', variable=self.edit_layer_var,
                        command=self._on_edit_layer).pack(side='left')
        ttk.Radiobutton(erow, text='Reference binary', value='ref', variable=self.edit_layer_var,
                        command=self._on_edit_layer).pack(side='left', padx=(8, 0))

        # --- Tools ---
        self._section(ctrl, 'Tools', 'Pick one tool. Hover a tool for what it does and its keyboard '
                                     'shortcut.')
        self.mode_var = tk.StringVar(value='pen')
        tgrid = ttk.Frame(ctrl)
        tgrid.pack(anchor='w', fill='x')
        self._tool_buttons = {}
        style = 'Toolbutton'
        try:
            ttk.Style(self).layout('Toolbutton')
        except tk.TclError:
            style = 'TRadiobutton'
        for i, (key, label, sc, tip) in enumerate(TOOLS):
            b = ttk.Radiobutton(tgrid, text=f'{label} ({sc})', value=key, variable=self.mode_var,
                                style=style, command=lambda k=key: self._set_tool(k))
            b.grid(row=i // 2, column=i % 2, sticky='ew', padx=1, pady=1)
            attach_tooltip(b, tip)
            self._tool_buttons[key] = b
        tgrid.columnconfigure(0, weight=1)
        tgrid.columnconfigure(1, weight=1)

        # --- Tool options (contextual) ---
        self._section(ctrl, 'Tool Options', 'Settings for the selected tool. Only the ones that apply '
                                            'are shown.')
        self._opts = ttk.Frame(ctrl)
        self._opts.pack(anchor='w', fill='x')
        self.brush_var = tk.DoubleVar(value=DEFAULT_SIZES['pen'])
        self._opt_size = ttk.Frame(self._opts)
        self._slider(self._opt_size, 'Size', self.brush_var, 0.5, SIZE_SLIDER_MAX, self._on_size_change,
                     fmt='{:.1f}', resolution=0.1, type_max=SIZE_MAX,
                     help_text='Radius of the pen, eraser or mask pen, in pixels of the binary (at 2x a binary '
                               'pixel is half an image pixel). 0.5 draws a 1 px line. '
                               f'{MOD} + mouse wheel over the image changes it and goes past the end of the '
                               f'slider, up to {SIZE_MAX:.0f}; a typed size can too.')
        self.snap_var = tk.BooleanVar(value=True)
        self._opt_snap = ttk.Frame(self._opts)
        ttk.Checkbutton(self._opt_snap, text='Snap to ends', variable=self.snap_var,
                        command=self._update_snap_marker).pack(side='left')
        help_icon(self._opt_snap, 'Line and Spline clicks jump to the nearest fringe end (marked with a '
                                  'green circle), so a join is exactly connected.')
        self.poly_var = tk.BooleanVar(value=False)
        self._opt_poly = ttk.Frame(self._opts)
        ttk.Checkbutton(self._opt_poly, text='Polyline', variable=self.poly_var,
                        command=self._cancel_shape).pack(side='left')
        help_icon(self._opt_poly, f'Each click adds a corner of a line made of straight pieces. Enter or {MOD}+click '
                                  'finishes it (one undo step); Esc cancels.')
        self._opt_ruler = ttk.Frame(self._opts)
        ttk.Label(self._opt_ruler, text='Lock:').pack(side='left')
        self.ruler_lock_var = tk.StringVar(value='None')
        ttk.Combobox(self._opt_ruler, textvariable=self.ruler_lock_var, values=['None', 'Horizontal', 'Vertical'],
                     width=10, state='readonly').pack(side='left', padx=(4, 0))
        help_icon(self._opt_ruler, 'Keep the ruler horizontal or vertical while you drag.')
        self._opt_hint = ttk.Label(self._opts, text='', wraplength=int(250 * s))

        # --- Fix-ups ---
        self._section(ctrl, 'Fix-ups', 'Automatic repairs on the binary selected under Editing.')
        lrow = ttk.Frame(ctrl)
        lrow.pack(anchor='w', fill='x', pady=1)
        ttk.Button(lrow, text='Link Endpoints', command=self._link_endpoints).pack(side='left')
        ttk.Label(lrow, text='Angle°').pack(side='left', padx=(6, 2))
        self.angle_deg_var = tk.StringVar(value='40')
        ttk.Spinbox(lrow, from_=0, to=90, width=3, textvariable=self.angle_deg_var, validate='key',
                    validatecommand=numeric_validator(lrow)).pack(side='left')
        ttk.Label(lrow, text='Max gap (px)').pack(side='left', padx=(6, 2))
        self.link_tol_var = tk.StringVar(value='10')
        ttk.Spinbox(lrow, from_=1, to=300, width=4, textvariable=self.link_tol_var, validate='key',
                    validatecommand=numeric_validator(lrow)).pack(side='left')
        help_icon(lrow, 'Draws a straight 1 px stroke between pairs of fringe ends that are closer than Max gap '
                        'pixels of the image and whose joining line is within Angle degrees of horizontal: '
                        '0 joins only horizontally, 90 joins at any angle up to vertical. Only ends that '
                        'point at each other are joined, never across to the next fringe. Only the new '
                        'strokes are added; nothing else in the binary changes.')
        mrow = ttk.Frame(ctrl)
        mrow.pack(anchor='w', fill='x', pady=1)
        self.btn_merge_overlay = ttk.Button(mrow, text='Merge Binaries', command=self.merge_overlay, state='disabled')
        self.btn_merge_overlay.pack(side='left')
        help_icon(mrow, 'Turns everything into the shot binary layer: the reference binary (at its nudged '
                        'position) is copied into the shot binary and is no longer a separate layer. '
                        'Asks first. Enabled when a reference binary is loaded.')

        # --- View ---
        self._section(ctrl, 'View', 'What is drawn on screen. None of these change your data or '
                                    'what gets saved.')
        self.show_shot_img = tk.BooleanVar(value=True)
        self.show_ref_img = tk.BooleanVar(value=True)
        self.show_shot_bin = tk.BooleanVar(value=True)
        self.show_ref_bin = tk.BooleanVar(value=True)
        self.shot_opacity_var = tk.DoubleVar(value=1.0)
        self.shot_brightness_var = tk.DoubleVar(value=0.0)
        self.ref_brightness_var = tk.DoubleVar(value=0.0)
        self.shot_binary_opacity_var = tk.DoubleVar(value=1.0)
        self.ref_binary_opacity_var = tk.DoubleVar(value=0.7)
        r = lambda: self._schedule_view()

        def layer_header(text, var, help_text):
            row = ttk.Frame(ctrl)
            row.pack(anchor='w', fill='x', pady=(4, 0))
            ttk.Checkbutton(row, text=text, variable=var, command=r).pack(side='left')
            help_icon(row, help_text)

        layer_header('Shot image', self.show_shot_img, 'The shot interferogram. Lower its opacity to see the '
                                                       'reference image underneath. Brightness is relative to an '
                                                       'automatic contrast stretch done on load.')
        self._slider(ctrl, 'Opacity', self.shot_opacity_var, 0.0, 1.0, r)
        self._slider(ctrl, 'Brightness', self.shot_brightness_var, -2.0, 2.0, r, fmt='{:+.1f}', resolution=0.1)
        layer_header('Reference image', self.show_ref_img, 'The reference interferogram, drawn under the '
                                                           'shot image.')
        self._slider(ctrl, 'Brightness', self.ref_brightness_var, -2.0, 2.0, r, fmt='{:+.1f}', resolution=0.1)
        layer_header('Shot binary', self.show_shot_bin, 'The shot binary in red, with its gray mask in gray. The saved '
                                                         'file is black fringes on white.')
        self._slider(ctrl, 'Opacity', self.shot_binary_opacity_var, 0.0, 1.0, r)
        layer_header('Reference binary', self.show_ref_bin, 'The reference binary in cyan.')
        self._slider(ctrl, 'Opacity', self.ref_binary_opacity_var, 0.0, 1.0, r)

        crow = ttk.Frame(ctrl)
        crow.pack(anchor='w', fill='x', pady=(6, 0))
        self.show_components_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(crow, text='Color Fringes', variable=self.show_components_var, command=r).pack(side='left')
        help_icon(crow, 'Gives every connected fringe its own color, so breaks and merges stand out.')
        mirow = ttk.Frame(ctrl)
        mirow.pack(anchor='w', fill='x', pady=(2, 0))
        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mirow, text='Mirror at x =', variable=self.mirror_var,
                        command=self._on_mirror_change).pack(side='left')
        self.mirror_x_var = tk.StringVar(value='')
        self.mirror_spin = ttk.Spinbox(mirow, from_=0, to=99999, increment=1, width=7, textvariable=self.mirror_x_var,
                                       validate='key', validatecommand=numeric_validator(mirow))
        self.mirror_spin.pack(side='left', padx=(4, 0))
        self.mirror_x_var.trace_add('write', lambda *a: self._on_mirror_change())
        help_icon(mirow, 'View only. Reflects the shot binary about this x (in pixels of the image, as the '
                         'cursor readout shows it; numbers only) and draws it in blue under the red lines. '
                         'Never saved.')
        nrow = ttk.Frame(ctrl)
        nrow.pack(anchor='w', fill='x', pady=(4, 0))
        ttk.Button(nrow, text='Reset Nudge', command=self.reset_nudge).pack(side='left')
        self.nudge_var = tk.StringVar(value='Nudge: 0, 0 px')
        ttk.Label(nrow, textvariable=self.nudge_var).pack(side='left', padx=(8, 0))
        help_icon(nrow, 'The arrow keys move the reference image and reference binary together one pixel of '
                        'the binary at a time (half an image pixel at 2x; hold a key to keep moving), and '
                        f'{MOD} + right or middle drag moves them with the mouse, so you can see where the shot '
                        'and reference line up. The offset is shown in pixels of the image. Merge Binaries uses '
                        'the moved position. A saved reference binary is not moved. Reset Nudge puts them back. '
                        'Every nudge can be undone.')

        # --- footer ---
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
        urow = ttk.Frame(ctrl)
        urow.pack(anchor='w', fill='x')
        ttk.Button(urow, text='Undo', command=self.undo).pack(side='left')
        ttk.Button(urow, text='Redo', command=self.redo).pack(side='left', padx=(6, 0))
        help_icon(urow, f'Undo ({MOD}+Z) and Redo ({MOD}+Y or {MOD}+Shift+Z). Keeps the last {self.MAX_UNDO} '
                        'edits, including opening layers, merging, nudges and grid switches.')
        self.ruler_dist_var = tk.StringVar(value='Dist: 0.00 px')
        ttk.Label(ctrl, textvariable=self.ruler_dist_var, font=fixed_font(self)).pack(anchor='w')
        ttk.Frame(ctrl, height=int(40 * s)).pack(fill='x')

        # --- right: status + canvas ---
        right = ttk.Frame(self)
        right.pack(side='left', fill='both', expand=True, padx=(0, 8), pady=8)
        # messages go to the status line at the bottom of the window (one status line);
        # this label only keeps the latest text and is not shown
        self.status = ttk.Label(right, text='', anchor='w')
        self.canvas = tk.Canvas(right, bg='gray20', highlightthickness=0)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.btn_fit = overlay_button(self.canvas, 'Fit', self.fit_view)
        attach_tooltip(self.btn_fit, 'Show the whole image', side='left')
        self._prompt = CanvasPrompt(self.canvas, 'Open a binary or a shot image to start, or use Send to Editor '
                                                 'on the Detection tab.',
                                    [('Open Shot Binary', self.open_binary), ('Open Shot Image', self._load_shot)])
        self._prompt.show()

    def _bind_events(self):
        c = self.canvas
        c.bind('<Configure>', self._on_canvas_configure, add='+')
        # clicking the image takes the keyboard back from a text box, so shortcuts work again
        for b in (1, 2, 3):
            c.bind(f'<ButtonPress-{b}>', lambda e: c.focus_set())
        c.bind('<ButtonPress-1>', self._on_left_down, add='+')
        c.bind('<B1-Motion>', self._on_left_drag)
        c.bind('<ButtonRelease-1>', self._on_left_up)
        bind_pan(c, self._on_pan_start, self._on_pan_move, self._on_pan_end)
        c.bind('<Motion>', self._on_mouse_move)
        c.bind('<Leave>', self._on_mouse_leave)
        bind_wheel(c, self._on_wheel)
        self.winfo_toplevel().bind('<KeyPress>', self._on_key, add='+')

    def _on_key(self, event):
        """Keyboard shortcuts, only while the Editor is on screen and the user is
        not typing in a box. Undo is {MOD}+Z whatever Caps Lock says."""
        try:
            if not self.winfo_ismapped():
                return
            w = self.focus_get()
            if w is not None and w.winfo_class() in TEXT_WIDGETS:
                return
        except Exception:
            pass
        ks = event.keysym or ''
        if mod_pressed(event):
            k = ks.lower()
            if k == 'z':
                self.redo() if shift_pressed(event) else self.undo()
                return 'break'
            if k == 'y':
                self.redo()
                return 'break'
            return
        if ks in ('Left', 'Right', 'Up', 'Down'):
            dx, dy = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}[ks]
            self.nudge(dx, dy)
            return 'break'
        if ks == 'Escape':
            if self._tool_points:
                self._cancel_shape()
            elif self._check_line is not None:
                self._clear_check()
                self.set_status('Connection check cleared')
            return 'break'
        if ks in ('Return', 'KP_Enter'):
            self._finish_shape()
            return 'break'
        if len(ks) == 1:
            for key, label, sc, tip in TOOLS:
                if ks.upper() == sc:
                    self._set_tool(key)
                    return 'break'

    # ============================================================ helpers ==
    def set_status(self, text):
        try:
            self.status.config(text=text)
        except Exception:
            pass
        try:
            self._status_cb(text)
        except Exception:
            pass

    def current_file_text(self):
        n = self._names
        parts = []
        if n.get('binary'):
            parts.append(n['binary'])
        if n.get('shot'):
            parts.append('on ' + n['shot'])
        if not parts:
            return 'Editor: nothing loaded'
        return 'Editor: ' + ' '.join(parts) + f'  ({self.gw} x {self.gh} px, {self.grid_factor}x)'

    def _report_file(self):
        try:
            self._file_cb(self.current_file_text())
        except Exception:
            pass

    def _binary_dir(self):
        return self._bin_dir if os.path.isdir(self._bin_dir) else APP_DIR

    # ------------------------------------------------------------ settings --
    def get_settings(self):
        sizes = dict(self._sizes)
        try:
            if self.mode_var.get() in SIZED_TOOLS:
                sizes[self.mode_var.get()] = float(self.brush_var.get())
        except Exception:
            pass
        return dict(sizes=sizes, link_angle=self.angle_deg_var.get(), link_tol=self.link_tol_var.get(),
                    snap=bool(self.snap_var.get()), polyline=bool(self.poly_var.get()),
                    plate_dir=self._plate_dir, bin_dir=self._bin_dir)

    def apply_settings(self, d):
        if not isinstance(d, dict):
            return
        for k, v in (d.get('sizes') or {}).items():
            if k in SIZED_TOOLS and isinstance(v, (int, float)) and 0.5 <= v <= SIZE_MAX:
                self._sizes[k] = float(v)
        if self.mode_var.get() in SIZED_TOOLS:
            self.brush_var.set(self._sizes[self.mode_var.get()])
        for key, var, lo, hi in (('link_angle', self.angle_deg_var, 0, 90), ('link_tol', self.link_tol_var, 1, 300)):
            try:
                v = float(d.get(key))
                if lo <= v <= hi:
                    var.set(f'{v:g}')
            except (TypeError, ValueError):
                pass
        for key, var in (('snap', self.snap_var), ('polyline', self.poly_var)):
            if isinstance(d.get(key), bool):
                var.set(d[key])
        for key, attr in (('plate_dir', '_plate_dir'), ('bin_dir', '_bin_dir')):
            v = d.get(key)
            if isinstance(v, str) and os.path.isdir(v):
                setattr(self, attr, v)

    def _layer(self, name):
        return {'mask': self.mask, 'gray': self.gray_mask, 'ovl': self.overlay_mask,
                'shot': self._shot_img, 'ref': self._ref_img, 'check': self._check_px}[name]

    def _set_layer(self, name, arr):
        if name == 'mask':
            self.mask = arr
        elif name == 'gray':
            self.gray_mask = arr
        elif name == 'ovl':
            self.overlay_mask = arr
        elif name == 'shot':
            self._shot_img = arr
        elif name == 'ref':
            self._ref_img = arr

    def _edit_target(self):
        """Layer name of the binary the tools edit, or None."""
        if self.edit_layer_var.get() == 'ref':
            return 'ovl' if self.overlay_mask is not None else None
        return 'mask'

    def _invalidate(self, name=None):
        self._hover = None
        names = [name] if name else ['mask', 'gray', 'ovl', 'shot', 'ref']
        if not name:
            self._pyr.clear()
            self._samples.clear()
        for n in names:
            self._ver[n] = self._ver.get(n, 0) + 1
            self._ends_cache.pop(n, None)
        if 'mask' in names:
            self._labels_dirty = True

    def _touched(self, name, x0, y0, x1, y1):
        """A rectangle of a layer changed in place: patch the zoomed-out copies
        there and drop the caches that depend on the whole layer."""
        self._pyr_patch(name, x0, y0, x1, y1)
        self._ends_cache.pop(name, None)
        self._hover = None
        self._live[name] = self._live.get(name, 0) + 1
        if name == 'mask':
            self._labels_dirty = True

    def _doc_hash(self, which='shot'):
        h = hashlib.blake2b(digest_size=16)
        if which == 'shot':
            arrs = (self.mask, self.gray_mask)
        else:
            arrs = (self.overlay_mask,)
        for a in arrs:
            if a is None:
                h.update(b'none')
            else:
                h.update(str(a.shape).encode())
                h.update(np.ascontiguousarray(a).data)
        return h.hexdigest()

    def has_unsaved_shot(self):
        if self._doc_hash() == self._saved_hash:
            return False
        # a cleared binary is not progress
        return bool((self.mask == 0).any() or (self.gray_mask > 0).any())

    def has_unsaved_ref(self):
        return self.overlay_mask is not None and self._doc_hash('ref') != self._saved_ref_hash

    def unsaved_summary(self):
        out = []
        if self.has_unsaved_shot():
            out.append('Editor: changes to the shot binary' + (f' ({self._names["binary"]})' if self._names.get('binary') else '')
                       + ' are not saved')
        if self.has_unsaved_ref():
            out.append('Editor: changes to the reference binary are not saved')
        return out

    def _confirm_discard(self, action):
        lines = self.unsaved_summary()
        if not lines:
            return True
        return messagebox.askyesno('Unsaved changes', '\n'.join(lines) + f'.\n\n{action} anyway? Those changes will '
                                   'be lost.', icon='warning', parent=self.winfo_toplevel())

    def _has_content(self):
        return bool(self._loaded or (self.mask == 0).any() or (self.gray_mask > 0).any()
                    or self.overlay_mask is not None or self._shot_src is not None or self._ref_src is not None)

    # =================================================== undo / redo ======
    def _state(self):
        return dict(gh=self.gh, gw=self.gw, grid_factor=self.grid_factor, native_hw=self.native_hw,
                    names=dict(self._names), mask=self.mask, gray=self.gray_mask, ovl=self.overlay_mask,
                    shot_src=self._shot_src, ref_src=self._ref_src, shot=self._shot_img, ref=self._ref_img,
                    shot_raw=self._shot_raw, nudge=self._nudge, loaded=self._loaded)

    def _restore_state(self, st):
        old_factor = self.grid_factor
        self.gh, self.gw = st['gh'], st['gw']
        self.grid_factor = st['grid_factor']
        self.native_hw = st['native_hw']
        self._names = dict(st['names'])
        self.mask, self.gray_mask, self.overlay_mask = st['mask'], st['gray'], st['ovl']
        self._shot_src, self._ref_src = st['shot_src'], st['ref_src']
        self._shot_img, self._ref_img = st['shot'], st['ref']
        self._shot_raw = st.get('shot_raw')
        self._nudge = st.get('nudge', (0, 0))
        self._loaded = st['loaded']
        self._show_nudge()
        self._invalidate()
        if self.grid_factor != old_factor:
            self._grid_cb(self.grid_factor)

    @staticmethod
    def _entry_arrays(items):
        for it in items:
            if it[0] == 'patch':
                yield it[4]; yield it[5]
            elif it[0] == 'layer':
                for a in (it[2], it[3]):
                    if a is not None:
                        yield a
            elif it[0] == 'state':
                for st in (it[1], it[2]):
                    for k in ('mask', 'gray', 'ovl', 'shot', 'ref', 'shot_src', 'ref_src', 'shot_raw'):
                        if st.get(k) is not None:
                            yield st[k]

    def _history_bytes(self):
        """Memory held by undo and redo; layers shared between entries count once."""
        seen = {}
        for _l, items in self._undo + self._redo:
            for a in self._entry_arrays(items):
                seen[id(a)] = a.nbytes
        return sum(seen.values())

    def _push_undo(self, label, items):
        self._undo.append((label, items))
        self._redo.clear()
        while len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        while len(self._undo) > 1 and self._history_bytes() > self.MAX_UNDO_BYTES:
            self._undo.pop(0)

    def _begin(self, names):
        names = [n for n in names if self._layer(n) is not None]
        self._snap_before = {n: self._layer(n).copy() for n in names}

    def _commit(self, label='edit', rect=None, extra=None):
        """Store what changed since _begin as an undo entry. rect (x0, y0, x1, y1)
        limits the comparison to where a stroke went, which keeps it fast."""
        snap = self._snap_before
        self._snap_before = None
        items = []
        for name, before in (snap or {}).items():
            after = self._layer(name)
            if after is None or after.shape != before.shape:
                continue
            H, W = before.shape[:2]
            if rect is not None:
                x0, y0, x1, y1 = rect
                x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
                x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
            else:
                x0, y0, x1, y1 = 0, 0, W, H
            diff = before[y0:y1, x0:x1] != after[y0:y1, x0:x1]
            rows = np.flatnonzero(diff.any(axis=1))
            if rows.size == 0:
                continue
            cols = np.flatnonzero(diff[rows[0]:rows[-1] + 1].any(axis=0))
            ya, yb, xa, xb = y0 + rows[0], y0 + rows[-1] + 1, x0 + cols[0], x0 + cols[-1] + 1
            items.append(('patch', name, ya, xa, before[ya:yb, xa:xb].copy(), after[ya:yb, xa:xb].copy()))
            self._touched(name, xa, ya, xb, yb)
        items += list(extra or [])
        if not items:
            return False
        self._push_undo(label, items)
        return True

    def _apply_items(self, items, use_before):
        seq = list(reversed(items)) if use_before else list(items)
        for it in seq:
            if it[0] == 'patch':
                _k, name, y0, x0, before, after = it
                arr = self._layer(name)
                if arr is None:
                    continue
                patch = before if use_before else after
                h, w = patch.shape[:2]
                arr[y0:y0 + h, x0:x0 + w] = patch
                self._touched(name, x0, y0, x0 + w, y0 + h)
            elif it[0] == 'layer':
                _k, name, before, after = it
                self._set_layer(name, before if use_before else after)
                self._invalidate(name)
            elif it[0] == 'state':
                self._restore_state(it[1] if use_before else it[2])
            elif it[0] == 'nudge':
                self._set_nudge(it[1] if use_before else it[2])

    def _after_history(self):
        try:
            self.btn_merge_overlay.config(state='normal' if self.overlay_mask is not None else 'disabled')
        except Exception:
            pass
        if self._has_content():
            self._prompt.hide()
        else:
            self._prompt.show()
        if self.overlay_mask is None and self.edit_layer_var.get() == 'ref':
            self.edit_layer_var.set('shot')
        self._ruler_points = []
        self._report_file()
        self._schedule_view()

    def undo(self):
        if self._painting:
            return
        self._nudge_t = 0.0                 # the next nudge starts a new undo step
        if not self._undo:
            self.set_status('Nothing to undo')
            return
        label, items = self._undo.pop()
        self._apply_items(items, use_before=True)
        self._redo.append((label, items))
        self.set_status(f'Undid: {label}')
        self._after_history()

    def redo(self):
        if self._painting:
            return
        if not self._redo:
            self.set_status('Nothing to redo')
            return
        label, items = self._redo.pop()
        self._apply_items(items, use_before=False)
        self._undo.append((label, items))
        self.set_status(f'Redid: {label}')
        self._after_history()

    def _clear_history(self):
        self._undo.clear()
        self._redo.clear()
        self._snap_before = None

    # ============================================================ nudge ==
    def _ref_offset(self, name):
        """Where a layer is drawn relative to the working grid: the reference image and
        reference binary sit at the nudge offset, everything else at (0, 0)."""
        return self._nudge if name in ('ovl', 'ref') else (0, 0)

    def _nudge_text(self, d=None):
        """An offset (working-grid pixels) as the user reads it: '+0.5, -1.5 px' in
        pixels of the image."""
        dx, dy = self._nudge if d is None else d
        f = max(1, self.grid_factor)
        sx = lambda v: ('+' if v > 0 else '') + image_px(v, f)
        return f'{sx(dx)}, {sx(dy)} px'

    def _show_nudge(self):
        try:
            self.nudge_var.set('Nudge: ' + self._nudge_text())
        except Exception:
            pass

    def _set_nudge(self, d):
        self._nudge = (int(d[0]), int(d[1]))
        self._show_nudge()
        self._samples.clear()
        self._set_hover(None)
        self._schedule_view()

    def _has_reference(self):
        return self.overlay_mask is not None or self._ref_img is not None

    def nudge(self, dx, dy, group=True):
        """Move the reference image and binary by (dx, dy) working-grid pixels. Presses
        that follow each other quickly (a held arrow key) are one undo step."""
        if self._painting:
            return
        if not self._has_reference():
            self.set_status('Nothing to nudge: open a reference image or reference binary first')
            return
        before = self._nudge
        after = (before[0] + int(dx), before[1] + int(dy))
        now = time.monotonic()
        last = self._undo[-1] if self._undo else None
        if (group and last is not None and last[0] == 'Nudge' and not self._redo
                and now - self._nudge_t < NUDGE_GROUP_S and len(last[1]) == 1 and last[1][0][0] == 'nudge'):
            self._undo[-1] = ('Nudge', [('nudge', last[1][0][1], after)])
        else:
            self._push_undo('Nudge', [('nudge', before, after)])
        self._nudge_t = now
        self._set_nudge(after)
        self.set_status('Reference image and binary nudged to ' + self._nudge_text(after))

    def reset_nudge(self):
        if self._nudge == (0, 0):
            self.set_status('Nudge is already 0, 0')
            return
        self._push_undo('Reset nudge', [('nudge', self._nudge, (0, 0))])
        self._nudge_t = 0.0
        self._set_nudge((0, 0))
        self.set_status(f'Reference image and binary back where they were loaded. {MOD}+Z undoes this.')

    # ======================================================== grid size ==
    def set_grid_factor(self, factor):
        """Switch the working grid between the shot image's own size (1x) and
        twice that (2x). Called when the user flips the switch at the top.
        Returns False if the user cancels (nothing changes then)."""
        factor = max(1, int(factor))
        if factor == self.grid_factor:
            return True
        old = self._state()
        up = factor > self.grid_factor
        if not self._has_content():
            # nothing loaded: just resize the empty layers
            h, w = self.native_hw if self.native_hw else (2048, 2048)
            self.grid_factor = factor
            self.gh, self.gw = h * factor, w * factor
            self.mask = np.full((self.gh, self.gw), 255, np.uint8)
            self.gray_mask = np.zeros((self.gh, self.gw), np.uint8)
            self._saved_hash = self._doc_hash()
            self._invalidate()
            self.fit_view()
            self.set_status(f'Working grid: {factor}x')
            self._report_file()
            return True
        if not up and (self.gh % 2 or self.gw % 2):
            messagebox.showerror('Switch to 1x', f'This binary is {self.gw} x {self.gh} px, which cannot be halved. '
                                 'It was made at 1x.', parent=self.winfo_toplevel())
            return False
        ink_mask = (self.mask == 0).any()
        ink_ovl = self.overlay_mask is not None and (self.overlay_mask == 0).any()
        if up:
            new_mask = np.where(upsample_lines_2x(self.mask == 0), 0, 255).astype(np.uint8)
            new_gray = np.repeat(np.repeat(self.gray_mask, 2, axis=0), 2, axis=1)
            new_ovl = None if self.overlay_mask is None else \
                np.where(upsample_lines_2x(self.overlay_mask == 0), 0, 255).astype(np.uint8)
            gh, gw = self.gh * 2, self.gw * 2
            msg = (f'Working grid 2x ({gw} x {gh}). The binaries were scaled up along their lines, so they '
                   'gain no new detail; Send to Editor again from Detection for a true 2x trace. '
                   f'{MOD}+Z undoes the switch.')
        else:
            what = []
            if ink_mask:
                what.append('The shot binary will be cleared.')
            if ink_ovl:
                what.append('The reference binary will be cleared.')
            if what:
                what.append('(Shrinking 1 px fringe lines to half size merges fringes that are only 1 or 2 px '
                            'apart, so they cannot be kept.)')
            if (self.gray_mask > 0).any():
                what.append('The gray mask is kept and scaled down.')
            if self._shot_src is not None or self._ref_src is not None:
                what.append('The shot and reference images are kept and scaled down.')
            what.append('The Detection tab re-detects at 1x.')
            what.append(f'You can undo this with {MOD}+Z.')
            if not messagebox.askokcancel('Switch to 1x grid?',
                                          f'Switch the working grid from 2x ({self.gw} x {self.gh}) to 1x '
                                          f'({self.gw // 2} x {self.gh // 2})?\n\n' + '\n'.join(what),
                                          icon='warning', parent=self.winfo_toplevel()):
                return False
            gh, gw = self.gh // 2, self.gw // 2
            new_mask = np.full((gh, gw), 255, np.uint8)
            new_gray = np.where(reduce_any_2x(self.gray_mask > 0), 255, 0).astype(np.uint8)
            new_ovl = None if self.overlay_mask is None else np.full((gh, gw), 255, np.uint8)
            msg = (f'Working grid 1x ({gw} x {gh}).' + (' Binaries cleared.' if (ink_mask or ink_ovl) else '')
                   + f' {MOD}+Z undoes the switch.')
        self.grid_factor = factor
        self.gh, self.gw = gh, gw
        self.mask, self.gray_mask, self.overlay_mask = new_mask, new_gray, new_ovl
        self._shot_img = self._plate_for(self._shot_src, old['shot'], up)
        self._ref_img = self._plate_for(self._ref_src, old['ref'], up)
        nx, ny = self._nudge
        self._nudge = (nx * 2, ny * 2) if up else (int(round(nx / 2.0)), int(round(ny / 2.0)))
        self._show_nudge()
        if self.native_hw is None:
            self.native_hw = (gh // factor, gw // factor)
        new = self._state()
        self._push_undo(f'Switch to {factor}x grid', [('state', old, new)])
        self._invalidate()
        self._ruler_points = []
        self._zoom = self._zoom * (old['gw'] / float(self.gw))
        self._ox *= self.gw / float(old['gw'])
        self._oy *= self.gh / float(old['gh'])
        self._snap_view()
        self._schedule_view()
        self.set_status(msg)
        self._report_file()
        return True

    def _plate_for(self, src, cur, up):
        if src is not None:
            return plate_on_grid(src, self.grid_factor)
        if cur is None:
            return None
        return cv2.resize(cur, (self.gw, self.gh), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)

    def _set_factor_quietly(self, factor):
        """The loaded files say which grid this is: follow them and move the switch."""
        if factor != self.grid_factor:
            self.grid_factor = factor
            self._grid_cb(factor)

    # ======================================================= data in/out ==
    def receive_from_detection(self, binary, shot_img, saved_path=None, name='', factor=1):
        """Called by the Detection tab's Send to Editor button."""
        if not self._confirm_discard('Replace them with the new detection'):
            return False
        b = np.asarray(binary)
        if b.dtype != np.uint8:
            b = b.astype(np.uint8)
        gh, gw = b.shape[:2]
        sh, sw = shot_img.shape[:2]
        k = max(1, int(round(gh / float(sh))))
        dropped = self._drop_mismatched((gh, gw), (sh, sw))
        self.gh, self.gw = gh, gw
        self.native_hw = (sh, sw)
        self._set_factor_quietly(k)
        self.mask, self.gray_mask = classify_binary(b)
        self._shot_raw = sanitize_image(shot_img)[0]
        self._shot_src = to_display(self._shot_raw)
        self._shot_img = plate_on_grid(self._shot_src, k)
        self._names['binary'] = os.path.basename(saved_path) if saved_path else 'detection'
        self._names['shot'] = name or None
        self._after_new_document()
        self.set_status('Received {} from Detection{}{}'.format(
            name or 'binary', f' (saved as {os.path.basename(saved_path)})' if saved_path else '',
            '. ' + dropped if dropped else ''))
        return True

    def _drop_mismatched(self, grid_hw, native_hw):
        """Reference layers that do not fit a new grid are removed. Returns a note."""
        notes = []
        if self.overlay_mask is not None and self.overlay_mask.shape[:2] != tuple(grid_hw):
            self.overlay_mask = None
            self._names['ovl'] = None
            notes.append('reference binary')
        if self._ref_src is not None and native_hw is not None and self._ref_src.shape[:2] != tuple(native_hw):
            self._ref_src = None
            self._ref_img = None
            self._names['ref'] = None
            notes.append('reference image')
        if self.overlay_mask is None and self._ref_src is None and self._nudge != (0, 0):
            self._nudge = (0, 0)
            self._show_nudge()
        return ('Removed the ' + ' and '.join(notes) + ' (different size)') if notes else ''

    def _after_new_document(self):
        self._loaded = True
        self._prompt.hide()
        if self._ref_src is not None:
            self._ref_img = plate_on_grid(self._ref_src, self.grid_factor)
        self._clear_history()
        self._clear_check()
        self._invalidate()
        self._saved_hash = self._doc_hash()
        self._saved_ref_hash = self._doc_hash('ref') if self.overlay_mask is not None else None
        self._ruler_points = []
        self._tool_points = []
        self._clear_preview()
        self.after(30, self.fit_view)
        self._after_history()

    def get_mask(self):
        result = self.mask.copy()
        result[self.gray_mask > 0] = 128
        return result

    @staticmethod
    def _read_binary(path, fringes=True):
        """(image, note): a binary file as uint8 with SpellBook's values (0 = fringe,
        255 = background, 128 = gray mask), or (None, '') if it cannot be read.
        0/1 files and other bit depths are scaled to 0 to 255. With fringes=True a
        binary of white fringes on black is inverted, and the note says so."""
        img = imread_any(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, ''
        if img.ndim == 3:
            if img.dtype not in (np.uint8, np.uint16, np.float32):
                img = img.astype(np.float32)
            img = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 \
                else img[..., 0]
        if img.dtype != np.uint8:
            img = np.rint(cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
        elif img.max() <= 1:
            img = (img * 255).astype(np.uint8)        # 0/1 binaries
        note = ''
        if fringes and np.count_nonzero(img < 128) > np.count_nonzero(img > 128):
            # fringe lines cover a small part of a binary, so one that is mostly
            # "fringe" has white fringes on black
            img = np.where(img == 128, 128, 255 - img).astype(np.uint8)
            note = 'It had white fringes on black and was inverted'
        return img, note

    def _ask_open(self, title, binary=True):
        p = filedialog.askopenfilename(parent=self.winfo_toplevel(), title=title,
                                       initialdir=self._binary_dir() if binary else self._plate_dir,
                                       filetypes=image_filetypes('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'))
        if p and binary:
            self._bin_dir = os.path.dirname(p)
        return p

    def _factor_for(self, size, native):
        """k in (1, 2) with size == native * k, else None."""
        for k in (1, 2):
            if (native[0] * k, native[1] * k) == tuple(size):
                return k
        return None

    def open_binary(self):
        path = self._ask_open('Open shot binary')
        if not path:
            return
        img, read_note = self._read_binary(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image', parent=self.winfo_toplevel())
            return
        if not self._confirm_discard('Open another binary'):
            return
        bh, bw = img.shape[:2]
        note = ''
        matched = False
        if self.native_hw is not None and (self._shot_src is not None or self._ref_src is not None):
            k = self._factor_for((bh, bw), self.native_hw)
            if k is None:
                H, W = self.native_hw
                if not messagebox.askyesno('Size does not match',
                                           f'This binary is {bw} x {bh} px, but the shot image is {W} x {H} px '
                                           f'(a binary for it is {W} x {H} at 1x or {2 * W} x {2 * H} at 2x).\n\n'
                                           'Open the binary on its own, without the shot and reference images?',
                                           parent=self.winfo_toplevel()):
                    return
                self._shot_src = self._shot_img = self._ref_src = self._ref_img = self._shot_raw = None
                self._names['shot'] = self._names['ref'] = None
                self._nudge = (0, 0)
                self._show_nudge()
                self.native_hw = None
                note = ' (shot image removed: different size)'
            else:
                matched = True
                self._set_factor_quietly(k)
                self._shot_img = plate_on_grid(self._shot_src, k)
        if not matched:
            k = self.grid_factor
            if k == 2 and (bh % 2 or bw % 2):
                k = 1
            self._set_factor_quietly(k)
            self.native_hw = (bh // k, bw // k)
        self.gh, self.gw = bh, bw
        dropped = self._drop_mismatched((bh, bw), self.native_hw)
        self.mask, self.gray_mask = classify_binary(img)
        self._names['binary'] = os.path.basename(path)
        self._after_new_document()
        gray_px = int(np.sum(self.gray_mask > 0))
        msg = f'Loaded {os.path.basename(path)} ({bw} x {bh} px, {self.grid_factor}x grid){note}'
        if gray_px:
            msg += f' with {gray_px} gray mask pixels'
        for extra in (read_note, dropped):
            if extra:
                msg += '. ' + extra
        self.set_status(msg)

    def open_overlay_binary(self):
        path = self._ask_open('Open reference binary')
        if not path:
            return
        img, read_note = self._read_binary(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image', parent=self.winfo_toplevel())
            return
        ov = classify_binary(img)[0]          # a gray mask in a reference binary is ignored
        oh, ow = ov.shape[:2]
        if (oh, ow) == (self.gh, self.gw):
            pass
        elif (oh * 2, ow * 2) == (self.gh, self.gw):
            ov = np.where(upsample_lines_2x(ov == 0), 0, 255).astype(np.uint8)
        else:
            extra = ' It looks like a 2x binary: switch the working grid to 2x first.' \
                if (oh, ow) == (self.gh * 2, self.gw * 2) else ''
            messagebox.showerror('Size does not match',
                                 f'The reference binary is {ow} x {oh} px but the working grid is '
                                 f'{self.gw} x {self.gh} px.{extra}', parent=self.winfo_toplevel())
            return
        old = self._state()
        self.overlay_mask = ov
        self._names['ovl'] = os.path.basename(path)
        self._invalidate('ovl')
        self._push_undo('Open reference binary', [('state', old, self._state())])
        self._saved_ref_hash = self._doc_hash('ref')
        self.set_status('Reference binary loaded. ' + (read_note + '. ' if read_note else '')
                        + 'Edit it by choosing Reference binary under Editing, or Merge Binaries to copy it into the '
                        'shot binary.')
        self._after_history()

    def _load_plate(self, which):
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(),
                                          title='Open shot image' if which == 'shot' else 'Open reference image',
                                          initialdir=self._plate_dir,
                                          filetypes=image_filetypes('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))
        if not path:
            return
        self._plate_dir = os.path.dirname(path)
        img = imread_any(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image', parent=self.winfo_toplevel())
            return
        nh, nw = img.shape[:2]
        old = self._state()
        blank = not ((self.mask == 0).any() or (self.gray_mask > 0).any() or self.overlay_mask is not None
                     or (self._shot_src is not None and which == 'ref') or (self._ref_src is not None and which == 'shot'))
        other = self._ref_src if which == 'shot' else self._shot_src
        if other is not None and (nh, nw) != other.shape[:2]:
            oh, ow = other.shape[:2]
            messagebox.showerror('Size does not match',
                                 f'The image is {nw} x {nh} px but the {"reference" if which == "shot" else "shot"} '
                                 f'image is {ow} x {oh} px. Both images must come from the same camera.',
                                 parent=self.winfo_toplevel())
            return
        k = None if blank else self._factor_for((self.gh, self.gw), (nh, nw))
        if k is None:
            if not blank:
                messagebox.showerror('Size does not match',
                                     f'The image is {nw} x {nh} px but the binary is {self.gw} x {self.gh} px. An '
                                     f'image for this binary is {self.gw} x {self.gh} (1x) or '
                                     f'{self.gw // 2} x {self.gh // 2} (2x).', parent=self.winfo_toplevel())
                return
            # empty editor: the image sets the grid
            k = self.grid_factor
            self.gh, self.gw = nh * k, nw * k
            self.mask = np.full((self.gh, self.gw), 255, np.uint8)
            self.gray_mask = np.zeros((self.gh, self.gw), np.uint8)
            self._saved_hash = self._doc_hash()
        self._set_factor_quietly(k)
        self.native_hw = (nh, nw)
        raw = sanitize_image(img)[0]
        src = to_display(raw)
        if which == 'shot':
            self._shot_src = src
            self._shot_raw = raw
            self._shot_img = plate_on_grid(src, k)
            self._names['shot'] = os.path.basename(path)
        else:
            self._ref_src = src
            self._ref_img = plate_on_grid(src, k)
            self._names['ref'] = os.path.basename(path)
        self._loaded = True
        self._invalidate()
        self._push_undo('Open shot image' if which == 'shot' else 'Open reference image', [('state', old, self._state())])
        self.set_status(f'Loaded {"shot" if which == "shot" else "reference"} image: {os.path.basename(path)} '
                        f'({nw} x {nh} px, shown at {k}x)')
        self._prompt.hide()
        self.after(30, self.fit_view)          # show the whole image that was just loaded
        self._after_history()

    def _load_shot(self):
        self._load_plate('shot')

    def _load_reference(self):
        self._load_plate('ref')

    def open_gray_mask(self):
        path = self._ask_open('Open gray mask')
        if not path:
            return
        img, _note = self._read_binary(path, fringes=False)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image', parent=self.winfo_toplevel())
            return
        g = gray_mask_from_file(img)
        h, w = g.shape[:2]
        if (h, w) == (self.gh, self.gw):
            pass
        elif (h * 2, w * 2) == (self.gh, self.gw):
            g = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
        elif (h, w) == (self.gh * 2, self.gw * 2):
            g = reduce_any_2x(g)
        else:
            messagebox.showerror('Size does not match',
                                 f'The gray mask is {w} x {h} px but the working grid is {self.gw} x {self.gh} px '
                                 '(a gray mask must be the same size, half or twice the size).',
                                 parent=self.winfo_toplevel())
            return
        self._begin(['mask'])
        new_gray = np.where(g, 255, 0).astype(np.uint8)
        self.mask[new_gray > 0] = 255          # masked pixels hold no fringes
        self._invalidate('mask')
        extra = [('layer', 'gray', self.gray_mask, new_gray)]
        self.gray_mask = new_gray
        self._invalidate('gray')
        self._commit('Open gray mask', extra=extra)
        self.set_status(f'Loaded gray mask: {os.path.basename(path)}')
        self._after_history()

    def _ask_save(self, title, name):
        try:
            os.makedirs(EDITED_DIR, exist_ok=True)
        except Exception:
            pass
        p = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title=title, initialdir=self._binary_dir(),
                                         initialfile=name, defaultextension='.png', filetypes=[('PNG', '*.png')])
        if p:
            self._bin_dir = os.path.dirname(p)
        return p

    def _default_name(self, suffix=''):
        base = self._names.get('binary') or self._names.get('shot') or 'binary'
        stem = os.path.splitext(base)[0]
        return f'{stem}{suffix}.png'

    def _write(self, path, arr, what):
        try:
            written = imwrite_png(path, arr)
        except Exception as e:
            messagebox.showerror('Save error', str(e), parent=self.winfo_toplevel())
            return None
        note = '' if written == path else ' (binaries are always saved as PNG)'
        self.set_status(f'Saved {what}: {os.path.basename(written)}{note}')
        return written

    def save_binary(self):
        path = self._ask_save('Save shot binary', self._default_name())
        if not path:
            return
        written = self._write(path, self.get_mask(), 'shot binary')
        if written:
            self._saved_hash = self._doc_hash()
            self._names['binary'] = os.path.basename(written)
            self._report_file()

    def save_reference_binary(self):
        if self.overlay_mask is None:
            messagebox.showinfo('Nothing to save', 'No reference binary loaded', parent=self.winfo_toplevel())
            return
        path = self._ask_save('Save reference binary', self._names.get('ovl') or 'reference_binary.png')
        if not path:
            return
        if self._write(path, self.overlay_mask, 'reference binary'):
            self._saved_ref_hash = self._doc_hash('ref')

    def save_gray_mask(self):
        path = self._ask_save('Save gray mask', self._default_name('_graymask'))
        if not path:
            return
        # the same convention as a shot binary: 128 = gray mask, 255 = not masked
        self._write(path, np.where(self.gray_mask > 0, 128, 255).astype(np.uint8), 'gray mask')

    # ============================================================ tools ==
    def _set_tool(self, mode):
        if mode not in dict((t[0], t) for t in TOOLS):
            return
        if self._painting:
            return
        self._clear_preview()
        self._tool_points = []
        if mode != 'ruler':
            self._ruler_points = []
            self._redraw_ruler()
        old = self.mode_var.get()
        if old != mode and 'check' in (old, mode):
            self._schedule_view()          # Connection Check draws the binary in black
        if mode != 'delete':
            self._set_hover(None)
        if old in SIZED_TOOLS:
            try:
                self._sizes[old] = float(self.brush_var.get())
            except Exception:
                pass
        self.mode_var.set(mode)
        if mode in SIZED_TOOLS:
            self.brush_var.set(self._sizes.get(mode, DEFAULT_SIZES[mode]))
        for w in (self._opt_size, self._opt_snap, self._opt_poly, self._opt_ruler, self._opt_hint):
            w.pack_forget()
        hints = {
            'mask': 'Hold Shift while painting to remove mask. Masking removes the fringes under it.',
            'spline': f'{MOD}+click adds the last point and finishes. Enter also finishes; Esc cancels.',
            'delete': 'Click a fringe line to delete that whole piece (highlighted in yellow). Zoom in to pick '
                      'a line exactly.',
            'check': 'Drag a line across the fringes. The ones it touches turn orange; fringes that stay black '
                     'are not connected to them. The orange stays while you use other tools; Esc clears it.',
        }
        if mode in SIZED_TOOLS:
            self._opt_size.pack(anchor='w', fill='x')
        if mode == 'line':
            self._opt_poly.pack(anchor='w', fill='x')
        if mode in ('line', 'spline'):
            self._opt_snap.pack(anchor='w', fill='x')
        if mode == 'ruler':
            self._opt_ruler.pack(anchor='w', fill='x')
        if hints.get(mode):
            self._opt_hint.config(text=hints[mode])
            self._opt_hint.pack(anchor='w', fill='x', pady=(2, 0))
        label = [t[1] for t in TOOLS if t[0] == mode][0]
        self.set_status(f'Tool: {label}')
        self._update_cursor()
        self._update_snap_marker()

    def _on_edit_layer(self):
        if self.edit_layer_var.get() == 'ref' and self.overlay_mask is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
        else:
            self.set_status('Now editing the ' + ('reference binary (cyan)' if self.edit_layer_var.get() == 'ref'
                                                  else 'shot binary (red)'))
        self._update_cursor()
        self._update_snap_marker()

    def _on_size_change(self):
        if self.mode_var.get() in SIZED_TOOLS:
            self._sizes[self.mode_var.get()] = float(self.brush_var.get())
        self._update_cursor()

    def _on_mirror_change(self):
        if self.mirror_var.get():
            try:
                float(self.mirror_x_var.get().strip())
            except ValueError:
                self.set_status('Mirror: type the x position (image pixels) in the box')
        self._schedule_view()

    # ----------------------------------------------------- coordinates ---
    def _to_img(self, cx, cy):
        """Canvas pixel -> working-grid coordinates of the pixel's center."""
        return self._ox + (cx + 0.5) / self._zoom, self._oy + (cy + 0.5) / self._zoom

    def _to_canvas(self, ix, iy):
        return (ix - self._ox) * self._zoom, (iy - self._oy) * self._zoom

    # ------------------------------------------------------- painting ---
    @staticmethod
    def _paint_disc(arr, xi, yi, r, value):
        """Set pixels whose center lies within r of (xi, yi). Always hits the pixel under (xi, yi)."""
        h, w = arr.shape[:2]
        x0 = int(max(0, math.floor(xi - r))); x1 = int(min(w - 1, math.ceil(xi + r)))
        y0 = int(max(0, math.floor(yi - r))); y1 = int(min(h - 1, math.ceil(yi + r)))
        if x1 < x0 or y1 < y0:
            return
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        disc = (xx + 0.5 - xi) ** 2 + (yy + 0.5 - yi) ** 2 <= r * r
        px, py = int(math.floor(xi)), int(math.floor(yi))
        if x0 <= px <= x1 and y0 <= py <= y1:
            disc[py - y0, px - x0] = True
        arr[y0:y1 + 1, x0:x1 + 1][disc] = value

    def _paint_at(self, x, y, r):
        """Paint at working-grid point (x, y) on the stroke's layer (at its nudge offset)."""
        name, value = self._paint_layer
        ox, oy = self._ref_offset(name)
        self._paint_disc(self._layer(name), x - ox, y - oy, r, value)
        if name == 'gray' and value == 255:
            self._paint_disc(self.mask, x, y, r, 255)      # masking removes the fringe underneath

    def _paint_segment(self, a, b, r):
        (x0, y0), (x1, y1) = a, b
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(0.25, r * 0.3)
        n = max(1, int(math.ceil(dist / step)))
        for i in range(1, n + 1):
            t = i / float(n)
            self._paint_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r)

    def _grow_rect(self, x0, y0, x1, y1):
        r = self._paint_rect
        self._paint_rect = (x0, y0, x1, y1) if r is None else (min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1))

    def _stroke_target(self, event):
        mode = self.mode_var.get()
        if mode == 'mask':
            return 'gray', (0 if shift_pressed(event) else 255)
        tgt = self._edit_target()
        if tgt is None:
            return None
        return tgt, (0 if mode == 'pen' else 255)

    def _redraw_canvas_rect(self, ia, ib, r):
        """Recomposite only the screen rectangle around an image-space segment."""
        (xa, ya), (xb, yb) = ia, ib
        ca = self._to_canvas(min(xa, xb) - r - 2, min(ya, yb) - r - 2)
        cb = self._to_canvas(max(xa, xb) + r + 2, max(ya, yb) + r + 2)
        self._render_rect(int(math.floor(ca[0])), int(math.floor(ca[1])), int(math.ceil(cb[0])) + 1,
                          int(math.ceil(cb[1])) + 1)

    def _touch(self, names, x0, y0, x1, y1):
        for n in names:
            self._touched(n, x0, y0, x1, y1)

    # ============================================================ mouse ==
    def _on_left_down(self, event):
        mode = self.mode_var.get()
        ix, iy = self._to_img(event.x, event.y)
        if mode in SIZED_TOOLS:
            tgt = self._stroke_target(event)
            if tgt is None:
                self.set_status('No reference binary loaded: Open > Reference Binary first')
                return
            name, value = tgt
            self._begin([name, 'mask'] if name == 'gray' else [name])
            self._paint_layer = (name, value)
            self._paint_rect = None
            r = max(0.5, float(self.brush_var.get()))
            self._paint_at(ix, iy, r)
            ox, oy = self._ref_offset(name)
            rect = (ix - ox - r - 1, iy - oy - r - 1, ix - ox + r + 2, iy - oy + r + 2)
            self._grow_rect(*rect)
            self._touch([name, 'mask'] if name == 'gray' else [name], *rect)
            self._painting = True
            self._last_paint_xy = (ix, iy)
            self._redraw_canvas_rect((ix, iy), (ix, iy), r)
        elif mode == 'delete':
            self._delete_fringe_at(ix, iy)
        elif mode == 'ruler':
            self._ruler_down(ix, iy)
        elif mode == 'line' and self.poly_var.get():
            # polyline: every click adds a corner; Ctrl/Cmd + click adds the last one and finishes
            self._tool_points.append(self._snap((ix, iy)))
            if mod_pressed(event) and len(self._tool_points) >= 2:
                self._finish_shape()
            else:
                self._update_preview(event.x, event.y)
        elif mode in ('line', 'check'):
            p = self._snap((ix, iy)) if mode == 'line' else (ix, iy)
            if not self._tool_points:
                self._tool_points = [p]
                self._drag_start = (event.x, event.y)
            else:
                self._tool_points.append(p)
                self._commit_shape()
        elif mode == 'spline':
            self._tool_points.append(self._snap((ix, iy)))
            if mod_pressed(event) and len(self._tool_points) >= 2:
                self._finish_shape()
            else:
                self._update_preview(event.x, event.y)
        self._last_mouse = (event.x, event.y)

    def _on_left_drag(self, event):
        mode = self.mode_var.get()
        self._last_mouse = (event.x, event.y)
        ix, iy = self._to_img(event.x, event.y)
        self._update_coords(ix, iy)
        if mode in SIZED_TOOLS and self._painting:
            name, _value = self._paint_layer
            r = max(0.5, float(self.brush_var.get()))
            self._paint_segment(self._last_paint_xy, (ix, iy), r)
            (ax, ay), (bx, by) = self._last_paint_xy, (ix, iy)
            ox, oy = self._ref_offset(name)
            rect = (min(ax, bx) - ox - r - 1, min(ay, by) - oy - r - 1, max(ax, bx) - ox + r + 2, max(ay, by) - oy + r + 2)
            self._grow_rect(*rect)
            self._touch([name, 'mask'] if name == 'gray' else [name], *rect)
            self._redraw_canvas_rect(self._last_paint_xy, (ix, iy), r)
            self._last_paint_xy = (ix, iy)
            self._update_cursor()
        elif mode == 'ruler':
            self._ruler_drag(ix, iy)
        elif mode in ('line', 'check', 'spline') and self._tool_points:
            self._update_preview(event.x, event.y)

    def _on_left_up(self, event):
        mode = self.mode_var.get()
        if mode in SIZED_TOOLS and self._painting:
            self._painting = False
            self._last_paint_xy = None
            name = self._paint_layer[0]
            self._paint_layer = None
            if name == 'mask':
                self._enforce_mask(self._paint_rect)
            label = {'pen': 'Pen stroke', 'erase': 'Eraser stroke', 'mask': 'Mask stroke'}[mode]
            self._commit(label, rect=self._paint_rect)
            self._paint_rect = None
            self._schedule_view()
            return
        if mode == 'ruler':
            self._ruler_up(*self._to_img(event.x, event.y))
            return
        if mode in ('line', 'check') and len(self._tool_points) == 1 and self._drag_start is not None \
                and not (mode == 'line' and self.poly_var.get()):
            dx = event.x - self._drag_start[0]; dy = event.y - self._drag_start[1]
            if math.hypot(dx, dy) > 3:                 # click-and-drag finishes the shape
                ix, iy = self._to_img(event.x, event.y)
                self._tool_points.append(self._snap((ix, iy)) if mode == 'line' else (ix, iy))
                self._commit_shape()

    def _enforce_mask(self, rect=None):
        """Masked pixels hold no fringes (the saved file stores 128 there)."""
        if rect is None:
            self.mask[self.gray_mask > 0] = 255
            return
        x0, y0, x1, y1 = rect
        x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        x1, y1 = min(self.gw, int(math.ceil(x1))), min(self.gh, int(math.ceil(y1)))
        if x1 > x0 and y1 > y0:
            sub = self.mask[y0:y1, x0:x1]
            sub[self.gray_mask[y0:y1, x0:x1] > 0] = 255

    def _on_pan_start(self, event):
        if mod_pressed(event) and self._has_reference() and not self._painting:
            # Ctrl/Cmd + right or middle drag: nudge the reference image and binary
            self._nudge_drag = (event.x, event.y, self._nudge)
            self._pan_start = None
            try:
                self.canvas.configure(cursor='fleur')
            except Exception:
                pass
            return 'break'
        self._nudge_drag = None
        self._pan_start = (event.x, event.y, self._ox, self._oy)
        try:
            self.canvas.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        nd = getattr(self, '_nudge_drag', None)
        if nd is not None:
            x0, y0, (nx, ny) = nd
            d = (nx + int(round((event.x - x0) / self._zoom)), ny + int(round((event.y - y0) / self._zoom)))
            if d != self._nudge:
                self._set_nudge(d)
            return 'break'
        if self._pan_start is None:
            return 'break'
        x0, y0, ox, oy = self._pan_start
        self._ox = ox - (event.x - x0) / self._zoom
        self._oy = oy - (event.y - y0) / self._zoom
        self._snap_view()
        self._schedule_view(fast=True)
        return 'break'

    def _on_pan_end(self, event):
        nd = getattr(self, '_nudge_drag', None)
        self._nudge_drag = None
        if nd is not None and self._nudge != nd[2]:
            self._push_undo('Nudge', [('nudge', nd[2], self._nudge)])
            self._nudge_t = 0.0
            self.set_status('Reference image and binary nudged to ' + self._nudge_text())
        self._pan_start = None
        try:
            self.canvas.configure(cursor='')
        except Exception:
            pass
        return 'break'

    def _on_mouse_move(self, event):
        self._last_mouse = (event.x, event.y)
        ix, iy = self._to_img(event.x, event.y)
        self._update_coords(ix, iy)
        self._update_cursor()
        mode = self.mode_var.get()
        if mode == 'delete':
            self._update_hover(ix, iy)
        if mode in ('line', 'spline', 'check') and self._tool_points:
            self._update_preview(event.x, event.y)
        if mode == 'ruler' and len(self._ruler_points) == 1:
            self._redraw_ruler()
        self._update_snap_marker()

    def _on_mouse_leave(self, event):
        self._last_mouse = None
        for item in (self._cursor_item, self._snap_item):
            if item is not None:
                self.canvas.delete(item)
        self._cursor_item = None
        self._snap_item = None
        self._set_hover(None)
        self._cursor_cb('')

    def _update_coords(self, ix, iy):
        """Cursor position (working grid) and the shot image's own value under it, for the bottom bar."""
        gx, gy = int(math.floor(ix)), int(math.floor(iy))
        if not (0 <= gx < self.gw and 0 <= gy < self.gh):
            self._cursor_cb('')
            return
        txt = position_text(gx, gy, self.grid_factor)
        raw = self._shot_raw
        if raw is not None:
            k = max(1, self.grid_factor)
            ny, nx = gy // k, gx // k
            if 0 <= ny < raw.shape[0] and 0 <= nx < raw.shape[1]:
                v = float(raw[ny, nx])
                txt += '   intensity ' + (f'{v:.0f}' if abs(v - round(v)) < 1e-6 else f'{v:.4g}')
        self._cursor_cb(txt)

    def _on_wheel(self, event, steps):
        if not steps:
            return 'break'
        if mod_pressed(event):
            if self.mode_var.get() in SIZED_TOOLS:
                f = 1.1 ** steps
                self.brush_var.set(round(max(0.5, min(SIZE_MAX, float(self.brush_var.get()) * f)), 1))
                self._on_size_change()
            return 'break'
        nz = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * (ZOOM_STEP ** steps)))
        gx = self._ox + event.x / self._zoom
        gy = self._oy + event.y / self._zoom
        self._zoom = nz
        self._ox = gx - event.x / nz
        self._oy = gy - event.y / nz
        self._snap_view()
        self._schedule_view(fast=True)
        return 'break'

    def _on_canvas_configure(self, event=None):
        if not self._fitted:
            self.fit_view()
        self._schedule_view()

    def _snap_view(self):
        """Keep the view offset on whole screen pixels so 1 px lines are drawn evenly."""
        z = self._zoom
        self._ox = round(self._ox * z) / z
        self._oy = round(self._oy * z) / z

    def fit_view(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        z = min(cw / float(self.gw), ch / float(self.gh))
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self._ox = (self.gw - cw / self._zoom) / 2.0
        self._oy = (self.gh - ch / self._zoom) / 2.0
        self._snap_view()
        self._fitted = True
        self._schedule_view()

    # ---------------------------------------------------- cursor / snap ---
    def _update_cursor(self):
        mode = self.mode_var.get()
        if self._last_mouse is None or mode not in SIZED_TOOLS:
            if self._cursor_item is not None:
                self.canvas.delete(self._cursor_item)
                self._cursor_item = None
            return
        x, y = self._last_mouse
        x += 0.5; y += 0.5
        r = max(1.0, float(self.brush_var.get()) * self._zoom)
        if mode == 'mask':
            color = '#888888'
        elif mode == 'pen':
            color = '#00aaff' if self.edit_layer_var.get() == 'ref' else '#00ff88'
        else:
            color = '#ff5555'
        if self._cursor_item is None:
            self._cursor_item = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, width=1)
        else:
            self.canvas.coords(self._cursor_item, x - r, y - r, x + r, y + r)
            self.canvas.itemconfig(self._cursor_item, outline=color)
        self.canvas.tag_raise(self._cursor_item)

    def _fringe_ends(self, name):
        if name in self._ends_cache:
            return self._ends_cache[name]
        arr = self._layer(name)
        if arr is None:
            return np.zeros(0, int), np.zeros(0, int)
        ys, xs = fringe_ends(arr == 0)
        self._ends_cache[name] = (ys, xs)
        return ys, xs

    def _snap_target(self, ix, iy):
        if not self.snap_var.get() or self.mode_var.get() not in ('line', 'spline'):
            return None
        name = self._edit_target()
        if name is None:
            return None
        ys, xs = self._fringe_ends(name)
        if len(ys) == 0:
            return None
        ox, oy = self._ref_offset(name)
        rad = max(1.0, 12.0 / self._zoom)
        sel = (np.abs(xs + ox + 0.5 - ix) <= rad) & (np.abs(ys + oy + 0.5 - iy) <= rad)
        if not sel.any():
            return None
        cx, cy = xs[sel] + ox + 0.5, ys[sel] + oy + 0.5
        d = (cx - ix) ** 2 + (cy - iy) ** 2
        k = int(np.argmin(d))
        if d[k] > rad * rad:
            return None
        return float(cx[k]), float(cy[k])

    def _snap(self, p):
        s = self._snap_target(*p)
        return s if s is not None else p

    def _update_snap_marker(self):
        s = None
        if self._last_mouse is not None:
            s = self._snap_target(*self._to_img(*self._last_mouse))
        if s is None:
            if self._snap_item is not None:
                self.canvas.delete(self._snap_item)
                self._snap_item = None
            return
        cx, cy = self._to_canvas(*s)
        r = 6
        if self._snap_item is None:
            self._snap_item = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#00ff00', width=2)
        else:
            self.canvas.coords(self._snap_item, cx - r, cy - r, cx + r, cy + r)
        self.canvas.tag_raise(self._snap_item)

    # ------------------------------------------------------- shapes -----
    def _clear_preview(self):
        for item in self._preview_items:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self._preview_items = []

    def _cancel_shape(self):
        if self._tool_points:
            self._tool_points = []
            self._clear_preview()
            self.set_status('Cancelled')

    def _update_preview(self, mx, my):
        self._clear_preview()
        if not self._tool_points:
            return
        mode = self.mode_var.get()
        cur = self._to_img(mx, my)
        if mode in ('line', 'spline'):
            cur = self._snap(cur)
        pts = self._tool_points + [cur]
        if mode == 'spline':
            pts = self._calculate_spline(pts)
        coords = []
        for px, py in pts:
            coords.extend(self._to_canvas(px, py))
        color = '#ff8800' if mode == 'check' else '#00ff88'
        if len(coords) >= 4:
            self._preview_items.append(self.canvas.create_line(*coords, fill=color, width=1, dash=(4, 4)))
        for px, py in self._tool_points:
            cx, cy = self._to_canvas(px, py)
            self._preview_items.append(self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill='#ff00ff', outline=''))

    def _finish_shape(self):
        """Enter or Ctrl/Cmd + click: finish a spline or a polyline."""
        mode = self.mode_var.get()
        if (mode == 'spline' or (mode == 'line' and self.poly_var.get())) and len(self._tool_points) >= 2:
            self._commit_shape()

    @staticmethod
    def _calculate_spline(points):
        if len(points) < 2:
            return points
        pts = np.array(points, dtype=np.float64)
        dist = np.sum(np.diff(pts, axis=0) ** 2, axis=1)
        pts = pts[np.insert(dist > 1e-4, 0, True)]
        if len(pts) == 2:
            return pts
        k = 3 if len(pts) > 3 else 2
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
            u_new = np.linspace(0, 1, max(100, len(pts) * 20))
            x_new, y_new = splev(u_new, tck)
            return np.column_stack((x_new, y_new))
        except Exception:
            return pts

    def _commit_shape(self):
        mode = self.mode_var.get()
        pts = self._tool_points
        self._tool_points = []
        self._drag_start = None
        self._clear_preview()
        if len(pts) < 2:
            return
        if mode == 'check':
            self._connection_check(pts[0], pts[1])
            return
        name = self._edit_target()
        if name is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
            return
        arr = self._layer(name)
        ox, oy = self._ref_offset(name)
        pts = [(x - ox, y - oy) for x, y in pts]          # into the layer's own pixels
        self._begin([name])
        if mode == 'line':
            corners = []
            for x, y in pts:
                c = (int(math.floor(x)), int(math.floor(y)))
                if not corners or c != corners[-1]:
                    corners.append(c)
            if len(corners) < 2:
                self._snap_before = None
                self.set_status('Points were identical. Cancelled.')
                return
            curve = np.array(corners, np.int32)
            cv2.polylines(arr, [curve], isClosed=False, color=0, thickness=1)
            label = 'Polyline' if len(corners) > 2 else 'Line'
        else:
            curve = np.floor(np.array(self._calculate_spline(pts), dtype=np.float64)).astype(np.int32)
            cv2.polylines(arr, [curve], isClosed=False, color=0, thickness=1)
            label = 'Spline'
        rect = (curve[:, 0].min() - 1, curve[:, 1].min() - 1, curve[:, 0].max() + 2, curve[:, 1].max() + 2)
        if name == 'mask':
            self._enforce_mask(rect)
        self._commit(label, rect=rect)
        self.set_status(label + ' drawn.')
        self._schedule_view()

    # ------------------------------------------------- delete fringe ----
    def _piece_at(self, ix, iy):
        """(layer, x, y) of the fringe pixel exactly under working-grid point (ix, iy)
        on the edited binary, or None."""
        name = self._edit_target()
        if name is None:
            return None
        arr = self._layer(name)
        ox, oy = self._ref_offset(name)
        x, y = int(math.floor(ix - ox)), int(math.floor(iy - oy))
        h, w = arr.shape
        if not (0 <= x < w and 0 <= y < h) or arr[y, x] != 0:
            return None
        return name, x, y

    def _delete_fringe_at(self, ix, iy):
        if self._edit_target() is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
            return
        hit = self._piece_at(ix, iy)
        if hit is None:
            self.set_status('No fringe under the cursor: click exactly on a line (zoom in to pick it)')
            return
        name, sx, sy = hit
        arr = self._layer(name)
        h, w = arr.shape
        ink = (arr == 0).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        _n, _img, _m, (bx, by, bw, bh) = cv2.floodFill(ink, ff, (sx, sy), 2, flags=8)
        comp = ink[by:by + bh, bx:bx + bw] == 2
        self._begin([name])
        arr[by:by + bh, bx:bx + bw][comp] = 255
        self._commit('Delete fringe', rect=(bx, by, bx + bw, by + bh))
        self.set_status(f'Deleted a fringe piece ({int(comp.sum())} px)')
        self._set_hover(None)
        self._schedule_view()

    # --------------------------------------------- delete hover highlight ---
    def _labels_of(self, name):
        """8-connected labels and bounding boxes of a binary layer (cached until it changes)."""
        if name == 'mask':
            labels = self._ensure_labels()[0]
            return labels, self._labels_stats
        key = (self._ver.get('ovl', 0), self._live.get('ovl', 0), id(self.overlay_mask))
        hit = self._ovl_labels
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        n, lab, st, _ = cv2.connectedComponentsWithStats((self.overlay_mask == 0).astype(np.uint8), connectivity=8)
        self._ovl_labels = (key, lab, st)
        return lab, st

    def _update_hover(self, ix, iy):
        hit = self._piece_at(ix, iy)
        if hit is None:
            self._set_hover(None)
            return
        name, x, y = hit
        cur = self._hover
        if cur is not None and cur[0] == name:
            bx, by, bw, bh = cur[2]
            if bx <= x < bx + bw and by <= y < by + bh and cur[3][y - by, x - bx]:
                return                      # still the same piece
        labels, st = self._labels_of(name)
        L = int(labels[y, x])
        if L <= 0 or L >= len(st):
            self._set_hover(None)
            return
        bx, by, bw, bh = (int(v) for v in st[L, :4])
        m = labels[by:by + bh, bx:bx + bw] == L
        self._set_hover((name, L, (bx, by, bw, bh), m))

    def _set_hover(self, h):
        if h is None and self._hover is None:
            return
        self._hover = h
        self._schedule_view()

    # ----------------------------------------------- connection check ---
    def _ensure_labels(self):
        if self._labels_cache is not None and not self._labels_dirty:
            return self._labels_cache
        bw = (self.mask == 0).astype(np.uint8)
        num, labels, stats, _c = cv2.connectedComponentsWithStats(bw, connectivity=8)
        self._labels_stats = stats
        idx = np.arange(num, dtype=np.uint32)
        x = (idx * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
        colors = np.stack([(x & np.uint32(0xFF)).astype(np.uint8),
                           ((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8),
                           ((x >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)], axis=1)
        c1 = idx % 3; c2 = (idx + 1) % 3
        colors[idx, c1] = np.maximum(colors[idx, c1], 150)
        colors[idx, c2] = np.minimum(colors[idx, c2], 100)
        colors[0] = (255, 255, 255)
        self._labels_cache = (labels, colors)
        self._labels_dirty = False
        self._labels_gen += 1
        return self._labels_cache

    def _connection_check(self, a, b):
        self._check_line = (a, b)
        self._check_shape = self.mask.shape
        self._refresh_check(force=True)
        self.set_status(f'Connection check: {len(self.magic2_labels)} fringes touched (orange). The orange stays '
                        'while you use other tools; Esc clears it.')
        self._schedule_view()

    def _refresh_check(self, force=False):
        """Keep the Connection Check orange up to date: the fringes its line touches
        in the shot binary as it is now. Redone after each finished edit (not while a
        stroke is being drawn), so cutting a merge turns the cut off fringe red again."""
        if self._check_line is None:
            return
        if self.mask.shape != self._check_shape:
            self._clear_check()                 # the grid changed: the line no longer fits
            return
        key = (id(self.mask), self._ver.get('mask', 0), self._live.get('mask', 0))
        if not force and (key == self._check_key or self._painting):
            return
        labels, _ = self._ensure_labels()
        (ax, ay), (bx, by) = self._check_line
        trip = np.zeros(self.mask.shape, np.uint8)
        cv2.line(trip, (int(ax), int(ay)), (int(bx), int(by)), 1, 2)
        touched = np.unique(labels[trip == 1])
        touched = touched[touched > 0]
        lut = np.zeros(len(self._labels_stats), bool)
        lut[touched] = True
        self._check_px = lut[labels]
        self.magic2_labels = set(touched.tolist())
        self._check_key = key
        self._invalidate('check')

    def _clear_check(self):
        if self._check_line is None and self._check_px is None:
            return
        self._check_line = self._check_shape = self._check_px = self._check_key = None
        self.magic2_labels = set()
        self._invalidate('check')
        self._schedule_view()

    # ---------------------------------------------------- link ends ----
    def _link_endpoints(self):
        name = self._edit_target()
        if name is None:
            self.set_status('No reference binary loaded to link endpoints on')
            return
        try:
            tol = float(self.link_tol_var.get())
            ang = float(self.angle_deg_var.get())
            if not (math.isfinite(tol) and math.isfinite(ang)) or tol < 1 or ang < 0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror('Link endpoints', 'Failed: Angle and Max gap must be numbers (Max gap at least 1, '
                                 'Angle 0 to 90).', parent=self.winfo_toplevel())
            return
        tol = min(300.0, tol)
        ang = min(90.0, ang)
        target = self._layer(name)
        ink = target == 0
        if not ink.any():
            self.set_status('0 endpoints linked (the binary is empty)')
            return
        blocked = (self.gray_mask > 0) if name == 'mask' else None
        # Max gap is in pixels of the image, like every length SpellBook shows
        links = link_endpoints(ink, tol * max(1, self.grid_factor), ang, blocked=blocked)
        if not links:
            self.set_status(f'0 endpoints linked (max gap {tol:g} px, angle {ang:g}°)')
            return
        self._begin([name])
        xs = [p[0] for l in links for p in l]; ys = [p[1] for l in links for p in l]
        for (p0, p1) in links:
            cv2.line(target, p0, p1, 0, 1)
        rect = (min(xs) - 1, min(ys) - 1, max(xs) + 2, max(ys) + 2)
        if name == 'mask':
            self._enforce_mask(rect)
        self._commit('Link endpoints', rect=rect)
        self.set_status(f'Linked {len(links)} endpoint pair{"s" if len(links) != 1 else ""} '
                        f'(max gap {tol:g} px, angle {ang:g}°)')
        self._schedule_view()

    # ----------------------------------------------------------- merge -----
    def merge_overlay(self):
        if self.overlay_mask is None:
            return
        nudge = '' if self._nudge == (0, 0) else f' at its nudged position ({self._nudge_text()})'
        if not messagebox.askokcancel('Merge Binaries?',
                                      'This turns everything into the shot binary layer: the reference binary'
                                      f'{nudge} is copied into the shot binary and is no longer a separate layer.'
                                      f'\n\nYou can undo it with {MOD}+Z.', parent=self.winfo_toplevel()):
            return
        self._begin(['mask'])
        ox, oy = self._nudge
        oh, ow = self.overlay_mask.shape[:2]
        h, w = self.mask.shape[:2]
        x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + ow), min(h, oy + oh)
        if x1 > x0 and y1 > y0:
            np.bitwise_and(self.mask[y0:y1, x0:x1], self.overlay_mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox],
                           out=self.mask[y0:y1, x0:x1])
        self._enforce_mask()
        extra = [('layer', 'ovl', self.overlay_mask, None)]
        self.overlay_mask = None
        self._invalidate('ovl')
        self._commit('Merge binaries', extra=extra)
        self.set_status(f'Reference binary merged into the shot binary. {MOD}+Z brings it back.')
        self._after_history()

    # ------------------------------------------------------------ ruler --
    def _apply_ruler_lock(self, anchor, p):
        lock = self.ruler_lock_var.get()
        if lock == 'Horizontal':
            return (p[0], anchor[1])
        if lock == 'Vertical':
            return (anchor[0], p[1])
        return p

    def _ruler_down(self, ix, iy):
        pt = (ix, iy)
        if len(self._ruler_points) == 2:
            for k in (0, 1):
                px, py = self._ruler_points[k]
                if math.hypot(ix - px, iy - py) * self._zoom < 15:
                    self._ruler_drag_idx = k
                    return
            self._ruler_points = [pt]
        elif len(self._ruler_points) == 1:
            self._ruler_points.append(self._apply_ruler_lock(self._ruler_points[0], pt))
        else:
            self._ruler_points = [pt]
        self._redraw_ruler()

    def _ruler_drag(self, ix, iy):
        if self._ruler_drag_idx is not None:
            k = self._ruler_drag_idx
            self._ruler_points[k] = self._apply_ruler_lock(self._ruler_points[1 - k], (ix, iy))
        self._redraw_ruler()

    def _ruler_up(self, ix, iy):
        if self._ruler_drag_idx is not None:
            self._ruler_drag_idx = None
        elif len(self._ruler_points) == 1:
            p0 = self._ruler_points[0]
            if math.hypot(p0[0] - ix, p0[1] - iy) * self._zoom > 3:
                self._ruler_points.append(self._apply_ruler_lock(p0, (ix, iy)))
        self._redraw_ruler()

    def _redraw_ruler(self):
        for item in self._ruler_items:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self._ruler_items = []
        if self.mode_var.get() != 'ruler' or not self._ruler_points:
            self.ruler_dist_var.set('Dist: 0.00 px')
            return
        p0 = self._ruler_points[0]
        if len(self._ruler_points) == 2:
            p1 = self._ruler_points[1]
        elif self._last_mouse is not None:
            p1 = self._apply_ruler_lock(p0, self._to_img(*self._last_mouse))
        else:
            p1 = p0
        c0 = self._to_canvas(*p0); c1 = self._to_canvas(*p1)
        self._ruler_items.append(self.canvas.create_line(c0[0], c0[1], c1[0], c1[1], fill='#00ffff', width=2, dash=(4, 4)))
        for cx, cy in (c0, c1):
            self._ruler_items.append(self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill='#ff00ff', outline='white'))
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(1, self.grid_factor)    # image pixels
        self.ruler_dist_var.set(f'Dist: {d:.2f} px')

    # =========================================================== render ==
    def _schedule_view(self, fast=False):
        if self._render_id is not None:
            return
        self._render_id = self.after(1 if fast else 8, self._render_full)

    def _render_full(self):
        self._render_id = None
        self._refresh_check()
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._frame_rgb is None or self._frame_rgb.shape[:2] != (ch, cw):
            self._frame_rgb = np.zeros((ch, cw, 3), np.uint8)
        self._frame_rgb[:] = self._compose(0, 0, cw, ch, full=True)
        self._push_frame()
        self._redraw_ruler()
        if self._tool_points and self._last_mouse is not None:
            self._update_preview(*self._last_mouse)
        self._update_cursor()
        self._update_snap_marker()

    def _push_frame(self):
        img = Image.fromarray(self._frame_rgb)
        if self._photo is not None and (self._photo.width(), self._photo.height()) == img.size \
                and hasattr(self._photo, 'paste'):
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        else:
            self.canvas.itemconfig(self._img_item, image=self._photo)
        self.canvas.tag_lower(self._img_item)

    def _push_rect(self, x0, y0, x1, y1):
        """Update only part of the on-screen image (used while drawing)."""
        if self._photo is None or (self._photo.width(), self._photo.height()) != \
                (self._frame_rgb.shape[1], self._frame_rgb.shape[0]):
            self._push_frame()
            return
        try:
            small = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(self._frame_rgb[y0:y1, x0:x1])))
            self.canvas.tk.call(str(self._photo), 'copy', str(small), '-to', x0, y0)
            self._small = small
        except Exception:
            self._push_frame()

    def _render_rect(self, cx0, cy0, cx1, cy1):
        """Recomposite the canvas rectangle [cx0, cx1) x [cy0, cy1)."""
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._frame_rgb is None or self._frame_rgb.shape[:2] != (ch, cw):
            self._schedule_view()
            return
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(cw, cx1), min(ch, cy1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        self._frame_rgb[cy0:cy1, cx0:cx1] = self._compose(cx0, cy0, cx1, cy1)
        self._push_rect(cx0, cy0, cx1, cy1)

    # Zoomed out, each layer is drawn from a cached reduced copy (a "level": 1/2,
    # 1/4, ... of the grid), so a frame never has to shrink a 4096 x 4096 array.
    # Fringe and mask levels keep a pixel lit if any pixel in its block is lit, so
    # 1 px lines never vanish. Levels are rebuilt only when that layer changes.
    def _level(self):
        z = self._zoom
        if z >= 1.0:
            return 0
        return max(0, int(math.ceil(math.log2(1.0 / z) - 1e-9)))

    def _pyr_patch(self, name, x0, y0, x1, y1):
        """Bring cached reduced copies of a binary up to date inside a changed
        rectangle, so a stroke shows while you draw."""
        base = self._layer(name)
        if base is None:
            return
        ver = self._ver.get(name, 0)
        for (n, kind, L), (v, data) in list(self._pyr.items()):
            if n != name or v != ver or kind not in ('ink', 'gray'):
                continue
            f = 2 ** L
            lx0, ly0 = max(0, int(math.floor(x0)) // f), max(0, int(math.floor(y0)) // f)
            lx1, ly1 = min(data.shape[1], -(-int(math.ceil(x1)) // f)), min(data.shape[0], -(-int(math.ceil(y1)) // f))
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            b = base[ly0 * f:ly1 * f, lx0 * f:lx1 * f]
            b = (b == 0) if kind == 'ink' else (b > 0)
            data[ly0:ly1, lx0:lx1] = b.reshape(ly1 - ly0, f, lx1 - lx0, f).any(axis=(1, 3))

    def _pyr_get(self, name, kind, L):
        """Reduced copy of a layer at level L (factor 2**L). kind: 'img', 'ink', 'gray', 'labels'."""
        if kind == 'labels':
            self._ensure_labels()
            ver = ('labels', self._labels_gen)
        else:
            ver = self._ver.get(name, 0)
        key = (name, kind, L)
        hit = self._pyr.get(key)
        if hit is not None and hit[0] == ver:
            return hit[1]
        base = self._ensure_labels()[0] if kind == 'labels' else self._layer(name)
        if base is None:
            return None
        f = 2 ** L
        H, W = base.shape[:2]
        h, w = max(1, H // f), max(1, W // f)
        if kind == 'img':
            data = cv2.resize(base[:h * f, :w * f], (w, h), interpolation=cv2.INTER_AREA)
        else:
            b = base[:h * f, :w * f]
            if kind == 'ink':
                b = b == 0
            elif kind == 'gray':
                b = b > 0
            b = b.reshape(h, f, w, f)
            data = b.max(axis=(1, 3)) if kind == 'labels' else b.any(axis=(1, 3))
        self._pyr[key] = (ver, data)
        return data

    def _sample_layer(self, name, kind, rect):
        """Draw layer `name` into canvas rectangle rect. Returns uint8 (img),
        bool (ink/gray) or int32 (labels) of the rect size, or None."""
        cx0, cy0, cx1, cy1 = rect
        Wd, Hd = cx1 - cx0, cy1 - cy0
        z = self._zoom
        L = self._level()
        f = 2 ** L
        base = self._layer(name) if kind != 'labels' else self._ensure_labels()[0]
        if base is None:
            return None
        dx, dy = (0, 0) if kind == 'labels' else self._ref_offset(name)
        vx, vy = self._ox - dx, self._oy - dy        # view origin in the layer's own pixels
        gx0 = vx + cx0 / z; gy0 = vy + cy0 / z
        gx1 = vx + cx1 / z; gy1 = vy + cy1 / z
        data = base if L == 0 else self._pyr_get(name, kind, L)
        dh, dw = data.shape[:2]
        lx0 = max(0, int(math.floor(gx0 / f)) - 1); ly0 = max(0, int(math.floor(gy0 / f)) - 1)
        lx1 = min(dw, int(math.ceil(gx1 / f)) + 1); ly1 = min(dh, int(math.ceil(gy1 / f)) + 1)
        if lx1 <= lx0 or ly1 <= ly0:
            return None
        sub = data[ly0:ly1, lx0:lx1]
        if L == 0:
            if kind == 'ink':
                sub = sub == 0
            elif kind == 'gray':
                sub = sub > 0
        a = z * f
        # level pixel k covers grid [k f, (k + 1) f): map pixel centers exactly
        tx = (lx0 * f + 0.5 * f - vx) * z - cx0 - 0.5
        ty = (ly0 * f + 0.5 * f - vy) * z - cy0 - 0.5
        M = np.float32([[a, 0, tx], [0, a, ty]])
        if kind == 'img':
            flags = cv2.INTER_NEAREST if a >= 4 else cv2.INTER_LINEAR
            return cv2.warpAffine(sub, M, (Wd, Hd), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        if kind == 'labels':
            out = cv2.warpAffine(sub.astype(np.float32), M, (Wd, Hd), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return out.astype(np.int32)
        out = cv2.warpAffine(sub.view(np.uint8) if sub.dtype == bool else sub.astype(np.uint8), M, (Wd, Hd),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0

    def _mirror_ink(self, rect):
        if not self.mirror_var.get():
            return None
        try:
            cxm = float(self.mirror_x_var.get().strip()) * max(1, self.grid_factor)   # image px -> grid
        except (ValueError, TypeError):
            return None
        if not math.isfinite(cxm):
            return None
        cx0, cy0, cx1, cy1 = rect
        Wd, Hd = cx1 - cx0, cy1 - cy0
        z = self._zoom
        L = self._level(); f = 2 ** L
        data = self.mask if L == 0 else self._pyr_get('mask', 'ink', L)
        dh, dw = data.shape[:2]
        gx0 = self._ox + cx0 / z; gy0 = self._oy + cy0 / z
        gx1 = self._ox + cx1 / z; gy1 = self._oy + cy1 / z
        lx0 = max(0, int(math.floor(gx0 / f)) - 1); ly0 = max(0, int(math.floor(gy0 / f)) - 1)
        lx1 = min(dw, int(math.ceil(gx1 / f)) + 1); ly1 = min(dh, int(math.ceil(gy1 / f)) + 1)
        if lx1 <= lx0 or ly1 <= ly0:
            return None
        # column x shows the column reflected about the center line x = cxm
        src = np.floor((2 * cxm - (np.arange(lx0, lx1) + 0.5) * f) / f).astype(np.int64)
        valid = (src >= 0) & (src < dw)
        sub = np.zeros((ly1 - ly0, lx1 - lx0), bool)
        if valid.any():
            cols = data[ly0:ly1, src[valid]]
            sub[:, valid] = (cols == 0) if L == 0 else cols
        a = z * f
        M = np.float32([[a, 0, (lx0 * f + 0.5 * f - self._ox) * z - cx0 - 0.5],
                        [0, a, (ly0 * f + 0.5 * f - self._oy) * z - cy0 - 0.5]])
        out = cv2.warpAffine(sub.view(np.uint8), M, (Wd, Hd), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0

    def _grid_inside(self, rect):
        z = self._zoom
        cx0, cy0, cx1, cy1 = rect
        xs = self._ox + (np.arange(cx0, cx1) + 0.5) / z
        ys = self._oy + (np.arange(cy0, cy1) + 0.5) / z
        return ((ys >= 0) & (ys < self.gh))[:, None] & ((xs >= 0) & (xs < self.gw))[None, :]

    def _sample(self, name, kind, rect, full):
        """Sample with a cache for whole-canvas frames, so changing an opacity or
        brightness only re-blends and does not resample every layer."""
        if not full:
            return self._sample_layer(name, kind, rect)
        src_name = 'mask' if kind == 'labels' else name
        key = (name, kind)
        if kind == 'labels':
            self._ensure_labels()
        stamp = (self._zoom, self._ox, self._oy, rect, self._ver.get(src_name, 0), self._live.get(src_name, 0),
                 self._labels_gen if kind == 'labels' else None, id(self._layer(src_name)), self._ref_offset(name))
        hit = self._samples.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        out = self._sample_layer(name, kind, rect)
        self._samples[key] = (stamp, out)
        return out

    def _compose(self, cx0, cy0, cx1, cy1, full=False):
        Wd, Hd = cx1 - cx0, cy1 - cy0
        rect = (cx0, cy0, cx1, cy1)
        out = np.empty((Hd, Wd, 3), np.uint8)
        out[:] = OUTSIDE
        inside = self._grid_inside(rect)
        if not inside.any():
            return out
        gray = None
        if self._ref_img is not None and self.show_ref_img.get():
            r = self._sample('ref', 'img', rect, full)
            if r is not None:
                b = float(self.ref_brightness_var.get())
                gray = r if b == 0 else cv2.convertScaleAbs(r, alpha=2.0 ** b)
        if self._shot_img is not None and self.show_shot_img.get():
            sh = self._sample('shot', 'img', rect, full)
            if sh is not None:
                b = float(self.shot_brightness_var.get())
                if b != 0:
                    sh = cv2.convertScaleAbs(sh, alpha=2.0 ** b)
                a = float(self.shot_opacity_var.get())
                if a >= 0.999:
                    gray = sh
                else:
                    under = gray if gray is not None else np.full((Hd, Wd), 255, np.uint8)
                    gray = cv2.addWeighted(sh, a, under, 1.0 - a, 0)
        if gray is None:
            gray = np.full((Hd, Wd), 255, np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if self.show_shot_bin.get():
            ink = self._sample('mask', 'ink', rect, full)
            mir = self._mirror_ink(rect)
            if mir is not None:
                rgb[mir & ~ink] = BLUE           # view only, under the shot binary
            if ink is not None and ink.any():
                colorize = self.show_components_var.get()
                if self.mode_var.get() == 'check':
                    rgb[ink] = BLACK              # Connection Check: the fringes it did not touch are black
                elif colorize:
                    labels, colors = self._ensure_labels()
                    lab = self._sample('mask', 'labels', rect, full)
                    col = colors[lab]
                    col[lab == 0] = RED           # drawn since the last relabel
                    rgb[ink] = col[ink]
                else:
                    a = float(self.shot_binary_opacity_var.get())
                    if a >= 0.999:
                        rgb[ink] = RED
                    else:
                        rgb[ink] = (rgb[ink].astype(np.float32) * (1.0 - a) + np.array(RED, np.float32) * a).astype(np.uint8)
                if self._check_px is not None and self._check_px.shape == self.mask.shape:
                    # the fringes the Connection Check touched stay orange whatever the tool
                    cm = self._sample('check', 'gray', rect, full)
                    if cm is not None:
                        rgb[cm & ink] = ORANGE
            g = self._sample('gray', 'gray', rect, full)
            if g is not None:
                rgb[g] = (128, 128, 128)
        if self.overlay_mask is not None and self.show_ref_bin.get():
            ov = self._sample('ovl', 'ink', rect, full)
            if ov is not None and ov.any():
                a = float(self.ref_binary_opacity_var.get())
                rgb[ov] = (rgb[ov].astype(np.float32) * (1.0 - a) + np.array(CYAN, np.float32) * a).astype(np.uint8)
        hv = self._hover
        if hv is not None and self.mode_var.get() == 'delete':
            name, _L, (bx, by, bw, bh), m = hv
            dx, dy = self._ref_offset(name)
            hm = self._sample_patch(m, bx + dx, by + dy, rect)
            if hm is not None:
                rgb[hm] = YELLOW
        rgb[~inside] = OUTSIDE
        return rgb

    def _sample_patch(self, m, gx, gy, rect):
        """A small bool array whose pixel (0, 0) sits at working-grid pixel (gx, gy),
        drawn into canvas rectangle rect (thickened when zoomed out so it stays visible)."""
        cx0, cy0, cx1, cy1 = rect
        z = self._zoom
        src = m
        if z < 1:
            k = int(min(15, math.ceil(1.0 / z))) | 1
            src = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        M = np.float32([[z, 0, (gx + 0.5 - self._ox) * z - cx0 - 0.5],
                        [0, z, (gy + 0.5 - self._oy) * z - cy0 - 0.5]])
        out = cv2.warpAffine(src.view(np.uint8) if src.dtype == bool else src.astype(np.uint8), M,
                             (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
        return out > 0
