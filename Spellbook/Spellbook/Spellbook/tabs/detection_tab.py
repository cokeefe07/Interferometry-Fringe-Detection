"""
Detection tab.

Fringe center lines are found by fringe_detection.ridge_detector, which follows
the local fringe direction and spacing instead of using fixed horizontal line
kernels. Heavy work runs in a background thread and is cached in stages, so
moving a slider only redoes the part of the analysis that slider affects:

  Max Fringe Angle, Smoothing Along Fringes -> full analysis (a few seconds)
  Sensitivity, Gap Bridging, Minimum Fringe Length, Noise Rejection
                                             -> clean-up only (about a second)

The whole image is always traced. The working grid is the image's own size
(1x) or twice that (2x); at 2x the center lines sit between the image pixels.
The viewer only draws the part of the image that is on screen, so zooming and
panning stay fast at 2x.
"""
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from fringe_detection.ui_helpers import (make_slider_row, help_icon, attach_tooltip, bold_font, bind_wheel,
                                         bind_pan, scrollable_panel, imread_any, imwrite_png, image_filetypes,
                                         sanitize_image, ui_scale, overlay_button, CanvasPrompt, position_text,
                                         log_error)
from fringe_detection.ridge_detector import RidgeDetector, DEFAULTS, auto_optimize, Cancelled

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EDITED_DIR = os.path.join(APP_DIR, 'EditedImages')     # binaries SpellBook writes
INPUT_DIR = os.path.join(APP_DIR, 'InputImages')       # a place to keep the shots to trace

ZOOM_STEP = 1.15        # zoom factor per mouse wheel notch (both tabs)
ZOOM_MIN, ZOOM_MAX = 0.02, 64.0


def grid_label(factor):
    return f'{int(factor)}x'


AUTO_HELP = ('Tries the settings of all six sliders over their useful range on this image and keeps the '
             'combination that gives the most good fringe: lines that follow the fringes where the fringe '
             'pattern is orderly, with few broken ends or lines squeezed between two fringes (where fringes '
             'merge). Stray pieces count only a little, since they are quick to delete. '
             'The result does not depend on where the sliders were. The status line counts down while it '
             'runs and the button becomes Cancel. The sliders are locked while it runs and are set to the '
             'result at the end, so you can fine tune after.')


def make_folders():
    for d in (INPUT_DIR, EDITED_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


class DetectionTabFrame(ttk.Frame):
    """Detection tab: detect fringe center lines and show them over the shot."""

    # label, detector parameter, min, max, is_int, resolution, default, tooltip
    SLIDERS = [
        ('Sensitivity', 'sensitivity', 0.0, 1.0, False, 0.01, DEFAULTS['sensitivity'],
         'How faint a fringe can be and still be traced. Raise it to pick up more fringes near '
         'the target; lower it if noise starts being traced.'),
        ('Smoothing Along Fringes (px)', 'smooth_len', 0.0, 25.0, False, 0.1, DEFAULTS['smooth_len'],
         'Averages the image along each fringe (never across it) to suppress speckle. Higher '
         'gives straighter, less broken lines in noisy areas; too high can make lines drift '
         'across dense, curved fringes. Automatically reduced where fringes are close together. '
         'In pixels of the image. Re-runs the full analysis.'),
        ('Max Fringe Angle°', 'max_angle', 0.0, 90.0, False, 1.0, DEFAULTS['max_angle'],
         'Steepest fringe tilt (from horizontal) that is traced. 0 keeps only horizontal fringes; '
         '90 keeps every angle up to vertical. Anything steeper than the setting, such as vertical '
         'streaks at the target edge, is filtered out. Re-runs the full analysis.'),
        ('Gap Bridging (px)', 'gap_bridge', 0, 40, True, 1, DEFAULTS['gap_bridge'],
         'Largest break in a fringe that is joined automatically, in pixels of the image. A gap is '
         'only bridged if the two ends line up along the fringe direction and the bridge does not '
         'cross a dark fringe.'),
        ('Minimum Fringe Length (px)', 'min_length', 5, 200, True, 1, DEFAULTS['min_length'],
         'Pieces shorter than this are deleted. In pixels of the image: at 2x it is scaled for you, '
         'so the same setting means the same length on the image on both grids.'),
        ('Noise Rejection', 'mask_level', 0.0, 1.0, False, 0.01, DEFAULTS['mask_level'],
         'Removes small fringes where the fringe pattern is weak or disordered (shadows, the dark '
         'band above the plume). The weaker the pattern around a piece, the longer it has to be to '
         'stay, so high settings also remove long pieces there. 0 keeps everything.'),
    ]
    STAGE_A_KEYS = ('smooth_len', 'max_angle')

    def __init__(self, master, status_callback=None, file_callback=None, cursor_callback=None):
        super().__init__(master)
        self._status_cb = status_callback or (lambda txt: None)
        self._file_cb = file_callback or (lambda txt: None)
        self._cursor_cb = cursor_callback or (lambda txt: None)
        # Set by the app: callable(binary_uint8, shot_float, saved_path, name, factor) -> bool
        self.send_to_editor_cb = None

        # data
        self.src_img = None          # image as float32 (native size)
        self._src_path = None
        self._load_note = ''         # said with every result: NaN pixels replaced on load
        self._img_gen = 0            # bumped on every load; results from older images are dropped
        self._grid_factor = 1
        self._det = None
        self._binary_mask = None     # uint8, 0 = fringe, 255 = background (what gets saved)
        self._traced = None          # bool, True = fringe pixel, working grid
        self._result_key = None      # (image generation, grid factor) of the shown result
        self._result_saved = False   # the shown result has been saved or sent to the Editor
        self._disp_native = None
        self._last_load_dir = INPUT_DIR if os.path.isdir(INPUT_DIR) else APP_DIR
        self._last_save_dir = EDITED_DIR if os.path.isdir(EDITED_DIR) else APP_DIR
        self.robust_range = tk.BooleanVar(value=True)

        # worker / scheduling
        self._q = queue.Queue()
        self._busy = False
        self._pending = None
        self._debounce_id = None
        self._job_id = 0
        self._auto = None            # dict while Auto-Optimize is queued or running
        self._suspend_param_cb = False

        # viewer: zoom = screen px per working-grid px; (ox, oy) = grid coords at canvas (0, 0)
        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._photo = None
        self._img_item = None
        self._pan_start = None
        self._render_id = None
        self._view_cache = None      # (view, image, trace, resampled view) of the last frame

        self._build_ui()
        self._attach_handlers()
        self.set_status('Load an image to start')
        self._poll_id = self.after(50, self._poll_queue)

    def destroy(self):
        for aid in (getattr(self, '_poll_id', None), self._debounce_id, self._render_id):
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
        if self._auto is not None:
            self._auto['event'].set()
        super().destroy()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        container, ctrl = scrollable_panel(self, 280)
        container.pack(side='left', fill='y', padx=8, pady=8)
        self._panel_container = container
        s = ui_scale(self)

        left_title = ttk.Frame(ctrl)
        left_title.pack(anchor='w', fill='x')
        ttk.Label(left_title, text='Fringe Detection', font=bold_font(self, 1)).pack(side='left')
        help_icon(left_title, 'Mouse wheel or two finger scroll: zoom.  Drag with any mouse button: pan.\n'
                              'Fit (top right of the image) shows the whole image.\n'
                              'The whole image is traced. Bright fringes are traced as red lines.')

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(anchor='w', fill='x', pady=4)
        self.btn_load = ttk.Button(btn_row, text='Load Image', command=self.load_image_dialog)
        self.btn_load.pack(side='left')
        attach_tooltip(self.btn_load, 'Open a shot (TIFF, PNG, JPEG or BMP, any bit depth). The whole image is '
                                      'traced right away.')
        self.btn_save = ttk.Button(btn_row, text='Save Fringes as Binary', command=self.save_result)
        self.btn_save.pack(side='left', padx=(6, 0))
        attach_tooltip(self.btn_save, 'Save the traced fringes as a PNG binary (0 = fringe, 255 = background) '
                                      'wherever you choose.')

        send_row = ttk.Frame(ctrl)
        send_row.pack(anchor='w', fill='x', pady=(2, 0))
        self.btn_send = ttk.Button(send_row, text='Send to Editor', command=self.send_to_editor)
        self.btn_send.pack(side='left', fill='x', expand=True)
        help_icon(send_row, 'Saves the current fringes to the EditedImages folder (named after the shot, '
                            'never overwriting an older file) and opens them in the Editor with the shot '
                            'image underneath, already aligned.')

        auto_row = ttk.Frame(ctrl)
        auto_row.pack(anchor='w', fill='x', pady=(4, 2))
        self.btn_auto = ttk.Button(auto_row, text='Auto-Optimize', command=self.auto_optimize)
        self.btn_auto.pack(side='left', fill='x', expand=True)
        self._auto_help = help_icon(auto_row, AUTO_HELP)

        brow = ttk.Frame(ctrl)
        brow.pack(anchor='w', fill='x')
        ttk.Checkbutton(brow, text='Brighten Image', variable=self.robust_range,
                        command=self._on_brighten_change).pack(side='left', padx=6)
        help_icon(brow, 'Stretches the display contrast so a few hot pixels cannot make the image dark. '
                        'Display only: detection is not affected.')

        self._slider_meta = {}
        self._vars = {}
        self._scales = {}
        self.bg_opacity = tk.DoubleVar(value=1.0)
        s_bg = make_slider_row(ctrl, 'Image Opacity', self.bg_opacity, 0.0, 1.0, resolution=0.01,
                               command=lambda _=None: self._schedule_view())
        self._slider_meta['Image Opacity'] = {'scale': s_bg, 'var': self.bg_opacity, 'is_int': False,
                                              'frm': 0.0, 'to': 1.0, 'key': 'Image Opacity'}
        help_icon(s_bg.master, 'Fades the image toward white so the red lines stand out. 1 shows the image '
                               'at full contrast. Click the number to type a value.')

        # messages go to the status line at the bottom of the window (one status line);
        # this label only keeps the latest text and is not shown
        self.status = ttk.Label(ctrl, text='')
        self.progress = ttk.Progressbar(ctrl, mode='determinate', length=int(200 * s), maximum=100)
        self.progress.pack(anchor='w', fill='x', pady=(6, 4))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(ctrl, text='Detection Settings', font=bold_font(self)).pack(anchor='w', pady=(4, 2))

        for (label, key, frm, to, is_int, res, default, tip) in self.SLIDERS:
            var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=float(default))
            sc = make_slider_row(ctrl, label, var, frm, to, resolution=res, is_int=is_int,
                                 command=lambda _=None, k=key: self.on_param_change(k))
            self._slider_meta[label] = {'scale': sc, 'var': var, 'is_int': is_int, 'frm': frm, 'to': to, 'key': key}
            self._vars[key] = var
            self._scales[key] = sc
            help_icon(sc.master, tip + ' Click the number to type a value.')
        ttk.Frame(ctrl, height=int(40 * s)).pack(fill='x')

        # viewer
        img_frame = ttk.Frame(self)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        img_frame.pack_propagate(False)
        self.fringe_canvas = tk.Canvas(img_frame, bg='black', highlightthickness=0)
        self.fringe_canvas.pack(fill='both', expand=True)
        self.btn_fit = overlay_button(self.fringe_canvas, 'Fit', self.fit_view)
        attach_tooltip(self.btn_fit, 'Show the whole image', side='left')
        self._prompt = CanvasPrompt(self.fringe_canvas, 'Load an interferogram to trace its fringes.',
                                    [('Load Image', self.load_image_dialog)])
        self._prompt.show()

    def _attach_handlers(self):
        c = self.fringe_canvas
        for b in (1, 2, 3):
            c.bind(f'<ButtonPress-{b}>', lambda e: c.focus_set(), add='+')
        bind_wheel(c, self._on_wheel)
        bind_pan(c, self._on_pan_start, self._on_pan_move, self._on_pan_end, buttons=(1, 2, 3))
        c.bind('<Configure>', lambda e: self._schedule_view(), add='+')
        c.bind('<Motion>', self._on_motion, add='+')
        c.bind('<Leave>', lambda e: self._cursor_cb(''), add='+')

    # ------------------------------------------------------------ helpers --
    def set_status(self, txt):
        try:
            self.status.config(text=txt)
        except Exception:
            pass
        try:
            self._status_cb(txt)
        except Exception:
            pass

    def current_file_text(self):
        if self.src_img is None:
            return 'No image loaded'
        h, w = self.src_img.shape[:2]
        return f'{os.path.basename(self._src_path or "image")}  ({w} x {h} px)'

    def _params(self):
        """Slider values, snapped to what the sliders show (main thread only)."""
        p = {}
        for (label, key, frm, to, is_int, res, default, tip) in self.SLIDERS:
            v = float(self._vars[key].get())
            v = round(round(v / res) * res, 10)
            p[key] = int(round(v)) if is_int else float(v)
        return p

    def _set_params(self, p):
        self._suspend_param_cb = True
        try:
            for (label, key, frm, to, is_int, res, default, tip) in self.SLIDERS:
                if key in p:
                    v = p[key]
                    meta = self._slider_meta[label]
                    # widen the slider range if the optimizer went outside it
                    if v < meta['frm'] or v > meta['to']:
                        meta['frm'] = min(meta['frm'], v); meta['to'] = max(meta['to'], v)
                        meta['scale'].config(from_=meta['frm'], to=meta['to'])
                    self._vars[key].set(int(round(v)) if is_int else float(v))
        finally:
            self._suspend_param_cb = False

    def _set_sliders_locked(self, locked):
        flag = ['disabled'] if locked else ['!disabled']
        for sc in self._scales.values():
            try:
                sc.state(flag)
            except Exception:
                pass

    # ------------------------------------------------------------ settings --
    def get_settings(self):
        """What is remembered for the next session."""
        d = {k: v for k, v in self._params().items()}
        d['brighten'] = bool(self.robust_range.get())
        d['image_opacity'] = float(self.bg_opacity.get())
        d['load_dir'] = self._last_load_dir
        d['save_dir'] = self._last_save_dir
        return d

    def apply_settings(self, d):
        """Restore remembered settings (anything missing or invalid is skipped)."""
        if not isinstance(d, dict):
            return
        p = {}
        for (label, key, frm, to, is_int, res, default, tip) in self.SLIDERS:
            v = d.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                v = min(max(float(v), frm), to)
                p[key] = int(round(v)) if is_int else round(round(v / res) * res, 10)
        self._set_params(p)
        if isinstance(d.get('brighten'), bool):
            self.robust_range.set(d['brighten'])
        op = d.get('image_opacity')
        if isinstance(op, (int, float)) and 0.0 <= op <= 1.0:
            self.bg_opacity.set(round(float(op), 2))
        for key, attr in (('load_dir', '_last_load_dir'), ('save_dir', '_last_save_dir')):
            v = d.get(key)
            if isinstance(v, str) and os.path.isdir(v):
                setattr(self, attr, v)

    def _result_is_current(self):
        return (self._binary_mask is not None and self._result_key == (self._img_gen, self._grid_factor)
                and not self._busy and self._pending is None and self._debounce_id is None)

    def unsaved_result(self):
        """True when fringes are traced on the loaded image and were never saved or sent."""
        return self.src_img is not None and self._binary_mask is not None and not self._result_saved

    # ------------------------------------------------------------ image IO --
    def load_image_dialog(self):
        p = filedialog.askopenfilename(parent=self.winfo_toplevel(), title='Load interferogram',
                                       initialdir=self._last_load_dir,
                                       filetypes=image_filetypes('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))
        if not p:
            return
        self._last_load_dir = os.path.dirname(p)
        self.load_image(p)

    def load_image(self, path):
        img = imread_any(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror('Load error', f'Could not read {path}', parent=self.winfo_toplevel())
            return
        if self._auto is not None:
            self._cancel_auto('Auto-Optimize cancelled: a new image was loaded')
        img, n_bad = sanitize_image(img)
        self.src_img = img
        self._src_path = path
        self._img_gen += 1
        self._det = RidgeDetector(self.src_img, self._params())
        self._traced = None
        self._binary_mask = None
        self._result_key = None
        self._result_saved = False
        self._make_display_image()
        self._prompt.hide()
        h, w = self.src_img.shape[:2]
        msg = f'Loaded {os.path.basename(path)} ({w} x {h} px)'
        self._load_note = ''
        if n_bad:
            self._load_note = f' ({n_bad} NaN or infinite pixels were replaced by the median)'
            msg += self._load_note
        self.set_status(msg)
        self._file_cb(self.current_file_text())
        self.after(60, self.fit_view)
        self.start_render_now()

    def _make_display_image(self):
        if self.src_img is None:
            return
        work = self.src_img
        if self.robust_range.get():
            lo, hi = np.percentile(work, (0.1, 99.9))
        else:
            lo, hi = float(work.min()), float(work.max())
        if hi <= lo:
            lo, hi = float(work.min()), float(work.max())
        if hi <= lo:
            self._disp_native = np.zeros(work.shape, np.uint8)
        else:
            self._disp_native = np.clip((work - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    def set_grid_factor(self, factor):
        """1 = the image's own size, 2 = twice that. The analysis itself always
        runs on the image; only the center-line step and the clean-up run on the
        chosen grid, so switching is quick and the 2x lines keep sub-pixel detail."""
        factor = max(1, int(factor))
        if factor == self._grid_factor:
            return
        if self._auto is not None:
            self._cancel_auto('Auto-Optimize cancelled: the working grid changed')
        old = self._grid_factor
        self._grid_factor = factor
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * old / float(factor)))
        self._ox *= factor / float(old)
        self._oy *= factor / float(old)
        self._snap_view()
        self._binary_mask = None
        self._traced = None
        self._result_key = None
        self._result_saved = False
        if self.src_img is not None:
            h, w = self.src_img.shape[:2]
            self.set_status(f'Working grid {grid_label(factor)} ({w * factor} x {h * factor} px): tracing again')
            self.start_render_now()
        else:
            self.set_status(f'Working grid set to {grid_label(factor)}. Load an image to start.')
        self._schedule_view()

    def _suggested_name(self):
        stem = os.path.splitext(os.path.basename(self._src_path or 'shot'))[0]
        h, w = self._binary_mask.shape[:2]
        size = f'{w}' if w == h else f'{w}x{h}'
        return f'{stem}_fringes_{size}'

    def _ready_to_save(self, what):
        """True when the trace on screen is the current one; otherwise says why not."""
        if self.src_img is None:
            messagebox.showinfo(what, 'Load an image first', parent=self.winfo_toplevel())
            return False
        if self._result_is_current():
            return True
        if self._busy or self._pending is not None or self._debounce_id is not None:
            msg = 'Detection is still updating. Try again when the status line says "Rendered".'
        else:
            msg = 'There is no trace of this image to save (the last detection failed).'
        messagebox.showinfo(what, msg, parent=self.winfo_toplevel())
        return False

    def save_result(self):
        if not self._ready_to_save('Save Fringes as Binary'):
            return
        make_folders()
        start = self._last_save_dir if os.path.isdir(self._last_save_dir) else APP_DIR
        p = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title='Save fringes as binary',
                                         initialdir=start, initialfile=self._suggested_name() + '.png',
                                         defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        try:
            written = imwrite_png(p, self._binary_mask)
            self._last_save_dir = os.path.dirname(written)
            self._result_saved = True
            note = '' if written == p else ' (binaries are always saved as PNG)'
            self.set_status(f'Saved: {os.path.basename(written)}{note}')
        except Exception as e:
            messagebox.showerror('Save error', str(e), parent=self.winfo_toplevel())

    def _auto_save_path(self):
        os.makedirs(EDITED_DIR, exist_ok=True)
        stem = self._suggested_name()
        path = os.path.join(EDITED_DIR, stem + '.png')
        k = 2
        while os.path.exists(path):
            path = os.path.join(EDITED_DIR, f'{stem}_{k}.png')
            k += 1
        return path

    def send_to_editor(self):
        if not self._ready_to_save('Send to Editor'):
            return
        try:
            path = imwrite_png(self._auto_save_path(), self._binary_mask)
        except Exception as e:
            messagebox.showerror('Save error', str(e), parent=self.winfo_toplevel())
            return
        self._result_saved = True
        sent = False
        if self.send_to_editor_cb is not None:
            try:
                sent = self.send_to_editor_cb(self._binary_mask, self.src_img, path,
                                              os.path.basename(self._src_path or 'shot'), self._grid_factor)
            except Exception:
                self.winfo_toplevel().report_callback_exception(*sys.exc_info())
                sent = False
        if sent is False:
            self.set_status(f'Saved {os.path.basename(path)} (not sent to the Editor)')
        else:
            self.set_status(f'Saved {os.path.basename(path)} and sent to the Editor')

    # ---------------------------------------------------------- scheduling --
    def on_param_change(self, key=None):
        if self._suspend_param_cb or self._auto is not None:
            return
        # Debounce: ttk.Scale fires continuously while dragging. Wait until the
        # slider has been still briefly; stage-A sliders wait a little longer.
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except Exception:
                pass
        delay = 450 if key in self.STAGE_A_KEYS else 200
        self._debounce_id = self.after(delay, self.start_render_now)

    def start_render_now(self):
        self._debounce_id = None
        if self.src_img is None or self._det is None:
            self.set_status('No image loaded')
            return
        job = ('detect', self._params(), self._grid_factor, self._img_gen)
        if self._busy:
            if self._pending is None or self._pending[0] == 'detect':
                self._pending = job
            self.set_status('Updating (queued)...')
            return
        self._launch(job)

    def _launch(self, job):
        self._busy = True
        self._job_id += 1
        jid = self._job_id
        kind = job[0]
        try:
            if kind == 'auto':
                self.progress.stop()
                self.progress.config(mode='determinate', maximum=1000)
                self.progress['value'] = 0
            else:
                self.progress.config(mode='indeterminate')
                self.progress.start(12)
        except Exception:
            pass
        if kind == 'detect':
            try:
                self.btn_auto.state(['disabled'])
            except Exception:
                pass
        stop = self._auto['event'].is_set if (kind == 'auto' and self._auto is not None) else None
        threading.Thread(target=self._worker, args=(jid, job, self._det, stop), daemon=True).start()

    def _worker(self, jid, job, det, stop):
        """Runs off the main thread. Never touches Tk; talks through the queue."""
        kind, params, up, gen = job
        try:
            if kind == 'detect':
                prog = lambda msg: self._q.put(('status', jid, msg))
                det.p.update(params)
                b = det.detect(up=up, progress=prog)
                self._q.put(('result', jid, (kind, gen, up, params, b, None)))
            elif kind == 'auto':
                prog = lambda msg, frac=None: self._q.put(('auto_progress', jid, (msg, frac)))
                best, b, report = auto_optimize(det, up=up, progress=prog, should_stop=stop)
                self._q.put(('result', jid, (kind, gen, up, best, b, report)))
        except Cancelled:
            self._q.put(('cancelled', jid, (kind, gen)))
        except Exception as e:
            self._q.put(('error', jid, (kind, gen, '{}: {}\n\n{}'.format(type(e).__name__, e, traceback.format_exc()))))
        finally:
            self._q.put(('done', jid, kind))

    def _poll_queue(self):
        try:
            while True:
                kind, jid, payload = self._q.get_nowait()
                if kind == 'status':
                    if jid == self._job_id and self._auto is None:
                        self.set_status(payload)
                elif kind == 'auto_progress':
                    self._on_auto_progress(*payload)
                elif kind == 'result':
                    jkind, gen, up, params, b, report = payload
                    if gen != self._img_gen or up != self._grid_factor:
                        continue          # result for an image or grid that is no longer shown
                    if jkind == 'auto':
                        self._auto = None
                        self._set_params(params)
                        self._det.p.update(params)
                        self._set_result(b)
                        self.set_status(report)
                    else:
                        self._set_result(b)
                elif kind == 'cancelled':
                    jkind, gen = payload
                    if jkind == 'auto':
                        self._finish_auto_ui()
                        if self._det is not None:
                            self._det.p.update(self._params())
                        msg = (self._auto or {}).get('cancel_msg') or 'Auto-Optimize cancelled. The sliders were not changed.'
                        self._auto = None
                        self.set_status(msg)
                elif kind == 'error':
                    jkind, gen, traceback_text = payload
                    print(traceback_text)
                    path = log_error(traceback_text)
                    if jkind == 'auto':
                        # the sliders were never changed, so the trace on screen still matches them
                        if self._det is not None:
                            self._det.p.update(self._params())
                    elif gen == self._img_gen:
                        # the trace on screen no longer matches the sliders: it cannot be saved or sent
                        self._traced = None
                        self._binary_mask = None
                        self._result_key = None
                        self._schedule_view()
                    self._auto = None
                    self._finish_auto_ui()
                    where = f'\n\nThe details were saved to\n{path}' if path else ''
                    messagebox.showerror('Detection error', traceback_text.split('\n\n')[0] + where,
                                         parent=self.winfo_toplevel())
                    self.set_status('Detection failed (see the error message)')
                elif kind == 'done':
                    if jid != self._job_id:
                        continue
                    self._busy = False
                    try:
                        self.progress.stop()
                        self.progress.config(mode='determinate', maximum=100)
                        self.progress['value'] = 0
                    except Exception:
                        pass
                    if payload == 'auto' or self._auto is None:
                        self._finish_auto_ui()
                    nxt = self._pending
                    self._pending = None
                    if nxt is not None:
                        if nxt[0] == 'detect':
                            # always re-read the sliders: they may have moved again
                            nxt = ('detect', self._params(), self._grid_factor, self._img_gen)
                        elif nxt[0] == 'auto':
                            nxt = ('auto', None, self._grid_factor, self._img_gen)
                        self._launch(nxt)
        except queue.Empty:
            pass
        except Exception:
            self.winfo_toplevel().report_callback_exception(*sys.exc_info())
        try:
            self._poll_id = self.after(50, self._poll_queue)
        except Exception:
            pass

    def _set_result(self, b):
        self._traced = b
        self._binary_mask = np.where(b, 0, 255).astype(np.uint8)
        self._result_key = (self._img_gen, self._grid_factor)
        self._result_saved = False
        n = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)[0] - 1
        self.set_status(f'Rendered: {n} fringe segments' + self._load_note)
        self._schedule_view()

    # --------------------------------------------------------- auto-optimize --
    def auto_optimize(self):
        if self._auto is not None:
            self._cancel_auto()
            return
        if self.src_img is None or self._det is None:
            messagebox.showinfo('Auto-Optimize', 'Load an image first', parent=self.winfo_toplevel())
            return
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except Exception:
                pass
            self._debounce_id = None
        self._auto = {'event': threading.Event(), 't0': time.time(), 'cancel_msg': None}
        self._set_sliders_locked(True)
        try:
            self.btn_auto.config(text='Cancel Auto-Optimize')
            self.btn_auto.state(['!disabled'])
        except Exception:
            pass
        job = ('auto', None, self._grid_factor, self._img_gen)
        if self._busy:
            self._pending = job
            self.set_status('Auto-Optimize starts when the current update finishes...')
            return
        self.set_status('Auto-Optimize: estimating how long this takes...')
        self._launch(job)

    def _cancel_auto(self, msg=None):
        a = self._auto
        if a is None:
            return
        a['cancel_msg'] = msg
        a['event'].set()
        if self._pending is not None and self._pending[0] == 'auto':
            # it never started
            self._pending = None
            self._auto = None
            self._finish_auto_ui()
            self.set_status(msg or 'Auto-Optimize cancelled. The sliders were not changed.')
        else:
            self.set_status('Cancelling Auto-Optimize...')

    def _on_auto_progress(self, msg, frac):
        if self._auto is None:
            return
        self.set_status(msg)
        if frac is not None:
            try:
                self.progress['value'] = int(1000 * max(0.0, min(1.0, frac)))
            except Exception:
                pass

    def _finish_auto_ui(self):
        self._set_sliders_locked(False)
        try:
            self.btn_auto.config(text='Auto-Optimize')
            self.btn_auto.state(['!disabled'])
        except Exception:
            pass

    # ------------------------------------------------------------- viewer --
    def _on_brighten_change(self):
        if self.src_img is not None:
            self._make_display_image()
        self._view_cache = None
        self._schedule_view()

    def _image_size(self):
        if self.src_img is None:
            return None
        h, w = self.src_img.shape[:2]
        return w * self._grid_factor, h * self._grid_factor

    def _snap_view(self):
        """Keep the view offset on whole screen pixels, so 1 px lines are never
        split unevenly by the nearest-neighbor drawing."""
        z = self._zoom
        self._ox = round(self._ox * z) / z
        self._oy = round(self._oy * z) / z

    def fit_view(self, force=True):
        sz = self._image_size()
        if sz is None:
            return
        iw, ih = sz
        cw = self.fringe_canvas.winfo_width(); ch = self.fringe_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            if force:
                self.after(120, lambda: self.fit_view(force=False))
            return
        z = min(cw / float(iw), ch / float(ih))
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self._ox = (iw - cw / self._zoom) / 2.0
        self._oy = (ih - ch / self._zoom) / 2.0
        self._snap_view()
        self._schedule_view()

    def _on_motion(self, event):
        """Cursor position and the raw image value under it, for the bottom bar."""
        if self.src_img is None:
            self._cursor_cb('')
            return
        gx = self._ox + (event.x + 0.5) / self._zoom
        gy = self._oy + (event.y + 0.5) / self._zoom
        h, w = self.src_img.shape[:2]
        s = self._grid_factor
        if not (0 <= gx < w * s and 0 <= gy < h * s):
            self._cursor_cb('')
            return
        v = float(self.src_img[int(gy // s), int(gx // s)])
        val = f'{v:.0f}' if abs(v - round(v)) < 1e-6 else f'{v:.4g}'
        self._cursor_cb(f'{position_text(int(gx), int(gy), s)}   intensity {val}')

    def _on_wheel(self, event, steps):
        if self.src_img is None or not steps:
            return 'break'
        nz = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * (ZOOM_STEP ** steps)))
        # keep the image point under the cursor fixed
        ix = self._ox + event.x / self._zoom
        iy = self._oy + event.y / self._zoom
        self._zoom = nz
        self._ox = ix - event.x / nz
        self._oy = iy - event.y / nz
        self._snap_view()
        self._schedule_view(fast=True)
        return 'break'

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self._ox, self._oy)
        try:
            self.fringe_canvas.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return 'break'
        x0, y0, ox, oy = self._pan_start
        self._ox = ox - (event.x - x0) / self._zoom
        self._oy = oy - (event.y - y0) / self._zoom
        self._snap_view()
        self._schedule_view(fast=True)
        return 'break'

    def _on_pan_end(self, event):
        self._pan_start = None
        try:
            self.fringe_canvas.configure(cursor='')
        except Exception:
            pass
        return 'break'

    def _schedule_view(self, fast=False):
        if self._render_id is not None:
            return
        self._render_id = self.after(1 if fast else 10, self._render_view)

    @staticmethod
    def _affine(scale_src, start_src, ox, z, dst0):
        """Translation for warpAffine so that source pixel k (covering grid
        [(start_src + k) * scale_src, (start_src + k + 1) * scale_src)) lands
        on the right canvas pixels; dst0 = canvas coordinate of dst pixel 0."""
        a = scale_src * z
        return a, (start_src * scale_src + 0.5 * scale_src - ox) * z - dst0 - 0.5

    def _render_view(self):
        """Draw only the visible part of the image at screen resolution. The
        resampled image and lines are cached per view, so changing the opacity
        only re-blends."""
        self._render_id = None
        c = self.fringe_canvas
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        if self._disp_native is None:
            return
        s = self._grid_factor
        base = self._disp_native
        key = (cw, ch, s, self._zoom, self._ox, self._oy)
        hit = self._view_cache
        # the arrays themselves are kept in the cache, so a new image or trace can never look like the old one
        if hit is None or hit[0] != key or hit[1] is not base or hit[2] is not self._traced:
            hit = (key, base, self._traced, self._sample_view(base, cw, ch))
            self._view_cache = hit
        v = hit[3]
        out = np.zeros((ch, cw, 3), np.uint8)
        if v is not None:
            dx0, dy0, dx1, dy1, bg, ms = v
            op = float(self.bg_opacity.get())
            if op < 0.999:
                bg = cv2.addWeighted(bg, op, np.full_like(bg, 255), 1.0 - op, 0)
            tile = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            if ms is not None:
                tile[ms] = (255, 0, 0)
            out[dy0:dy1, dx0:dx1] = tile
        img = Image.fromarray(out)
        if self._photo is not None and (self._photo.width(), self._photo.height()) == (cw, ch) \
                and hasattr(self._photo, 'paste'):
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
        if self._img_item is None:
            self._img_item = c.create_image(0, 0, anchor='nw', image=self._photo)
        else:
            c.itemconfig(self._img_item, image=self._photo)

    def _sample_view(self, base, cw, ch):
        """Visible part of the image and of the traced lines, resampled to the
        canvas: (dx0, dy0, dx1, dy1, gray image, line mask or None), or None."""
        s = self._grid_factor
        H, W = base.shape[0] * s, base.shape[1] * s
        z = self._zoom
        x0 = max(0.0, self._ox); y0 = max(0.0, self._oy)
        x1 = min(float(W), self._ox + cw / z); y1 = min(float(H), self._oy + ch / z)
        if not (x1 > x0 and y1 > y0):
            return None
        dx0 = int(round((x0 - self._ox) * z)); dy0 = int(round((y0 - self._oy) * z))
        dx1 = min(cw, int(round((x1 - self._ox) * z))); dy1 = min(ch, int(round((y1 - self._oy) * z)))
        dw, dh = dx1 - dx0, dy1 - dy0
        if dw <= 0 or dh <= 0:
            return None
        nx0, ny0 = max(0, int(x0 // s) - 1), max(0, int(y0 // s) - 1)
        nx1, ny1 = min(base.shape[1], int(np.ceil(x1 / s)) + 1), min(base.shape[0], int(np.ceil(y1 / s)) + 1)
        crop = base[ny0:ny1, nx0:nx1]
        cz = z * s
        if cz < 1:
            # shrink with area averaging first, then place
            f = max(1, int(np.floor(1.0 / cz)))
            hh, ww = crop.shape[0] // f, crop.shape[1] // f
            if hh >= 1 and ww >= 1:
                red = cv2.resize(crop[:hh * f, :ww * f], (ww, hh), interpolation=cv2.INTER_AREA)
            else:
                red, f = crop, 1
            ax, tx = self._affine(s * f, nx0 / f, self._ox, z, dx0)
            ay, ty = self._affine(s * f, ny0 / f, self._oy, z, dy0)
            M = np.float32([[ax, 0, tx], [0, ay, ty]])
            bg = cv2.warpAffine(red, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            ax, tx = self._affine(s, nx0, self._ox, z, dx0)
            ay, ty = self._affine(s, ny0, self._oy, z, dy0)
            M = np.float32([[ax, 0, tx], [0, ay, ty]])
            interp = cv2.INTER_NEAREST if cz >= 4 else cv2.INTER_LINEAR
            bg = cv2.warpAffine(crop, M, (dw, dh), flags=interp, borderMode=cv2.BORDER_REPLICATE)
        ms = None
        tr = self._traced
        if tr is not None and tr.shape == (H, W):
            gx0, gy0 = max(0, int(x0) - 1), max(0, int(y0) - 1)
            gx1, gy1 = min(W, int(np.ceil(x1)) + 1), min(H, int(np.ceil(y1)) + 1)
            m = tr[gy0:gy1, gx0:gx1]
            if z < 1:
                # max-pool so 1 px lines stay visible when zoomed out
                f = max(1, int(np.floor(1.0 / z)))
                hh, ww = m.shape[0] // f, m.shape[1] // f
                if hh >= 1 and ww >= 1:
                    src = m[:hh * f, :ww * f].reshape(hh, f, ww, f).any(axis=(1, 3))
                else:
                    src, f = m, 1
                ax, tx = self._affine(f, gx0 / f, self._ox, z, dx0)
                ay, ty = self._affine(f, gy0 / f, self._oy, z, dy0)
            else:
                ax, tx = self._affine(1, gx0, self._ox, z, dx0)
                ay, ty = self._affine(1, gy0, self._oy, z, dy0)
                src = m
            M3 = np.float32([[ax, 0, tx], [0, ay, ty]])
            ms = cv2.warpAffine(np.ascontiguousarray(src).view(np.uint8), M3, (dw, dh), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return dx0, dy0, dx1, dy1, bg, ms
