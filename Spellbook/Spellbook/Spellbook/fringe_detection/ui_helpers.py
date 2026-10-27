"""Small Tk helpers shared by the Detection and Editor tabs.

Everything here is about behaving the same on Windows, macOS and Linux:
mouse buttons, wheel and trackpad deltas, the Command key on macOS, theme
colors and fonts, and reading or writing images whose path contains
non-ASCII characters (OpenCV's own imread/imwrite cannot on Windows).
"""
import datetime
import json
import os
import sys
import tempfile
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import cv2
import numpy as np

from . import VERSION

IS_MAC = sys.platform == 'darwin'
IS_WIN = sys.platform.startswith('win')

# Label used in help text for the main shortcut modifier.
MOD = 'Cmd' if IS_MAC else 'Ctrl'

SHIFT = 0x0001
CONTROL = 0x0004
COMMAND = 0x0008      # Mod1, which Tk uses for the Command key on macOS


def tk_version(widget):
    try:
        return float(str(widget.tk.call('info', 'patchlevel'))[:3])
    except Exception:
        return 8.6


def mod_pressed(event):
    """True when the shortcut modifier is held: Ctrl everywhere, and also
    Command on macOS."""
    st = int(getattr(event, 'state', 0) or 0)
    if st & CONTROL:
        return True
    return IS_MAC and bool(st & COMMAND)


def shift_pressed(event):
    return bool(int(getattr(event, 'state', 0) or 0) & SHIFT)


# ------------------------------------------------------------------ wheel ---
def wheel_steps(event, widget=None):
    """Mouse wheel or trackpad scroll as a number of 'notches' (positive = up).

    Windows reports 120 per notch (touchpads less). Tk 8.6 on macOS reports
    about 1 per notch and many small events for a trackpad. Tk 8.6 on Linux
    sends Button-4/5. Tk 8.7 and 9 report 120 per notch everywhere.
    """
    num = getattr(event, 'num', None)
    d = getattr(event, 'delta', 0)
    try:
        d = int(d)
    except Exception:
        d = 0
    if num in (4, 5) and d == 0:
        return 1.0 if num == 4 else -1.0
    if d == 0:
        return 0.0
    if IS_MAC and widget is not None and tk_version(widget) < 8.7:
        steps = d / 3.0
    else:
        steps = d / 120.0
    return max(-4.0, min(4.0, steps))


def touchpad_dy(event):
    """Vertical part of a Tk 9 <TouchpadScroll> event, in pixels (positive = up)."""
    try:
        d = int(event.delta)
    except Exception:
        return 0
    dy = d & 0xFFFF
    if dy >= 0x8000:
        dy -= 0x10000
    return dy


def bind_wheel(widget, callback):
    """Bind wheel and trackpad scrolling on every platform and Tk version.
    callback(event, steps) gets the scroll in notches (positive = up)."""
    widget.bind('<MouseWheel>', lambda e: callback(e, wheel_steps(e, widget)), add='+')
    widget.bind('<Button-4>', lambda e: callback(e, 1.0), add='+')
    widget.bind('<Button-5>', lambda e: callback(e, -1.0), add='+')
    try:
        # Tk 9 on macOS and Windows: two finger trackpad scrolling
        acc = {'dy': 0.0}

        def tp(e):
            acc['dy'] += touchpad_dy(e)
            steps = acc['dy'] / 40.0
            if abs(steps) >= 0.25:
                acc['dy'] = 0.0
                return callback(e, max(-4.0, min(4.0, steps)))
            return 'break'
        widget.bind('<TouchpadScroll>', tp, add='+')
    except tk.TclError:
        pass


PAN_BUTTONS = (2, 3)   # middle and right button on every platform (macOS swaps their numbers)


def bind_pan(widget, start, move, end, buttons=PAN_BUTTONS):
    for b in buttons:
        widget.bind(f'<ButtonPress-{b}>', start, add='+')
        widget.bind(f'<B{b}-Motion>', move, add='+')
        widget.bind(f'<ButtonRelease-{b}>', end, add='+')


# ------------------------------------------------------------- theme bits ---
def theme_colors(widget):
    """(background, foreground) of the current ttk theme, for plain Tk widgets."""
    st = ttk.Style(widget)
    bg = st.lookup('TFrame', 'background') or st.lookup('.', 'background')
    fg = st.lookup('TLabel', 'foreground') or st.lookup('.', 'foreground')
    try:
        widget.winfo_rgb(bg)
    except Exception:
        bg = widget.winfo_toplevel().cget('background')
    try:
        widget.winfo_rgb(fg)
    except Exception:
        fg = 'black'
    return bg, fg


def bold_font(widget, delta=0):
    f = tkfont.nametofont('TkDefaultFont', root=widget).copy()
    f.configure(weight='bold')
    if delta:
        f.configure(size=f.cget('size') + (delta if f.cget('size') > 0 else -delta))
    return f


def fixed_font(widget):
    return tkfont.nametofont('TkFixedFont', root=widget)


def ui_scale(widget):
    """1.0 at 96 dpi, 1.5 at 144 dpi, and so on (pixel sizes are scaled by this)."""
    try:
        return max(1.0, float(widget.tk.call('tk', 'scaling')) / (96.0 / 72.0))
    except Exception:
        return 1.0


def attach_tooltip(widget, text, side='right', wraplength=240):
    tip = {'win': None}

    def show_tip(_e=None):
        if tip['win'] is not None:
            return
        win = tk.Toplevel(widget)
        tip['win'] = win
        try:
            win.wm_overrideredirect(True)
        except Exception:
            pass
        frame = ttk.Frame(win, borderwidth=1, relief='solid')
        frame.pack()
        ttk.Label(frame, text=text, justify='left', padding=6,
                  wraplength=int(wraplength * ui_scale(widget))).pack()
        try:
            win.update_idletasks()
            wrx = widget.winfo_rootx(); wry = widget.winfo_rooty()
            ww = widget.winfo_width(); wh = widget.winfo_height()
            win_w = win.winfo_width() or win.winfo_reqwidth()
            win_h = win.winfo_height() or win.winfo_reqheight()
            left = str(side).lower() == 'left'
            x = int(wrx - 12 - win_w) if left else int(wrx + ww + 12)
            y = int(wry + wh * 0.5 - win_h * 0.5)
            sw, sh = widget.winfo_screenwidth(), widget.winfo_screenheight()
            if 0 <= wrx < sw and 0 <= wry < sh:
                # keep it on the screen (only checked on the main screen, whose size Tk knows)
                if not left and x + win_w > sw:
                    x = int(wrx - 12 - win_w)
                elif left and x < 0:
                    x = int(wrx + ww + 12)
                x = min(x, sw - win_w)
                y = min(y, sh - win_h)
            win.wm_geometry(f'+{max(0, x)}+{max(0, y)}')
        except Exception:
            pass

    def hide_tip(_e=None):
        w = tip.get('win')
        if w is not None:
            try:
                w.destroy()
            except Exception:
                pass
            tip['win'] = None

    widget.bind('<Enter>', show_tip, add='+')
    widget.bind('<Leave>', hide_tip, add='+')
    widget.bind('<ButtonPress>', hide_tip, add='+')


def help_icon(parent, text, side='left', pack_side='right'):
    """Small round '?' that shows text on hover, drawn in the theme's colors."""
    bg, fg = theme_colors(parent)
    s = ui_scale(parent)
    font = tkfont.nametofont('TkDefaultFont', root=parent)
    n = max(int(round(18 * s)), int(font.metrics('linespace')) + 4)
    c = tk.Canvas(parent, width=n, height=n, highlightthickness=0, bg=bg, bd=0)
    m = max(1, int(round(n / 9.0)))
    c.create_oval(m, m, n - m, n - m, outline=fg, width=1)
    c.create_text(n / 2.0, n / 2.0, text='?', fill=fg, font=font)
    c.pack(side=pack_side, padx=(0, 4))
    attach_tooltip(c, text, side=side)
    return c


def scrollable_panel(parent, width):
    """Fixed-width column with a vertical scrollbar that appears when the
    contents are taller than the window. Wheel and trackpad scrolling work
    anywhere over the panel. Returns the inner frame to fill."""
    s = ui_scale(parent)
    container = ttk.Frame(parent, width=int(width * s))
    container.pack_propagate(False)
    bg, _fg = theme_colors(parent)
    canvas = tk.Canvas(container, highlightthickness=0, bg=bg, bd=0)
    sb = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    inner = ttk.Frame(canvas)
    win = canvas.create_window((0, 0), window=inner, anchor='nw')
    state = {'shown': None}

    def refresh(_e=None):
        canvas.configure(scrollregion=(0, 0, inner.winfo_reqwidth(), inner.winfo_reqheight()))
        need = inner.winfo_reqheight() > canvas.winfo_height() + 1
        if need != state['shown']:
            state['shown'] = need
            canvas.pack_forget(); sb.pack_forget()
            if need:
                sb.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
        if not need:
            canvas.yview_moveto(0)

    inner.bind('<Configure>', lambda e: (refresh(), fit_width()))
    canvas.bind('<Configure>', lambda e: (canvas.itemconfig(win, width=e.width), refresh()))
    canvas.pack(side='left', fill='both', expand=True)

    def on_wheel(_e, steps):
        if state['shown']:
            canvas.yview_scroll(int(-round(steps * 2)) or (-1 if steps > 0 else 1), 'units')
        return 'break'

    def bind_tree(w):
        # the wheel over any widget in the panel, a slider included, scrolls the
        # panel (on_wheel returns 'break', so a slider never changes by accident)
        bind_wheel(w, on_wheel)
        for ch in w.winfo_children():
            bind_tree(ch)

    def fit_width():
        # never cut off a row: widen the column if its contents ask for more
        # (macOS buttons and fonts are larger than on Windows)
        try:
            need = inner.winfo_reqwidth() + max(16, sb.winfo_reqwidth()) + 2
            if need > container.winfo_reqwidth():
                container.configure(width=need)
        except tk.TclError:
            pass

    bind_wheel(canvas, on_wheel)
    container.after_idle(lambda: bind_tree(inner))
    container.rebind_wheel = lambda: bind_tree(inner)
    return container, inner


# ---------------------------------------------------------------- sliders ---
def _decimals(res):
    s = ('%.10f' % res).rstrip('0')
    return len(s.split('.')[1]) if '.' in s else 0


def make_slider_row(parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None,
                    command=None):
    """Label above a slider with its value on the right. The value snaps to
    `resolution` (what is shown is exactly what is used), and the arrow keys
    move a focused slider by one step. Returns the ttk.Scale."""
    if command is None:
        def command(_=None):
            return
    if resolution is None:
        resolution = 1 if is_int else 0.01
    if fmt is None:
        fmt = '{:.%df}' % _decimals(resolution) if not is_int else '{:d}'
    ttk.Label(parent, text=label_text).pack(anchor='w')
    row = ttk.Frame(parent)
    row.pack(fill='x')
    return _slider_widgets(row, var, frm, to, resolution, is_int, fmt, command)


def make_inline_slider(row, var, frm, to, resolution=0.01, fmt='{:.2f}', command=None, type_max=None):
    """Slider plus value label packed into an existing row (Editor layout).
    type_max: a typed value may go past the end of the slider, up to this."""
    if command is None:
        def command(_=None):
            return
    return _slider_widgets(row, var, frm, to, resolution, False, fmt, command, val_width=5, type_max=type_max)


def numeric_validator(widget, allow_float=True, allow_negative=False):
    """validatecommand that lets only a number be typed into an Entry or Spinbox."""
    def ok(text):
        if text in ('', '.', '-', '-.'):
            return text in ('', '.') or allow_negative
        if text.startswith('-'):
            if not allow_negative:
                return False
            text = text[1:]
        if not allow_float:
            return text.isdigit()
        parts = text.split('.')
        return len(parts) <= 2 and all(p.isdigit() or p == '' for p in parts)
    return (widget.register(ok), '%P')


def _slider_widgets(row, var, frm, to, resolution, is_int, fmt, command, val_width=6, type_max=None):
    state = {'busy': False}

    def snap(v):
        lo, hi = float(scale.cget('from')), float(scale.cget('to'))
        v = min(max(float(v), min(lo, hi)), max(lo, hi))
        v = round(round(v / resolution) * resolution, 10)
        return int(round(v)) if is_int else v

    def on_move(value):
        if state['busy']:
            return
        state['busy'] = True
        try:
            v = snap(value)
            try:
                cur = var.get()
            except Exception:
                cur = None
            if cur != v:
                var.set(v)
        finally:
            state['busy'] = False
        command(value)

    scale = ttk.Scale(row, from_=frm, to=to, variable=var, command=on_move)
    scale.pack(side='left', fill='x', expand=True)
    scale.snap = snap
    scale.resolution = resolution

    def step(sign):
        try:
            v = snap(float(var.get()) + sign * resolution)
        except Exception:
            return 'break'
        try:
            if 'disabled' in scale.state():
                return 'break'
        except Exception:
            pass
        var.set(v)
        command(v)
        return 'break'

    for key, sign in (('<Left>', -1), ('<Down>', -1), ('<Right>', 1), ('<Up>', 1)):
        scale.bind(key, lambda e, s=sign: step(s))
    val_var = tk.StringVar()

    def _update_val(*_a):
        try:
            v = var.get()
            val_var.set(fmt.format(int(round(v))) if is_int else fmt.format(float(v)))
        except Exception:
            val_var.set('')

    _update_val()
    var.trace_add('write', lambda *a: _update_val())
    lbl = ttk.Label(row, textvariable=val_var, width=val_width, anchor='e', cursor='xterm')
    lbl.pack(side='left', padx=(6, 0))

    def start_edit(_e=None):
        # click the value to type one; Enter or clicking elsewhere applies it, Esc cancels
        try:
            if 'disabled' in scale.state():
                return
        except Exception:
            pass
        if state.get('entry') is not None:
            return
        ent = ttk.Entry(row, width=val_width, justify='right', validate='key',
                        validatecommand=numeric_validator(row, allow_float=not is_int,
                                                          allow_negative=min(frm, to) < 0))
        state['entry'] = ent
        ent.insert(0, val_var.get())
        ent.select_range(0, 'end')
        ent.place(in_=lbl, x=0, y=0, relwidth=1, relheight=1)
        ent.focus_set()

        def finish(apply):
            if state.get('entry') is not ent:
                return 'break'
            state['entry'] = None
            text = ent.get().strip()
            try:
                ent.destroy()
            except Exception:
                pass
            try:
                row.winfo_toplevel().focus_set()
            except Exception:
                pass
            if not apply or text in ('', '.', '-', '-.'):
                return 'break'
            try:
                v = float(text)
                if type_max is not None and v > float(scale.cget('to')):
                    # past the end of the slider, up to type_max
                    v = round(round(min(v, float(type_max)) / resolution) * resolution, 10)
                else:
                    v = snap(v)
            except ValueError:
                row.bell()
                return 'break'
            if v != var.get():
                var.set(v)
                command(v)
            return 'break'
        ent.bind('<Return>', lambda e: finish(True))
        ent.bind('<KP_Enter>', lambda e: finish(True))
        ent.bind('<Escape>', lambda e: finish(False))
        ent.bind('<FocusOut>', lambda e: finish(True))
        return 'break'

    lbl.bind('<Button-1>', start_edit)
    attach_tooltip(lbl, 'Click to type a value')
    scale.value_label = lbl
    return scale


def overlay_button(canvas, text, command, corner='ne', pad=8):
    """Small button that stays in a corner of a canvas (above the image)."""
    btn = ttk.Button(canvas, text=text, command=command, width=max(4, len(text) + 1))
    item = canvas.create_window(0, 0, window=btn, anchor=corner)

    def place(_e=None):
        w, h = canvas.winfo_width(), canvas.winfo_height()
        x = w - pad if 'e' in corner else pad
        y = pad if 'n' in corner else h - pad
        canvas.coords(item, x, y)
        canvas.tag_raise(item)
    canvas.bind('<Configure>', place, add='+')
    canvas.after_idle(place)
    btn.place_overlay = place
    return btn


class CanvasPrompt:
    """Message and buttons in the middle of an empty canvas, showing the
    first thing to do. show()/hide()."""

    def __init__(self, canvas, text, buttons):
        self.canvas = canvas
        frm = ttk.Frame(canvas, padding=12)
        ttk.Label(frm, text=text, justify='center', anchor='center',
                  wraplength=int(360 * ui_scale(canvas))).pack(pady=(0, 8))
        row = ttk.Frame(frm)
        row.pack()
        self.buttons = []
        for label, cmd in buttons:
            b = ttk.Button(row, text=label, command=cmd)
            b.pack(side='left', padx=4)
            self.buttons.append(b)
        self.frame = frm
        self.item = None
        canvas.bind('<Configure>', lambda e: self._place(), add='+')

    def _place(self):
        if self.item is not None:
            self.canvas.coords(self.item, self.canvas.winfo_width() / 2.0, self.canvas.winfo_height() / 2.0)

    def show(self):
        if self.item is None:
            self.item = self.canvas.create_window(0, 0, window=self.frame, anchor='center')
        self._place()

    def hide(self):
        if self.item is not None:
            self.canvas.delete(self.item)
            self.item = None

    @property
    def shown(self):
        return self.item is not None


# ------------------------------------------------------------- settings ---
ERROR_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spellbook_errors.log')


def log_error(text):
    """Append an error report to spellbook_errors.log in the SpellBook folder (or
    in the temp folder when that one cannot be written). Returns the file used,
    or None."""
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = (f'--- {stamp}  SpellBook {VERSION}, Python {sys.version.split()[0]}, {sys.platform}\n'
             f'{text.rstrip()}\n\n')
    for path in (ERROR_LOG, os.path.join(tempfile.gettempdir(), 'spellbook_errors.log')):
        try:
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(entry)
            return path
        except Exception:
            continue
    return None


def load_settings(path):
    """Settings saved by an earlier session (a small JSON file), or {}."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(path, d):
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        return True
    except Exception:
        return False


# ------------------------------------------------------------- file input ---
def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread that also works for paths with non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_png(path, img):
    """Write img as PNG (lossless). Returns the path actually written, which
    always ends in .png."""
    root, ext = os.path.splitext(path)
    if ext.lower() != '.png':
        path = root + '.png' if ext.lower() in ('.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp') \
            else path + '.png'
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    buf.tofile(path)
    return path


def position_text(gx, gy, factor=1):
    """Cursor position for the status bar, from working-grid pixel (gx, gy).
    Positions are in pixels of the image on both grids, like every other length
    SpellBook shows; at 2x the position on the binary is added."""
    if factor <= 1:
        return f'x {gx}  y {gy}'

    def img(v):
        x = v / factor
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
    return f'x {img(gx)}  y {img(gy)}  (binary {gx}, {gy})'


def image_px(v, factor=1, digits=1):
    """A length or position on the working grid, in pixels of the image."""
    x = v / float(max(1, factor))
    return f'{x:.0f}' if abs(x - round(x)) < 1e-9 else f'{x:.{digits}f}'


def image_filetypes(*exts):
    """File dialog patterns; adds upper-case variants on Linux, where the Tk
    dialog matches case-sensitively."""
    pats = ['*' + e for e in exts]
    if sys.platform.startswith('linux'):
        pats += ['*' + e.upper() for e in exts]
    return [('Images', tuple(pats)), ('All files', '*')]


def sanitize_image(img):
    """Gray float32 image with NaN and infinite pixels replaced by the median of
    the finite ones. Returns (image, number_of_replaced_pixels)."""
    # a float32 copy first, which cvtColor also takes when the file holds 32 bit integers or float64
    a = np.asarray(img).astype(np.float32)
    if a.ndim == 3:
        a = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY) if a.shape[2] >= 3 else a[..., 0]
    bad = ~np.isfinite(a)
    n = int(bad.sum())
    if n:
        good = a[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        a = a.copy()
        a[bad] = fill
    return a, n
