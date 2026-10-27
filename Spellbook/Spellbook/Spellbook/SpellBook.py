import os
import sys
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from fringe_detection import VERSION as __version__


def _windows_dpi_awareness():
    # Without this Windows stretches the whole window on high-DPI screens and it looks blurry.
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.OleDLL('shcore').SetProcessDpiAwareness(1)
        except Exception:
            pass


_windows_dpi_awareness()

from tabs.detection_tab import DetectionTabFrame, grid_label, make_folders, APP_DIR  # noqa: E402
from tabs.fringe_editor import FringeEditorFrame  # noqa: E402
from fringe_detection.ui_helpers import ui_scale, help_icon, load_settings, save_settings, log_error  # noqa: E402

SETTINGS_FILE = os.path.join(APP_DIR, 'spellbook_settings.json')

GRID_HELP = ('1x traces on the image\'s own pixels. 2x traces on a grid twice as fine: the fringes come out '
             'more accurate because a line can sit between two image pixels, and tracing reaches further into '
             'regions where fringes are less than a pixel apart at 1x (near the target). 2x binaries are '
             'twice the size of the image. Both tabs always use the same grid.')


class SpellBookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f'SpellBook {__version__} — Fringe Detection')
        s = ui_scale(self)
        w = min(int(1100 * s), max(800, self.winfo_screenwidth() - 40))
        h = min(int(720 * s), max(560, self.winfo_screenheight() - 80))
        self.geometry(f'{w}x{h}')
        self.minsize(int(760 * min(s, 1.25)), int(480 * min(s, 1.25)))
        try:
            self.after(100, self.lift)
            self.after(120, lambda: self.attributes('-topmost', True))
            self.after(700, lambda: self.attributes('-topmost', False))
        except Exception:
            pass
        make_folders()
        settings = load_settings(SETTINGS_FILE)

        # --- Header strip: grid switch in the top-right corner ---
        # The switch is global because the Detection and Editor tabs must agree on a grid.
        header = ttk.Frame(self)
        header.pack(side='top', fill='x', padx=6, pady=(4, 0))
        ttk.Button(header, text='Help', command=self.open_help).pack(side='left')
        self.grid_factor_var = tk.IntVar(value=1)
        help_icon(header, GRID_HELP, side='left', pack_side='right')
        ttk.Radiobutton(header, text='2x', variable=self.grid_factor_var,
                        value=2, command=self._on_grid_change).pack(side='right', padx=(0, 4))
        ttk.Radiobutton(header, text='1x', variable=self.grid_factor_var,
                        value=1, command=self._on_grid_change).pack(side='right', padx=(0, 4))
        ttk.Label(header, text='Working grid:').pack(side='right', padx=(0, 6))
        self._grid_factor = 1

        # --- Bottom bar: current file on the left, status message, cursor readout on the right ---
        bottom = ttk.Frame(self)
        bottom.pack(fill='x', side='bottom')
        self.file_label = ttk.Label(bottom, text='No image loaded', anchor='w')
        self.file_label.pack(side='left', padx=(6, 0))
        ttk.Separator(bottom, orient='vertical').pack(side='left', fill='y', padx=8, pady=2)
        self.cursor_label = ttk.Label(bottom, text='', anchor='e')
        self.cursor_label.pack(side='right', padx=(8, 8))
        self.statusbar = ttk.Label(bottom, text='Ready', anchor='w')
        self.statusbar.pack(side='left', fill='x', expand=True)
        self._file_text = {}
        self._status_text = {}       # the latest message of each tab; the bar shows the current tab's

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Detection tab
        self._detection_frame = DetectionTabFrame(self.notebook, status_callback=lambda t: self._set_status('det', t),
                                                  file_callback=lambda t: self._set_file('det', t),
                                                  cursor_callback=self._set_cursor)
        self.notebook.add(self._detection_frame, text='Detection')

        # Editor tab
        self._editor_frame = FringeEditorFrame(self.notebook, file_callback=lambda t: self._set_file('edit', t),
                                               status_callback=lambda t: self._set_status('edit', t),
                                               grid_callback=self._sync_grid_from_editor,
                                               cursor_callback=self._set_cursor)
        self.notebook.add(self._editor_frame, text='Editor')
        self._detection_frame.send_to_editor_cb = self._send_to_editor
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: (self._show_file(), self._show_status(),
                                                                 self._set_cursor('')))
        self._set_file('det', self._detection_frame.current_file_text())
        self._set_file('edit', self._editor_frame.current_file_text())

        # settings remembered from the last session
        try:
            self._detection_frame.apply_settings(settings.get('detection', {}))
            self._editor_frame.apply_settings(settings.get('editor', {}))
            if settings.get('grid') == 2:
                self.grid_factor_var.set(2)
                self._on_grid_change()
        except Exception:
            # a damaged settings file only costs the remembered settings
            text = traceback.format_exc()
            sys.stderr.write(text)
            log_error('Could not restore the remembered settings:\n' + text)

        self.protocol('WM_DELETE_WINDOW', self.on_close)

    # ------------------------------------------------------------- errors ---
    def report_callback_exception(self, exc, val, tb):
        """Anything unexpected: say so in a dialog and keep the details in
        spellbook_errors.log, since there is often no console to show them."""
        text = ''.join(traceback.format_exception(exc, val, tb))
        sys.stderr.write(text)
        path = log_error(text)
        where = f'\n\nThe details were saved to\n{path}\nPlease send that file with a note of what you did.' if path else ''
        try:
            messagebox.showerror('SpellBook error', f'Something went wrong: {exc.__name__}: {val}{where}', parent=self)
        except Exception:
            pass

    # ------------------------------------------------------------ bottom bar --
    def _set_file(self, which, text):
        self._file_text[which] = text
        self._show_file()

    def _current_tab(self):
        try:
            return 'edit' if self.notebook.index(self.notebook.select()) == 1 else 'det'
        except Exception:
            return 'det'

    def _show_file(self):
        self.file_label.config(text=self._file_text.get(self._current_tab(), ''))

    def _set_status(self, which, txt):
        self._status_text[which] = txt
        self._show_status()

    def _show_status(self):
        try:
            self.statusbar.config(text=self._status_text.get(self._current_tab(), ''))
        except Exception:
            pass

    def set_status(self, txt):
        """A message about both tabs (the working grid)."""
        for which in ('det', 'edit'):
            self._status_text[which] = txt
        self._show_status()

    def open_help(self):
        path = Path(APP_DIR) / 'help.html'
        if not path.exists():
            messagebox.showinfo('Help', 'help.html is missing from the SpellBook folder.', parent=self)
            return
        try:
            opened = webbrowser.open(path.resolve().as_uri())
        except Exception:
            opened = False
        if not opened:
            messagebox.showinfo('Help', f'Could not open a web browser. The manual is\n{path.resolve()}',
                                parent=self)

    def _set_cursor(self, txt):
        try:
            self.cursor_label.config(text=txt)
        except Exception:
            pass

    # ---------------------------------------------------------------- grid ---
    def _send_to_editor(self, binary, shot_img, saved_path, name, factor):
        ok = self._editor_frame.receive_from_detection(binary, shot_img, saved_path, name, factor)
        if ok:
            self.notebook.select(self._editor_frame)
        return ok

    def _on_grid_change(self):
        """Apply the grid switch to both tabs, keeping them in lockstep."""
        factor = int(self.grid_factor_var.get())
        if factor == self._grid_factor:
            return
        # The Editor converts its layers first; it asks before anything is lost and
        # can refuse, in which case both tabs stay where they were.
        try:
            ok = self._editor_frame.set_grid_factor(factor)
        except Exception:
            self.report_callback_exception(*sys.exc_info())
            ok = False
        if ok is False:
            self.grid_factor_var.set(self._grid_factor)
            return
        self._apply_grid(factor)

    def _sync_grid_from_editor(self, factor):
        """The Editor changed grid on its own (undo or redo of a grid switch, or a
        binary that is twice the size of the shot image): follow it."""
        factor = int(factor)
        self.grid_factor_var.set(factor)
        if factor != self._grid_factor:
            self._apply_grid(factor)

    def _apply_grid(self, factor):
        self._grid_factor = factor
        try:
            self._detection_frame.set_grid_factor(factor)
        except Exception:
            self.report_callback_exception(*sys.exc_info())
        self.set_status(f'Working grid: {grid_label(factor)}')

    # --------------------------------------------------------------- close ---
    def save_settings(self):
        d = {'version': __version__, 'grid': int(self._grid_factor)}
        try:
            d['detection'] = self._detection_frame.get_settings()
        except Exception:
            pass
        try:
            d['editor'] = self._editor_frame.get_settings()
        except Exception:
            pass
        save_settings(SETTINGS_FILE, d)

    def on_close(self):
        lines = []
        try:
            lines += self._editor_frame.unsaved_summary()
        except Exception:
            pass
        try:
            if self._detection_frame.unsaved_result():
                lines.append('Detection: the fringes traced on ' + self._detection_frame.current_file_text().split('  (')[0]
                             + ' have not been saved or sent to the Editor')
        except Exception:
            pass
        if lines:
            msg = ('Are you sure you want to close SpellBook? This progress will be lost:\n\n'
                   + '\n'.join('• ' + l for l in lines))
            if not messagebox.askyesno('Close SpellBook?', msg, icon='warning', parent=self):
                return
        self.save_settings()
        try:
            self.destroy()
        except Exception:
            pass


def main():
    try:
        app = SpellBookApp()
    except Exception:
        text = traceback.format_exc()
        sys.stderr.write(text)
        path = log_error(text)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror('SpellBook could not start', text.strip().splitlines()[-1]
                                 + (f'\n\nThe details were saved to\n{path}' if path else ''), parent=root)
            root.destroy()
        except Exception:
            pass
        raise SystemExit(1)
    app.mainloop()


if __name__ == '__main__':
    main()
