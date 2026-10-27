"""Quick check after copying the updated files into the SpellBook folder.

Run it from the SpellBook folder (the one with SpellBook.py):

    python check_files.py

It compiles every .py file (catches copy and paste damage), imports the
modules SpellBook needs and checks they are at least the versions in
requirements.txt, and runs one small detection and one Editor link on a
synthetic image without opening a window. It ends with ALL OK.
"""
import compileall
import os
import re
import sys
import traceback

here = os.path.dirname(os.path.abspath(__file__))
app = here if os.path.exists(os.path.join(here, 'SpellBook.py')) else os.path.join(here, 'EditedFringeDetection')
sys.path.insert(0, app)
ok = True

print('Python', sys.version.split()[0], 'in', app)
if sys.version_info < (3, 10):
    print('FAILED: SpellBook needs Python 3.10 or newer')
    ok = False
# skip a virtual environment or git folder kept inside the SpellBook folder
skip = re.compile(r'(^|[\\/])(\.?venv[^\\/]*|env|site-packages|\.git)([\\/]|$)')
if not compileall.compile_dir(app, quiet=1, force=True, rx=skip):
    print('FAILED: a file does not compile (see above)')
    ok = False


def version_tuple(v):
    return tuple(int(x) for x in re.findall(r'\d+', str(v))[:3])


# minimum versions, read from requirements.txt
minimum = {}
try:
    with open(os.path.join(app, 'requirements.txt')) as f:
        for line in f:
            m = re.match(r'\s*([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)', line)
            if m:
                minimum[m.group(1).lower()] = m.group(2)
except OSError:
    print('  (requirements.txt not found, package versions not checked)')

for mod, pip_name in (('numpy', 'numpy'), ('cv2', 'opencv-python'), ('scipy', 'scipy'),
                      ('skimage', 'scikit-image'), ('PIL', 'Pillow')):
    try:
        m = __import__(mod)
    except Exception as e:
        print(f'FAILED: cannot import {mod} ({pip_name}): {e}')
        ok = False
        continue
    ver = getattr(m, '__version__', '?')
    need = minimum.get(pip_name.lower())
    print(f'  {mod:8s} {ver}')
    if need and version_tuple(ver) < version_tuple(need):
        print(f'FAILED: {pip_name} {ver} is too old, SpellBook needs {need} or newer. '
              f'Run: python -m pip install -r requirements.txt')
        ok = False
try:
    import tkinter
    print(f'  tkinter  Tk {tkinter.TkVersion}')
except Exception as e:
    print(f'FAILED: cannot import tkinter: {e}')
    ok = False

try:
    import numpy as np
    from fringe_detection.ridge_detector import RidgeDetector, DEFAULTS
    from tabs.fringe_editor import link_endpoints, upsample_lines_2x
    from tabs import detection_tab  # noqa: F401
    import SpellBook  # noqa: F401  (does not open a window)
    y, x = np.mgrid[0:256, 0:256]
    img = (1000 + 400 * np.cos(2 * np.pi * y / 8.0 + 0.002 * (x - 128) ** 2)).astype(np.float32)
    b = RidgeDetector(img, dict(DEFAULTS)).detect(up=1)
    print(f'  detection on a test pattern: {int(b.sum())} fringe pixels')
    b2 = RidgeDetector(img, dict(DEFAULTS)).detect(up=2)
    print(f'  detection at 2x: {b2.shape[1]} x {b2.shape[0]}')
    cut = b.copy()
    cut[:, 120:126] = False
    print(f'  Link Endpoints test: {len(link_endpoints(cut, 10, 40))} joins')
    if b.sum() < 1000:
        print('FAILED: the test pattern was not traced')
        ok = False
except Exception:
    traceback.print_exc()
    ok = False

print('ALL OK' if ok else 'PROBLEMS FOUND')
sys.exit(0 if ok else 1)
