# SpellBook

SpellBook traces the bright fringes of laser-plasma interferograms and lets you correct the trace by hand. The result is a binary PNG with one 1 px line per bright fringe, ready for fringe numbering and phase extraction (for example with Magic).

It has two tabs:

* **Detection** finds the fringe center lines automatically. Auto-Optimize picks the detection settings for each shot.
* **Editor** is for fixing what detection could not: drawing missing fringes, deleting stray pieces, joining broken fringes, and masking out areas without fringes.

Version 2.0.0. See `CHANGELOG.md` for what changed.

## Install

You need Python 3.10 or newer. The installers from python.org include Tkinter, which SpellBook uses for its window.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_files.py
python SpellBook.py
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_files.py
python SpellBook.py
```

`check_files.py` is a quick self test: it checks that every file is intact and that every package imports and is new enough, then traces a small test pattern without opening a window. It ends with `ALL OK`. If it says a package is too old, run `python -m pip install -r requirements.txt` again.

On Linux, install Tkinter with your package manager if it is missing (for example `sudo apt install python3-tk`).

## Folders

| Folder or file | What it is for |
|---|---|
| `InputImages/` | A place for the shots you want to trace. Load dialogs start here. |
| `EditedImages/` | Binaries written by SpellBook. Send to Editor saves here automatically, and Save dialogs start here. You can save anywhere else from the dialog. |
| `spellbook_settings.json` | Created when you close SpellBook. Remembers the detection sliders, the working grid, Editor tool sizes and the last folders. Delete it to go back to the defaults. |
| `spellbook_errors.log` | Only created if something goes wrong: SpellBook shows a message and saves the details here. Send it along when you report a problem. |

## Typical workflow

1. **Detection tab:** click **Load Image** and pick a shot (TIFF, PNG, JPEG or BMP, any bit depth). The whole image is traced right away and the fringes are drawn as red lines.
2. Click **Auto-Optimize**. It tries the detection settings over their useful range and sets the sliders to the best combination for this shot (see below). You can fine tune the sliders afterwards.
3. Click **Send to Editor**. The trace is saved to `EditedImages/` and opens in the Editor with the shot underneath.
4. **Editor tab:** fix the trace, mask out areas without fringes with the Mask Pen, and save with **Save > Shot Binary**.

## Working grid: 1x or 2x

The switch at the top right sets the grid for both tabs.

* **1x** traces on the image's own pixels. A 2048 x 2048 shot gives a 2048 x 2048 binary.
* **2x** traces on a grid twice as fine (a 2048 x 2048 shot gives a 4096 x 4096 binary). Fringes come out more accurate because a line can sit between two image pixels, and tracing reaches further into regions where fringes are less than a pixel apart at 1x, such as near the target.

Every length and position SpellBook shows or asks for is in pixels of the image (camera pixels), on both grids: the Detection sliders, the cursor readout, the Ruler, Mirror at x, Link Endpoints' Max gap and the nudge offset. So a number means the same thing at 1x and 2x. At 2x the cursor readout also gives the position on the binary. Only the brush sizes are in pixels of the binary, so size 0.5 always draws a 1 px line.

## Detection tab

| Control | What it does |
|---|---|
| Load Image | Opens a shot and traces it. The view resets to show the whole image. |
| Save Fringes as Binary | Saves the current trace as a PNG wherever you choose. |
| Send to Editor | Saves the trace to `EditedImages/` (named after the shot, never overwriting an older file) and opens it in the Editor with the shot image underneath, already aligned. |
| Auto-Optimize | Finds the best detection settings for this shot (see below). While it runs the button reads Cancel Auto-Optimize, the status line counts down, and the sliders are locked. |
| Brighten Image | Stretches the display contrast so a few hot pixels cannot make the image dark. Display only. |
| Image Opacity | Fades the image toward white so the red lines stand out. Display only. |

Detection settings (click a slider's number to type a value):

| Slider | What it does |
|---|---|
| Sensitivity | How faint a fringe can be and still be traced. Raise it to pick up more fringes near the target; lower it if noise starts being traced. |
| Smoothing Along Fringes (px) | Averages the image along each fringe, never across it, to suppress speckle. Higher gives straighter, less broken lines in noisy areas; too high can make lines drift across dense, curved fringes. It is reduced automatically where fringes are close together. |
| Max Fringe Angle° | Steepest fringe tilt from horizontal that is traced. 0 keeps only horizontal fringes; 90 keeps every angle up to vertical. Structures steeper than the setting, such as vertical streaks at the target edge, are filtered out. |
| Gap Bridging (px) | Largest break in a fringe that is joined automatically. A gap is only bridged if both ends line up along the fringe and the bridge does not cross a dark fringe. |
| Minimum Fringe Length (px) | Pieces shorter than this are deleted. |
| Noise Rejection | Removes small fringes where the fringe pattern is weak or disordered (shadows, the dark band above the plume). The weaker the pattern around a piece, the longer it has to be to stay, so high settings also remove long pieces there. 0 keeps everything. |

Smoothing and Max Fringe Angle rerun the full analysis (a few seconds for a 2048 x 2048 shot). The other sliders only redo the clean up and update in about a second.

### How Auto-Optimize works

Auto-Optimize traces the shot many times with different settings and keeps the trace that would need the least hand editing.

1. It starts at Smoothing 9 px and Max Fringe Angle 60 (90 when most of the fringes are steeper than 60 degrees).
2. It tries every value of Sensitivity, Noise Rejection, Minimum Fringe Length and Gap Bridging over the whole useful range, one slider at a time, and keeps each improvement.
3. It steps Max Fringe Angle up (70, then 85) while the trace improves, or down to 55 when the first step up does not help (from a start at 90 it steps down: 85, 70 and so on), and then fine tunes the other sliders again.
4. It steps Smoothing up (12, 15 px) or down (6, 3 px) the same way and fine tunes again.
5. At 2x the search runs on the 1x grid, which is about four times faster, and the three best settings are compared again at 2x.

Every trace gets a score for how much hand editing it saves. Each pixel of real fringe counts for it. Broken ends inside the fringe area and lines squeezed between two fringes (which is where fringes merge) count against it. Stray pieces count only a little, because they are quick to delete, and least near the edge of the fringe area. There is also a small preference for less smoothing, because heavy smoothing can make lines drift across dense fringes. Whether a piece is a real fringe is judged from how orderly the fringe pattern is around it, measured the same way for every setting. The weights of the score were tuned on hand-edited shots, so that the settings it picks need the least editing.

While it runs, the status line counts down, the button reads Cancel Auto-Optimize, and the sliders are locked. At the end the sliders show the result, so you can fine tune from there. The result is the same whatever the sliders were before.

## Editor tab

The Editor shows the shot binary in red over the shot image. A reference binary (cyan) and a reference image can be loaded underneath to compare with.

**Files**

* **Open:** Shot Binary, Reference Binary, Shot Image, Reference Image, Gray Mask. Images must come from the same camera as the binary: a shot image is the binary's size at 1x, or half of it at 2x.
* **Save:** Shot Binary (with its gray mask), Reference Binary, Gray Mask. Binaries are always saved as PNG. A gray mask file uses the same values as a binary: 128 = masked, 255 = not masked (gray mask files from SpellBook 1, with 255 = masked, still open).

**Editing** picks which binary the tools change: the shot binary or the reference binary. Mask Pen always edits the gray mask of the shot.

**Tools** (shortcut letter in brackets)

| Tool | Use |
|---|---|
| Pen (P) | Draw fringe pixels. Size 0.5 draws a 1 px line. |
| Eraser (E) | Remove fringe pixels. |
| Line (L) | Straight 1 px line: click and drag, or click the start and then the end. With Polyline on, every click adds a corner and Enter or Ctrl+click (Cmd+click on a Mac) finishes. |
| Spline (S) | Smooth 1 px curve through your clicks. Ctrl+click (Cmd+click) adds the last point and finishes; Enter also finishes. |
| Delete Fringe (D) | Click exactly on a line to delete that whole connected piece. The piece under the cursor is highlighted in yellow first. |
| Mask Pen (M) | Mark out areas without fringes to isolate the proper region. Masking removes the fringes underneath. Hold Shift while painting to remove mask. |
| Ruler (R) | Measure a distance in image pixels. Lock keeps it horizontal or vertical. |
| Connection Check (K) | Drag a line across fringes: every fringe it touches turns orange. Fringes that stay black are not connected to them. The orange stays while you use other tools and follows your edits (cutting a merge turns the cut off part red again); Esc clears it. |

With **Snap to ends** on, Line and Spline clicks jump to the nearest fringe end (green circle), so a join is exactly connected.

**Fix-ups**

* **Link Endpoints** adds a straight 1 px stroke between pairs of fringe ends that are closer than Max gap (image pixels) and whose joining line is within Angle degrees of horizontal: 0 joins only horizontally, 90 joins at any angle up to vertical. Only ends that point at each other are joined, never across to the next fringe. Nothing else in the binary changes.
* **Merge Binaries** turns everything into the shot binary layer: the reference binary, at its nudged position, is copied into the shot binary. It asks first.

**View** (none of these change your data)

* Opacity and Brightness for the shot image, the reference image and both binaries.
* **Color Fringes** gives every connected fringe its own color, so breaks and merges stand out.
* **Mirror at x =** reflects the shot binary about that x position (image pixels, as the cursor readout shows it) and draws it in blue under the red lines.
* **Nudge:** the arrow keys move the reference image and reference binary together one pixel of the binary at a time (half an image pixel at 2x; hold a key to keep moving), and Ctrl + right drag (Cmd + right drag on a Mac) moves them with the mouse, so you can see where the shot and reference line up. The offset next to Reset Nudge is in image pixels. Reset Nudge puts them back. A saved reference binary is not moved.

**Undo and Redo** cover every change, including opening layers, merging, nudges and grid switches. A held arrow key is one undo step.

## Mouse and keyboard

| Action | Windows and Linux | macOS |
|---|---|---|
| Zoom | Mouse wheel | Mouse wheel or two finger scroll |
| Pan | Right or middle drag (any button on the Detection tab) | Right or middle drag, or two finger click and drag |
| Show the whole image | Fit button, top right of the image | same |
| Brush size (past the slider's 60, up to 500) | Ctrl + wheel | Cmd + wheel |
| Undo / Redo | Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z | Cmd+Z / Cmd+Shift+Z |
| Finish a spline or polyline | Enter or Ctrl+click | Enter or Cmd+click |
| Cancel a spline or polyline, or clear the Connection Check | Esc | Esc |
| Nudge reference | Arrow keys, or Ctrl + right drag | Arrow keys, or Cmd + right drag |

The bottom bar is the one status line: it shows the file in the current tab, that tab's latest message, and the cursor position (image pixels) with the image value (camera counts) under it. **Help** at the top left opens the manual (help.html) in your browser.

## Binary format

PNG, one value per pixel: **0** = fringe, **255** = background, **128** = gray mask (areas without usable fringes, which phase extraction should skip). When reading a binary, anything below 128 counts as fringe and exactly 128 as gray mask. Binaries saved as 0 and 1, at 16 bit, or as white fringes on black are recognized and read the same way (the status line says when one was inverted). Detected fringe lines are 1 px wide and separate fringes never touch, not even diagonally, so 8-connected labeling gives one label per fringe.

## How detection works

1. Hot pixels are removed and an FFT filter takes out the slowly varying background and any structure steeper than Max Fringe Angle.
2. The local fringe direction and spacing are measured everywhere.
3. The image is averaged along the fringes (following their curvature) to suppress speckle, then divided by the local fringe amplitude so faint and bright fringes look alike.
4. Ridge strength is measured across the fringes at several widths, so dense fringes near the target and wide fringes far from it are both found.
5. Center lines are found on the working grid (at 2x they can sit between image pixels), thresholded, cut wherever two fringes would touch, cleaned of short pieces, and short gaps along a fringe are bridged.

Wide fringes (a period of 20 px or more) are analyzed on a reduced copy of the image and traced back on the full grid. Only bright fringes are traced.

## Troubleshooting

| Problem | What to try |
|---|---|
| SpellBook shows "Something went wrong" | The details are in `spellbook_errors.log` in the SpellBook folder. Send that file with a note of what you did. |
| A slider seems to do nothing | Wait for the status line to say Rendered; changes are applied as soon as the previous update finishes. |
| Fringes near the target are merged or missing | Switch to 2x, then Auto-Optimize again. |
| Vertical streaks at the target edge are traced | Lower Max Fringe Angle, or mask them in the Editor. |
| The window is cut off on a small or high DPI screen | The control panels scroll; use the mouse wheel over them. |
| Images with non-ASCII characters in the path | Supported for reading and writing. |
