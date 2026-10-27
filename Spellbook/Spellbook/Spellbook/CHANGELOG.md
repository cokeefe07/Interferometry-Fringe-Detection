# Changelog

## 2.0.0

Detection
* New fringe detector that follows the local fringe direction and spacing instead of fixed horizontal line kernels; dense fringes near the target and curved fringes are traced as single 1 px lines, and separate fringes never touch.
* Working grid switch: 1x (the image's own pixels) or 2x (twice as fine). Works for any camera size, not only 2048 x 2048.
* Auto-Optimize searches all six detection settings over their useful range and sets the sliders to the result; countdown, Cancel button, sliders locked while it runs. Its score was tuned so that the settings it picks need the least hand editing on hand-edited shots, with stray pieces counting only a little because they are quick to delete.
* Max Fringe Angle now runs from 0 (horizontal only) to 90 (every angle up to vertical); wide fringes (period 20 px and more) and faint fringes are traced.
* Noise Rejection starts at 0.25 instead of 0.85, which deleted some long real fringes; on hand-edited shots the first trace now needs less editing.
* Click a slider's number to type a value. Settings and folders are remembered between sessions.
* Fit button on the image, cursor position and image value in the bottom bar, file name of the current tab in the bottom left.
* Region selection, Edit Sliders and Show Flattened Image removed.
* Images with NaN or infinite pixels are cleaned on load; paths with non-ASCII characters work.

Editor
* Undo and redo for every change (pixel edits, opening layers, gray masks, merging, nudges, grid switches), fast and memory bounded.
* Masking removes the fringes underneath, exactly as the saved PNG has them. Binaries are always saved as PNG.
* Link Endpoints only adds strokes between ends that point at each other along a fringe, within the angle and distance set; it never changes anything else. Angle runs from 0 (horizontal joins only) to 90 (any angle up to vertical).
* Line tool polyline mode; Spline finishes with Ctrl+click (Cmd+click on macOS) or Enter.
* Ctrl + wheel (Cmd + wheel on macOS) or a typed size takes the pen, eraser and mask pen past the Size slider's 60, up to 500.
* Delete Fringe deletes only the piece under the clicked pixel and highlights it first.
* Nudge (arrow keys or Ctrl + right drag) moves the reference image and binary together; a held key is one undo step.
* Merge Binaries asks before turning everything into the shot binary layer, and can be undone.
* Gray mask files use the same values as binaries (128 = masked, 255 = not masked); old gray mask files still open.
* Binaries saved as 0 and 1, at 16 bit, or as white fringes on black open correctly.
* Connection Check shows touched fringes in orange and the rest in black. The orange stays while you use other tools and follows your edits; Esc clears it.
* Crop tool removed (use the Mask Pen).
* Faster drawing and redrawing at 4096 x 4096.

General
* Every length and position shown or typed is in pixels of the image on both grids (sliders, cursor readout, Ruler, Mirror at x, Link Endpoints Max gap, nudge offset); at 2x the readout also gives the position on the binary.
* One status line at the bottom of the window, showing the current tab's messages.
* Version in the window title, and a Help button that opens help.html.
* Works on Windows, macOS and Linux with Tk 8.6 and Tk 9: mouse buttons, trackpad scrolling, Cmd shortcuts on macOS, high DPI screens, theme fonts and colors.
* Asks before closing when work would be lost.
* Unexpected errors show a message and are saved to spellbook_errors.log instead of disappearing.
* `InputImages/` and `EditedImages/` folders inside the SpellBook folder.
* `check_files.py` self test, which also checks that the installed packages are new enough.
* Unused code removed (mixins/, zoom_handler.py, shading_pipeline.py, fringe_utils.py); the window class is now SpellBookApp.
* A .gitignore keeps `__pycache__`, your settings file and the contents of InputImages and EditedImages out of the repository.
* requirements.txt lists scipy and the minimum version of every package (Pillow 10 or newer is needed for Tk 9).
