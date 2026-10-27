SpellBook 2.0.0
Fringe tracing and editing for laser-plasma interferograms.

The full guide is in README.md, and in help.html (the Help button in SpellBook opens it).

Install (Python 3.10 or newer)
  Windows PowerShell:
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
  macOS or Linux:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Check the install (prints ALL OK):
    python check_files.py

Run:
    python SpellBook.py

Folders
  InputImages/   put the shots to trace here (load dialogs start here)
  EditedImages/  binaries written by SpellBook (Send to Editor saves here)

Binaries are PNG files: 0 = fringe, 255 = background, 128 = gray mask.
