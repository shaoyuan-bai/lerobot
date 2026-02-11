#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch lerobot.datasets.video_utils to NOT crash training when timestamps violate tolerance.
Instead: log warning and fall back to nearest loaded frame.

- Finds the installed module path via import.
- Backs up original file to *.bak
- Replaces the assert-block in:
  - decode_video_frames_torchvision
  - decode_video_frames_torchcodec
"""

import os
import re
import shutil
import sys
from pathlib import Path


def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}")
    sys.exit(code)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str):
    p.write_text(s, encoding="utf-8")


def backup_file(p: Path) -> Path:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"[OK] backup created: {bak}")
    else:
        print(f"[INFO] backup already exists: {bak}")
    return bak


def patch_assert_block(src: str) -> tuple[str, int]:
    """
    Replace the exact assert-block pattern used in LeRobot video_utils.py:

        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), ( ... )

    with:

        is_within_tol = min_ < tolerance_s
        if not is_within_tol.all():
            bad = (~is_within_tol).nonzero(as_tuple=False).squeeze(-1)
            logging.warning(...)

    We keep the rest of the function unchanged; it already uses argmin_ to pick nearest frames.
    """

    # This regex matches:
    #  is_within_tol = min_ < tolerance_s
    #  assert is_within_tol.all(), ( ...multi-line... )
    pattern = re.compile(
        r"""
        (?P<prefix>
            \n[ \t]*is_within_tol[ \t]*=[ \t]*min_[ \t]*<[ \t]*tolerance_s[ \t]*\n
        )
        (?P<assert_block>
            [ \t]*assert[ \t]+is_within_tol\.all\(\)[ \t]*,[ \t]*\(
            .*?
            \n[ \t]*\)[ \t]*\n
        )
        """,
        re.VERBOSE | re.DOTALL,
    )

    repl = r"""\g<prefix>        if not is_within_tol.all():
            bad = (~is_within_tol).nonzero(as_tuple=False).squeeze(-1)

            # Don't crash training. Fall back to nearest loaded frame(s).
            # argmin_ already points to the closest loaded timestamp for each query timestamp.
            try:
                max_err = float(min_[bad].max().item()) if bad.numel() > 0 else float("nan")
            except Exception:
                max_err = float("nan")

            logging.warning(
                "[LeRobot][Video] timestamp mismatch detected; "
                "fallback to nearest frame | "
                f"bad_count={int(bad.numel())} "
                f"max_error={max_err:.4f}s "
                f"tolerance={tolerance_s:.4f}s "
                f"video={video_path} "
                f"backend={backend if 'backend' in locals() else 'torchcodec'}"
            )
"""

    new_src, n = pattern.subn(repl, src)
    return new_src, n


def main():
    # Import the module to locate the file being used by your environment.
    try:
        import lerobot.datasets.video_utils as vu  # noqa
    except Exception as e:
        die(f"Cannot import lerobot.datasets.video_utils. Are you in the (lerobot) env? Error: {e}")

    mod_path = Path(vu.__file__).resolve()
    print(f"[INFO] lerobot.datasets.video_utils = {mod_path}")

    if not mod_path.exists():
        die(f"Module file does not exist: {mod_path}")

    src = read_text(mod_path)

    # Quick sanity check: ensure both functions exist in file content.
    if "def decode_video_frames_torchvision" not in src:
        die("decode_video_frames_torchvision not found in this file; patch aborted.")
    if "def decode_video_frames_torchcodec" not in src:
        die("decode_video_frames_torchcodec not found in this file; patch aborted.")

    # Backup
    backup_file(mod_path)

    # Apply patch (expect 2 replacements: one in torchvision, one in torchcodec)
    new_src, n = patch_assert_block(src)

    if n == 0:
        die(
            "Did not find the expected assert-block pattern to patch.\n"
            "This file may already be patched or has different formatting.\n"
            f"File: {mod_path}"
        )

    print(f"[OK] patched assert-block occurrences: {n}")

    # Write back
    write_text(mod_path, new_src)
    print(f"[OK] wrote patched file: {mod_path}")

    # Verify patched markers exist
    verify = read_text(mod_path)
    if "[LeRobot][Video] timestamp mismatch detected" not in verify:
        die("Patch write succeeded but marker string not found; something went wrong.", code=2)

    print("[DONE] Patch applied successfully.")
    print("Now re-run your training command. If timestamps mismatch, it will WARN but continue.")


if __name__ == "__main__":
    main()