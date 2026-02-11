#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--fps", type=float, default=None, help="不填则从 meta/info.json 读取")
    ap.add_argument("--shrink_frames", type=float, default=2.0, help="to_timestamp 往回缩多少帧（默认2帧）")
    ap.add_argument("--chunk", type=str, default="chunk-000")
    ap.add_argument("--episodes_file", type=str, default="file-000.parquet")
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    ep_path = root / "meta" / "episodes" / args.chunk / args.episodes_file

    if not info_path.exists():
        raise SystemExit(f"[ERROR] missing {info_path}")
    if not ep_path.exists():
        raise SystemExit(f"[ERROR] missing {ep_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(args.fps) if args.fps is not None else float(info.get("fps", 30))
    shrink_s = float(args.shrink_frames) / fps

    df = pd.read_parquet(ep_path)

    # 找出所有 videos/*/to_timestamp 列
    to_cols = [c for c in df.columns if c.endswith("/to_timestamp") and c.startswith("videos/")]
    if not to_cols:
        raise SystemExit("[ERROR] no videos/*/to_timestamp columns found in episodes parquet")

    for c in to_cols:
        df[c] = df[c].astype("float64") - shrink_s

    out = ep_path if args.inplace else ep_path.with_suffix(".shrunk.parquet")
    df.to_parquet(out, index=False)

    print("[OK] wrote:", out)
    print("[INFO] shrink_s =", shrink_s, "seconds")
    print("[INFO] modified columns:", to_cols)


if __name__ == "__main__":
    main()
