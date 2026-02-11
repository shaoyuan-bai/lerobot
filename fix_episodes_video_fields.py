#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--chunk", type=str, default="chunk-000")
    ap.add_argument("--data_file", type=str, default="file-000.parquet")
    ap.add_argument("--episodes_file", type=str, default="file-000.parquet")
    ap.add_argument("--inplace", action="store_true", help="覆盖原 episodes parquet；不加则生成 *.fixed.parquet")
    ap.add_argument("--chunk_index", type=int, default=0, help="chunk-000 -> 0")
    ap.add_argument("--file_index", type=int, default=0, help="file-000 -> 0")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    data_path = root / "data" / args.chunk / args.data_file
    ep_path = root / "meta" / "episodes" / args.chunk / args.episodes_file

    if not info_path.exists():
        raise SystemExit(f"[ERROR] info.json not found: {info_path}")
    if not data_path.exists():
        raise SystemExit(f"[ERROR] data parquet not found: {data_path}")
    if not ep_path.exists():
        raise SystemExit(f"[ERROR] episodes parquet not found: {ep_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))

    # video keys from info.json
    features = info.get("features", {}) or {}
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if not video_keys:
        raise SystemExit("[ERROR] No video keys found in meta/info.json (dtype=video).")

    df = pd.read_parquet(data_path)
    need_cols = {"episode_index", "timestamp"}
    miss = need_cols - set(df.columns)
    if miss:
        raise SystemExit(f"[ERROR] data parquet missing columns: {sorted(miss)}")

    # episode timestamp ranges
    ts = (
        df.groupby("episode_index", as_index=False)
        .agg(from_ts=("timestamp", "min"), to_ts=("timestamp", "max"))
    )
    ts["to_ts"] = ts["to_ts"].astype("float64") + (1.0 / fps)

    ep = pd.read_parquet(ep_path)
    if "episode_index" not in ep.columns:
        raise SystemExit(f"[ERROR] episodes parquet missing episode_index. Columns: {list(ep.columns)}")

    merged = ep.merge(ts, on="episode_index", how="left")
    if merged["from_ts"].isna().any() or merged["to_ts"].isna().any():
        bad = merged[merged["from_ts"].isna() | merged["to_ts"].isna()][["episode_index"]]
        raise SystemExit(f"[ERROR] Some episodes have no timestamp range in data parquet. Examples:\n{bad.head(10)}")

    # add per-video fields
    for vid_key in video_keys:
        merged[f"videos/{vid_key}/from_timestamp"] = merged["from_ts"].astype("float64")
        merged[f"videos/{vid_key}/to_timestamp"] = merged["to_ts"].astype("float64")
        merged[f"videos/{vid_key}/chunk_index"] = int(args.chunk_index)
        merged[f"videos/{vid_key}/file_index"] = int(args.file_index)

    merged = merged.drop(columns=["from_ts", "to_ts"])

    out_path = ep_path if args.inplace else ep_path.with_suffix(".fixed.parquet")
    ensure_dir(out_path.parent)
    merged.to_parquet(out_path, index=False)

    print("[OK] wrote:", out_path)
    print("[INFO] video keys:", video_keys)
    # print a few columns to verify
    check_cols = ["episode_index"]
    for vid_key in video_keys:
        check_cols += [
            f"videos/{vid_key}/from_timestamp",
            f"videos/{vid_key}/to_timestamp",
            f"videos/{vid_key}/chunk_index",
            f"videos/{vid_key}/file_index",
        ]
    print("[INFO] head:\n", merged[check_cols].head(2))


if __name__ == "__main__":
    main()
