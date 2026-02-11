#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="LeRobot v3 数据集根目录（包含 data/, meta/, videos/）",
    )
    ap.add_argument(
        "--chunk",
        type=str,
        default="chunk-000",
        help="chunk 目录名（默认 chunk-000）",
    )
    ap.add_argument(
        "--data_file",
        type=str,
        default="file-000.parquet",
        help="data chunk 下的 parquet 文件名（默认 file-000.parquet）",
    )
    args = ap.parse_args()

    root = Path(args.dataset_root)
    data_path = root / "data" / args.chunk / args.data_file

    if not data_path.exists():
        raise SystemExit(f"[ERROR] data parquet not found: {data_path}")

    # 读取帧级数据
    df = pd.read_parquet(data_path)

    required_cols = {"episode_index", "index"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] data parquet missing required columns: {sorted(missing)}")

    # 可选列：frame_index / task_index
    has_frame = "frame_index" in df.columns
    has_task = "task_index" in df.columns

    sort_cols = ["episode_index"]
    if has_frame:
        sort_cols.append("frame_index")
    else:
        sort_cols.append("index")

    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    agg = {
        "start_index": ("index", "min"),
        "end_index": ("index", "max"),
        "length": ("index", "size"),
    }
    if has_frame:
        agg["start_frame"] = ("frame_index", "min")
        agg["end_frame"] = ("frame_index", "max")
    if has_task:
        agg["task_index"] = ("task_index", "first")

    ep_meta = (
        df_sorted.groupby("episode_index", as_index=False)
        .agg(**{k: v for k, v in agg.items()})
        .sort_values("episode_index")
        .reset_index(drop=True)
    )

    # 输出到 meta/episodes/chunk-000/file-000.parquet（对齐 record100time）
    out_dir = root / "meta" / "episodes" / args.chunk
    ensure_dir(out_dir)
    out_path = out_dir / "file-000.parquet"

    ep_meta.to_parquet(out_path, index=False)

    print("[OK] episodes parquet written:")
    print("  ", out_path)
    print("[INFO] episodes rows:", len(ep_meta))
    print("[INFO] columns:", list(ep_meta.columns))


if __name__ == "__main__":
    main()
