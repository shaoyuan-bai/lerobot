#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--chunk", type=str, default="chunk-000")
    ap.add_argument("--inplace", action="store_true", help="直接覆盖原文件；不加则生成 *.fixed.parquet")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    ep_path = root / "meta" / "episodes" / args.chunk / "file-000.parquet"
    if not ep_path.exists():
        raise SystemExit(f"[ERROR] episodes parquet not found: {ep_path}")

    df = pd.read_parquet(ep_path)

    # 允许两种来源：start/end_index 或 dataset_from/to_index 已存在
    has_start_end = {"start_index", "end_index"}.issubset(df.columns)
    has_dataset_from_to = {"dataset_from_index", "dataset_to_index"}.issubset(df.columns)

    if not has_start_end and not has_dataset_from_to:
        raise SystemExit(
            "[ERROR] episodes parquet has neither (start_index,end_index) nor (dataset_from_index,dataset_to_index).\n"
            f"Columns: {list(df.columns)}"
        )

    # 若缺 dataset_from/to_index，就用 start/end 生成
    if "dataset_from_index" not in df.columns:
        df["dataset_from_index"] = df["start_index"].astype("int64")

    if "dataset_to_index" not in df.columns:
        # 通常 loader 需要右开区间，所以 to = end + 1
        if "end_index" in df.columns:
            df["dataset_to_index"] = (df["end_index"].astype("int64") + 1)
        else:
            # 如果只有 dataset_from/to，本分支不会走到；留个兜底
            df["dataset_to_index"] = (df["dataset_from_index"].astype("int64") + df.get("length", 0).astype("int64"))

    # 可选：补一个 episode_length（有些实现会用这个名字）
    if "episode_length" not in df.columns and "length" in df.columns:
        df["episode_length"] = df["length"].astype("int64")

    # 保证关键列类型
    for c in ["episode_index", "dataset_from_index", "dataset_to_index"]:
        if c in df.columns:
            df[c] = df[c].astype("int64")

    # 简单自检：to > from
    bad = (df["dataset_to_index"] <= df["dataset_from_index"]).sum()
    if bad:
        raise SystemExit(f"[ERROR] Found {bad} episodes with dataset_to_index <= dataset_from_index")

    out_path = ep_path if args.inplace else ep_path.with_suffix(".fixed.parquet")
    df.to_parquet(out_path, index=False)

    print("[OK] wrote:", out_path)
    print("[INFO] columns:", list(df.columns))
    print("[INFO] head:\n", df.head(3))


if __name__ == "__main__":
    main()
