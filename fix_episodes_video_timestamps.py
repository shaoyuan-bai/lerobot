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
    args = ap.parse_args()

    root = Path(args.dataset_root)

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"[ERROR] info.json not found: {info_path}")

    data_path = root / "data" / args.chunk / args.data_file
    if not data_path.exists():
        raise SystemExit(f"[ERROR] data parquet not found: {data_path}")

    ep_path = root / "meta" / "episodes" / args.chunk / args.episodes_file
    if not ep_path.exists():
        raise SystemExit(f"[ERROR] episodes parquet not found: {ep_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))

    # 从 info.json 的 features 中找 dtype=video 的 key
    features = info.get("features", {}) or {}
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if not video_keys:
        raise SystemExit("[ERROR] No video keys found in meta/info.json features (dtype=video).")

    # 读取 data parquet，按 episode 得到 timestamp 范围
    df = pd.read_parquet(data_path)
    need_cols = {"episode_index", "timestamp"}
    miss = need_cols - set(df.columns)
    if miss:
        raise SystemExit(f"[ERROR] data parquet missing columns: {sorted(miss)}")

    # 每个 episode 的时间范围
    ts = (
        df.groupby("episode_index", as_index=False)
        .agg(from_ts=("timestamp", "min"), to_ts=("timestamp", "max"))
    )
    ts["to_ts"] = ts["to_ts"].astype("float64") + (1.0 / fps)  # 右开区间更安全

    # 读取 episodes parquet 并合并
    ep = pd.read_parquet(ep_path)

    if "episode_index" not in ep.columns:
        raise SystemExit(f"[ERROR] episodes parquet has no episode_index. Columns: {list(ep.columns)}")

    merged = ep.merge(ts, on="episode_index", how="left")

    # 检查有没有 episode 没匹配到 timestamp
    if merged["from_ts"].isna().any() or merged["to_ts"].isna().any():
        bad = merged[merged["from_ts"].isna() | merged["to_ts"].isna()][["episode_index"]]
        raise SystemExit(f"[ERROR] Some episodes have no timestamp range in data parquet. Examples:\n{bad.head(10)}")

    # 为每个 video_key 补充 4 列
    for vid_key in video_keys:
        col_from = f"videos/{vid_key}/from_timestamp"
        col_to = f"videos/{vid_key}/to_timestamp"
        merged[col_from] = merged["from_ts"].astype("float64")
        merged[col_to] = merged["to_ts"].astype("float64")

    # 清理临时列
    merged = merged.drop(columns=["from_ts", "to_ts"])

    out_path = ep_path if args.inplace else ep_path.with_suffix(".fixed.parquet")
    ensure_dir(out_path.parent)
    merged.to_parquet(out_path, index=False)

    print("[OK] wrote:", out_path)
    print("[INFO] video keys:", video_keys)
    # 打印一下是否确实存在你缺的列
    sample_cols = []
    for vid_key in video_keys:
        sample_cols += [f"videos/{vid_key}/from_timestamp", f"videos/{vid_key}/to_timestamp"]
    print("[INFO] added cols:", sample_cols[:6], ("..." if len(sample_cols) > 6 else ""))
    print("[INFO] head:\n", merged[["episode_index"] + sample_cols].head(3))


if __name__ == "__main__":
    main()
