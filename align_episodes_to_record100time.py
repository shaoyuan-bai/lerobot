#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


def vec_stats_from_series(series: pd.Series) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """Compute per-dimension min/max/mean/std for a series of vectors (list/ndarray)."""
    arrs = []
    for x in series:
        if x is None:
            continue
        try:
            v = np.asarray(x, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        arrs.append(v)

    if not arrs:
        return ([], [], [], [], 0)

    m = int(max(v.shape[0] for v in arrs))
    A = np.vstack([np.pad(v, (0, m - v.shape[0]), constant_values=np.nan) for v in arrs])

    mn = np.nanmin(A, axis=0).tolist()
    mx = np.nanmax(A, axis=0).tolist()
    mean = np.nanmean(A, axis=0).tolist()
    std = np.nanstd(A, axis=0).tolist()
    return (mn, mx, mean, std, int(len(arrs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_root", type=str, required=True, help="要修复的数据集根目录（lerobot_v3_dataset_aligned）")
    ap.add_argument("--ref_root", type=str, required=True, help="参考正确数据集根目录（record100time）")
    ap.add_argument("--chunk", type=str, default="chunk-000")
    ap.add_argument("--data_file", type=str, default="file-000.parquet")
    ap.add_argument("--episodes_file", type=str, default="file-000.parquet")
    ap.add_argument("--inplace", action="store_true", help="覆盖 target 的 episodes parquet；不加则输出 *.aligned.parquet")
    args = ap.parse_args()

    target_root = Path(args.target_root)
    ref_root = Path(args.ref_root)

    tgt_ep_path = target_root / "meta" / "episodes" / args.chunk / args.episodes_file
    tgt_data_path = target_root / "data" / args.chunk / args.data_file
    tgt_tasks_path = target_root / "meta" / "tasks.parquet"
    ref_ep_path = ref_root / "meta" / "episodes" / args.chunk / args.episodes_file

    if not ref_ep_path.exists():
        raise SystemExit(f"[ERROR] ref episodes parquet not found: {ref_ep_path}")
    if not tgt_ep_path.exists():
        raise SystemExit(f"[ERROR] target episodes parquet not found: {tgt_ep_path}")
    if not tgt_data_path.exists():
        raise SystemExit(f"[ERROR] target data parquet not found: {tgt_data_path}")

    ref_ep = pd.read_parquet(ref_ep_path)
    ref_cols: List[str] = list(ref_ep.columns)

    tgt_ep = pd.read_parquet(tgt_ep_path)
    data = pd.read_parquet(tgt_data_path)

    if "episode_index" not in tgt_ep.columns:
        raise SystemExit("[ERROR] target episodes parquet missing episode_index")
    if "episode_index" not in data.columns:
        raise SystemExit("[ERROR] target data parquet missing episode_index")
    if "timestamp" not in data.columns:
        raise SystemExit("[ERROR] target data parquet missing timestamp")

    # ---------- ensure base columns ----------
    # tasks: list[str]
    if "tasks" not in tgt_ep.columns:
        if tgt_tasks_path.exists() and "task_index" in tgt_ep.columns:
            tasks_df = pd.read_parquet(tgt_tasks_path)
            # index -> task string, col task_index
            idx_to_task: Dict[int, str] = {}
            for task_str, row in tasks_df.iterrows():
                try:
                    idx_to_task[int(row["task_index"])] = str(task_str)
                except Exception:
                    pass

            def map_task_list(x):
                t = idx_to_task.get(int(x), "")
                return [t] if t != "" else [""]

            tgt_ep["tasks"] = tgt_ep["task_index"].apply(map_task_list)
        else:
            tgt_ep["tasks"] = [[""] for _ in range(len(tgt_ep))]

    for c, v in [
        ("data/chunk_index", 0),
        ("data/file_index", 0),
        ("meta/episodes/chunk_index", 0),
        ("meta/episodes/file_index", 0),
    ]:
        if c not in tgt_ep.columns:
            tgt_ep[c] = v

    # dataset_from/to
    if "dataset_from_index" not in tgt_ep.columns:
        if "start_index" in tgt_ep.columns:
            tgt_ep["dataset_from_index"] = tgt_ep["start_index"].astype("int64")
        else:
            raise SystemExit("[ERROR] missing dataset_from_index and no start_index to infer.")
    if "dataset_to_index" not in tgt_ep.columns:
        if "end_index" in tgt_ep.columns:
            tgt_ep["dataset_to_index"] = (tgt_ep["end_index"].astype("int64") + 1)
        elif "length" in tgt_ep.columns:
            tgt_ep["dataset_to_index"] = tgt_ep["dataset_from_index"].astype("int64") + tgt_ep["length"].astype("int64")
        else:
            raise SystemExit("[ERROR] missing dataset_to_index and cannot infer.")

    if "length" not in tgt_ep.columns:
        tgt_ep["length"] = (tgt_ep["dataset_to_index"] - tgt_ep["dataset_from_index"]).astype("int64")

    # ---------- build stats tables (index=episode_index) ----------
    g = data.groupby("episode_index", sort=True)

    # scalar stats (no merge level_1 issues)
    scalar_keys = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    scalar_frames = []
    for key in scalar_keys:
        if key not in data.columns:
            continue
        tmp = g[key].agg(["min", "max", "mean", "count"])
        # std with ddof=0
        tmp_std = g[key].apply(lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0)))
        tmp["std"] = tmp_std
        tmp = tmp.rename(columns={
            "min": f"stats/{key}/min",
            "max": f"stats/{key}/max",
            "mean": f"stats/{key}/mean",
            "std": f"stats/{key}/std",
            "count": f"stats/{key}/count",
        })
        scalar_frames.append(tmp)

    scalar_stats = pd.concat(scalar_frames, axis=1) if scalar_frames else pd.DataFrame(index=tgt_ep["episode_index"].unique())

    # vector stats for action/state
    vec_stats_parts = []

    if "action" in data.columns:
        act = g["action"].apply(lambda s: pd.Series(vec_stats_from_series(s), index=[
            "stats/action/min", "stats/action/max", "stats/action/mean", "stats/action/std", "stats/action/count"
        ]))
        vec_stats_parts.append(act)

    if "observation.state" in data.columns:
        st = g["observation.state"].apply(lambda s: pd.Series(vec_stats_from_series(s), index=[
            "stats/observation.state/min", "stats/observation.state/max", "stats/observation.state/mean",
            "stats/observation.state/std", "stats/observation.state/count"
        ]))
        vec_stats_parts.append(st)

    vec_stats = pd.concat(vec_stats_parts, axis=1) if vec_stats_parts else pd.DataFrame(index=scalar_stats.index)

    # images stats columns exist -> fill NaN/0 (we don't have pixel stats)
    for vid_key in ["observation.images.handeye", "observation.images.fixed"]:
        for c in [f"stats/{vid_key}/min", f"stats/{vid_key}/max", f"stats/{vid_key}/mean", f"stats/{vid_key}/std"]:
            if c not in vec_stats.columns:
                vec_stats[c] = np.nan
        cc = f"stats/{vid_key}/count"
        if cc not in vec_stats.columns:
            vec_stats[cc] = 0

    # ---------- attach stats to tgt_ep via join on episode_index ----------
    tgt_ep = tgt_ep.set_index("episode_index", drop=False)
    tgt_ep = tgt_ep.join(scalar_stats, how="left")
    tgt_ep = tgt_ep.join(vec_stats, how="left")
    tgt_ep = tgt_ep.reset_index(drop=True)

    # ---------- ensure any remaining ref columns exist ----------
    for c in ref_cols:
        if c not in tgt_ep.columns:
            if c.endswith("/count") or c.endswith("_index") or c.endswith("chunk_index") or c.endswith("file_index"):
                tgt_ep[c] = 0
            else:
                tgt_ep[c] = np.nan

    # keep only ref columns and order
    aligned = tgt_ep[ref_cols].copy()

    # cast common index columns to int64 where applicable
    int_cols = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.handeye/chunk_index",
        "videos/observation.images.handeye/file_index",
        "videos/observation.images.fixed/chunk_index",
        "videos/observation.images.fixed/file_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ]
    for c in int_cols:
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(0).astype("int64")

    out_path = tgt_ep_path if args.inplace else tgt_ep_path.with_suffix(".aligned.parquet")
    aligned.to_parquet(out_path, index=False)

    print("[OK] aligned episodes parquet written:")
    print(" ", out_path)
    print("[INFO] columns =", len(aligned.columns))
    print("[INFO] head:")
    print(aligned.head(2))


if __name__ == "__main__":
    main()
