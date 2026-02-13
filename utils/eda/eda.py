#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def safe_imread(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img


def img_info(img: np.ndarray) -> dict:
    if img.ndim == 2:
        h, w = img.shape
        c = 1
        dtype = str(img.dtype)
        mn, mx = float(np.min(img)), float(np.max(img))
        mean, std = float(np.mean(img)), float(np.std(img))
        return {"h": h, "w": w, "c": c, "dtype": dtype, "min": mn, "max": mx, "mean": mean, "std": std}
    elif img.ndim == 3:
        h, w, c = img.shape
        dtype = str(img.dtype)
        mn, mx = float(np.min(img)), float(np.max(img))
        mean, std = float(np.mean(img)), float(np.std(img))
        return {"h": h, "w": w, "c": c, "dtype": dtype, "min": mn, "max": mx, "mean": mean, "std": std}
    else:
        return {"h": None, "w": None, "c": None, "dtype": str(img.dtype), "min": None, "max": None, "mean": None, "std": None}


def to_rgb_for_plot(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img[:, :, 0]


def plot_histogram(img: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(7, 4))
    if img.ndim == 2:
        data = img.ravel()
        plt.hist(data, bins=256)
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Count")
    else:
        rgb = to_rgb_for_plot(img)
        if rgb.ndim == 2:
            data = rgb.ravel()
            plt.hist(data, bins=256)
            plt.title(title)
            plt.xlabel("Intensity")
            plt.ylabel("Count")
        else:
            labels = ["R", "G", "B"] if rgb.shape[2] == 3 else [f"Ch{i}" for i in range(rgb.shape[2])]
            for ch in range(min(rgb.shape[2], 3)):
                plt.hist(rgb[:, :, ch].ravel(), bins=256, alpha=0.5, label=labels[ch])
            plt.title(title)
            plt.xlabel("Intensity")
            plt.ylabel("Count")
            plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_samples(paths, out_path: Path, title: str, n=12):
    paths = list(paths)[:n]
    if not paths:
        return
    cols = 4
    rows = int(np.ceil(len(paths) / cols))
    plt.figure(figsize=(4 * cols, 3.4 * rows))
    for i, p in enumerate(paths, start=1):
        img = safe_imread(p)
        if img is None:
            continue
        plt.subplot(rows, cols, i)
        vis = to_rgb_for_plot(img)
        if vis.ndim == 2:
            plt.imshow(vis, cmap="gray")
        else:
            plt.imshow(vis)
        plt.title(p.name, fontsize=9)
        plt.axis("off")
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def summarize_group(records):
    if not records:
        return {}

    hs = np.array([r["h"] for r in records if r["h"] is not None], dtype=float)
    ws = np.array([r["w"] for r in records if r["w"] is not None], dtype=float)
    cs = np.array([r["c"] for r in records if r["c"] is not None], dtype=float)

    means = np.array([r["mean"] for r in records if r["mean"] is not None], dtype=float)
    stds = np.array([r["std"] for r in records if r["std"] is not None], dtype=float)
    mins = np.array([r["min"] for r in records if r["min"] is not None], dtype=float)
    maxs = np.array([r["max"] for r in records if r["max"] is not None], dtype=float)

    sizes = [(r["h"], r["w"], r["c"]) for r in records if r["h"] is not None]
    unique_sizes = sorted(set(sizes))

    dtypes = sorted(set(r["dtype"] for r in records if r.get("dtype") is not None))

    def stat(arr):
        if arr.size == 0:
            return None
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "count": len(records),
        "unique_sizes": unique_sizes[:25],
        "unique_sizes_more": max(0, len(unique_sizes) - 25),
        "channels_set": sorted(set(int(x) for x in cs)) if cs.size else [],
        "dtypes": dtypes,
        "h": stat(hs),
        "w": stat(ws),
        "mean": stat(means),
        "std": stat(stds),
        "minv": stat(mins),
        "maxv": stat(maxs),
    }


def markdown_table(rows, headers):
    def esc(x):
        s = str(x)
        return s.replace("\n", " ").replace("|", "\\|")
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(esc(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("utils/eda/outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    all_image_paths = sorted([p for p in data_root.rglob("*") if is_image_file(p)])
    if not all_image_paths:
        print(f"No images found under: {data_root}")
        sys.exit(1)

    records = []
    for p in all_image_paths:
        img = safe_imread(p)
        if img is None:
            continue
        info = img_info(img)
        info.update({
            "path": str(p),
            "relpath": str(p.relative_to(data_root)),
            "group": str(p.relative_to(data_root).parts[0]) if len(p.relative_to(data_root).parts) > 0 else "",
        })
        records.append(info)

    groups = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)

    report_lines = []
    report_lines.append(f"# Image Dataset EDA Report")
    report_lines.append("")
    report_lines.append(f"- Data root: `{data_root}`")
    report_lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    report_lines.append(f"- Total images read: **{len(records)}**")
    report_lines.append("")

    overall = summarize_group(records)
    report_lines.append("## Overall Summary")
    report_lines.append("")
    report_lines.append(f"- Unique channel counts: **{overall.get('channels_set', [])}**")
    report_lines.append(f"- Dtypes: **{overall.get('dtypes', [])}**")
    if overall.get("h"):
        report_lines.append(f"- Height (px): min={overall['h']['min']:.0f}, max={overall['h']['max']:.0f}, mean={overall['h']['mean']:.1f}, std={overall['h']['std']:.1f}")
    if overall.get("w"):
        report_lines.append(f"- Width (px): min={overall['w']['min']:.0f}, max={overall['w']['max']:.0f}, mean={overall['w']['mean']:.1f}, std={overall['w']['std']:.1f}")
    if overall.get("mean"):
        report_lines.append(f"- Pixel mean: min={overall['mean']['min']:.2f}, max={overall['mean']['max']:.2f}, mean={overall['mean']['mean']:.2f}, std={overall['mean']['std']:.2f}")
    if overall.get("std"):
        report_lines.append(f"- Pixel std: min={overall['std']['min']:.2f}, max={overall['std']['max']:.2f}, mean={overall['std']['mean']:.2f}, std={overall['std']['std']:.2f}")
    report_lines.append("")

    overall_samples_path = out_root / "samples_overall.png"
    plot_samples([Path(r["path"]) for r in records], overall_samples_path, "Random Samples (Overall)", n=min(12, len(records)))
    report_lines.append("### Sample Grid (Overall)")
    report_lines.append("")
    report_lines.append(f"![samples_overall]({overall_samples_path.name})")
    report_lines.append("")

    overall_hist_path = out_root / "hist_overall.png"
    max_for_hist = []
    for r in records:
        p = Path(r["path"])
        img = safe_imread(p)
        if img is None:
            continue
        if img.ndim == 2:
            max_for_hist.append(img.ravel())
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
            max_for_hist.append(gray.ravel())
        if len(max_for_hist) >= 20:
            break
    if max_for_hist:
        stacked = np.concatenate(max_for_hist, axis=0)
        plt.figure(figsize=(7, 4))
        plt.hist(stacked, bins=256)
        plt.title("Overall Intensity Histogram (from up to 20 images, grayscale)")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(overall_hist_path, dpi=160)
        plt.close()
        report_lines.append("### Intensity Histogram (Overall, Grayscale)")
        report_lines.append("")
        report_lines.append(f"![hist_overall]({overall_hist_path.name})")
        report_lines.append("")

    report_lines.append("## Per-Folder Summaries")
    report_lines.append("")

    folder_rows = []
    for gname in sorted(groups.keys()):
        gsum = summarize_group(groups[gname])
        folder_rows.append({
            "folder": gname,
            "count": gsum.get("count", 0),
            "channels": gsum.get("channels_set", []),
            "dtypes": gsum.get("dtypes", []),
            "unique_sizes_top": gsum.get("unique_sizes", [])[:5],
        })
    report_lines.append(markdown_table(folder_rows, ["folder", "count", "channels", "dtypes", "unique_sizes_top"]))
    report_lines.append("")

    for gname in sorted(groups.keys()):
        grec = groups[gname]
        gsum = summarize_group(grec)

        report_lines.append(f"### `{gname}`")
        report_lines.append("")
        report_lines.append(f"- Images: **{gsum.get('count', 0)}**")
        report_lines.append(f"- Channels: **{gsum.get('channels_set', [])}**")
        report_lines.append(f"- Dtypes: **{gsum.get('dtypes', [])}**")
        us = gsum.get("unique_sizes", [])
        more = gsum.get("unique_sizes_more", 0)
        report_lines.append(f"- Unique sizes (top): **{us}**" + (f" (+{more} more)" if more else ""))
        report_lines.append("")

        sample_path = out_root / f"samples_{gname}.png"
        plot_samples([Path(r["path"]) for r in grec], sample_path, f"Samples: {gname}", n=min(12, len(grec)))
        report_lines.append(f"![samples_{gname}]({sample_path.name})")
        report_lines.append("")

        rep_img_path = Path(grec[0]["path"])
        rep_img = safe_imread(rep_img_path)
        if rep_img is not None:
            hist_path = out_root / f"hist_{gname}.png"
            plot_histogram(rep_img, hist_path, f"Histogram (Representative): {gname} ({rep_img_path.name})")
            report_lines.append(f"![hist_{gname}]({hist_path.name})")
            report_lines.append("")

        detail_rows = []
        for r in sorted(grec, key=lambda x: x["relpath"])[:20]:
            detail_rows.append({
                "file": r["relpath"],
                "shape": f"{r['h']}x{r['w']}x{r['c']}",
                "dtype": r["dtype"],
                "mean": f"{r['mean']:.2f}",
                "std": f"{r['std']:.2f}",
                "min": f"{r['min']:.0f}",
                "max": f"{r['max']:.0f}",
            })
        report_lines.append("#### File-level Preview (first 20)")
        report_lines.append("")
        report_lines.append(markdown_table(detail_rows, ["file", "shape", "dtype", "mean", "std", "min", "max"]))
        report_lines.append("")

    report_path = out_root / "eda_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
