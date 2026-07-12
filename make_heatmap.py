# encoding=utf-8
# make_heatmap.py — 把眼动 CSV / 坐标点 画成注视热力图
#
# 用法:
#   python make_heatmap.py 数据.csv                 # 自动推断屏幕尺寸
#   python make_heatmap.py 数据.csv 1920 1080       # 指定屏幕宽高(像素)
#   python make_heatmap.py 数据.csv 1920 1080 --bg 首帧.png
#   python make_heatmap.py 数据.csv 1920 1080 --sigma 4 --bins 140 --cmap magma
#
# 说明:
#   - 优先用 filtered_gaze_position(滤波后坐标); 缺失时回退 calibrated。
#   - 坐标为屏幕像素(左上角原点), 与录制时 cfg.screen_size 一致。
#   - 也可被 demo.py 直接 import: render_heatmap(xs, ys, W, H, out_path, ...)
#     其中 bg 可传 BGR 帧(numpy), 把热力图叠加到视频画面上。

import sys
import csv
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面后端, 服务器/脚本环境安全
import matplotlib.pyplot as plt


def load_positions(path):
    """从 CSV 读取有效注视点像素坐标, 返回 (xs, ys) numpy 数组。"""
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = row.get("filtered_gaze_position_x")
            y = row.get("filtered_gaze_position_y")
            if x in (None, "NA", "") or y in (None, "NA", ""):
                x = row.get("calibrated_gaze_position_x")
                y = row.get("calibrated_gaze_position_y")
            if x in (None, "NA", "") or y in (None, "NA", ""):
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                continue
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def gaussian_blur(h, sigma):
    """可分离高斯模糊(numpy 实现, 避免 scipy 依赖)。"""
    if sigma <= 0:
        return h
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    h = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), axis=1, arr=h)
    h = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), axis=0, arr=h)
    return h


def render_heatmap(xs, ys, W, H, out_path, *,
                   sigma=3.0, bins=120, bg=None, cmap="inferno",
                   caption=None, title=None, colorbar=False, max_alpha=0.85):
    """
    绘制并保存一张注视热力图。

    参数:
      xs, ys   : 注视点像素坐标(屏幕坐标, 左上角原点)
      W, H     : 屏幕尺寸(像素)
      out_path : 输出 PNG 路径
      bg       : None(浅灰底), 或背景图路径(str), 或 BGR 帧(numpy)
      cmap     : 颜色映射
      caption  : 左下角叠加文字(如 "0–5 s · n=123")
      title    : 顶部标题(若给定会留出上边距)
      colorbar : 是否画颜色条(仅无背景时建议开)
      max_alpha: 热点最大不透明度
    """
    W = int(W); H = int(H)
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)

    if len(xs) == 0:
        # 没有有效点: 画一张"无数据"占位图
        ax.imshow(np.full((H, W, 3), 245, dtype=np.uint8),
                  origin="upper", extent=[0, W, H, 0], aspect="auto")
        ax.text(0.5, 0.5, "No valid gaze points", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return

    # 2D 直方图 + 高斯平滑 -> 归一化密度
    H2, _, _ = np.histogram2d(ys, xs, bins=bins, range=[[0, H], [0, W]])
    H2 = gaussian_blur(H2, sigma)
    if H2.max() > 0:
        H2 = H2 / H2.max()
    density = H2
    # 上采样到屏幕分辨率, 才能与背景图逐像素混合
    density = cv2.resize(density, (W, H), interpolation=cv2.INTER_LINEAR)
    density = np.clip(density, 0, 1)

    # 背景图处理
    bg_img = None
    if bg is not None:
        if isinstance(bg, str):
            tmp = cv2.imread(bg)
            if tmp is not None:
                bg_img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        else:  # 假设是 BGR numpy 帧
            bg_img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) if bg.ndim == 3 else bg
        if bg_img is not None:
            bg_img = cv2.resize(bg_img, (W, H))

    needs_margin = bool(title) or colorbar
    if needs_margin:
        fig.subplots_adjust(left=0.02, right=0.98,
                            top=0.90 if title else 0.98, bottom=0.02)
    else:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    cmap_obj = plt.get_cmap(cmap)
    rgba = cmap_obj(density)
    alpha = (density ** 0.55)[..., None] * max_alpha   # 低密度透明, 高密度不透明
    heat = rgba[..., :3] * 255.0

    if bg_img is not None:
        base = bg_img.astype(np.float32)
    else:
        base = np.full((H, W, 3), 245.0, dtype=np.float32)

    out = np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    im = ax.imshow(out, origin="upper", extent=[0, W, H, 0], aspect="auto")

    # 屏幕边框(无背景时更明显)
    if bg_img is None:
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color="#9aa0a6", lw=1.5)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)        # y 轴向下(屏幕坐标)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if caption:
        ax.text(0.013, 0.045, caption, transform=ax.transAxes,
                fontsize=14, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fixation density")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="眼动 CSV -> 注视热力图")
    ap.add_argument("csv", help="眼动数据 CSV 路径")
    ap.add_argument("width", nargs="?", type=int, default=None, help="屏幕宽(像素)")
    ap.add_argument("height", nargs="?", type=int, default=None, help="屏幕高(像素)")
    ap.add_argument("--sigma", type=float, default=3.0, help="高斯平滑系数(默认3)")
    ap.add_argument("--bins", type=int, default=120, help="网格分辨率(默认120)")
    ap.add_argument("--bg", default=None, help="可选: 叠加到该背景图/视频帧")
    ap.add_argument("--cmap", default="inferno", help="颜色映射(默认 inferno)")
    args = ap.parse_args()

    xs, ys = load_positions(args.csv)
    if len(xs) == 0:
        print("没有有效的注视点数据, 无法生成热力图。")
        return

    W = args.width if args.width else int(np.ceil(xs.max())) + 20
    H = args.height if args.height else int(np.ceil(ys.max())) + 20

    out_png = args.csv.rsplit(".", 1)[0] + "_heatmap.png"
    title = f"Gaze Heatmap  (n={len(xs)} points)"
    # 无背景时画颜色条 + 标题; 有背景时满屏发光 + 角标
    render_heatmap(xs, ys, W, H, out_png, sigma=args.sigma, bins=args.bins,
                   bg=args.bg, cmap=args.cmap,
                   caption=(None if args.bg else None),
                   title=(title if args.bg is None else None),
                   colorbar=(args.bg is None))
    print(f"已保存热力图: {out_png}  (屏幕 {W}x{H}, 有效点 {len(xs)})")


if __name__ == "__main__":
    main()
