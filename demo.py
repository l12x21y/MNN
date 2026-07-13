# encoding=utf-8
# demo.py — 眼动追踪 + 视频播放 记录 demo (单屏版, 流畅优化)
#
# 用法:
#   python demo.py                 # 直接用下面 VIDEO_PATH 设置的视频
#   python demo.py 其它路径.mp4    # 也可在命令行覆盖视频路径
#
# 实验流程(单屏):
#   1) 运行脚本, 自动全屏校准(看圆点, 空格继续)
#   2) 校准完成出现说明页, 按 [空格] 开始播放视频并记录
#   3) 被试观看全屏视频(屏幕上不显示任何注视点, 避免干扰实验)
#   4) 视频自然播放完毕后自动停止(按 [ESC] 也可提前结束)
#   5) 整个测试的数据与可视化写入同名文件夹:
#        gaze_测试开始时间/gaze_测试开始时间.csv      # 原始记录
#        gaze_测试开始时间/1.png, 2.png, ...          # 每 5 秒一张注视热力图(干净版, 无视频画面)
#        gaze_测试开始时间/1_video.png, 2_video.png   # 同窗口的带视频画面版
#
# 输出 CSV 每行: timestamp(电脑时钟纳秒), datetime(可读电脑时间),
#   原始/校准后/滤波后 注视点像素坐标, 左右眼开合度, 追踪状态, 触发标记
#
# 流畅性设计原则:
#   - 录制频率由 RECORD_HZ(默认 10Hz) 控制: 摄像头线程每 1/RECORD_HZ 秒才跑一次
#     脸检测+眼动推理并写一条样本。录得越慢, 播放期被推理占用的 CPU/GIL 越少, 视频越流畅。
#   - 播放视频期间【只记录数据】, 绝不做任何可视化(不渲染热力图、不拷贝视频帧作底图、
#     不起后台渲染线程), 保证视频播放完全不卡顿。
#   - 每帧显示用 pygame.image.frombuffer 直接封装 RGB 帧(不转置/不重建 surface)。
#   - 设置 CAP_PROP_BUFFERSIZE=1 降低解码延迟, 并用 clock.tick(fps) 按原始帧率匀速播放。
#   - 热力图等可视化全部在【数据记录完成之后】统一离线生成(见 main 末尾的
#     _generate_visualizations): 干净版直接由 CSV 统计得到; 带视频画面版从目标视频按
#     窗口中点时间抽取关键帧作底图。播放循环一结束就先 stop_sampling + save_data,
#     整个播放期间没有任何可视化代码运行, 可视化不可能拖慢视频。
#   - 该原则对 CPU / GPU 两套都生效; 差异只在 MNN 推理走 CPU 还是 GPU(见 USE_CPU/USE_GPU)。

# ===================== CPU / GPU 切换（二选一）=====================
# 把"要用的那一套"对应行的值改成 1，另一行改成 0。
#   USE_GPU=1 → MNN 眼动推理自动选择最佳 GPU 后端:
#                优先 NVIDIA CUDA(6), 其次 Vulkan(4), 最后 OpenCL(2);
#                若全不可用则回退 CPU(0)。
#   USE_CPU=1 → 强制 CPU（backend=0）。
# 注意: 此开关必须在 import GazeFollower 之前设置——GazeFollower 类定义时会
#       构造默认的 MNN 估计器并读取该值决定走 CPU 还是 GPU。
#
# 当前: CPU 模式 —— 先不考虑 GPU，确保实验可运行。
USE_CPU = 1     # CPU 套：强制 MNN 推理走 CPU（backend=0）
USE_GPU = 0     # GPU 套：MNN 自动探测最佳 GPU 后端（CUDA→Vulkan→OpenCL→CPU）。
# ===========================================================================

import os
import sys
import time

import cv2
import numpy as np
import pygame
from datetime import datetime
from screeninfo import get_monitors

# 限制 OpenCV 并行线程数: 避免视频解码和摄像头同时占用过多 CPU 线程,
# 减轻与主线程的竞争。
cv2.setNumThreads(1)

# 必须在 import GazeFollower 之前确定后端：GazeFollower 类定义时会构造默认 MNN
# 估计器，并读取 GAZEFOLLOWER_MNN_BACKEND 决定走 CPU 还是 GPU。
if USE_CPU == 1 and USE_GPU != 1:
    os.environ["GAZEFOLLOWER_MNN_BACKEND"] = "0"   # 强制 CPU
    _ACTIVE_SET = "CPU（backend=0）"
elif USE_GPU == 1:
    # 不设环境变量 → MGazeNetGazeEstimator 按 [6,4,2,0] 自动探测,
    # 优先 NVIDIA CUDA backend=6, 其次 Vulkan(4)、OpenCL(2), 最后 CPU(0)。
    # RTX 4060 上 CUDA 和 Vulkan 均应可用。
    if "GAZEFOLLOWER_MNN_BACKEND" in os.environ:
        del os.environ["GAZEFOLLOWER_MNN_BACKEND"]
    _ACTIVE_SET = "GPU自动探测（CUDA→Vulkan→OpenCL→CPU）"
else:
    # 两套都置 1 或都置 0：自动选择（CUDA→Vulkan→OpenCL→CPU）
    if "GAZEFOLLOWER_MNN_BACKEND" in os.environ:
        del os.environ["GAZEFOLLOWER_MNN_BACKEND"]
    _ACTIVE_SET = "GPU自动探测（CUDA→Vulkan→OpenCL→CPU）"

from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera
from gazefollower.misc import DefaultConfig
from make_heatmap import render_heatmap


# ===================== 配置区(按需修改) =====================
# 视频位置: 直接改这一行设置要播放的视频文件
# 也可在命令行传参覆盖: python demo.py 你的视频路径.mp4
# 若命令行未传参且默认路径不存在, 脚本会自动提示你输入路径。
VIDEO_PATH = r"E:\HCI+\video\001.mp4"

# 数据记录频率(Hz): 摄像头线程每 1/RECORD_HZ 秒只跑一次脸检测+眼动推理并写一条样本。
# 当前 10Hz(每 0.1 秒一条)。该频率直接决定录制 CSV 的采样密度, 也是播放期 CPU/GIL
# 占用的主要来源 —— 录得越慢, 留给主线程解码/显示视频的算力越多, 视频越流畅。
# 若机器仍偏卡, 可再下调到 8 / 5 Hz; 若想更密的可视化, 可上调(代价是更占 CPU)。
RECORD_HZ = 10

# 摄像头采集分辨率(宽, 高)。越低 → MediaPipe 处理越快 → CPU 负担越小 → 视频越流畅。
# 320x240 为最优平衡(MediaPipe 处理量仅为 640x480 的 1/4, 不影响校准精度)。
# 若校准精度受影响可改回 (640, 480)。
CAM_WIDTH, CAM_HEIGHT = 320, 240

# 每隔多少秒生成一组热力图(默认 5 秒)
WINDOW_SEC = 5

# 是否生成"带视频画面"版热力图(同名 *_video.png)。
# 若机器太慢/觉得没必要, 改成 False: 只出干净版(n.png), 省去每窗口的视频帧拷贝与混合开销。
SAVE_VIDEO_BG = True
# ===========================================================


# ===================== 可视化(录制完成后离线生成, 不在播放期进行) =====================

def _extract_keyframe(cap, t_sec, W, H):
    """从已打开的 VideoCapture 抽取 t_sec 附近的一帧(BGR, 已缩放到 WxH),
    用作带画面版热力图底图。返回 BGR numpy 帧; 失败返回 None。"""
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    except Exception:
        pass
    frame = None
    # 顺读 2 帧, 规避部分编码器 seek 后首帧不准确的问题
    for _ in range(2):
        ret, fr = cap.read()
        if ret and fr is not None:
            frame = fr
        else:
            break
    if frame is None:
        return None
    if frame.shape[1] != W or frame.shape[0] != H:
        frame = cv2.resize(frame, (W, H))
    return frame  # 保持 BGR, render_heatmap 内部会做 BGR2RGB


def _generate_visualizations(out_dir, out_csv, video_path, W, H, window_sec, save_video_bg):
    """录制完成后统一生成热力图(全部离线, 不占用播放期任何算力):
       - 干净版: 直接由 CSV 统计得到, 浅灰底;
       - 带画面版(可选): 从目标视频按窗口中点时间抽取关键帧作底图。
    返回生成的组数(>=0)。
    """
    import csv

    # 1) 解析 CSV
    #    关键修复: 视频起点(trigger==1)的锚定, 必须与"该行注视点是否有效"解耦。
    #    标记视频起点的那一帧往往还没算出有效注视点(filtered_x/y = NA),
    #    若先因无效跳过, 会丢失起点锚, 导致后面所有有效点被一并丢弃(出现 0 张图)。
    #    因此分两遍: 第一遍只找起点; 第二遍才收集有效注视点。

    # 第一遍: 定位视频起点(无视 x/y 是否有效)
    video_start_ns = None
    with open(out_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 14:
                continue
            ts = row[0].strip()
            trigger = row[13].strip()
            if video_start_ns is None and trigger == "1":
                try:
                    video_start_ns = int(ts)
                except ValueError:
                    video_start_ns = None
                    continue
                break  # 找到起点即可, 不必扫完整个文件
    # 兜底: 全程没有 trigger(纯录制、没放视频)时, 退回以首行时间戳为起点
    if video_start_ns is None:
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 14:
                    continue
                try:
                    video_start_ns = int(row[0].strip())
                except ValueError:
                    continue
                break
    if video_start_ns is None:
        print("[热力图] 无法定位起点(CSV 无有效时间戳), 跳过可视化。")
        return 0

    # 第二遍: 收集有效注视点, 按窗口切分
    windows = {}  # win_idx -> list[(x, y)]
    with open(out_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 14:
                continue
            ts, fx, fy = row[0].strip(), row[6].strip(), row[7].strip()
            if fx in ("", "NA") or fy in ("", "NA"):
                continue
            try:
                ts_ns = int(ts)
                x, y = float(fx), float(fy)
            except ValueError:
                continue
            vt = (ts_ns - video_start_ns) / 1e9
            if vt < 0:
                continue  # 视频起点之前的点(校准/等待期)不计入
            wi = int(vt // window_sec)
            windows.setdefault(wi, []).append((x, y))

    if not windows:
        print("[热力图] 没有有效注视点, 跳过可视化。")
        return 0

    cap = None
    if save_video_bg and video_path:
        cap = cv2.VideoCapture(video_path)

    img_count = 0
    for wi in range(max(windows.keys()) + 1):
        pts = windows.get(wi)
        if not pts:
            continue
        img_count += 1
        t0 = wi * window_sec
        t1 = t0 + window_sec
        caption = f"{t0:g}–{t1:g} s · n={len(pts)}"
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        p_clean = os.path.join(out_dir, f"{img_count}.png")
        render_heatmap(xs, ys, W, H, p_clean, bg=None, caption=caption, cmap="inferno")
        if cap is not None:
            bg = _extract_keyframe(cap, (t0 + t1) / 2.0, W, H)
            if bg is not None:
                p_video = os.path.join(out_dir, f"{img_count}_video.png")
                render_heatmap(xs, ys, W, H, p_video, bg=bg, caption=caption, cmap="inferno")
        print(f"[热力图] 已生成 {p_clean}" +
              (f" 和 {p_video}" if cap is not None and bg is not None else ""))

    if cap is not None:
        cap.release()
    return img_count



def _show_text_centered(screen, lines, font, color=(255, 255, 255), bg=(30, 30, 30)):
    """在屏幕中央显示多行文字。"""
    screen.fill(bg)
    line_h = font.get_height()
    total = line_h * len(lines) + 8 * (len(lines) - 1)
    y = (screen.get_height() - total) // 2
    for line in lines:
        surf = font.render(line, True, color)
        rect = surf.get_rect(center=(screen.get_width() // 2, y + line_h // 2))
        screen.blit(surf, rect)
        y += line_h + 8
    pygame.display.flip()


def _resolve_video_path() -> str:
    """确定视频路径：命令行参数 > VIDEO_PATH 配置 > 交互输入。"""
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.isfile(VIDEO_PATH):
        path = VIDEO_PATH
    else:
        print(f"[提示] 默认视频路径不存在：{VIDEO_PATH}")
        path = input("请输入视频文件路径（直接回车跳过则不播放视频）: ").strip()
        if not path:
            print("[提示] 未指定视频，将以黑屏模式记录（仅录注视数据）。")
            return ""
    return path


def main():
    print(f"[模式] 当前 MNN 后端: {_ACTIVE_SET}    |   采样频率: {RECORD_HZ} Hz")
    video_path = _resolve_video_path()
    if video_path and not os.path.isfile(video_path):
        print(f"[警告] 视频文件不存在: {video_path}，将以黑屏模式记录。")
        video_path = ""

    # 单屏: 始终使用主显示器(0 号)
    monitors = get_monitors()
    mon = monitors[0]
    W, H = mon.width, mon.height

    # 用该屏分辨率作为注视坐标的映射基准
    cfg = DefaultConfig()
    cfg.screen_size = np.array([W, H], dtype=np.int32)

    pygame.init()
    gf = GazeFollower(
        camera=WebCamCamera(img_width=CAM_WIDTH, img_height=CAM_HEIGHT),
        config=cfg)

    # 1) 校准(全屏, 按提示完成)
    print("Calibration...")
    gf.calibrate()

    # 2) 全屏窗口, 仅播放视频, 不叠加任何注视点
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Stimulus")

    font = pygame.font.SysFont("Microsoft YaHei", 32)

    _show_text_centered(screen, [
        "Eye-Tracking Recording",
        "",
        "校准已完成。",
        "按 [空格] 开始" + ("播放视频并记录" if video_path else "记录"),
        "视频播完自动结束 / 按 [ESC] 提前结束",
    ], font)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gf.release()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    gf.release()
                    return
                if event.key == pygame.K_SPACE:
                    waiting = False

    # 3) 本次测试的文件夹: 以测试开始时间命名(与 CSV 同名)
    test_start = datetime.now()
    stamp = test_start.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"gaze_{stamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{out_dir}.csv")

    # 校准已完成, 录制前把摄像头的处理/记录频率降到 RECORD_HZ,
    # 释放 CPU 给主线程的视频解码与显示
    gf.camera.set_process_fps(RECORD_HZ)
    gf.start_sampling()

    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[警告] 无法打开视频: {video_path}，将以黑屏模式记录。")
            cap = None
        else:
            # 降低解码缓冲, 避免读到滞后帧(减小播放延迟与抖动)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 30
    if not fps or fps <= 0:
        fps = 30

    clock = pygame.time.Clock()
    video_started = False
    running = True
    rec_start_wall = None        # 视频(记录)开始的墙上时钟

    # 零分配显示: 复用同一块内存 + 单一 surface(与 rgb_buf 共享内存)。
    # 这样每帧不再做 cvtColor 拷贝 + tobytes(6MB) + 新建 surface, 杜绝 GC 抖动(卡顿主因之一)。
    rgb_buf = np.empty((H, W, 3), dtype=np.uint8)
    surf = pygame.image.frombuffer(rgb_buf, (W, H), "RGB")

    # ===== 播放期: 只显示视频 + 记录数据, 绝不做任何可视化(无热力图/无拷贝/无后台线程) =====
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 背景: 视频帧(无注视点叠加)
        if cap:
            ret, frame = cap.read()
            if not ret:
                # 视频自然播完 -> 自动结束
                running = False
                break
            if not video_started:
                video_started = True
                rec_start_wall = time.time()
                gf.send_trigger(1)  # 标记视频开始时刻(CSV 中 trigger=1)

            # 零分配显示: 把 BGR 帧原地翻转通道写入复用的 rgb_buf(BGR->RGB),
            # 再 blit 那个与 rgb_buf 共享内存的 surface, 每帧零额外分配。
            if frame.shape[1] != W or frame.shape[0] != H:
                small = cv2.resize(frame, (W, H))
                rgb_buf[:] = small[:, :, ::-1]
            else:
                rgb_buf[:] = frame[:, :, ::-1]
            screen.blit(surf, (0, 0))  # surf 与 rgb_buf 共享内存, 显示即刚写入的帧
        else:
            if rec_start_wall is None:
                rec_start_wall = time.time()
            screen.fill((0, 0, 0))

        # 左上角录制指示(红点), 不叠加注视点, 避免干扰被试
        pygame.draw.circle(screen, (255, 0, 0), (40, 40), 12)
        pygame.display.flip()

        clock.tick(fps)  # 按视频原始帧率匀速播放

    # ===== 录制收尾(先停记录, 再离线做可视化) =====
    if cap:
        cap.release()
    gf.stop_sampling()
    gf.save_data(out_csv)
    gf.release()

    # ===== 数据记录完成后再统一生成热力图(完全离线, 不占用播放期任何算力) =====
    print("[信息] 数据记录完成, 开始离线生成热力图(不影响刚才的播放流畅度)...")
    img_count = _generate_visualizations(out_dir, out_csv, video_path, W, H,
                                         WINDOW_SEC, SAVE_VIDEO_BG)

    # 结束提示
    _show_text_centered(screen, [
        "记录完成",
        "",
        f"数据: {out_dir}/{out_dir}.csv",
        f"热力图: {out_dir}/1.png ... ({img_count} 组" +
        (" , 另含 *_video.png" if SAVE_VIDEO_BG else " , 仅干净版)"),
    ], font)
    pygame.time.delay(1500)
    pygame.quit()
    print(f"已保存数据到 {out_csv}")
    print(f"已生成 {img_count} 组热力图(每 {WINDOW_SEC} 秒一组, " +
          ("含干净版与带画面版" if SAVE_VIDEO_BG else "仅干净版(未生成视频底图版)") +
          f"), 位于 {out_dir}/")


if __name__ == "__main__":
    main()
