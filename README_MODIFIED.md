# GazeFollower (Modified Version)

> This project is a modified version based on:
> Original repository: https://github.com/GanchengZhu/GazeFollower
>
> The original work is created by Gancheng Zhu et al. and licensed under
> Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
>
> Modifications in this version are made for academic research purposes.

## What this version is

This is a modified build of [GazeFollower](https://github.com/GanchengZhu/GazeFollower), an open-source
deep-learning gaze-tracking system for web cameras. It is adapted for **controlled gaze-tracking
experiments**: the subject watches a fullscreen stimulus video while gaze data is recorded, and
visualizations (attention heatmaps) are generated **offline, after** the recording session.

The core difference vs. upstream: **during playback the program only records data — no on-screen gaze
overlay and no live heatmap rendering** — so the stimulus video plays with zero stutter. All
visualizations are produced in a single offline pass once recording stops.

## Key modifications

- **Offline visualization.** Heatmaps are generated *after* recording finishes
  (`_generate_visualizations` in `demo.py`), never during playback. This keeps the video perfectly smooth.
- **Zero-allocation playback display.** Each video frame is written in place into a reused `rgb_buf`
  and blitted via a single `pygame.Surface` that shares that buffer's memory — no per-frame copy,
  `cvtColor`, `tobytes`, or `Surface` allocation — eliminating GC jitter (the main source of stutter).
- **No gaze overlay on the stimulus.** During recording the screen shows only the video (plus a small
  red "recording" dot); no crosshair / gaze dot is drawn, so the subject is not disturbed.
- **Two-pass CSV parsing (bug fix).** The video-start anchor (`trigger == 1`) is located in a first pass
  independent of gaze validity, then valid gaze points are collected in a second pass. This fixes the
  "0 heatmaps" bug where the start frame had no valid gaze yet and caused all later points to be dropped.
- **Per-window attention heatmaps.** After recording, gaze points are binned into time windows
  (default every 5 s) and rendered as:
  - `N.png` — clean heatmap (light-gray background), and
  - `N_video.png` — the same window over a keyframe extracted from the stimulus video (optional, `SAVE_VIDEO_BG`).
- **CPU / GPU backend switch.** The MNN inference backend is selectable via `USE_CPU` / `USE_GPU` at the
  top of `demo.py` (must be set *before* importing `GazeFollower`). Default is **CPU**; the Vulkan/GPU
  path is kept but disabled here because it is unstable on this machine during calibration. Playback
  smoothness comes from the architecture, not from the backend.
- **MediaPipe face alignment kept as default** (good blink / eyelid-openness detection); the BlazeFace
  alternative is available but not used in this experiment flow.
- **Capture rate limiting.** `WebCamCamera` samples at `process_fps` (default 15) to bound CPU load.

## Recorded data (CSV)

Each row in `gaze_<timestamp>/gaze_<timestamp>.csv` contains:

| Column | Meaning |
| --- | --- |
| `timestamp` | computer clock, nanoseconds |
| `datetime` | human-readable local time |
| raw / calibrated / filtered gaze X,Y | pixel coordinates of the gaze point |
| left / right eye openness | eyelid openness (0–100) |
| tracking state | whether the face / gaze was tracked |
| trigger | event marker (`1` = stimulus video started) |

## How to run

1. Install dependencies (see `requirements.txt` / `setup.py`):
   `MNN`, `mediapipe`, `opencv-python`, `pygame`, `numpy`, `scipy`, `screeninfo`, `pandas`.
2. Edit the config block at the top of `demo.py`:
   - `VIDEO_PATH` — the stimulus video to play (or pass it as a CLI argument).
   - `WINDOW_SEC` — heatmap window length in seconds (default `5`).
   - `SAVE_VIDEO_BG` — also emit `N_video.png` heatmaps over video frames (`True` / `False`).
   - `USE_CPU` / `USE_GPU` — backend selection (default CPU).
3. Run:
   ```bash
   python demo.py                      # uses VIDEO_PATH
   python demo.py path/to/other.mp4    # override video path
   ```
4. Experiment flow:
   1. Fullscreen calibration runs automatically (look at the dots; follow on-screen prompts).
   2. After calibration, a "press [SPACE] to start" screen appears.
   3. Press **[SPACE]** — the stimulus video plays fullscreen and gaze data is recorded.
   4. When the video ends (or you press **[ESC]**) recording stops automatically.
   5. Heatmaps are generated offline into `gaze_<timestamp>/` and a "记录完成" screen is shown.

## Output

```
gaze_2026-07-13_12-00-00/
├── gaze_2026-07-13_12-00-00.csv   # raw recording
├── 1.png, 2.png, ...              # clean heatmaps (one per WINDOW_SEC)
└── 1_video.png, 2_video.png, ...  # heatmaps over video keyframes (if SAVE_VIDEO_BG)
```

## License

This modified version is distributed under the same terms as the original:
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See `LICENSE-CC-BY-NC-SA`.
